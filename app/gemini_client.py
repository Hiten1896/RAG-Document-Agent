from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("docagent.gemini")


class GeminiExhaustedError(RuntimeError):
    def __init__(self, message: str, kind: str, last_error: Optional[Exception] = None):
        super().__init__(message)
        self.kind = kind
        self.last_error = last_error


def _parse_csv_env(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_api_keys() -> List[str]:
    keys: List[str] = []
    for k in _parse_csv_env(os.getenv("GEMINI_API_KEYS")):
        if k not in keys:
            keys.append(k)
    for single in (os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY")):
        if single and single not in keys:
            keys.append(single)
    return keys


def _load_models() -> List[str]:
    models: List[str] = []
    single = os.getenv("LLM_MODEL")
    if single:
        models.append(single)
    for m in _parse_csv_env(os.getenv("LLM_MODELS")):
        if m not in models:
            models.append(m)
    if not models:
        # `gemini-2.5-flash` and `gemini-1.5-flash` used to be the defaults and
        # both now 404 for newly issued API keys ("no longer available to new
        # users"), which took /api/query down with a 503 on every request.
        #
        # `gemini-3.5-flash` (used here previously) does not exist as a model
        # id at all — every call to it 404'd immediately, so the first model
        # in the list was dead on arrival and every request paid the cost of
        # falling through to the second entry.
        #
        # Order matters more than it looks: the google-genai SDK retries a 429
        # internally with backoff for ~35s before raising, so a first entry that
        # is out of quota adds that delay to every single answer. A concrete,
        # verified, currently-GA model goes first; the floating "latest" alias
        # is the fallback that keeps this from rotting again as Google renames
        # things.
        models = ["gemini-2.5-flash", "gemini-flash-latest"]
    return models


class GeminiClient:
    _TRANSIENT_MARKERS = ("timeout", "timed out", "503", "unavailable", "connection")
    _QUOTA_MARKERS = ("quota", "rate limit", "429", "resource_exhausted")
    _AUTH_MARKERS = ("api key", "unauthenticated", "permission", "invalid_argument")
    _MODEL_MARKERS = ("not_found", "no longer available", "404")

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        temperature: float = 0.2,
        max_retries_per_attempt: int = 2,
        max_output_tokens: int = 1024,
    ):
        self.api_keys = api_keys if api_keys is not None else _load_api_keys()
        self.models = models if models is not None else _load_models()
        self.temperature = temperature
        self.max_retries_per_attempt = max_retries_per_attempt
        # Response latency scales with how many tokens the model generates, so
        # an unbounded max lets one verbose answer take far longer than the
        # question needed. 1024 tokens is generous for a grounded QA answer
        # with citations while cutting off runaway generations that were the
        # single biggest lever on perceived "it's taking too long" latency.
        # `max_tokens` is the constructor kwarg ChatGoogleGenerativeAI actually
        # reads (not `max_output_tokens`, which is the raw Gemini API field
        # name but isn't what this LangChain wrapper accepts).
        self.max_output_tokens = max_output_tokens

        if not self.api_keys:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEYS or GEMINI_API_KEY in .env"
            )

        self._current_key_index = 0
        # (model, key) -> ChatGoogleGenerativeAI. Constructing this object sets
        # up the genai SDK's own HTTP/gRPC transport, which is real, measurable
        # overhead — previously `_client_for` built a fresh one on every single
        # `invoke()` call (i.e. every query), so every answer paid that setup
        # cost on top of the actual model latency. There are at most
        # len(models) * len(api_keys) distinct combinations ever used, so
        # caching them on the singleton means that setup happens once per
        # combination for the life of the process instead of once per query.
        self._clients: Dict[tuple[str, str], ChatGoogleGenerativeAI] = {}

    @staticmethod
    def _classify(message: str) -> str:
        m = message.lower()
        if any(marker in m for marker in GeminiClient._QUOTA_MARKERS):
            return "quota"
        if any(marker in m for marker in GeminiClient._AUTH_MARKERS):
            return "auth"
        if any(marker in m for marker in GeminiClient._MODEL_MARKERS):
            return "model_not_found"
        if any(marker in m for marker in GeminiClient._TRANSIENT_MARKERS):
            return "transient"
        return "unknown"

    def _client_for(self, model: str, key: str) -> ChatGoogleGenerativeAI:
        cache_key = (model, key)
        client = self._clients.get(cache_key)
        if client is None:
            client = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=key,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            self._clients[cache_key] = client
        return client

    def invoke(self, prompt: str):
        last_error: Optional[Exception] = None
        last_kind = "unknown"
        num_keys = len(self.api_keys)
        start_key_index = self._current_key_index
        next_start_key_index = start_key_index

        for model in self.models:
            for offset in range(num_keys):
                key_index = (start_key_index + offset) % num_keys
                key = self.api_keys[key_index]
                client = self._client_for(model, key)

                for attempt in range(1, self.max_retries_per_attempt + 1):
                    try:
                        response = client.invoke(prompt)
                        self._current_key_index = key_index
                        return response
                    except Exception as e:
                        message = str(e)
                        kind = self._classify(message)
                        last_error = e

                        if last_kind == "unknown" or kind != "transient":
                            last_kind = kind

                        if kind == "transient" and attempt < self.max_retries_per_attempt:
                            time.sleep(2 ** (attempt - 1))
                            continue

                        if kind == "quota":
                            next_start_key_index = (key_index + 1) % num_keys
                        break

        self._current_key_index = next_start_key_index
        raise GeminiExhaustedError(
            f"All Gemini models/keys exhausted. Last error: {last_error}",
            kind=last_kind,
            last_error=last_error,
        )