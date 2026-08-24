from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

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
        models = ["gemini-2.5-flash", "gemini-1.5-flash"]
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
    ):
        self.api_keys = api_keys if api_keys is not None else _load_api_keys()
        self.models = models if models is not None else _load_models()
        self.temperature = temperature
        self.max_retries_per_attempt = max_retries_per_attempt

        if not self.api_keys:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEYS or GEMINI_API_KEY in .env"
            )

        self._current_key_index = 0

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
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=key, temperature=self.temperature
        )

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