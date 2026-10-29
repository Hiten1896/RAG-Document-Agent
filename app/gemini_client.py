from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import requests
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("docagent.llm")


class GeminiExhaustedError(RuntimeError):
    def __init__(
        self,
        message: str,
        kind: str,
        last_error: Optional[Exception] = None,
        attempts: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.last_error = last_error
        self.attempts = attempts or []


class ProviderUnavailableError(RuntimeError):
    pass


@dataclass
class LLMResult:
    response: Any
    provider: str
    model: str
    attempts: List[Dict[str, Any]]


@dataclass
class LLMStreamResult:
    first_chunk: str
    chunks: Iterator[str]
    provider: str
    model: str
    attempts: List[Dict[str, Any]]


def _chunk_to_text(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def _parse_csv_env(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_api_keys() -> List[str]:
    keys: List[str] = []
    for key in _parse_csv_env(os.getenv("GEMINI_API_KEYS")):
        if key not in keys:
            keys.append(key)
    for single in (os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY")):
        if single and single not in keys:
            keys.append(single)
    return keys


def _load_models() -> List[str]:
    models: List[str] = []
    single = os.getenv("LLM_MODEL")
    if single:
        models.append(single)
    for model in _parse_csv_env(os.getenv("LLM_MODELS")):
        if model not in models:
            models.append(model)
    return models or ["gemini-2.5-flash", "gemini-flash-latest"]


class GeminiClient:
    _TRANSIENT_MARKERS = ("timeout", "timed out", "503", "unavailable", "connection")
    _QUOTA_MARKERS = ("quota", "rate limit", "429", "resource_exhausted")
    _AUTH_MARKERS = ("api key", "unauthenticated", "permission", "invalid_argument", "403")
    _MODEL_MARKERS = ("not_found", "no longer available", "404")

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ):
        self.api_keys = api_keys if api_keys is not None else _load_api_keys()
        self.models = models if models is not None else _load_models()
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        if not self.api_keys:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEYS or GEMINI_API_KEY in .env"
            )
        self._current_key_index = 0
        self._clients: Dict[Tuple[str, str], ChatGoogleGenerativeAI] = {}

    @staticmethod
    def _classify(message: str) -> str:
        lowered = message.lower()
        if any(marker in lowered for marker in GeminiClient._QUOTA_MARKERS):
            return "quota"
        if any(marker in lowered for marker in GeminiClient._AUTH_MARKERS):
            return "auth"
        if any(marker in lowered for marker in GeminiClient._MODEL_MARKERS):
            return "model_not_found"
        if any(marker in lowered for marker in GeminiClient._TRANSIENT_MARKERS):
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
                timeout=15,
                max_retries=0,
            )
            self._clients[cache_key] = client
        return client

    def invoke(self, prompt: str) -> LLMResult:
        last_kind = "unknown"
        last_error: Optional[Exception] = None
        attempts: List[Dict[str, str]] = []
        start_key_index = self._current_key_index
        for model in self.models:
            for offset in range(len(self.api_keys)):
                key_index = (start_key_index + offset) % len(self.api_keys)
                started = time.perf_counter()
                try:
                    response = self._client_for(model, self.api_keys[key_index]).invoke(prompt)
                    attempts.append(
                        {
                            "provider": "gemini",
                            "model": model,
                            "status": "success",
                            "ms": round((time.perf_counter() - started) * 1000),
                        }
                    )
                    self._current_key_index = key_index
                    return LLMResult(response, "gemini", model, attempts)
                except Exception as exc:
                    last_error = exc
                    last_kind = self._classify(str(exc))
                    attempts.append(
                        {
                            "provider": "gemini",
                            "model": model,
                            "status": last_kind,
                            "ms": round((time.perf_counter() - started) * 1000),
                        }
                    )
                    # A model/key gets one call per request; routing decides fallback.
                    continue
        raise GeminiExhaustedError(
            "All Gemini models/keys exhausted.",
            kind=last_kind,
            last_error=last_error,
            attempts=attempts,
        )

    def open_stream(self, prompt: str) -> LLMStreamResult:
        last_kind = "unknown"
        last_error: Optional[Exception] = None
        attempts: List[Dict[str, Any]] = []
        start_key_index = self._current_key_index
        for model in self.models:
            for offset in range(len(self.api_keys)):
                key_index = (start_key_index + offset) % len(self.api_keys)
                started = time.perf_counter()
                iterator: Optional[Iterator[Any]] = None
                try:
                    iterator = iter(
                        self._client_for(model, self.api_keys[key_index]).stream(prompt)
                    )
                    first_chunk = ""
                    for chunk in iterator:
                        first_chunk = _chunk_to_text(chunk)
                        if first_chunk:
                            break
                    if not first_chunk:
                        raise RuntimeError("Gemini stream returned no text.")

                    self._current_key_index = key_index
                    attempts.append(
                        {
                            "provider": "gemini",
                            "model": model,
                            "status": "success",
                            "ms": round((time.perf_counter() - started) * 1000),
                        }
                    )

                    def remaining_chunks(
                        stream_iterator: Iterator[Any] = iterator,
                    ) -> Iterator[str]:
                        try:
                            for chunk in stream_iterator:
                                text = _chunk_to_text(chunk)
                                if text:
                                    yield text
                        finally:
                            close = getattr(stream_iterator, "close", None)
                            if callable(close):
                                close()

                    return LLMStreamResult(
                        first_chunk, remaining_chunks(), "gemini", model, attempts
                    )
                except Exception as exc:
                    if iterator is not None:
                        close = getattr(iterator, "close", None)
                        if callable(close):
                            close()
                    last_error = exc
                    last_kind = self._classify(str(exc))
                    attempts.append(
                        {
                            "provider": "gemini",
                            "model": model,
                            "status": last_kind,
                            "ms": round((time.perf_counter() - started) * 1000),
                        }
                    )
        raise GeminiExhaustedError(
            "All Gemini models/keys exhausted.",
            kind=last_kind,
            last_error=last_error,
            attempts=attempts,
        )


class GroqClient:
    base_url = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str, model: Optional[str], temperature: float = 0.2):
        self.api_key = api_key
        self.temperature = temperature
        self._session = requests.Session()
        self.model = model or self._discover_model()

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _discover_model(self) -> str:
        response = self._session.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=30,
        )
        if response.status_code in {401, 403}:
            raise ProviderUnavailableError("Groq authentication failed.")
        response.raise_for_status()
        models = response.json().get("data", [])
        preferred_markers = ("qwen", "gpt-oss", "llama", "mixtral", "mistral", "gemma")
        excluded_markers = (
            "prompt-guard",
            "safeguard",
            "whisper",
            "orpheus",
            "embed",
            "rerank",
        )
        candidates = [
            item.get("id", "")
            for item in models
            if item.get("id")
            and any(marker in item.get("id", "").lower() for marker in preferred_markers)
            and not any(marker in item.get("id", "").lower() for marker in excluded_markers)
        ]
        if not candidates:
            raise ProviderUnavailableError("Groq returned no general-purpose chat model.")
        return candidates[0]

    def invoke(self, prompt: str) -> LLMResult:
        started = time.perf_counter()
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "temperature": self.temperature,
                "tools": [],
                "max_tokens": 600,
                "reasoning_format": "hidden",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        if response.status_code in {401, 403}:
            raise ProviderUnavailableError("Groq authentication failed.")
        if response.status_code == 429:
            raise ProviderUnavailableError("Groq rate limit reached.")
        if response.status_code >= 500:
            raise ProviderUnavailableError("Groq service unavailable.")
        response.raise_for_status()
        return LLMResult(
            response.json()["choices"][0]["message"]["content"],
            "groq",
            self.model,
            [
                {
                    "provider": "groq",
                    "model": self.model,
                    "status": "success",
                    "ms": round((time.perf_counter() - started) * 1000),
                }
            ],
        )

    def open_stream(self, prompt: str) -> LLMStreamResult:
        started = time.perf_counter()
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "temperature": self.temperature,
                "tools": [],
                "max_tokens": 600,
                "reasoning_format": "hidden",
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=(15, 60),
            stream=True,
        )
        try:
            if response.status_code in {401, 403}:
                raise ProviderUnavailableError("Groq authentication failed.")
            if response.status_code == 429:
                raise ProviderUnavailableError("Groq rate limit reached.")
            if response.status_code >= 500:
                raise ProviderUnavailableError("Groq service unavailable.")
            response.raise_for_status()

            def read_chunks() -> Iterator[str]:
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    payload = json.loads(data)
                    choices = payload.get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("delta") or {}).get("content")
                    if isinstance(content, str) and content:
                        yield content
                    elif isinstance(content, list):
                        text = "".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict)
                            and isinstance(block.get("text"), str)
                        )
                        if text:
                            yield text

            iterator = iter(read_chunks())
            first_chunk = next(iterator, "")
            if not first_chunk:
                raise ProviderUnavailableError("Groq stream returned no text.")

            def remaining_chunks() -> Iterator[str]:
                try:
                    yield from iterator
                finally:
                    response.close()

            return LLMStreamResult(
                first_chunk,
                remaining_chunks(),
                "groq",
                self.model,
                [
                    {
                        "provider": "groq",
                        "model": self.model,
                        "status": "success",
                        "ms": round((time.perf_counter() - started) * 1000),
                    }
                ],
            )
        except Exception:
            response.close()
            raise


class LLMClient:
    def __init__(self):
        self.temperature = 0.2
        self.provider_order = [
            item.lower()
            for item in _parse_csv_env(os.getenv("LLM_PROVIDER_ORDER", "gemini,groq"))
            if item.lower() in {"gemini", "groq"}
        ] or ["gemini", "groq"]
        self.gemini: Optional[GeminiClient] = None
        self.groq: Optional[GroqClient] = None
        self._disabled_until: Dict[str, float] = {}
        self._disabled_until_restart: set[str] = set()
        if _load_api_keys():
            self.gemini = GeminiClient()
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            try:
                self.groq = GroqClient(groq_key, os.getenv("GROQ_MODEL") or None, self.temperature)
            except (ProviderUnavailableError, requests.RequestException) as exc:
                logger.warning("Groq provider unavailable at startup: %s", type(exc).__name__)
        if self.gemini is None and self.groq is None:
            raise RuntimeError("No LLM provider is configured. Set a Gemini or Groq API key.")
        configured = [name for name, client in (("gemini", self.gemini), ("groq", self.groq)) if client]
        logger.info("LLM providers configured: %s", ", ".join(configured))

    def _provider_client(self, provider: str):
        return self.gemini if provider == "gemini" else self.groq

    def invoke(
        self,
        prompt: str,
        groq_prompt: Optional[str] = None,
        forced_provider: Optional[str] = None,
    ) -> LLMResult:
        providers = [forced_provider] if forced_provider else self.provider_order
        attempts: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None
        for provider in providers:
            client = self._provider_client(provider)
            if client is None or provider in self._disabled_until_restart:
                continue
            if self._disabled_until.get(provider, 0) > time.monotonic():
                continue
            model = getattr(client, "model", "configured")
            attempt_started = time.perf_counter()
            try:
                result = client.invoke(groq_prompt if provider == "groq" and groq_prompt else prompt)
                result.attempts = attempts + result.attempts
                return result
            except GeminiExhaustedError as exc:
                last_error = exc
                attempts.extend(exc.attempts)
                if exc.kind == "quota":
                    self._disabled_until[provider] = time.monotonic() + 600
                    break
                elif exc.kind == "auth":
                    self._disabled_until_restart.add(provider)
            except (ProviderUnavailableError, requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                status = "429" if "rate limit" in str(exc).lower() else "unavailable"
                attempts.append(
                    {
                        "provider": provider,
                        "model": model,
                        "status": status,
                        "ms": round((time.perf_counter() - attempt_started) * 1000),
                    }
                )
                if status == "429":
                    self._disabled_until[provider] = time.monotonic() + 600
                elif "authentication" in str(exc).lower():
                    self._disabled_until_restart.add(provider)
        raise ProviderUnavailableError("All configured LLM providers are unavailable.") from last_error

    def open_stream(
        self, prompt: str, groq_prompt: Optional[str] = None
    ) -> LLMStreamResult:
        attempts: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None
        for provider in self.provider_order:
            client = self._provider_client(provider)
            if client is None or provider in self._disabled_until_restart:
                continue
            if self._disabled_until.get(provider, 0) > time.monotonic():
                continue

            model = getattr(client, "model", "configured")
            attempt_started = time.perf_counter()
            try:
                provider_prompt = (
                    groq_prompt
                    if provider == "groq" and groq_prompt
                    else prompt
                )
                result = client.open_stream(provider_prompt)
                result.attempts = attempts + result.attempts
                return result
            except GeminiExhaustedError as exc:
                last_error = exc
                attempts.extend(exc.attempts)
                if exc.kind == "quota":
                    self._disabled_until[provider] = time.monotonic() + 600
                elif exc.kind == "auth":
                    self._disabled_until_restart.add(provider)
            except (ProviderUnavailableError, requests.RequestException) as exc:
                last_error = exc
                status = "429" if "rate limit" in str(exc).lower() else "unavailable"
                attempts.append(
                    {
                        "provider": provider,
                        "model": model,
                        "status": status,
                        "ms": round((time.perf_counter() - attempt_started) * 1000),
                    }
                )
                if status == "429":
                    self._disabled_until[provider] = time.monotonic() + 600
                elif "authentication" in str(exc).lower():
                    self._disabled_until_restart.add(provider)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "%s stream failed before its first text chunk: %s",
                    provider,
                    type(exc).__name__,
                )
                attempts.append(
                    {
                        "provider": provider,
                        "model": model,
                        "status": "unavailable",
                        "ms": round((time.perf_counter() - attempt_started) * 1000),
                    }
                )
        raise ProviderUnavailableError(
            "All configured LLM providers are unavailable."
        ) from last_error
