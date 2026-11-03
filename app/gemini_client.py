from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from collections import deque
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import requests
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("docagent.llm")


class ProviderLimiter:
    """Small process-local token bucket; never waits longer than ten seconds."""

    def __init__(self, provider: str):
        self.provider = provider
        try:
            rpm = float(os.getenv(f"{provider.upper()}_RPM", "8" if provider == "gemini" else "20"))
        except ValueError:
            rpm = 8 if provider == "gemini" else 20
        self.interval = 60.0 / max(0.1, rpm)
        self._calls: deque[float] = deque()
        self._lock = Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60:
                    self._calls.popleft()
                wait = max(0.0, (self._calls[-1] + self.interval - now) if self._calls else 0.0)
                if wait <= 10:
                    if wait:
                        time.sleep(wait)
                    self._calls.append(time.monotonic())
                    return
            raise ProviderUnavailableError(f"{self.provider.title()} is busy; retry in {max(1, round(wait))} s.", "busy", wait)

    def recent(self) -> int:
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= 60:
                self._calls.popleft()
            return len(self._calls)


class GeminiExhaustedError(RuntimeError):
    def __init__(
        self,
        message: str,
        kind: str,
        last_error: Optional[Exception] = None,
        attempts: Optional[List[Dict[str, Any]]] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.last_error = last_error
        self.attempts = attempts or []
        self.retry_after = retry_after


class ProviderUnavailableError(RuntimeError):
    def __init__(self, message: str, kind: str = "unavailable", retry_after: Optional[float] = None):
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after


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


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite"
).strip()
GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"


def _retry_after(response: requests.Response) -> Optional[float]:
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value else None
    except ValueError:
        return None


class GeminiClient:
    _TRANSIENT_MARKERS = (
        "timeout",
        "timed out",
        "503",
        "504",
        "gateway timeout",
        "unavailable",
        "connection",
    )
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
        self.models = list(dict.fromkeys(models or [GEMINI_MODEL]))
        self.model = self.models[0]
        self.fallback_model = (
            GEMINI_FALLBACK_MODEL
            if GEMINI_FALLBACK_MODEL not in self.models
            else ""
        )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        if not self.api_keys:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEYS or GEMINI_API_KEY in .env"
            )
        self._current_key_index = 0
        self._clients: Dict[Tuple[str, str], ChatGoogleGenerativeAI] = {}
        self._check_models()

    def _check_models(self) -> None:
        try:
            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": self.api_keys[0], "pageSize": 1000},
                timeout=15,
            )
            response.raise_for_status()
            available = {
                str(item.get("name", "")).removeprefix("models/")
                for item in response.json().get("models", [])
            }
            configured = [model for model in self.models + [self.fallback_model] if model]
            missing = [model for model in configured if model not in available]
            if missing:
                flash_models = sorted(
                    model for model in available
                    if "flash" in model.lower() or "flash-lite" in model.lower()
                )
                logger.warning(
                    "Configured Gemini model(s) not found: %s. Available flash/flash-lite "
                    "models: %s",
                    ", ".join(missing),
                    ", ".join(flash_models) or "(none reported)",
                )
        except requests.RequestException as exc:
            logger.warning("Could not check Gemini models at startup: %s", type(exc).__name__)

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

    def invoke(self, prompt: str, feature: str = "unknown") -> LLMResult:
        last_kind = "unknown"
        last_error: Optional[Exception] = None
        attempts: List[Dict[str, str]] = []
        start_key_index = self._current_key_index
        for model in self.models + ([self.fallback_model] if self.fallback_model else []):
            for offset in range(min(1, len(self.api_keys))):
                key_index = (start_key_index + offset) % len(self.api_keys)
                started = time.perf_counter()
                try:
                    response = self._client_for(model, self.api_keys[key_index]).invoke(prompt)
                    logger.info(
                        "llm_call feature=%s provider=gemini model=%s attempt=%s "
                        "prompt_chars=%s status=200",
                        feature,
                        model,
                        len(attempts) + 1,
                        len(prompt),
                    )
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
                    logger.info("llm_call feature=%s provider=gemini model=%s attempt=%s prompt_chars=%s status=%s", feature, model, len(attempts) + 1, len(prompt), GeminiClient._classify(str(exc)))
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
                    if last_kind != "model_not_found":
                        raise GeminiExhaustedError(
                            "Gemini request failed.",
                            kind=last_kind,
                            last_error=last_error,
                            attempts=attempts,
                        )
                    if model == self.fallback_model:
                        break
        raise GeminiExhaustedError(
            "All Gemini models/keys exhausted.",
            kind=last_kind,
            last_error=last_error,
            attempts=attempts,
            retry_after=60 if last_kind == "quota" else None,
        )

    def open_stream(self, prompt: str, feature: str = "query") -> LLMStreamResult:
        last_kind = "unknown"
        last_error: Optional[Exception] = None
        attempts: List[Dict[str, Any]] = []
        start_key_index = self._current_key_index
        for model in self.models + ([self.fallback_model] if self.fallback_model else []):
            for offset in range(min(1, len(self.api_keys))):
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

                    logger.info("llm_call feature=%s provider=gemini model=%s attempt=%s prompt_chars=%s status=200", feature, model, len(attempts) + 1, len(prompt))

                    self._current_key_index = key_index
                    attempts.append(
                        {
                            "provider": "gemini",
                            "model": model,
                            "status": "success",
                            "ms": round((time.perf_counter() - started) * 1000),
                        }
                    )
                    if last_kind != "model_not_found":
                        raise GeminiExhaustedError(
                            "Gemini stream request failed.",
                            kind=last_kind,
                            last_error=last_error,
                            attempts=attempts,
                        )
                    if model == self.fallback_model:
                        break

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
            retry_after=60 if last_kind == "quota" else None,
        )


class GroqClient:
    base_url = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str, model: Optional[str], temperature: float = 0.2):
        self.api_key = api_key
        self.temperature = temperature
        self._session = requests.Session()
        self.model = model or GROQ_DEFAULT_MODEL
        self.fallback_model = None
        self._check_model()

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _models(self) -> List[str]:
        response = self._session.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=30,
        )
        if response.status_code in {401, 403}:
            raise ProviderUnavailableError("Groq authentication failed.", "auth")
        if response.status_code == 429:
            raise ProviderUnavailableError("Groq rate limit reached.", "quota", _retry_after(response))
        response.raise_for_status()
        return [
            item.get("id", "")
            for item in response.json().get("data", [])
            if item.get("id")
        ]

    def _check_model(self) -> None:
        try:
            models = self._models()
        except (ProviderUnavailableError, requests.RequestException) as exc:
            logger.warning("Could not check Groq models at startup: %s", type(exc).__name__)
            return
        if self.model not in models:
            logger.warning(
                "Configured Groq model not found: %s. Available models include: %s",
                self.model,
                ", ".join(models[:20]) or "(none reported)",
            )

    def _discover_model(self) -> str:
        models = self._models()
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

    def invoke(self, prompt: str, feature: str = "unknown") -> LLMResult:
        started = time.perf_counter()
        response = self._post(prompt, feature, retry_5xx=True)
        return LLMResult(
            response.json()["choices"][0]["message"]["content"],
            "groq", self.model,
            [{"provider": "groq", "model": self.model, "status": "success", "ms": round((time.perf_counter() - started) * 1000)}],
        )

    def _post(self, prompt: str, feature: str, retry_5xx: bool = False) -> requests.Response:
        for attempt in range(2 if retry_5xx else 1):
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
            logger.info("llm_call feature=%s provider=groq model=%s attempt=%s prompt_chars=%s status=%s", feature, self.model, attempt + 1, len(prompt), response.status_code)
            if response.status_code == 429:
                raise ProviderUnavailableError("Groq rate limit reached.", "quota", _retry_after(response))
            if response.status_code == 404:
                raise ProviderUnavailableError("Groq model was not found.", "model_not_found")
            if response.status_code >= 500 and attempt == 0:
                continue
            if response.status_code in {401, 403}:
                raise ProviderUnavailableError("Groq authentication failed.", "auth")
            if response.status_code >= 500:
                raise ProviderUnavailableError("Groq service unavailable.", "transient")
            response.raise_for_status()
            return response
        return response

    def open_stream(self, prompt: str, feature: str = "query") -> LLMStreamResult:
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
        logger.info("llm_call feature=%s provider=groq model=%s attempt=1 prompt_chars=%s status=%s", feature, self.model, len(prompt), response.status_code)
        try:
            if response.status_code in {401, 403}:
                raise ProviderUnavailableError("Groq authentication failed.", "auth")
            if response.status_code == 429:
                raise ProviderUnavailableError("Groq rate limit reached.", "quota", _retry_after(response))
            if response.status_code == 404:
                raise ProviderUnavailableError("Groq model was not found.", "model_not_found")
            if response.status_code >= 500:
                raise ProviderUnavailableError("Groq service unavailable.", "transient")
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
        self._last_error_kind: Dict[str, str] = {}
        self._limiters = {"gemini": ProviderLimiter("gemini"), "groq": ProviderLimiter("groq")}
        self._events: Dict[str, deque[Tuple[float, str]]] = {"gemini": deque(), "groq": deque()}
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
        feature: str = "unknown",
    ) -> LLMResult:
        providers = [forced_provider] if forced_provider else self.provider_order
        attempts: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None
        last_retry_after: Optional[float] = None
        for provider in providers:
            client = self._provider_client(provider)
            if client is None or provider in self._disabled_until_restart:
                continue
            if self._disabled_until.get(provider, 0) > time.monotonic():
                continue
            model = getattr(client, "model", "configured")
            attempt_started = time.perf_counter()
            try:
                limiter = getattr(self, "_limiters", {}).get(provider)
                if limiter is not None:
                    limiter.acquire()
                events = getattr(self, "_events", {}).get(provider)
                if events is not None:
                    events.append((time.monotonic(), feature))
                result = client.invoke(groq_prompt if provider == "groq" and groq_prompt else prompt, feature)
                result.attempts = attempts + result.attempts
                return result
            except GeminiExhaustedError as exc:
                last_error = exc
                attempts.extend(exc.attempts)
                self._last_error_kind[provider] = exc.kind
                if exc.kind == "quota":
                    retry_after = exc.retry_after or 60
                    last_retry_after = retry_after
                    self._disabled_until[provider] = time.monotonic() + retry_after
                elif exc.kind == "auth":
                    self._disabled_until_restart.add(provider)
                elif exc.kind == "unknown":
                    raise exc.last_error or exc
            except (ProviderUnavailableError, requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                kind = getattr(exc, "kind", "transient")
                if kind == "unavailable" and "rate limit" in str(exc).lower():
                    kind = "quota"
                last_retry_after = getattr(exc, "retry_after", None)
                self._last_error_kind[provider] = kind
                attempts.append(
                    {
                        "provider": provider,
                        "model": model,
                        "status": kind,
                        "ms": round((time.perf_counter() - attempt_started) * 1000),
                    }
                )
                if kind == "quota":
                    self._disabled_until[provider] = time.monotonic() + max(1, min(getattr(exc, "retry_after", None) or 60, 3600))
                elif kind == "auth" or "authentication" in str(exc).lower():
                    self._disabled_until_restart.add(provider)
        kinds = [self._last_error_kind.get(provider) for provider in providers if self._last_error_kind.get(provider)]
        kind = "quota" if kinds and all(item == "quota" for item in kinds) else ("auth" if kinds and all(item == "auth" for item in kinds) else "unavailable")
        raise ProviderUnavailableError(
            "All configured LLM providers are unavailable.", kind, last_retry_after
        ) from last_error

    def status(self) -> Dict[str, Dict[str, Any]]:
        now = time.monotonic()
        by_feature: Dict[str, Dict[str, int]] = {}
        for provider, events in self._events.items():
            while events and now - events[0][0] >= 60:
                events.popleft()
            counts: Dict[str, int] = {}
            for _, feature in events:
                counts[feature] = counts.get(feature, 0) + 1
            by_feature[provider] = counts
        return {
            provider: {
                "configured": self._provider_client(provider) is not None,
                "configured_model": getattr(self._provider_client(provider), "model", None),
                "fallback_model": getattr(self._provider_client(provider), "fallback_model", None),
                "blocked_until": (time.time() + max(0, until - now)) if (until := self._disabled_until.get(provider, 0)) > now else None,
                "last_error_kind": self._last_error_kind.get(provider),
                "calls_last_60s": self._limiters[provider].recent(),
                "calls_last_60s_by_feature": by_feature[provider],
            }
            for provider in ("gemini", "groq")
        }

    def open_stream(
        self, prompt: str, groq_prompt: Optional[str] = None, feature: str = "query"
    ) -> LLMStreamResult:
        attempts: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None
        last_retry_after: Optional[float] = None
        if not hasattr(self, "_last_error_kind"):
            self._last_error_kind = {}
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
                limiter = getattr(self, "_limiters", {}).get(provider)
                if limiter is not None:
                    limiter.acquire()
                events = getattr(self, "_events", {}).get(provider)
                if events is not None:
                    events.append((time.monotonic(), feature))
                try:
                    result = client.open_stream(provider_prompt, feature)
                except TypeError as exc:
                    if "positional argument" not in str(exc):
                        raise
                    result = client.open_stream(provider_prompt)
                result.attempts = attempts + result.attempts
                return result
            except GeminiExhaustedError as exc:
                last_error = exc
                attempts.extend(exc.attempts)
                self._last_error_kind[provider] = exc.kind
                if exc.kind == "quota":
                    retry_after = exc.retry_after or 60
                    last_retry_after = retry_after
                    self._disabled_until[provider] = time.monotonic() + retry_after
                elif exc.kind == "auth":
                    self._disabled_until_restart.add(provider)
            except (ProviderUnavailableError, requests.RequestException) as exc:
                last_error = exc
                kind = getattr(exc, "kind", "transient")
                if kind == "unavailable" and "rate limit" in str(exc).lower():
                    kind = "quota"
                last_retry_after = getattr(exc, "retry_after", None)
                self._last_error_kind[provider] = kind
                status = "429" if kind == "quota" else kind
                attempts.append(
                    {
                        "provider": provider,
                        "model": model,
                        "status": status,
                        "ms": round((time.perf_counter() - attempt_started) * 1000),
                    }
                )
                if kind == "quota":
                    self._disabled_until[provider] = time.monotonic() + max(1, min(last_retry_after or 60, 3600))
                elif "authentication" in str(exc).lower():
                    self._disabled_until_restart.add(provider)
            except Exception as exc:
                last_error = exc
                self._last_error_kind[provider] = "unavailable"
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
        kinds = [
            self._last_error_kind.get(provider)
            for provider in self.provider_order
            if self._last_error_kind.get(provider)
        ]
        kind = "quota" if kinds and all(item == "quota" for item in kinds) else (
            "auth" if kinds and all(item == "auth" for item in kinds) else "unavailable"
        )
        raise ProviderUnavailableError(
            "All configured LLM providers are unavailable.", kind, last_retry_after
        ) from last_error
