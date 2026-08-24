"""
Gemini answer-generation client: key rotation, model fallback, and retry.

The old setup was a single `ChatGoogleGenerativeAI` instance built once at
import time from one hardcoded model name and one API key. That was fine
until it wasn't — twice now, Google has retired the configured model out
from under this project (1.5-flash, then 2.5-flash), and a single key
means the first 429 from Gemini's free-tier quota takes down every
question until someone notices and waits it out.

This module fixes both:

  - Multiple keys, tried in order. A quota/rate-limit error on the
    current key rotates to the next one before failing, instead of
    surfacing the 429 to the user immediately. One exhausted free-tier
    key no longer means the app is down if a second key is configured.
  - Multiple models, tried in order. A "model not found / retired"
    error falls through to the next configured model. If Google retires
    the primary model again, the app degrades to the fallback instead
    of breaking outright.
  - Retry with backoff on transient failures (5xx, timeouts) before
    exhausting a key/model and moving to the next one — a single dropped
    connection shouldn't burn through the whole rotation.

This intentionally mirrors the resilience pattern already used for Jina
embeddings in main.py (batched retries with backoff) rather than
inventing a new one — same shape, adapted to Gemini's specific error
types (quota vs. auth vs. model-not-found need different responses).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("docagent.gemini")


class GeminiExhaustedError(RuntimeError):
    """Raised when every configured (key, model) combination failed.
    Carries the last underlying exception and a rough classification so
    the caller (main.py's /query handler) can map it to the right HTTP
    status without re-parsing error text itself."""

    def __init__(self, message: str, kind: str, last_error: Optional[Exception] = None):
        super().__init__(message)
        self.kind = kind  # "quota" | "auth" | "model_not_found" | "unknown"
        self.last_error = last_error


def _parse_csv_env(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_api_keys() -> List[str]:
    """GEMINI_API_KEYS (plural, comma-separated) is the primary config for
    rotation. GEMINI_API_KEY / GOOGLE_API_KEY (singular, either name) are
    kept working for backward compatibility with existing .env files —
    both are folded in if present, duplicates removed, order preserved."""
    keys: List[str] = []
    for k in _parse_csv_env(os.getenv("GEMINI_API_KEYS")):
        if k not in keys:
            keys.append(k)
    for single in (os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY")):
        if single and single not in keys:
            keys.append(single)
    return keys


def _load_models() -> List[str]:
    """LLM_MODELS (plural, comma-separated, ordered) is the primary config
    for fallback. LLM_MODEL (singular) still works and is prepended if not
    already present in LLM_MODELS, keeping older .env files valid.

    The default list carries one fallback beyond the primary so a fresh
    checkout survives a single Google retirement without any .env edit at
    all — that's exactly the failure this project has already hit twice.
    """
    models: List[str] = []
    single = os.getenv("LLM_MODEL")
    if single:
        models.append(single)
    for m in _parse_csv_env(os.getenv("LLM_MODELS")):
        if m not in models:
            models.append(m)
    if not models:
        models = ["gemini-3.6-flash", "gemini-2.5-flash"]
    return models


@dataclass
class _Attempt:
    key_index: int
    model: str


class GeminiClient:
    """Answer-generation client with key rotation and model fallback.

    Usage mirrors a single `ChatGoogleGenerativeAI`: call `.invoke(prompt)`
    and get back a message object with `.text`/`.content`, same as before.
    Internally it walks (model, key) combinations — outer loop over
    models (so a retired model is abandoned quickly across all keys),
    inner loop over keys (so a quota hit rotates within the same model
    before falling back to a different one) — retrying transient errors
    with backoff at each combination before moving on.
    """

    # Transient error signals worth retrying in place before rotating —
    # a dropped connection or a momentary 503 is not the same problem as
    # an exhausted quota or a retired model, and doesn't deserve burning
    # through the whole key/model list on the first hiccup.
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
                "No Gemini API key found. Set GEMINI_API_KEYS (comma-separated "
                "for multiple keys) or GEMINI_API_KEY / GOOGLE_API_KEY in .env "
                "to a key from https://aistudio.google.com/."
            )

        # Which key index to start from — advances on a quota hit so the
        # NEXT query starts from the key that's still known-good, rather
        # than re-trying the exhausted one first every time.
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
        return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=self.temperature)

    def invoke(self, prompt: str):
        """Try every (model, key) combination in order, retrying transient
        errors in place, until one succeeds or all are exhausted. Raises
        GeminiExhaustedError with the most informative failure classification
        seen across the whole attempt (quota/auth/model_not_found win over
        a generic "unknown", since those give the caller something
        actionable to tell the user)."""
        last_error: Optional[Exception] = None
        last_kind = "unknown"
        num_keys = len(self.api_keys)
        # Base index for the rotation, fixed for the duration of this whole
        # invoke() call. `self._current_key_index` is only updated once, at
        # the very end (success or exhaustion) — mutating it mid-rotation
        # would shift the base that `offset` is computed against and skip
        # or repeat keys instead of visiting each exactly once.
        start_key_index = self._current_key_index
        next_start_key_index = start_key_index

        for model in self.models:
            # Start each model's key rotation from wherever the previous
            # quota hit left off, not always index 0 — see
            # _current_key_index above.
            for offset in range(num_keys):
                key_index = (start_key_index + offset) % num_keys
                key = self.api_keys[key_index]
                client = self._client_for(model, key)

                for attempt in range(1, self.max_retries_per_attempt + 1):
                    try:
                        response = client.invoke(prompt)
                        # Success — remember this key as the one to prefer
                        # next time, so a later transient failure on an
                        # earlier key doesn't keep re-triggering rotation.
                        self._current_key_index = key_index
                        return response
                    except Exception as e:
                        message = str(e)
                        kind = self._classify(message)
                        last_error = e
                        # Prefer a more specific classification over a
                        # vaguer one already recorded, so the final error
                        # shown to the user reflects the most useful signal
                        # seen across the whole rotation.
                        if last_kind == "unknown" or kind != "transient":
                            last_kind = kind

                        if kind == "transient" and attempt < self.max_retries_per_attempt:
                            wait = 2 ** (attempt - 1)
                            logger.warning(
                                "Gemini call transient failure (model=%s, key=#%d, attempt %d/%d): %s "
                                "— retrying in %ds",
                                model, key_index, attempt, self.max_retries_per_attempt, message, wait,
                            )
                            time.sleep(wait)
                            continue

                        # Not transient (or out of in-place retries) —
                        # stop retrying this exact (model, key) and let the
                        # outer loops decide what to try next.
                        logger.warning(
                            "Gemini call failed (model=%s, key=#%d, kind=%s): %s",
                            model, key_index, kind, message,
                        )
                        if kind == "quota":
                            # Remember to start the NEXT invoke() call from
                            # the following key — recorded locally and only
                            # committed to self._current_key_index once,
                            # after this whole call finishes, so it can't
                            # perturb the offset arithmetic above.
                            next_start_key_index = (key_index + 1) % num_keys
                        break  # stop retrying this (model, key); try the next one

        # Every (model, key) combination failed for every model.
        self._current_key_index = next_start_key_index
        raise GeminiExhaustedError(
            f"All Gemini models/keys exhausted. Last error: {last_error}",
            kind=last_kind,
            last_error=last_error,
        )