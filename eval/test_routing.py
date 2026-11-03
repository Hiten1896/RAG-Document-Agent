import unittest
from collections import deque
from unittest.mock import patch

from app.gemini_client import (
    GeminiClient,
    GeminiExhaustedError,
    LLMClient,
    LLMResult,
    ProviderLimiter,
    ProviderUnavailableError,
)


class FakeProvider:
    model = "fake-model"
    fallback_model = None

    def __init__(self, response="ok", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def invoke(self, prompt, feature):
        self.calls.append(feature)
        if self.error:
            raise self.error
        return LLMResult(
            self.response,
            "fake",
            self.model,
            [{"provider": "fake", "model": self.model, "status": "success"}],
        )


def router(gemini, groq):
    client = object.__new__(LLMClient)
    client.routes = {"text": ["groq", "gemini"], "visual": ["gemini", "groq"]}
    client.provider_order = client.routes["text"]
    client.gemini = gemini
    client.groq = groq
    client._disabled_until = {}
    client._disabled_until_restart = set()
    client._last_error_kind = {}
    client._limiters = {
        "gemini": type("Limiter", (), {"acquire": lambda self: None})(),
        "groq": type("Limiter", (), {"acquire": lambda self: None})(),
    }
    client._events = {"gemini": [], "groq": []}
    return client


class RoutingTests(unittest.TestCase):
    def test_text_prefers_groq_and_visual_prefers_gemini(self):
        gemini = FakeProvider()
        groq = FakeProvider()
        client = router(gemini, groq)

        self.assertEqual(client.invoke("q", feature="query").provider, "fake")
        self.assertEqual(groq.calls, ["query"])
        self.assertEqual(client.invoke("v", feature="diagram").provider, "fake")
        self.assertEqual(gemini.calls, ["diagram"])

    def test_limiter_skips_wait_over_eight_seconds(self):
        limiter = ProviderLimiter("test", rpm=4, max_wait=8)
        limiter.acquire()
        with self.assertRaises(ProviderUnavailableError) as raised:
            limiter.acquire()
        self.assertEqual(raised.exception.kind, "busy")
        self.assertGreater(raised.exception.retry_after, 8)

    def test_gemini_429_uses_second_key_same_model(self):
        class FakeGeminiProvider:
            def __init__(self, key):
                self.key = key

            def invoke(self, prompt):
                if self.key == "key-1":
                    raise RuntimeError("429 resource_exhausted")
                return "second key answered"

        with patch.object(GeminiClient, "_check_models"):
            client = GeminiClient(api_keys=["key-1", "key-2"], models=["model"])
        client._client_for = lambda model, key: FakeGeminiProvider(key)
        result = client.invoke("prompt", "query")
        self.assertEqual(result.response, "second key answered")
        self.assertEqual(result.model, "model")

    def test_gemini_rpd_skips_capped_pair_and_resets_at_pacific_midnight(self):
        with patch.object(GeminiClient, "_check_models"):
            client = GeminiClient(api_keys=["key-1"], models=["model"])
        client._model_rpd = {"model": 1}
        client._limiters[(0, "model")] = type(
            "Limiter", (), {"acquire": lambda self: None, "interval": 5, "recent": lambda self: 0}
        )()
        client._daily_calls.setdefault((0, "model"), deque()).append(1000)
        fake_provider = FakeProvider(response="should not be called")
        client._client_for = lambda model, key: fake_provider

        with patch("app.gemini_client._pacific_day_start", return_value=900):
            status = client.status()
            self.assertEqual(status["pairs"][0]["calls_today"], 1)
            self.assertEqual(status["pairs"][0]["rpd_limit"], 1)
            self.assertEqual(status["pairs"][0]["remaining"], 0)
            with self.assertRaises(GeminiExhaustedError):
                client.invoke("prompt", "query")
        self.assertEqual(fake_provider.calls, [])

        with patch("app.gemini_client._pacific_day_start", return_value=1100):
            status = client.status()
            self.assertEqual(status["pairs"][0]["calls_today"], 0)
            self.assertEqual(status["pairs"][0]["remaining"], 1)

    def test_all_gemini_exhausted_falls_to_groq(self):
        gemini = FakeProvider(
            error=GeminiExhaustedError("quota", "quota", retry_after=60)
        )
        groq = FakeProvider()
        result = router(gemini, groq).invoke("visual", feature="diagram")
        self.assertEqual(result.provider, "fake")
        self.assertEqual(gemini.calls, ["diagram"])
        self.assertEqual(groq.calls, ["diagram"])


if __name__ == "__main__":
    unittest.main()
