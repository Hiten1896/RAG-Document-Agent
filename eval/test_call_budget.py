"""Smoke simulation for the user action to provider-call contract."""

from collections import Counter

from app.gemini_client import LLMClient, LLMResult


class FakeProvider:
    model = "test-model"

    def __init__(self, calls):
        self.calls = calls

    def invoke(self, prompt, feature):
        self.calls.append(feature)
        return LLMResult("{}", "fake", self.model, [{"provider": "fake", "model": self.model, "status": "success"}])


def main() -> None:
    calls = []
    client = object.__new__(LLMClient)
    client.provider_order = ["gemini"]
    client.gemini = FakeProvider(calls)
    client.groq = None
    client._disabled_until = {}
    client._disabled_until_restart = set()
    client._last_error_kind = {}
    client._limiters = {"gemini": type("Limiter", (), {"acquire": lambda self: None, "recent": lambda self: len(calls)})()}
    client._events = {"gemini": []}

    client.invoke("question", feature="query")
    client.invoke("document context", feature="mindmap")
    client.invoke("answered question", feature="followup")

    counts = Counter(calls)
    expected = {"query": 1, "mindmap": 1, "followup": 1}
    assert dict(counts) == expected, f"expected {expected}, got {dict(counts)}"
    print(f"provider calls: {dict(counts)}")
    print("PASS: query=1, mindmap=1, follow-up=1")


if __name__ == "__main__":
    main()
