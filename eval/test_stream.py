import asyncio
import json
import time
import unittest
from unittest.mock import patch

from fastapi import Request
from fastapi.testclient import TestClient

from app import main
from app.gemini_client import (
    LLMClient,
    LLMStreamResult,
    ProviderUnavailableError,
)
from eval.test_mindmap import FakeEmbeddings, FakeVectorStore


class FakeStreamingLLM:
    def __init__(self, chunks):
        self.chunks = chunks
        self.prompts = []

    def open_stream(self, prompt, groq_prompt=None):
        selected_prompt = groq_prompt or prompt
        self.prompts.append(selected_prompt)
        first, remaining, provider, model = self.chunks
        return LLMStreamResult(first, iter(remaining), provider, model, [])


def _add_document(vector_store, session, chat, text):
    vector_store.add_texts(
        [text],
        [
            {
                "session_id": session,
                "chat_id": chat,
                "file_name": "facts.txt",
                "location": "page 1",
                "page": 1,
                "chunk_index": 0,
            }
        ],
    )


def _events(body):
    parsed = []
    for block in body.split("\n\n"):
        event_name = None
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if event_name and data_lines:
            parsed.append((event_name, json.loads("\n".join(data_lines))))
    return parsed


class QueryStreamTests(unittest.TestCase):
    def test_stream_returns_grounded_tokens_final_text_and_debug(self):
        session, chat = "stream", "zorvania"
        vector_store = FakeVectorStore()
        _add_document(
            vector_store,
            session,
            chat,
            "The Zorvania Institute was founded in 1987 by Dr. Anita Mehra.",
        )
        llm = FakeStreamingLLM(
            (
                "The Zorvania Institute was founded in ",
                ["1987 by Dr. Anita Mehra."],
                "groq",
                "stream-test",
            )
        )
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/query/stream",
                headers={
                    "X-Session-Id": session,
                    "X-Chat-Id": chat,
                    "X-Debug": "true",
                },
                json={"query": "Who founded the Zorvania Institute and when?"},
            )

        events = _events(response.text)
        meta = next(data for name, data in events if name == "meta")
        token_text = "".join(data["text"] for name, data in events if name == "token")
        done = next(data for name, data in events if name == "done")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        self.assertEqual(response.headers["cache-control"], "no-cache")
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        self.assertEqual(response.headers["x-debug"], "true")
        self.assertEqual(meta["provider"], "groq")
        self.assertEqual(meta["sources"][0]["page"], 1)
        self.assertEqual(token_text, done["answer"])
        self.assertIn("1987", done["answer"])
        self.assertEqual(done["debug"]["model"], "stream-test")

    def test_stream_preserves_exact_refusal(self):
        session, chat = "stream-refusal", "chat"
        vector_store = FakeVectorStore()
        _add_document(vector_store, session, chat, "The institute is in Blenth.")
        refusal = "I could not find this in the uploaded documents."
        llm = FakeStreamingLLM((refusal, [], "groq", "stream-test"))
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/query/stream",
                headers={"X-Session-Id": session, "X-Chat-Id": chat},
                json={"query": "What is the hostel fee?"},
            )

        events = _events(response.text)
        done = next(data for name, data in events if name == "done")
        self.assertEqual(done["answer"], refusal)

    def test_router_falls_back_before_first_token_and_starts_quota_cooldown(self):
        class FailingProvider:
            model = "gemini-test"

            def open_stream(self, prompt):
                raise ProviderUnavailableError("rate limit reached")

        class AvailableProvider:
            model = "groq-test"

            def open_stream(self, prompt):
                return LLMStreamResult("First token", iter([" tail"]), "groq", self.model, [])

        client = LLMClient.__new__(LLMClient)
        client.provider_order = ["gemini", "groq"]
        client.gemini = FailingProvider()
        client.groq = AvailableProvider()
        client._disabled_until = {}
        client._disabled_until_restart = set()

        result = client.open_stream("prompt")

        self.assertEqual(result.provider, "groq")
        self.assertEqual(result.first_chunk, "First token")
        self.assertEqual(result.attempts[0]["status"], "429")
        self.assertGreater(client._disabled_until["gemini"], time.monotonic())

    def test_midstream_failure_sends_interrupted_error_without_provider_switch(self):
        session, chat = "stream-error", "chat"
        vector_store = FakeVectorStore()
        _add_document(vector_store, session, chat, "The institute is in Blenth.")
        open_calls = []

        def broken_chunks():
            yield from ()
            raise RuntimeError("stream connection ended")

        class BrokenLLM:
            def open_stream(self, prompt, groq_prompt=None):
                open_calls.append(prompt)
                return LLMStreamResult(
                    "Partial answer", broken_chunks(), "gemini", "stream-test", []
                )

        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", BrokenLLM()),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/query/stream",
                headers={"X-Session-Id": session, "X-Chat-Id": chat},
                json={"query": "Where is the institute?"},
            )

        events = _events(response.text)
        error = next(data for name, data in events if name == "error")
        self.assertEqual(len(open_calls), 1)
        self.assertTrue(error["streamed"])
        self.assertIn("interrupted", error["message"].lower())
        self.assertNotIn("done", [name for name, _ in events])

    def test_closing_stream_closes_provider_iterator(self):
        session, chat = "stream-abort", "chat"
        vector_store = FakeVectorStore()
        _add_document(vector_store, session, chat, "The institute is in Blenth.")

        class ClosableIterator:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self):
                raise StopIteration

            def close(self):
                self.closed = True

        provider_iterator = ClosableIterator()

        class ClosableLLM:
            def open_stream(self, prompt, groq_prompt=None):
                return LLMStreamResult(
                    "First token", provider_iterator, "groq", "stream-test", []
                )

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        request = Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/query/stream",
                "raw_path": b"/api/query/stream",
                "query_string": b"",
                "headers": [],
                "client": ("test", 1),
                "server": ("test", 80),
            },
            receive,
        )

        async def close_after_meta():
            with (
                patch.object(main, "vector_store", vector_store),
                patch.object(main, "jina_embeddings", FakeEmbeddings()),
                patch.object(main, "_llm_client", ClosableLLM()),
                patch.object(main, "_llm_error", None),
            ):
                response = await main.stream_query_document(
                    main.QueryRequest(query="Where is the institute?"),
                    request,
                    session,
                    chat,
                )
                body = response.body_iterator
                await body.__anext__()
                await body.__anext__()
                await body.aclose()

        asyncio.run(close_after_meta())
        self.assertTrue(provider_iterator.closed)


if __name__ == "__main__":
    unittest.main()
