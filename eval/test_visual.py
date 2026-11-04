import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from eval.test_mindmap import FakeLLM, FakeVectorStore


def stored(session: str, chat: str, text: str, location: str = "page 1"):
    return {
        "session_id": session,
        "chat_id": chat,
        "file_name": "zorvania.txt",
        "location": location,
        "chunk_index": 0,
    }, text


class VisualEndpointTests(unittest.TestCase):
    def test_api_cors_preflight_allows_visual_request_headers(self):
        origin = (
            main._allowed_origins[0]
            if main._allowed_origins
            else "https://frontend.example"
        )
        with TestClient(main.app) as client:
            response = client.options(
                "/api/visual",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": (
                        "content-type,x-session-id,x-chat-id,x-debug"
                    ),
                },
            )

        self.assertEqual(response.status_code, 200)
        allowed_headers = response.headers["access-control-allow-headers"].lower()
        self.assertIn("x-session-id", allowed_headers)
        self.assertIn("x-chat-id", allowed_headers)
        self.assertIn("x-debug", allowed_headers)

    def request(self, payload, llm, vector_store, session="visual", chat="chat"):
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", object()),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            return client.post(
                "/api/visual",
                headers={"X-Session-Id": session, "X-Chat-Id": chat},
                json=payload,
            )

    def test_chart_rejects_invented_numbers(self):
        vector = FakeVectorStore()
        metadata, text = stored("visual", "chat", "The institute opened in 1987.")
        vector.add_texts([text], [metadata])
        llm = FakeLLM(json.dumps({
            "type": "bar", "title": "Counts", "labels": ["campus"], "values": [42], "source": "page 1"
        }))
        response = self.request({"kind": "chart"}, llm, vector)
        self.assertEqual(response.status_code, 422)
        self.assertIn("not found", response.json()["detail"])

    def test_chart_accepts_numbers_that_are_present_in_context(self):
        vector = FakeVectorStore()
        metadata, text = stored("visual", "chat", "Quarterly output was 12 in Q1 and 18 in Q2.")
        vector.add_texts([text], [metadata])
        llm = FakeLLM(json.dumps({
            "type": "bar", "title": "Quarterly output", "labels": ["Q1", "Q2"],
            "values": [12, 18], "source": "page 1"
        }))
        response = self.request({"kind": "chart"}, llm, vector)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["values"], [12.0, 18.0])

    def test_bad_json_gets_one_repair_retry(self):
        vector = FakeVectorStore()
        metadata, text = stored("visual", "chat", "The institute was founded in 1987.")
        vector.add_texts([text], [metadata])
        llm = FakeLLM("not json")
        response = self.request({"kind": "chart"}, llm, vector)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(llm.prompts), 2)

    def test_chat_isolation(self):
        vector = FakeVectorStore()
        first_metadata, first_text = stored("visual", "one", "First chat only")
        second_metadata, second_text = stored("visual", "two", "Second chat only")
        vector.add_texts([first_text, second_text], [first_metadata, second_metadata])
        llm = FakeLLM(json.dumps({
            "type": "flowchart", "nodes": [{"id": "a", "label": "First chat only", "source": "page 1"}], "edges": []
        }))
        self.request({"kind": "chart"}, llm, vector, chat="one")
        self.assertIn("First chat only", llm.prompts[0])
        self.assertNotIn("Second chat only", llm.prompts[0])


if __name__ == "__main__":
    unittest.main()
