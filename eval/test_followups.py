import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from eval.test_mindmap import FakeEmbeddings, FakeLLM, FakeVectorStore


class FollowupsEndpointTests(unittest.TestCase):
    def test_followups_return_three_fenced_questions_and_debug(self):
        llm = FakeLLM(
            "```json\n"
            + json.dumps(
                {
                    "questions": [
                        "Which year was the institute founded?",
                        "Where is the institute's main campus?",
                        "Which programs does the institute offer?",
                    ]
                }
            )
            + "\n```"
        )
        vector_store = FakeVectorStore()
        vector_store.add_texts(
            [
                "The Zorvania Institute was founded in 1987. Its main campus is in "
                "The main campus is located in Blenth. It offers Applied Robotics "
                "and Climate Data Science.",
                "Private context from another chat must not be used.",
            ],
            [
                {
                    "session_id": "followups",
                    "chat_id": "current",
                    "file_name": "facts.txt",
                    "location": "page 1",
                    "page": 1,
                    "chunk_index": 0,
                },
                {
                    "session_id": "followups",
                    "chat_id": "other",
                    "file_name": "private.txt",
                    "location": "page 1",
                    "page": 1,
                    "chunk_index": 0,
                },
            ],
        )
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/followups",
                headers={
                    "X-Session-Id": "followups",
                    "X-Chat-Id": "current",
                    "X-Debug": "true",
                },
                json={"question": "What does this document say about the institute?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["questions"]), 3)
        self.assertEqual(response.headers["x-debug"], "true")
        self.assertEqual(response.json()["debug"]["provider"], "groq")
        self.assertEqual(response.json()["debug"]["chunks_used"], 1)
        self.assertEqual(response.json()["debug"]["llm_attempts"], 1)
        self.assertIn("1987", llm.prompts[0])
        self.assertNotIn("Private context", llm.prompts[0])

    def test_followups_without_relevant_documents_returns_clear_error(self):
        llm = FakeLLM("")
        with (
            patch.object(main, "vector_store", FakeVectorStore()),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/followups",
                headers={"X-Session-Id": "empty", "X-Chat-Id": "chat"},
                json={"question": "What does the document say?"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("No relevant document content", response.json()["detail"])
        self.assertEqual(llm.prompts, [])

    def test_followups_retry_invalid_json_once_then_return_502(self):
        llm = FakeLLM(["not JSON", "still not JSON"])
        vector_store = FakeVectorStore()
        vector_store.add_texts(
            ["The Zorvania Institute was founded in 1987 in Blenth."],
            [
                {
                    "session_id": "invalid-followups",
                    "chat_id": "chat",
                    "file_name": "facts.txt",
                    "location": "document",
                    "chunk_index": 0,
                }
            ],
        )
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/followups",
                headers={"X-Session-Id": "invalid-followups", "X-Chat-Id": "chat"},
                json={"question": "When was the institute founded?"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(llm.prompts), 2)


if __name__ == "__main__":
    unittest.main()
