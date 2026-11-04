import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from eval.test_mindmap import FakeEmbeddings, FakeLLM, FakeVectorStore


class QueryPlanningTests(unittest.TestCase):
    def test_intent_classifies_broad_document_overview(self):
        llm = FakeLLM(
            json.dumps(
                {
                    "action": "summary",
                    "topic": "",
                    "retrieval_query": "document overview, main sections, and key points",
                    "overview": True,
                }
            )
        )
        with (
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/intent",
                json={"prompt": "What's inside this PDF?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "summary")
        self.assertTrue(response.json()["overview"])
        self.assertIn("overview", response.json()["retrieval_query"])
        self.assertIn("Classify and plan retrieval", llm.prompts[0])

    def test_semantic_query_uses_planned_search_terms_but_answers_original_question(self):
        embeddings = FakeEmbeddings()
        vector_store = FakeVectorStore()
        vector_store.add_texts(
            ["Refunds are available within 30 days of purchase."],
            [
                {
                    "session_id": "planned",
                    "chat_id": "chat",
                    "file_name": "policy.txt",
                    "location": "page 1",
                    "chunk_index": 0,
                }
            ],
        )
        llm = FakeLLM("Refunds are available within 30 days.")
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", embeddings),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/query",
                headers={
                    "X-Session-Id": "planned",
                    "X-Chat-Id": "chat",
                    "X-Debug": "true",
                },
                json={
                    "query": "What happens if I change my mind?",
                    "retrieval_query": "refund policy eligibility and return window",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            embeddings.queries,
            ["refund policy eligibility and return window"],
        )
        self.assertIn("What happens if I change my mind?", llm.prompts[0])
        self.assertIn("Refunds are available within 30 days", llm.prompts[0])

    def test_overview_samples_across_chat_without_embedding_the_vague_question(self):
        embeddings = FakeEmbeddings()
        vector_store = FakeVectorStore()
        vector_store.add_texts(
            [
                f"Section {index} explains subject {index}."
                for index in range(main.OVERVIEW_MAX_CHUNKS + 8)
            ]
            + ["Private text from another chat."],
            [
                {
                    "session_id": "overview",
                    "chat_id": "chat",
                    "file_name": "handbook.pdf",
                    "location": f"page {index + 1}",
                    "chunk_index": index,
                }
                for index in range(main.OVERVIEW_MAX_CHUNKS + 8)
            ]
            + [
                {
                    "session_id": "overview",
                    "chat_id": "other",
                    "file_name": "private.pdf",
                    "location": "page 1",
                    "chunk_index": 0,
                }
            ],
        )
        llm = FakeLLM("The PDF describes its major sections and topics.")
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", embeddings),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/query",
                headers={
                    "X-Session-Id": "overview",
                    "X-Chat-Id": "chat",
                    "X-Debug": "true",
                },
                json={
                    "query": "What is inside this PDF?",
                    "overview": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(embeddings.queries, [])
        self.assertEqual(len(response.json()["debug"]["chunks"]), main.OVERVIEW_MAX_CHUNKS)
        self.assertEqual(response.json()["debug"]["top_k"], main.OVERVIEW_MAX_CHUNKS)
        self.assertIn("Section 0", llm.prompts[0])
        self.assertIn("Section 19", llm.prompts[0])
        self.assertNotIn("Private text", llm.prompts[0])
        self.assertTrue(
            all("relevance_score" not in source for source in response.json()["sources"])
        )


if __name__ == "__main__":
    unittest.main()
