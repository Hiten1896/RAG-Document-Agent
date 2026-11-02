import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from eval.test_mindmap import FakeEmbeddings, FakeLLM, FakeVectorStore


class RecordingVectorStore(FakeVectorStore):
    def __init__(self):
        super().__init__()
        self.searches = []

    def similarity_search_by_vector_with_relevance_scores(
        self, embedding, k, filter
    ):
        self.searches.append((embedding, k, filter))
        return super().similarity_search_by_vector_with_relevance_scores(
            embedding, k, filter
        )


def _metadata(session, chat, file_name, location, chunk_index):
    return {
        "session_id": session,
        "chat_id": chat,
        "file_name": file_name,
        "location": location,
        "page": chunk_index + 1,
        "chunk_index": chunk_index,
    }


def _comparison(kind, topic, doc_a, source_a, doc_b, source_b):
    return {
        "kind": kind,
        "topic": topic,
        "document_a": doc_a,
        "source_a": source_a,
        "document_b": doc_b,
        "source_b": source_b,
    }


class CompareEndpointTests(unittest.TestCase):
    def test_compare_retrieves_each_file_separately_and_validates_sources(self):
        vector_store = RecordingVectorStore()
        vector_store.add_texts(
            [
                "The Zorvania Institute was founded in 1987. Annual tuition is "
                "42,500 rupees.",
                "The institute was founded in 1987. The library has 18,240 books.",
                "Private context from another chat must not be used.",
            ],
            [
                _metadata("compare-session", "current", "facts-a.txt", "page 1", 0),
                _metadata("compare-session", "current", "facts-b.txt", "page 2", 0),
                _metadata("compare-session", "other", "private.txt", "page 9", 0),
            ],
        )
        generated = {
            "comparisons": [
                _comparison(
                    "agreement",
                    "Founding year",
                    "Founded in 1987",
                    "facts-a.txt — page 1",
                    "Founded in 1987",
                    "facts-b.txt — page 2",
                ),
                _comparison(
                    "difference",
                    "Annual tuition",
                    "42,500 rupees",
                    "facts-a.txt — page 1",
                    "not found in facts-b.txt",
                    "not found in facts-b.txt",
                ),
            ]
        }
        llm = FakeLLM(f"```json\n{json.dumps(generated)}\n```")
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/compare",
                headers={
                    "X-Session-Id": "compare-session",
                    "X-Chat-Id": "current",
                    "X-Debug": "true",
                },
                json={
                    "file_a": "facts-a.txt",
                    "file_b": "facts-b.txt",
                    "focus": "Compare the founding year and tuition.",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["files"], ["facts-a.txt", "facts-b.txt"])
        self.assertEqual(len(payload["comparisons"]), 2)
        self.assertEqual(payload["comparisons"][0]["kind"], "agreement")
        self.assertEqual(
            payload["comparisons"][1]["document_b"], "not found in facts-b.txt"
        )
        self.assertEqual(response.headers["x-debug"], "true")
        self.assertEqual(
            payload["debug"]["chunks_by_file"],
            {"facts-a.txt": 1, "facts-b.txt": 1},
        )
        self.assertEqual(len(vector_store.searches), 2)
        self.assertEqual(
            {search[1] for search in vector_store.searches},
            {main.COMPARE_RETRIEVAL_K},
        )
        filters = [str(search[2]) for search in vector_store.searches]
        self.assertTrue(any("facts-a.txt" in item for item in filters))
        self.assertTrue(any("facts-b.txt" in item for item in filters))
        filter_payloads = [json.dumps(search[2]) for search in vector_store.searches]
        self.assertTrue(all('"session_id"' in item for item in filter_payloads))
        self.assertTrue(all('"chat_id"' in item for item in filter_payloads))
        self.assertTrue(all('"file_name"' in item for item in filter_payloads))
        self.assertIn("founded in 1987", llm.prompts[0].casefold())
        self.assertNotIn("Private context", llm.prompts[0])

    def test_compare_requires_two_distinct_files_in_active_chat(self):
        vector_store = FakeVectorStore()
        vector_store.add_texts(
            ["Only current-chat document.", "Other chat document."],
            [
                _metadata("compare-one", "chat", "one.txt", "page 1", 0),
                _metadata("compare-one", "other", "two.txt", "page 1", 0),
            ],
        )
        llm = FakeLLM("")
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            too_few = client.post(
                "/api/compare",
                headers={"X-Session-Id": "compare-one", "X-Chat-Id": "chat"},
                json={"file_a": "one.txt", "file_b": "two.txt"},
            )

            vector_store.add_texts(
                ["Second current-chat document."],
                [_metadata("compare-one", "chat", "two.txt", "page 1", 1)],
            )
            duplicate = client.post(
                "/api/compare",
                headers={"X-Session-Id": "compare-one", "X-Chat-Id": "chat"},
                json={"file_a": "one.txt", "file_b": "one.txt"},
            )

        self.assertEqual(too_few.status_code, 400)
        self.assertIn("at least two", too_few.json()["detail"])
        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("different documents", duplicate.json()["detail"])
        self.assertEqual(llm.prompts, [])

    def test_compare_rejects_file_not_uploaded_to_active_chat(self):
        vector_store = FakeVectorStore()
        vector_store.add_texts(
            [
                "A current chat document.",
                "A second current chat document.",
                "A private document.",
            ],
            [
                _metadata("compare-not-found", "chat", "one.txt", "page 1", 0),
                _metadata("compare-not-found", "chat", "two.txt", "page 1", 1),
                _metadata("compare-not-found", "other", "private.txt", "page 1", 2),
            ],
        )
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", FakeLLM("")),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/compare",
                headers={"X-Session-Id": "compare-not-found", "X-Chat-Id": "chat"},
                json={"file_a": "one.txt", "file_b": "private.txt"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertIn("No uploaded document", response.json()["detail"])

    def test_compare_rejects_unretrieved_source_label(self):
        vector_store = FakeVectorStore()
        vector_store.add_texts(
            ["The report states the budget is 100."],
            [_metadata("compare-source", "chat", "one.txt", "page 1", 0)],
        )
        vector_store.add_texts(
            ["The second report says the budget is 120."],
            [_metadata("compare-source", "chat", "two.txt", "page 2", 1)],
        )
        generated = {
            "comparisons": [
                _comparison(
                    "difference",
                    "Budget",
                    "100",
                    "forged.txt — page 1",
                    "120",
                    "two.txt — page 2",
                )
            ]
        }
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", FakeLLM(json.dumps(generated))),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/compare",
                headers={"X-Session-Id": "compare-source", "X-Chat-Id": "chat"},
                json={"file_a": "one.txt", "file_b": "two.txt"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("invalid source", response.json()["detail"])

    def test_compare_caps_context_separately_for_large_documents(self):
        vector_store = FakeVectorStore()
        for file_name in ("large-a.txt", "large-b.txt"):
            vector_store.add_texts(
                [
                    f"{file_name} chunk {index}: CommonFact is stated. "
                    + ("supporting text " * 120)
                    for index in range(8)
                ],
                [
                    _metadata("compare-large", "chat", file_name, f"page {index + 1}", index)
                    for index in range(8)
                ],
            )
        generated = {
            "comparisons": [
                _comparison(
                    "agreement",
                    "Common fact",
                    "CommonFact is stated",
                    "large-a.txt — page 1",
                    "CommonFact is stated",
                    "large-b.txt — page 1",
                )
            ]
        }
        llm = FakeLLM(json.dumps(generated))
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/compare",
                headers={
                    "X-Session-Id": "compare-large",
                    "X-Chat-Id": "chat",
                    "X-Debug": "true",
                },
                json={"file_a": "large-a.txt", "file_b": "large-b.txt"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        context_sizes = response.json()["debug"]["context_chars_by_file"]
        self.assertLessEqual(
            context_sizes["large-a.txt"], main.COMPARE_MAX_TEXT_CHARS_PER_FILE + 1000
        )
        self.assertLessEqual(
            context_sizes["large-b.txt"], main.COMPARE_MAX_TEXT_CHARS_PER_FILE + 1000
        )
        self.assertLessEqual(len(llm.prompts[0]), main.GROQ_MAX_CONTEXT_CHARS)
        self.assertLessEqual(len(llm.prompts), 2)

    def test_compare_retries_invalid_json_once(self):
        vector_store = FakeVectorStore()
        vector_store.add_texts(
            ["Facts from A."], [_metadata("compare-json", "chat", "a.txt", "page 1", 0)]
        )
        vector_store.add_texts(
            ["Facts from B."], [_metadata("compare-json", "chat", "b.txt", "page 1", 0)]
        )
        llm = FakeLLM(["invalid", "still invalid"])
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "jina_embeddings", FakeEmbeddings()),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/compare",
                headers={"X-Session-Id": "compare-json", "X-Chat-Id": "chat"},
                json={"file_a": "a.txt", "file_b": "b.txt"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(llm.prompts), 2)


if __name__ == "__main__":
    unittest.main()
