import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from eval.test_mindmap import FakeLLM, FakeVectorStore


def _quiz_question(answer, source, question="When was the institute founded?"):
    return {
        "question": question,
        "options": [answer, "1974", "1998", "2005"],
        "correct_answer": answer,
        "explanation": f"The document states {answer}.",
        "source": source,
    }


def _seed_document(store, text, session="quiz-session", chat="current", source="facts.txt"):
    store.add_texts(
        [text],
        [
            {
                "session_id": session,
                "chat_id": chat,
                "file_name": source,
                "location": "page 1",
                "page": 1,
                "chunk_index": 0,
            }
        ],
    )


class QuizEndpointTests(unittest.TestCase):
    def test_quiz_returns_grounded_fenced_json_and_debug(self):
        response_data = {
            "questions": [
                _quiz_question("1987", "facts.txt — page 1"),
                _quiz_question("42,500 rupees", "facts.txt — page 1", "What is the annual tuition fee?"),
            ]
        }
        llm = FakeLLM(f"```json\n{json.dumps(response_data)}\n```")
        vector_store = FakeVectorStore()
        _seed_document(
            vector_store,
            "The Zorvania Institute was founded in 1987. The annual tuition "
            "fee is 42,500 rupees.",
        )
        vector_store.add_texts(
            ["Other chat private value is 9182."],
            [
                {
                    "session_id": "quiz-session",
                    "chat_id": "other",
                    "file_name": "private.txt",
                    "location": "page 1",
                    "chunk_index": 1,
                }
            ],
        )
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            result = client.post(
                "/api/quiz",
                headers={
                    "X-Session-Id": "quiz-session",
                    "X-Chat-Id": "current",
                    "X-Debug": "true",
                },
                json={"count": 2, "difficulty": "easy"},
            )

        self.assertEqual(result.status_code, 200, result.text)
        payload = result.json()
        self.assertEqual(len(payload["questions"]), 2)
        self.assertEqual(payload["questions"][0]["correct_answer"], "1987")
        self.assertEqual(payload["questions"][0]["source"], "facts.txt — page 1")
        self.assertEqual(len(payload["questions"][0]["options"]), 4)
        self.assertEqual(result.headers["x-debug"], "true")
        self.assertEqual(payload["debug"]["chunks_used"], 1)
        self.assertEqual(payload["debug"]["llm_attempts"], 1)
        self.assertIn("Difficulty: easy", llm.prompts[0])
        self.assertNotIn("9182", llm.prompts[0])

    def test_quiz_without_uploaded_documents_returns_clear_error(self):
        llm = FakeLLM("")
        with (
            patch.object(main, "vector_store", FakeVectorStore()),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/quiz",
                headers={"X-Session-Id": "empty-quiz", "X-Chat-Id": "chat"},
                json={"count": 3, "difficulty": "medium"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Upload a document", response.json()["detail"])
        self.assertEqual(llm.prompts, [])

    def test_quiz_retries_invalid_json_once_then_returns_502(self):
        llm = FakeLLM(["not JSON", "still not JSON"])
        vector_store = FakeVectorStore()
        _seed_document(vector_store, "The institute was founded in 1987.")
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/quiz",
                headers={"X-Session-Id": "quiz-session", "X-Chat-Id": "current"},
                json={"count": 1, "difficulty": "medium"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(llm.prompts), 2)

    def test_quiz_retries_invalid_four_option_schema(self):
        invalid_question = _quiz_question("1987", "facts.txt — page 1")
        invalid_question["options"] = ["1987", "1987", "1998", "2005"]
        llm = FakeLLM(
            [json.dumps({"questions": [invalid_question]})] * 2
        )
        vector_store = FakeVectorStore()
        _seed_document(vector_store, "The institute was founded in 1987.")
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/quiz",
                headers={"X-Session-Id": "quiz-session", "X-Chat-Id": "current"},
                json={"count": 1, "difficulty": "medium"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(llm.prompts), 2)

    def test_quiz_rejects_answer_not_found_in_document(self):
        generated = {
            "questions": [
                _quiz_question("Atlantis", "facts.txt — page 1")
            ]
        }
        llm = FakeLLM(json.dumps(generated))
        vector_store = FakeVectorStore()
        _seed_document(vector_store, "The institute was founded in 1987.")
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/quiz",
                headers={"X-Session-Id": "quiz-session", "X-Chat-Id": "current"},
                json={"count": 1, "difficulty": "hard"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("could not be verified", response.json()["detail"])

    def test_quiz_caps_large_document_context_and_chunk_count(self):
        vector_store = FakeVectorStore()
        texts = [
            f"Anchor answer: GroundedValue. Chunk {index} contains "
            + ("supporting document text " * 24)
            for index in range(60)
        ]
        metadatas = [
            {
                "session_id": "large-quiz",
                "chat_id": "chat",
                "file_name": "large.txt",
                "location": "page 1",
                "chunk_index": index,
            }
            for index in range(60)
        ]
        vector_store.add_texts(texts, metadatas)
        generated = {
            "questions": [
                _quiz_question("GroundedValue", "large.txt — page 1")
            ]
        }
        llm = FakeLLM(json.dumps(generated))
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            response = client.post(
                "/api/quiz",
                headers={
                    "X-Session-Id": "large-quiz",
                    "X-Chat-Id": "chat",
                    "X-Debug": "true",
                },
                json={"count": 1, "difficulty": "easy"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["debug"]["chunks_used"], main.QUIZ_MAX_CHUNKS
        )
        self.assertLessEqual(
            response.json()["debug"]["context_chars"], main.QUIZ_MAX_TEXT_CHARS
        )
        self.assertLessEqual(len(llm.prompts[0]), main.GROQ_MAX_CONTEXT_CHARS)
        self.assertIn("Chunk 0", llm.prompts[0])
        self.assertIn("Chunk 59", llm.prompts[0])
        self.assertLessEqual(len(llm.prompts), 2)

    def test_quiz_rejects_wrong_question_count_and_excessive_count(self):
        generated = {"questions": [_quiz_question("1987", "facts.txt — page 1")]}
        llm = FakeLLM(json.dumps(generated))
        vector_store = FakeVectorStore()
        _seed_document(vector_store, "The institute was founded in 1987.")
        with (
            patch.object(main, "vector_store", vector_store),
            patch.object(main, "_llm_client", llm),
            patch.object(main, "_llm_error", None),
            TestClient(main.app) as client,
        ):
            wrong_count = client.post(
                "/api/quiz",
                headers={"X-Session-Id": "quiz-session", "X-Chat-Id": "current"},
                json={"count": 2, "difficulty": "medium"},
            )
            excessive_count = client.post(
                "/api/quiz",
                headers={"X-Session-Id": "quiz-session", "X-Chat-Id": "current"},
                json={"count": 11, "difficulty": "medium"},
            )

        self.assertEqual(wrong_count.status_code, 502)
        self.assertEqual(excessive_count.status_code, 422)


if __name__ == "__main__":
    unittest.main()
