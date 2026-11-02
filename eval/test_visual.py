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

    def test_diagram_is_bounded_grounded_and_mermaid_is_backend_generated(self):
        vector = FakeVectorStore()
        metadata, text = stored("visual", "chat", "The institute was founded in 1987. Paris is not here.")
        vector.add_texts([text], [metadata])
        llm = FakeLLM(json.dumps({
            "type": "flowchart",
            "nodes": [{"id": "a", "label": "Founded in 1987", "source": "page 1"}],
            "edges": [],
        }))
        result = self.request({"kind": "diagram"}, llm, vector).json()
        self.assertEqual(result["mermaid"], 'flowchart LR\n    a["Founded in 1987"]')
        self.assertNotIn("Paris", result["mermaid"])
        self.assertLessEqual(len(llm.prompts), 2)

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

    def test_explainer_cards_keep_sources_on_points_steps_numbers_and_glossary(self):
        vector = FakeVectorStore()
        metadata, text = stored("visual", "chat", "The institute was founded in 1987 and teaches robotics.")
        vector.add_texts([text], [metadata])
        llm = FakeLLM(json.dumps({
            "title": "Institute overview",
            "summary": {"text": "A document-grounded overview.", "source": "page 1"},
            "key_points": [
                {"text": "Founded in 1987", "source": "page 1"},
                {"text": "Teaches robotics", "source": "page 1"},
                {"text": "No outside facts", "source": "page 1"},
            ],
            "steps": [{"text": "Read the stated history", "source": "page 1"}],
            "key_numbers": [{"label": "Founded", "value": "1987", "source": "page 1"}],
            "glossary": [{"term": "Robotics", "definition": "A stated subject", "source": "page 1"}],
        }))
        result = self.request({"kind": "explainer"}, llm, vector).json()
        self.assertEqual(len(result["key_points"]), 3)
        self.assertEqual(result["key_numbers"][0]["source"], "page 1")
        self.assertEqual(result["glossary"][0]["source"], "page 1")
        self.assertNotIn("Paris", json.dumps(result))

    def test_bad_json_gets_one_repair_retry(self):
        vector = FakeVectorStore()
        metadata, text = stored("visual", "chat", "The institute was founded in 1987.")
        vector.add_texts([text], [metadata])
        llm = FakeLLM("not json")
        response = self.request({"kind": "diagram"}, llm, vector)
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
        self.request({"kind": "diagram"}, llm, vector, chat="one")
        self.assertIn("First chat only", llm.prompts[0])
        self.assertNotIn("Second chat only", llm.prompts[0])


if __name__ == "__main__":
    unittest.main()
