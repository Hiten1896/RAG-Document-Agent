import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"
REFUSAL_MARKERS = (
    "not in the uploaded documents",
    "not in the documents",
    "could not find",
)


def evaluate(item, answer):
    normalized = answer.casefold()
    if item["type"] == "unanswerable":
        return any(marker in normalized for marker in REFUSAL_MARKERS)
    return all(expected.casefold() in normalized for expected in item["expected"])


def excerpt(text, limit=120):
    return " ".join(text.split())[:limit]


def run(provider):
    questions = json.loads(
        (ROOT / "eval" / "grounding_questions.json").read_text(encoding="utf-8")
    )
    session_id = f"grounding-{provider}-{uuid.uuid4()}"
    headers = {"X-Session-Id": session_id, "X-Chat-Id": "evaluation"}
    document = ROOT / "eval" / "grounding_test_doc.txt"
    with document.open("rb") as handle:
        ingest = requests.post(
            f"{BASE_URL}/api/ingest",
            headers=headers,
            files={"file": (document.name, handle, "text/plain")},
            timeout=180,
        )
    ingest.raise_for_status()

    rows = []
    quota_error = False
    for index, item in enumerate(questions):
        if index:
            time.sleep(6)
        query_headers = {**headers, "X-Debug": "true"}
        if provider != "auto":
            query_headers["X-LLM-Provider"] = provider
        response = requests.post(
            f"{BASE_URL}/api/query",
            headers=query_headers,
            json={"query": item["question"]},
            timeout=180,
        )
        detail = ""
        payload = response.json()
        if response.ok:
            answer = str(payload.get("answer", ""))
            debug = payload.get("debug", {})
            chunks = debug.get("chunks", [])
            top = chunks[0] if chunks else {}
            result = "PASS" if evaluate(item, answer) else "FAIL"
            provider_model = f"{debug.get('provider', '-')} / {debug.get('model', '-')}"
        else:
            detail = str(payload.get("detail", f"HTTP {response.status_code}"))
            answer = detail
            debug = {}
            top = {}
            provider_model = "-"
            result = "ERROR" if response.status_code == 429 or "RESOURCE_EXHAUSTED" in detail else "FAIL"
            quota_error = result == "ERROR"
        rows.append(
            {
                "question": item["question"],
                "type": item["type"],
                "result": result,
                "answer": answer,
                "provider_model": provider_model,
                "source": top.get("source", "-"),
                "score": top.get("relevance_score", "-"),
                "debug": debug,
            }
        )
        if quota_error:
            break

    print("| # | Question | Type | Result | Provider/model | Answer excerpt | Top source | Score |")
    print("|---:|---|---|---|---|---|---|---:|")
    for index, row in enumerate(rows, 1):
        print(
            f"| {index} | {row['question']} | {row['type']} | {row['result']} | "
            f"{row['provider_model']} | {excerpt(row['answer'])} | "
            f"{row['source']} | {row['score']} |"
        )
    if quota_error:
        print("\nStopped after quota error; remaining questions were not run.")
    return rows


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("groq", "gemini", "auto"), default="auto")
    args = parser.parse_args()
    provider = args.provider
    rows = run(provider)
    return 1 if any(row["result"] in {"FAIL", "ERROR"} for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
