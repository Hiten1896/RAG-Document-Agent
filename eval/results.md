# Grounding evaluation results

Model used for Groq runs: `qwen/qwen3.8-27b`.

## Legacy prompt — Groq

| # | Result | Question | Answer excerpt |
|---:|---|---|---|
| 1 | PASS | Who founded the Zorvania Institute and when? | Dr. Anita Mehra; 1987 |
| 2 | PASS | What is the annual tuition fee? | ₹42,500 |
| 3 | PASS | Who is the director and when did they join? | Prof. Kavya Nambiar; 2019 |
| 4 | PASS | How many books does the library hold? | 18,240 |
| 5 | PASS | What is the mascot's name? | Tarn |
| 6 | PASS | At what temperature does water boil at sea level, according to the document? | 60°C |
| 7 | FAIL | What is the hostel fee? | Refused with different wording |
| 8 | FAIL | Who won the 2011 Cricket World Cup? | Refused with different wording |
| 9 | FAIL | What is the capital of France? | Refused with different wording |

**Pass count: 6/9**

## Strict prompt — Groq

| # | Result | Question | Answer excerpt |
|---:|---|---|---|
| 1 | PASS | Who founded the Zorvania Institute and when? | Dr. Anita Mehra; 1987 |
| 2 | PASS | What is the annual tuition fee? | ₹42,500 |
| 3 | PASS | Who is the director and when did they join? | Prof. Kavya Nambiar; 2019 |
| 4 | PASS | How many books does the library hold? | 18,240 |
| 5 | PASS | What is the mascot's name? | Tarn |
| 6 | PASS | At what temperature does water boil at sea level, according to the document? | 60°C |
| 7 | PASS | What is the hostel fee? | I could not find this in the uploaded documents. |
| 8 | PASS | Who won the 2011 Cricket World Cup? | I could not find this in the uploaded documents. |
| 9 | PASS | What is the capital of France? | I could not find this in the uploaded documents. |

**Pass count: 9/9**

## Automatic routing — strict

Gemini was quota-exhausted, so automatic routing used Groq without retrying
Gemini. **Pass count: 9/9.**

## Without-retrieval comparison — Groq, question only

| Question | Grounded app answer | Groq without context |
|---|---|---|
| Who founded the Zorvania Institute and when? | Dr. Anita Mehra; 1987 | Said the organization was not widely documented and may be fictional. |
| What is the annual tuition fee? | ₹42,500 | Asked for institution/program context. |
| What is the mascot's name? | Tarn | Asked which mascot was intended. |
| At what temperature does water boil at sea level, according to the document? | 60°C, according to the document | Answered the outside-knowledge value 100°C. |

## Summary

- Legacy prompt: 6/9 pass; its refusal wording was not exact enough for rows 7–9.
- Strict prompt: 9/9 pass; all three unanswerable questions used the exact refusal.
- Automatic routing: 9/9 pass; Gemini quota exhaustion was handled by Groq fallback.

## Mind-map API groundedness

`python -m unittest eval.test_mindmap -v` passed all 7 endpoint tests. The
Zorvania fixture test checks the generated labels against the document, rejects
an unsupported "Paris" node, and verifies depth/node limits. Additional tests
cover session/chat isolation, structured JSON repair, and bounded large-document
map/reduce behavior.

## Quiz API validation

`python -m unittest eval.test_quiz -v` passed all 7 deterministic endpoint tests.
The tests cover session/chat isolation, no-document errors, fenced and invalid
JSON, four-option schema validation, exact requested question counts, rejection
of an answer absent from the source text, and context limits on a large document.
These tests use a fake LLM provider; they validate grounding enforcement and
response behavior, not live-provider quiz quality.

## Per-file comparison API validation

`python -m unittest eval.test_compare -v` passed all 6 deterministic endpoint
tests. They verify independent session-plus-file retrieval for both selected
files, chat isolation, per-file context limits, citation-label validation,
the fewer-than-two-files guard, and the two-call invalid-JSON repair limit.
These tests use a fake LLM provider; they validate the endpoint contract and
retrieval boundaries, not live-provider comparison quality.

## Completed feature validation

| Feature | Validation | Result / limitation |
|---|---|---|
| Mind maps | 7 API tests; grounding, schema, isolation, and bounds | Passed |
| Citation navigation | Synthetic two-page PDF browser check | Page 2 opened for a page-2 citation; PDF text highlighting is not included |
| Follow-up questions | 3 API tests and browser interaction checks | Lazy loading, selected-chip submission, and refusal omission verified |
| Export | DOCX package generation and production build | Valid DOCX output; browser download/print dialog was not verified |
| Streaming | 5 API tests; frontend lint/build; `/api/query` grounding 9/9 | Passed locally; Render buffering remains unverified, so streaming stays disabled by default |
| Quiz and flashcards | 7 API tests; evidence-answer validation and large-context cap | Endpoint checks passed with a fake provider; live-provider quiz quality was not measured |
| Compare documents | 6 API tests; per-file filters, isolation, citations, and context limits | Endpoint checks passed with a fake provider; browser interaction was not verified |

Final local regression: `python -m unittest discover -s eval -p "test_*.py" -q`
passed all 28 tests; `eval/run_grounding_test.py` passed 9/9; frontend
`npm run lint` and `npm run build` both passed.
