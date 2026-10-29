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

## Quiz API validation

`python -m unittest eval.test_quiz -v` passed all 7 deterministic endpoint tests.
The tests cover session/chat isolation, no-document errors, fenced and invalid
JSON, four-option schema validation, exact requested question counts, rejection
of an answer absent from the source text, and context limits on a large document.
These tests use a fake LLM provider; they validate grounding enforcement and
response behavior, not live-provider quiz quality.
