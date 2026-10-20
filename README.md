# RAG Document Agent

FastAPI + Next.js RAG application for grounded document question answering.

## What it supports

The backend accepts `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.txt`, and `.md`.
Text is chunked and embedded through the remote Jina embeddings API. Retrieved
chunks are stored in an in-memory ChromaDB collection, so uploads disappear when
the backend restarts.

## LLM providers

Gemini is tried first by default, with Groq as the fallback:

```text
LLM_PROVIDER_ORDER=gemini,groq
```

The router uses temperature `0.2`, no tools, and no search grounding. A provider
that returns a quota/rate-limit response is placed behind a ten-minute in-memory
circuit breaker; authentication failures are skipped until restart. Groq model
selection uses `GROQ_MODEL`, or discovers a general-purpose chat model when it is
unset.

## Environment variables

Set these in a local `.env` file; never commit that file:

```env
GOOGLE_API_KEY=
JINA_API_KEY=
GROQ_API_KEY=
GROQ_MODEL=
LLM_PROVIDER_ORDER=gemini,groq
GROQ_MAX_CONTEXT_CHARS=12000
PROMPT_VERSION=strict
```

The strict prompt answers only from retrieved context and uses the exact refusal
`I could not find this in the uploaded documents.` when the context does not
contain the answer. Set `PROMPT_VERSION=legacy` only to compare the previous
prompt.

## API

- `POST /api/ingest` uploads and indexes one supported document.
- `POST /api/query` retrieves chunks and generates an answer.
- `DELETE /api/session/clear` removes one session/chat's vectors.

Use `X-Session-Id` and `X-Chat-Id` on ingest/query. Add `X-Debug: true` to query
responses to receive retrieved chunks, relevance scores, the exact prompt,
provider/model, routing attempts, and any Groq context trimming. The frontend's
Show evidence panel renders this information, including `Answered by provider /
model`.

Grounding behavior is verified by the fixtures and results in
[eval/results.md](eval/results.md), using
[eval/run_grounding_test.py](eval/run_grounding_test.py).

## Windows local run

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend defaults to the backend at
`http://127.0.0.1:8000`; override it with `NEXT_PUBLIC_API_BASE_URL` when needed.
