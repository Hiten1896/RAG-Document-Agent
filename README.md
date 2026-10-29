# RAG Document Agent

FastAPI + Next.js RAG application for grounded document question answering.

## What it supports

The backend accepts `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.txt`, and `.md`.
Text is chunked and embedded through the remote Jina embeddings API. Retrieved
chunks are stored in an in-memory ChromaDB collection, so uploads disappear when
the backend restarts.

The app can also generate a grounded, interactive mind map from all documents
in the active chat. It samples document chunks across their locations, keeps
source labels on map nodes, and caps each request at four LLM calls. The browser
renders maps with `markmap-lib` and `markmap-view`; maps can be fitted to the
screen and downloaded as SVG or PNG.

The **Quiz me** card generates up to ten document-grounded multiple-choice
questions. It samples and caps the context, validates each correct answer and
source against the selected chat's document text, and limits generation and
JSON repair to two LLM calls. The same questions can be used in score-tracked
quiz mode or as flip-style flashcards.

**Compare documents** retrieves evidence separately from each selected file in
the active chat, then returns cited agreements and differences in a table.
Each side is capped at four retrieved chunks and 4,000 context characters; the
structured response and JSON repair use at most two LLM calls.

Answer citations and source cards open the cited PDF page in the viewer. For
DOCX, PPTX, XLSX, CSV, TXT, and Markdown files, the viewer shows the extracted
passage and its page, slide, sheet, or document location when that text is
available in the answer evidence. PDF text highlighting is not currently
provided.

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

For the frontend, streaming can be enabled with
`NEXT_PUBLIC_ENABLE_STREAMING=true`. It is disabled by default until the
deployed Render backend has been verified not to buffer SSE responses.

The strict prompt answers only from retrieved context and uses the exact refusal
`I could not find this in the uploaded documents.` when the context does not
contain the answer. Set `PROMPT_VERSION=legacy` only to compare the previous
prompt.

## API

- `POST /api/ingest` uploads and indexes one supported document.
- `POST /api/query` retrieves chunks and generates an answer.
- `POST /api/query/stream` streams grounded answers as Server-Sent Events; the existing `/api/query` remains unchanged.
- `POST /api/followups` generates three document-grounded follow-up questions for a prior question.
- `POST /api/quiz` generates 1–10 multiple-choice questions from the active chat's documents.
- `POST /api/compare` compares two files with separate per-file retrieval and citations.
- `POST /api/mindmap` generates a structured mind map from the active chat's documents.
- `DELETE /api/session/clear` removes one session/chat's vectors.

Use `X-Session-Id` and `X-Chat-Id` on ingest/query/stream/followups/quiz/compare/mindmap. Add
`X-Debug: true` to query responses to receive retrieved chunks, relevance scores, the exact
prompt, provider/model, routing attempts, and any Groq context trimming. Mind
map responses include provider/model, chunk count, cache status, and timings
when requested. Follow-up generation is lazy: click **Suggest questions** below
an answer to request three questions grounded in fresh retrieval results. Its
optional debug response includes the provider/model, retrieved chunk count, and
timings. The frontend's Show evidence panel renders query information,
including `Answered by provider / model`.
Quiz requests sample at most 10 chunks and cap document context at 10,000
characters; `count` is limited to 10 and `difficulty` accepts `easy`, `medium`,
or `hard`. `X-Debug: true` returns sampling, provider, and timing details.
Comparison requires two different files already indexed in the same chat.
Each file is retrieved with its own session-plus-filename filter; debug output
reports per-file chunk and context counts.
Query sources carry a numeric `page` where available and a `location` label for
slides, sheets, or unpaginated text; the frontend uses these fields for citation
navigation.

Mind maps are cached in memory per session, chat, document set, and topic; they
are cleared with the chat and expire with its normal session cleanup. They need
no additional environment variables or Python dependencies.

## Export

Use **Export chat** or **Export this answer** to save a conversation as a Word
`.docx` file or open the browser's print dialog to save a PDF. Exports include
questions, answers, and sources; retrieved evidence is included only when its
Show evidence panel is open. An available mind map is rendered as a plain-text
image and included in both formats. Word export uses the pinned `docx` package;
PDF export uses the browser's built-in print-to-PDF support.

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
