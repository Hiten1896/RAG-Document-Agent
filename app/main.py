import logging
import os
import re
import requests
import tempfile
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

# LangChain & Google GenAI Imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# `langchain_community.vectorstores.Chroma` is the deprecated shim; the
# maintained integration lives in the `langchain-chroma` package (already in
# requirements.txt, and what rag_agent_project.ipynb uses).
from langchain_chroma import Chroma

load_dotenv()

logger = logging.getLogger("docagent")

# ── API Key Verification ──
# GEMINI_API_KEY (or GOOGLE_API_KEY) is used only for answer generation now —
# one call per question, not per chunk, so it's far less exposed to quota
# limits than embedding calls were.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "No Gemini API key found. Copy .env.example to .env and set "
        "GOOGLE_API_KEY (or GEMINI_API_KEY) to a key from "
        "https://aistudio.google.com/."
    )

# JINA_API_KEY powers embeddings (ingestion + query-time similarity search).
# Split from Gemini deliberately: embedding a document is dozens to hundreds
# of API calls, which repeatedly exhausted Gemini's free-tier embedding
# quota (429 RESOURCE_EXHAUSTED) even with small batches and backoff. Jina's
# free tier grants 1M tokens with no monthly call cap, which comfortably
# covers ingesting documents of unknown size handed over live, without also
# consuming the same quota pool the answer-generation calls depend on.
JINA_API_KEY = os.getenv("JINA_API_KEY")

if not JINA_API_KEY:
    raise RuntimeError(
        "No Jina API key found. Set JINA_API_KEY in .env to a key from "
        "https://jina.ai/embeddings/ (free tier, no card required)."
    )

# ── Paths ──
# Anchor persistence to the repository root rather than the process working
# directory. With a bare "./chroma_db", launching uvicorn from anywhere other
# than the repo root silently created a brand-new empty database, so previously
# ingested documents appeared to vanish.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = Path(
    os.getenv("CHROMA_PERSIST_DIR") or (PROJECT_ROOT / "data" / "chroma_db")
).resolve()
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Named explicitly instead of relying on Chroma's default "langchain"
# collection. `data/chroma_db` may hold orphaned collections from earlier
# embedding providers tried during development (local MiniLM at 384 dims,
# Gemini at 768/3072 dims) — those are dimensionally incompatible with Jina's
# 1024-dim vectors and would corrupt similarity search if mixed in. Naming
# this collection explicitly keeps it isolated; old collections are simply
# unused, not deleted, and can be removed manually if disk space matters.
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "docagent_jina")

# jina-embeddings-v3: 1024 dims, 8192-token context, free tier covers
# ingesting documents of unknown size handed over live without touching the
# Gemini quota that answer generation depends on. See JINA_API_KEY above for
# why this is a separate provider from the LLM.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v3")
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"

# The LLM call in /query is one request per question, not one per chunk, so
# it's far less likely to hit the same quota wall embeddings did.
# `gemini-1.5-flash` was retired first, then `gemini-2.5-flash` was ALSO
# retired for new-user access shortly after ("no longer available to new
# users" — see the 404 this threw). Gemini's model lineup moves fast enough
# that any hardcoded name here has a real chance of going stale again;
# `gemini-3.6-flash` is Google's own suggested replacement in that error and
# is confirmed present in this account's `ListModels` output. If this 404s
# again later, re-run `GET /v1beta/models?key=...` and swap in whatever's
# current — the error message itself usually names the successor.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024


class JinaEmbeddings:
    """Minimal LangChain-`Embeddings`-compatible wrapper around Jina AI's
    REST embeddings endpoint (https://api.jina.ai/v1/embeddings).

    Implemented with a plain `requests` call rather than pulling in a
    dedicated SDK — Jina's endpoint is OpenAI-schema-compatible, so a
    lightweight wrapper is all `langchain_chroma.Chroma` needs (it only
    requires `embed_documents(texts) -> list[list[float]]` and
    `embed_query(text) -> list[float]`).

    A single request batches multiple texts, which keeps Chroma's own
    ingestion batching (see EMBED_BATCH_SIZE below) as the only batching
    layer to reason about.
    """

    def __init__(self, api_key: str, model: str, timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _embed(self, texts: List[str], task: str) -> List[List[float]]:
        if not texts:
            return []
        try:
            response = requests.post(
                JINA_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts, "task": task},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"JinaEmbeddingError: could not reach Jina API: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"JinaEmbeddingError: {response.status_code} {response.text[:500]}"
            )

        data = response.json()
        # Jina returns `data` sorted by the `index` field, not necessarily
        # input order under concurrent batching — sort defensively so a
        # chunk's embedding always lines up with the chunk it was requested
        # for.
        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # `retrieval.passage`: the adapter Jina trained for indexing long
        # documents, as opposed to short queries — asymmetric retrieval
        # models score better when document and query sides use their
        # matching adapter rather than one adapter for both.
        return self._embed(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> List[float]:
        result = self._embed([text], task="retrieval.query")
        return result[0] if result else []


# ── FastAPI App Setup ──
app = FastAPI(title="DocAgent RAG Backend", version="2.1")

# `allow_origins=["*"]` together with `allow_credentials=True` is an invalid
# combination: browsers reject a credentialed request whose
# Access-Control-Allow-Origin is the "*" wildcard, so that config silently broke
# any future cookie/auth-bearing request. Use an explicit allowlist when
# credentials are needed, and only fall back to the wildcard without them.
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if _origins_env:
    ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
_allow_credentials = "*" not in ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Embedding & LLM Setup ──
embeddings = JinaEmbeddings(api_key=JINA_API_KEY, model=EMBEDDING_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(PERSIST_DIR),
    embedding_function=embeddings,
)

# ── Reset vector store on every startup ──
# Chroma persists to disk across restarts, but the frontend's chat history
# and this file's own `ingestion_jobs` tracker are in-memory and reset on
# every launch. That mismatch made documents "silently" accumulate across
# dev restarts — re-uploading the same PDF during testing just kept adding
# duplicate embeddings on top of old ones, growing the collection forever.
# Wiping the collection here means every server start begins from a clean,
# empty index, matching the fresh chat history the frontend already shows.
# Set CLEAR_CHROMA_ON_STARTUP=false in .env to opt out (e.g. in production,
# where documents should persist across deploys/restarts).
if os.getenv("CLEAR_CHROMA_ON_STARTUP", "true").strip().lower() not in ("false", "0", "no"):
    try:
        vectorstore.delete_collection()
    except Exception:
        # Nothing to delete yet (first-ever run) — safe to ignore.
        pass
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )
    logger.info("Cleared Chroma collection %r on startup.", COLLECTION_NAME)

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)


# ── Job status tracking ──
# In-memory status tracker for document ingestion jobs.
# Bounded to MAX_TRACKED_JOBS entries (LRU eviction) so long-running server
# instances don't leak memory as more documents get ingested over time.
# Every method takes a lock: ingestion runs in FastAPI's background-task
# threadpool while /status is served from other threads, and OrderedDict
# mutation is not safe across threads.
class _BoundedJobStore(MutableMapping):
    """LRU-bounded, thread-safe mapping of filename -> ingestion job dict.

    Subclasses MutableMapping so the full mapping protocol exists. The previous
    hand-rolled version only defined `__setitem__`/`__getitem__`/`__contains__`/
    `get`, so `len()`, iteration, `dict()`, and JSON serialization all failed —
    and `bool(store)` was always True because there was no `__len__`.

    Every method takes a lock: ingestion runs in FastAPI's background-task
    threadpool while /status is served from other threads, and concurrent
    `move_to_end` / `popitem` on an OrderedDict is not safe across threads.
    """

    def __init__(self, maxlen: int = 500):
        self._store: "OrderedDict[str, dict]" = OrderedDict()
        self._maxlen = maxlen
        self._lock = threading.RLock()

    def __setitem__(self, key: str, value: dict) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self._maxlen:
                self._store.popitem(last=False)  # evict oldest (O(1))

    def __getitem__(self, key: str) -> dict:
        with self._lock:
            value = self._store[key]
            self._store.move_to_end(key)
            return value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._store[key]

    def __iter__(self):
        with self._lock:
            return iter(list(self._store))

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


ingestion_jobs = _BoundedJobStore(maxlen=500)


# ── Data Models ──
class QueryRequest(BaseModel):
    """Accepts either `query` (the shape documented in the README and sent by
    the Next.js frontend) or `question`. The backend previously required only
    `question`, so every request from the frontend failed with a 422."""

    query: str = ""
    top_k: int = Field(default=4, ge=1, le=20)
    # Optional: restrict retrieval to one ingested document. With a single
    # shared collection and no filter, a question about the open document could
    # be answered entirely out of a different PDF, citing the wrong file.
    source: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _accept_question_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("query") and data.get("question"):
            data = {**data, "query": data["question"]}
        return data

    @model_validator(mode="after")
    def _require_non_empty(self) -> "QueryRequest":
        if not self.query.strip():
            raise ValueError("Provide a non-empty 'query' (or 'question').")
        return self


class SourceMetadata(BaseModel):
    source: str
    page: Optional[int] = 1
    chunk_type: Optional[str] = "text"
    relevance_score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceMetadata]


class TitleRequest(BaseModel):
    """A single user message to summarize into a short chat title."""

    message: str = ""

    @model_validator(mode="after")
    def _require_non_empty(self) -> "TitleRequest":
        if not self.message.strip():
            raise ValueError("Provide a non-empty 'message'.")
        return self


class TitleResponse(BaseModel):
    title: str


# ── Helpers ──
_UNSAFE_FILENAME_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]')


def sanitize_filename(raw: Optional[str]) -> str:
    """Reduce a client-supplied filename to a safe, path-free label.

    `os.path.join(temp_dir, f"upload_{filename}")` used the raw value, so a
    filename like `..\\..\\Windows\\Temp\\x.pdf` escaped the temp directory and
    the file was then unlinked in the `finally` block — an arbitrary-write and
    arbitrary-delete primitive. A `None` filename also crashed on `.endswith`.
    """
    if not raw:
        raise HTTPException(status_code=400, detail="A filename is required.")

    # Normalize first so lookalike/composed Unicode can't smuggle separators.
    name = unicodedata.normalize("NFKC", raw)
    # Take the last path component under BOTH separators; ntpath/posixpath
    # basename each only understand one.
    name = name.replace("\\", "/").split("/")[-1]
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")

    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return name[:200]


def _coerce_answer(message: Any) -> str:
    """`AIMessage.content` is `str | list[str | dict]` in langchain-core 1.x.

    Passing a list straight into `answer: str` raises a pydantic
    ValidationError that the broad `except` below turns into an opaque 500.
    That happens whenever the model emits thinking parts — and unconditionally
    on Gemini 3+, so it would detonate the moment LLM_MODEL is bumped.
    `AIMessage.text` flattens content blocks for us; fall back to manual
    flattening if a provider returns something without it.

    Gemini can also return a response with NO text part at all — e.g. it
    emitted only a "thinking"/reasoning block, hit a safety filter, or was
    truncated (`finish_reason` != "stop"). Previously that silently produced
    `""`, the frontend's `res.answer || 'No answer returned.'` fallback never
    triggered because the field itself was present-but-empty, and the message
    bubble rendered with no text — just the citation chips below it, which
    looked like "it only tells me the page". Detect that case and say why.
    """
    text = getattr(message, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    if callable(text):  # older langchain exposed text() as a method
        try:
            result = text()
            if isinstance(result, str) and result.strip():
                return result
        except Exception:
            pass

    content = getattr(message, "content", message)
    flattened: Optional[str] = None
    if isinstance(content, str):
        flattened = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        flattened = "".join(parts)

    if flattened and flattened.strip():
        return flattened

    # Nothing usable came back. Look at why, so the UI can show a real reason
    # instead of an empty bubble.
    finish_reason = None
    response_meta = getattr(message, "response_metadata", None)
    if isinstance(response_meta, dict):
        finish_reason = response_meta.get("finish_reason")

    if finish_reason and str(finish_reason).upper() not in ("STOP", "1"):
        reason = str(finish_reason).upper()
        if "SAFETY" in reason:
            return (
                "Gemini declined to answer this question because it was "
                "flagged by safety filters. Try rephrasing the question."
            )
        if "MAX_TOKENS" in reason or "LENGTH" in reason:
            return (
                "Gemini's response was cut off before finishing. Try asking "
                "a narrower question or reducing top_k."
            )
        return (
            f"Gemini did not return a usable answer (finish_reason={reason}). "
            "Try rephrasing the question."
        )

    return (
        "Gemini returned an empty response for this question. This can "
        "happen transiently — try asking again."
    )


# Chroma only accepts str/int/float/bool metadata values. PyMuPDF attaches a
# dozen PDF header fields (including `file_path`, which leaks the server's temp
# path) and any None among them raises on insert. Keep only what we cite.
_KEEP_METADATA = ("source", "page", "chunk_type")


def _clean_metadata(meta: dict, filename: str) -> dict:
    page = meta.get("page")
    cleaned = {
        "source": filename,
        # PyMuPDFLoader pages are 0-based; citations should read 1-based.
        "page": (page + 1) if isinstance(page, int) else 1,
        "chunk_type": "text",
    }
    return {k: cleaned[k] for k in _KEEP_METADATA}


def _drop_existing_chunks(filename: str) -> None:
    """Re-ingesting a document used to append a second full copy of its chunks,
    inflating retrieval with duplicates. Remove the prior copy first."""
    try:
        existing = vectorstore.get(where={"source": filename}, include=[])
        ids = existing.get("ids") or []
        if ids:
            vectorstore.delete(ids=ids)
    except Exception:
        # A missing collection or an unsupported filter must not abort ingestion.
        pass


# ── Background Worker ──
def process_pdf_in_background(file_path: str, filename: str):
    try:
        ingestion_jobs[filename] = {"status": "processing", "filename": filename}

        # 1. Load document (PyMuPDF is a C++-backed parser, significantly
        #    faster than pypdf for large textbook-sized PDFs)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No pages could be read from this PDF.")

        # 2. Chunk text efficiently (larger chunks = fewer embedding API calls)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(docs)

        # 3. Normalize metadata to the fields we actually cite
        for chunk in chunks:
            chunk.metadata = _clean_metadata(chunk.metadata, filename)

        # A scanned/image-only PDF splits into zero usable chunks. Reporting
        # "completed" for that was misleading — queries would find nothing.
        if not chunks:
            raise ValueError(
                "No extractable text found. This PDF may be a scanned image; "
                "OCR is required."
            )

        _drop_existing_chunks(filename)

        # 4. Embed & insert into ChromaDB in batches to avoid oversized
        #    single requests and to keep memory/latency predictable for
        #    large documents.
        #
        # Jina's free tier (100 RPM / 100K TPM) is generous enough that a
        # single ingestion won't realistically trip it the way Gemini's
        # embedding quota did — each batch is one HTTP call regardless of
        # size, not one call per chunk. Batches still exist to bound memory
        # and keep a single oversized request from timing out, and retries
        # cover ordinary transient failures (a dropped connection, a brief
        # 5xx) — not a quota wall, so plain exponential backoff is enough
        # here, unlike the Gemini path this replaced.
        BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))
        MAX_BATCH_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "3"))
        total_chunks = len(chunks)
        chunks_indexed = 0

        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            last_err: Optional[Exception] = None
            for attempt in range(1, MAX_BATCH_RETRIES + 1):
                try:
                    vectorstore.add_documents(batch)
                    chunks_indexed += len(batch)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    logger.warning(
                        "Embedding batch %d-%d failed (attempt %d/%d) for %s: %s",
                        i, i + len(batch), attempt, MAX_BATCH_RETRIES, filename, e,
                    )
                    if attempt < MAX_BATCH_RETRIES:
                        is_rate_limited = "429" in str(e)
                        # Jina's limit is per-minute like most providers; a
                        # short fixed wait clears it without the multi-minute
                        # stall Gemini's quota needed.
                        time.sleep(15 if is_rate_limited else 2 ** (attempt - 1))
            if last_err is not None:
                # Keep whatever was already indexed instead of dropping it —
                # a partial index is still queryable — but report the gap
                # clearly rather than claiming full success.
                raise RuntimeError(
                    f"Embedding failed after {MAX_BATCH_RETRIES} attempts on "
                    f"chunks {i}-{i + len(batch)} of {total_chunks} "
                    f"({chunks_indexed} chunks were indexed before this point): "
                    f"{type(last_err).__name__}: {last_err}"
                )

        # 5. Update completed status
        ingestion_jobs[filename] = {
            "status": "completed",
            "filename": filename,
            "pages": len(docs),
            "chunks": total_chunks,
            "text_chunks": total_chunks,
            "visual_chunks": 0,
        }
    except Exception as e:
        logger.exception("Ingestion failed for %s", filename)
        ingestion_jobs[filename] = {
            "status": "failed",
            "filename": filename,
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        # Single syscall on the common path instead of exists()+remove()
        # (avoids a redundant stat() call and a TOCTOU race).
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


# ── Endpoints ──


@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": "FastAPI RAG Pipeline Active",
        "collection": COLLECTION_NAME,
        "persist_dir": str(PERSIST_DIR),
    }


@app.post("/ingest")
@app.post("/upload")  # README documents /upload; keep both working.
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    filename = sanitize_filename(file.filename)

    # `.endswith(".pdf")` is case-sensitive, so a perfectly valid `REPORT.PDF`
    # was rejected while the frontend happily accepted it.
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Never build the temp path from client input. mkstemp also avoids the
    # collision where two users uploading "notes.pdf" overwrite each other.
    fd, temp_path = tempfile.mkstemp(prefix="docagent_", suffix=".pdf")
    os.close(fd)

    written = 0
    first_bytes = b""
    try:
        with open(temp_path, "wb") as buffer:
            # `shutil.copyfileobj` is blocking and ran directly inside this
            # `async def`, stalling the event loop for the whole upload — the
            # health check and every other request froze behind a large PDF.
            while chunk := await file.read(1024 * 1024):
                if not first_bytes:
                    first_bytes = chunk[:5]
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "File exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit."
                        ),
                    )
                await run_in_threadpool(buffer.write, chunk)

        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # The extension was the only gate, so a renamed .exe or zip bomb was
        # accepted and only failed deep inside the background task. Check the
        # actual PDF magic number instead of trusting the name.
        if not first_bytes.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="File is not a valid PDF (missing %PDF- header).",
            )
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise

    ingestion_jobs[filename] = {"status": "processing", "filename": filename}
    background_tasks.add_task(process_pdf_in_background, temp_path, filename)

    return {"filename": filename, "status": "processing", "message": "Ingestion initiated."}


def _already_indexed(filename: str) -> bool:
    """Chroma persists across restarts but `ingestion_jobs` does not, so after a
    server restart /status reported `not_found` for a fully indexed document and
    the UI showed 'Lost track of ingestion progress'. Consult the vector store
    before giving up."""
    try:
        found = vectorstore.get(where={"source": filename}, limit=1, include=[])
        return bool(found.get("ids"))
    except Exception:
        return False


@app.get("/status/{filename:path}")
@app.get("/ingest/status/{filename:path}")  # what the frontend requests
def check_status(filename: str):
    job = ingestion_jobs.get(filename)
    if job:
        return job
    if _already_indexed(filename):
        return {"status": "completed", "filename": filename}
    return {"status": "not_found", "filename": filename}


@app.post("/title", response_model=TitleResponse)
def generate_title(payload: TitleRequest):
    """Summarize a user's first message into a short chat-history title.

    Deliberately separate from /query: this needs no document retrieval, so
    it skips the vector store and Jina embeddings entirely and only spends
    one small Gemini call (same GEMINI_API_KEY /query already uses) — the
    RAG pipeline's LLM, reused for a much cheaper job than answering a
    question over document context.
    """
    message = payload.message.strip()

    prompt = f"""Summarize the following chat message into a short title for a
chat history list, similar to how ChatGPT or Claude name conversations.

Rules:
- 2 to 6 words.
- No quotation marks, no trailing punctuation.
- Plain title case, no markdown.
- Capture the topic, not the literal phrasing of a question.

Message:
{message}

Title:"""

    try:
        response = llm.invoke(prompt)
        title = _coerce_answer(response).strip().strip('"').strip("'")
        # Guard against the model ignoring the length instruction — a title
        # this UI truncates anyway shouldn't be allowed to explode into a
        # full sentence in the sidebar.
        if len(title) > 60:
            title = title[:60].rstrip() + "…"
        if not title:
            raise ValueError("empty title")
        return TitleResponse(title=title)
    except Exception:
        # Titling is a nicety, not core functionality — never let it 500 and
        # break sending a message. Frontend falls back to truncating the
        # message itself if this endpoint fails for any reason.
        logger.exception("Title generation failed; frontend will fall back to truncation.")
        raise HTTPException(status_code=503, detail="Title generation unavailable.")


@app.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest):
    question = payload.query.strip()

    try:
        search_filter = {"source": payload.source} if payload.source else None

        # Use the scoring variant so responses can carry `relevance_score`,
        # which the README and the frontend's SourceMetadata both advertise but
        # the retriever-only path never populated.
        try:
            scored = vectorstore.similarity_search_with_relevance_scores(
                question, k=payload.top_k, filter=search_filter
            )
        except Exception:
            scored = [
                (doc, None)
                for doc in vectorstore.similarity_search(
                    question, k=payload.top_k, filter=search_filter
                )
            ]

        if not scored:
            return QueryResponse(
                query=question,
                answer="No relevant content found in uploaded documents.",
                sources=[],
            )

        context_parts = []
        sources = []

        for doc, score in scored:
            context_parts.append(doc.page_content)
            page_num = doc.metadata.get("page", 1)
            sources.append(
                SourceMetadata(
                    source=doc.metadata.get("source", "Document"),
                    page=page_num if isinstance(page_num, int) else 1,
                    chunk_type=doc.metadata.get("chunk_type", "text"),
                    relevance_score=round(score, 4) if isinstance(score, (int, float)) else None,
                )
            )

        context_text = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant analyzing user documents.
Answer the following question using only the provided context below. If you do not know the answer based on the context, state that clearly.

Context:
{context_text}

Question: {question}

Answer:"""

        response = llm.invoke(prompt)

        return QueryResponse(
            query=question,
            answer=_coerce_answer(response),
            sources=sources,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Log the full exception server-side; return a short, non-leaky detail.
        # The old handler echoed raw Google API text straight to the browser.
        logger.exception("Query failed")
        message = str(e).lower()
        if "quota" in message or "rate limit" in message or "429" in message:
            raise HTTPException(
                status_code=429,
                detail="Gemini API quota exceeded. Please retry shortly.",
            )
        if "api key" in message or "unauthenticated" in message or "permission" in message:
            raise HTTPException(
                status_code=503,
                detail="Gemini API rejected the credentials. Check GOOGLE_API_KEY in .env.",
            )
        if "not_found" in message or "no longer available" in message:
            # Google periodically retires model names outright (this has
            # already happened twice for this project — 1.5-flash, then
            # 2.5-flash). The error text from Google usually names the
            # replacement, so surface it rather than a bare 500.
            raise HTTPException(
                status_code=503,
                detail=(
                    f"The configured Gemini model ('{LLM_MODEL}') is no longer "
                    "available. Google's response: "
                    f"{str(e)[:300]}. Update LLM_MODEL in .env to a model from "
                    "GET https://generativelanguage.googleapis.com/v1beta/models"
                    "?key=YOUR_KEY."
                ),
            )
        raise HTTPException(
            status_code=500,
            detail=f"Query failed ({type(e).__name__}). See server logs for details.",
        )