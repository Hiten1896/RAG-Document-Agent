import logging
import os
import re
import tempfile
import threading
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

# LangChain, HuggingFace & Google GenAI Imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# `langchain_community.vectorstores.Chroma` is the deprecated shim; the
# maintained integration lives in the `langchain-chroma` package (already in
# requirements.txt, and what rag_agent_project.ipynb uses).
from langchain_chroma import Chroma

# fitz (PyMuPDF) is used directly, not just via PyMuPDFLoader, to rasterize
# individual pages for OCR fallback. Reusing PyMuPDF for this (instead of
# adding poppler/pdf2image) avoids a second, OS-level dependency on top of
# Tesseract — poppler needs its own separate installer on Windows.
import fitz

# OCR is optional: pytesseract needs the Tesseract binary installed
# separately (not pip-installable), so a fresh clone without it would
# otherwise crash at import time before the server could even start.
# Import lazily-guarded so the rest of the app still works — scanned PDFs
# just get a clear "install Tesseract" error instead of ingestion silently
# being unavailable app-wide.
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

load_dotenv()

logger = logging.getLogger("docagent")

# ── API Key Verification (Supports both GEMINI_API_KEY and GOOGLE_API_KEY) ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "No Gemini API key found. Copy .env.example to .env and set "
        "GOOGLE_API_KEY (or GEMINI_API_KEY) to a key from "
        "https://aistudio.google.com/."
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
# collection. Embeddings are local HuggingFace MiniLM (384-dim) again, so this
# collection is dimensionally incompatible with any "docagent_gemini" data
# from a prior Gemini-embeddings run — those chunks won't be found by
# similarity search here and would need re-ingesting into this collection.
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "docagent_hf")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# `gemini-2.5-flash` has been retired from the v1beta model catalog for
# accounts provisioned after the 2.5 line was sunset — Google's own 404 for
# this key names `gemini-3.6-flash` as the direct replacement, so that's the
# new default. (An even newer `gemini-3.7-flash` also exists if a future
# migration is needed — set LLM_MODEL to override without a code change.)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024

# ── OCR Configuration ──
# A per-page character threshold, not a whole-document one. A 40-page PDF
# where only 2 pages are scanned photos still needs those 2 pages OCR'd —
# checking total document length would let a few good text pages mask a
# batch of image-only ones, silently losing that content instead of
# recovering it.
MIN_CHARS_PER_PAGE = int(os.getenv("MIN_CHARS_PER_PAGE", "20"))
# Render scanned pages at higher DPI than the PDF's default (usually 72) —
# Tesseract's accuracy drops sharply below ~200 DPI, especially on
# handwriting or small/photocopied text.
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
if os.getenv("TESSERACT_CMD"):
    if OCR_AVAILABLE:
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

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
# Embeddings run locally (HuggingFace sentence-transformers), not through the
# Gemini API. This removes the embedding path entirely from Gemini's quota —
# ingestion does one API call per question, not per chunk. The LLM below still
# uses Gemini for generation, so GEMINI_API_KEY is still required.
#
# `all-MiniLM-L6-v2` is fixed at 384 dimensions (no Matryoshka-style variable
# output like `gemini-embedding-001`), so there is no dimensionality knob to
# pin here — every call from this model produces 384-dim vectors by
# construction, which is what keeps ingestion and query embeddings compatible.
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(PERSIST_DIR),
    embedding_function=embeddings,
)

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


# ── OCR Helpers ──
def _ocr_page(pdf_doc: "fitz.Document", page_index: int) -> str:
    """Rasterize one page and run Tesseract on it.

    Only called for pages PyMuPDF's text layer already failed on, so this
    never runs on normal typed pages — it's purely the scanned/handwritten
    fallback path, keeping ingestion of typed PDFs exactly as fast as before.
    """
    page = pdf_doc[page_index]
    zoom = OCR_DPI / 72  # PyMuPDF's base unit is 72 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    # Tesseract's default English-only model reads clean typed scans well,
    # but does noticeably worse on handwriting. There's no free, reliably
    # accurate handwriting-OCR engine to swap in here, so this is a
    # best-effort pass — messy handwriting may still come out garbled, and
    # that's a genuine limitation, not a bug being masked.
    return pytesseract.image_to_string(img)


def _extract_pdf_pages(file_path: str, filename: str) -> tuple[list, dict]:
    """Extract text per page, OCR'ing any page whose text layer is empty or
    near-empty (scanned image, or a page that's a photo of handwriting).

    Returns (docs, page_stats) where docs is a list of langchain Documents
    (one per page that ended up with usable text) and page_stats records how
    many pages came from the text layer vs. OCR vs. were unrecoverable, so
    the ingestion job can report this honestly instead of a flat
    success/fail.
    """
    from langchain_core.documents import Document

    loader = PyMuPDFLoader(file_path)
    raw_docs = loader.load()  # one Document per page, 0-indexed page numbers

    if not raw_docs:
        raise ValueError("No pages could be read from this PDF.")

    docs: list = []
    stats = {"total_pages": len(raw_docs), "text_pages": 0, "ocr_pages": 0, "empty_pages": 0}

    pdf_doc = None
    try:
        for i, raw in enumerate(raw_docs):
            text = (raw.page_content or "").strip()

            if len(text) >= MIN_CHARS_PER_PAGE:
                stats["text_pages"] += 1
                docs.append(raw)
                continue

            # This page's text layer is empty or near-empty: likely a scanned
            # image, a photo of handwritten notes, or a blank/decorative
            # page. Try OCR before giving up on it.
            if not OCR_AVAILABLE:
                stats["empty_pages"] += 1
                continue

            if pdf_doc is None:
                pdf_doc = fitz.open(file_path)

            try:
                ocr_text = _ocr_page(pdf_doc, i).strip()
            except Exception as e:
                logger.warning("OCR failed on page %d of %s: %s", i + 1, filename, e)
                ocr_text = ""

            if len(ocr_text) >= MIN_CHARS_PER_PAGE:
                stats["ocr_pages"] += 1
                raw.page_content = ocr_text
                raw.metadata["chunk_type"] = "ocr"
                docs.append(raw)
            else:
                # Genuinely blank page, or handwriting OCR couldn't recover
                # enough to be useful. Counted, not silently dropped.
                stats["empty_pages"] += 1
    finally:
        if pdf_doc is not None:
            pdf_doc.close()

    return docs, stats


def _clean_metadata(meta: dict, filename: str) -> dict:
    page = meta.get("page")
    cleaned = {
        "source": filename,
        # PyMuPDFLoader pages are 0-based; citations should read 1-based.
        "page": (page + 1) if isinstance(page, int) else 1,
        # Preserve "ocr" if _extract_pdf_pages already tagged this page;
        # otherwise it's normal extracted text.
        "chunk_type": meta.get("chunk_type", "text"),
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

        # 1. Load document, OCR'ing any page whose text layer is empty or
        #    near-empty. Handles the mixed case (some typed pages, some
        #    scanned/handwritten pages in the same PDF) as well as fully
        #    scanned documents — previously a single page with no text
        #    layer just produced an empty-content Document that silently
        #    contributed nothing, with no indication that had happened.
        docs, page_stats = _extract_pdf_pages(file_path, filename)

        if not docs:
            if page_stats["empty_pages"] == page_stats["total_pages"] and not OCR_AVAILABLE:
                raise ValueError(
                    "No extractable text found on any page, and OCR is not "
                    "available on this server (Tesseract is not installed). "
                    "Install Tesseract OCR and the pytesseract/Pillow "
                    "packages to support scanned or handwritten PDFs."
                )
            raise ValueError(
                "No extractable text found on any page, even after OCR. "
                "This PDF may be blank, corrupted, or contain handwriting "
                "too unclear for OCR to recover."
            )

        # 2. Chunk text efficiently (larger chunks = fewer embedding API calls)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(docs)

        # 3. Normalize metadata to the fields we actually cite
        for chunk in chunks:
            chunk.metadata = _clean_metadata(chunk.metadata, filename)

        # Chunking can still land on zero real chunks if every recovered
        # page was just whitespace/punctuation that split_documents then
        # dropped. Distinct from the docs-empty case above.
        if not chunks:
            raise ValueError(
                "Pages were read, but no usable text chunks were produced. "
                "The document may contain only images, tables, or very "
                "sparse text that couldn't be meaningfully split."
            )

        _drop_existing_chunks(filename)

        # 4. Embed & insert into ChromaDB in batches. Embeddings run locally
        #    now (HuggingFace), so there's no API quota/rate limit to retry
        #    against — batching here is just to bound memory/latency per call
        #    for very large documents, not to defend against transient
        #    network failures.
        BATCH_SIZE = 100
        total_chunks = len(chunks)
        chunks_indexed = 0

        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            try:
                vectorstore.add_documents(batch)
                chunks_indexed += len(batch)
            except Exception as e:
                # Keep whatever was already indexed instead of dropping it —
                # a partial index is still queryable — but report the gap
                # clearly rather than claiming full success.
                raise RuntimeError(
                    f"Embedding failed on chunks {i}-{i + len(batch)} of "
                    f"{total_chunks} ({chunks_indexed} chunks were indexed "
                    f"before this point): {type(e).__name__}: {e}"
                )

        # 5. Update completed status. Surfacing page_stats here (rather than
        #    just "completed") is what makes a partially-scanned PDF visible
        #    to the user — e.g. "40 pages, 3 recovered via OCR, 1 unreadable"
        #    instead of a bare success that hides which pages didn't make it
        #    into the index and therefore can't be queried.
        ingestion_jobs[filename] = {
            "status": "completed",
            "filename": filename,
            "pages": page_stats["total_pages"],
            "pages_from_text": page_stats["text_pages"],
            "pages_from_ocr": page_stats["ocr_pages"],
            "pages_unreadable": page_stats["empty_pages"],
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


@app.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest):
    question = payload.query.strip()

    try:
        # If the caller scoped the question to one document, check whether
        # that document is still mid-ingestion or failed outright. Without
        # this, a question asked seconds after upload silently returns "no
        # relevant content found" — indistinguishable from a real empty
        # result — instead of telling the user ingestion isn't done yet.
        if payload.source:
            job = ingestion_jobs.get(payload.source)
            if job and job.get("status") == "processing":
                return QueryResponse(
                    query=question,
                    answer=(
                        f"'{payload.source}' is still being processed. "
                        "Please wait for ingestion to complete before asking "
                        "questions about it."
                    ),
                    sources=[],
                )
            if job and job.get("status") == "failed":
                return QueryResponse(
                    query=question,
                    answer=(
                        f"'{payload.source}' failed to ingest "
                        f"({job.get('error', 'unknown error')}), so there is "
                        "no content to search."
                    ),
                    sources=[],
                )

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
        any_ocr = False

        for doc, score in scored:
            chunk_type = doc.metadata.get("chunk_type", "text")
            if chunk_type == "ocr":
                any_ocr = True
                # Tag OCR'd context inline so the model can weigh it
                # appropriately — OCR output (especially from handwriting)
                # can contain misreadings, and the model should be able to
                # hedge on details from this text rather than repeat a
                # possible misread with full confidence.
                context_parts.append(
                    f"[OCR-extracted text, may contain recognition errors]\n"
                    f"{doc.page_content}"
                )
            else:
                context_parts.append(doc.page_content)

            page_num = doc.metadata.get("page", 1)
            sources.append(
                SourceMetadata(
                    source=doc.metadata.get("source", "Document"),
                    page=page_num if isinstance(page_num, int) else 1,
                    chunk_type=chunk_type,
                    relevance_score=round(score, 4) if isinstance(score, (int, float)) else None,
                )
            )

        context_text = "\n\n---\n\n".join(context_parts)

        ocr_note = (
            "\nSome context below was extracted via OCR from scanned or "
            "handwritten pages and may contain recognition errors. If a "
            "detail looks suspicious (e.g. a garbled word or number), say "
            "so rather than stating it with full confidence.\n"
            if any_ocr else ""
        )

        prompt = f"""You are a helpful assistant analyzing user documents.
Answer the following question using only the provided context below. If you do not know the answer based on the context, state that clearly.
{ocr_note}
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
        raise HTTPException(
            status_code=500,
            detail=f"Query failed ({type(e).__name__}). See server logs for details.",
        )