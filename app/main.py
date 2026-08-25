"""FastAPI app for the RAG Document Agent.

Session isolation: every chunk is tagged with the caller's `X-Session-Id` and
`X-Chat-Id`, and every retrieval filters on both. Without those headers a client
falls back to a shared `default_session:default_chat` bucket, so the frontend
must always send them (see frontend/src/lib/session.ts).

Vectors live in an in-memory Chroma collection. That is deliberate — Render's
free tier has 512MB of RAM and no persistent disk — but it means a process
restart or an idle spin-down drops every uploaded document.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import requests
from chromadb import EphemeralClient
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from .document_loaders import (
    LEGACY_FORMAT_ADVICE,
    SUPPORTED_EXTENSIONS,
    load_document,
)
from .gemini_client import GeminiClient, GeminiExhaustedError

# Must run before the os.getenv calls below. Without this the whole service used
# to die at import time with "No Gemini API key found", because .env was never
# read. On Render the real env vars are already set, so this is a no-op there.
load_dotenv()

logger = logging.getLogger("docagent.api")

# TLS-inspecting antivirus and corporate proxies (Avast Web Shield, Zscaler,
# Kaspersky) re-sign every certificate with a private root that is installed in
# the OS trust store but absent from certifi's bundle. Python then rejects all
# outbound HTTPS with CERTIFICATE_VERIFY_FAILED while curl and browsers work
# fine, which looks like a broken API key but isn't. Opting into the OS trust
# store resolves it without ever disabling verification. Off by default because
# Render's Linux image already has a correct CA bundle.
if os.getenv("USE_SYSTEM_CERT_STORE", "").strip().lower() in {"1", "true", "yes"}:
    try:
        import truststore

        truststore.inject_into_ssl()
        logger.info("Outbound TLS is using the OS certificate trust store.")
    except ImportError:
        logger.warning(
            "USE_SYSTEM_CERT_STORE is set but the 'truststore' package is not "
            "installed; falling back to certifi."
        )

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v2-base-en")

# Jina rejects oversized requests, so chunks go out in batches rather than as one
# giant payload. 64 keeps each request small; 4 workers is enough to hide network
# latency without hammering Render's shared CPU.
JINA_BATCH_SIZE = int(os.getenv("JINA_BATCH_SIZE", "64"))
JINA_MAX_WORKERS = int(os.getenv("JINA_MAX_WORKERS", "4"))
JINA_TIMEOUT_SECONDS = 60
JINA_MAX_ATTEMPTS = 3

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Bound the work so a huge upload returns a clear error instead of hanging until
# the platform kills the request.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "1500"))

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
CLEANUP_INTERVAL_SECONDS = 300

RETRIEVAL_K = 4


# ------------------------------------------------------------------------------
# 2. Embeddings
# ------------------------------------------------------------------------------
class JinaEmbeddings(Embeddings):
    """Jina's REST embedding API behind the LangChain Embeddings interface.

    Calling the API over HTTP rather than running a local model is what keeps the
    service inside 512MB — torch + sentence-transformers alone would exceed it.

    Requests are batched and issued concurrently. Before this, `embed_documents`
    POSTed every chunk of a document in a single request, which is why large PDFs
    failed outright: a 200-page file produces ~600 inputs and the request either
    got rejected or timed out.
    """

    def __init__(
        self,
        api_key: str,
        model: str = JINA_MODEL,
        batch_size: int = JINA_BATCH_SIZE,
        max_workers: int = JINA_MAX_WORKERS,
    ):
        self.url = "https://api.jina.ai/v1/embeddings"
        self.model = model
        self.batch_size = max(1, batch_size)
        # One Session for the process so TLS handshakes and TCP connections are
        # reused across batches instead of being rebuilt for every call.
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="jina"
        )

    def _post_batch(self, batch: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": batch}
        last_error: Optional[BaseException] = None

        for attempt in range(1, JINA_MAX_ATTEMPTS + 1):
            try:
                res = self._session.post(
                    self.url, json=payload, timeout=JINA_TIMEOUT_SECONDS
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                if res.ok:
                    # Jina returns an `index` per item. Sorting on it means a
                    # reordered response can't silently misalign embeddings with
                    # their chunks.
                    items = sorted(
                        res.json().get("data", []), key=lambda d: d.get("index", 0)
                    )
                    return [item["embedding"] for item in items]

                if res.status_code == 429 or res.status_code >= 500:
                    last_error = RuntimeError(
                        f"Jina returned {res.status_code}: {res.text[:200]}"
                    )
                else:
                    # A 4xx that isn't rate limiting (bad key, input too long)
                    # will fail identically on retry.
                    raise RuntimeError(
                        f"Jina embedding request failed "
                        f"({res.status_code}): {res.text[:300]}"
                    )

            if attempt < JINA_MAX_ATTEMPTS:
                backoff = 2 ** (attempt - 1)
                logger.warning(
                    "Jina batch failed (attempt %s/%s), retrying in %ss: %s",
                    attempt,
                    JINA_MAX_ATTEMPTS,
                    backoff,
                    last_error,
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"Jina embedding request failed after {JINA_MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        if len(batches) == 1:
            return self._post_batch(batches[0])

        # Executor.map preserves input order, so flattening keeps every embedding
        # aligned with the chunk it came from.
        embedded = list(self._executor.map(self._post_batch, batches))
        return [vector for batch in embedded for vector in batch]

    def embed_query(self, text: str) -> List[float]:
        result = self._post_batch([text])
        return result[0] if result else []


jina_embeddings = JinaEmbeddings(api_key=JINA_API_KEY) if JINA_API_KEY else None

if jina_embeddings is None:
    logger.warning("JINA_API_KEY is not set — /api/ingest and /api/query will 503.")

# In-memory (ephemeral) Chroma client — zero disk storage.
chroma_client = EphemeralClient()
vector_store = Chroma(
    client=chroma_client,
    collection_name="ephemeral_docs",
    embedding_function=jina_embeddings,
)

# Replaces the old manual character-slicing loop. Splitting on paragraph and
# sentence boundaries first keeps chunks semantically whole, which measurably
# improves what retrieval returns for the same chunk size.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# ------------------------------------------------------------------------------
# 3. Gemini (lazily constructed)
# ------------------------------------------------------------------------------
_gemini_client: Optional[GeminiClient] = None
_gemini_error: Optional[str] = None


def get_gemini_client() -> GeminiClient:
    """Build the Gemini client on first use.

    This used to be a module-level `GeminiClient()`. A missing or misspelled API
    key therefore raised during import and took the entire service down — every
    route 502'd, including the health check. Deferring it means a bad key
    degrades to a 503 on /api/query only.
    """
    global _gemini_client, _gemini_error

    if _gemini_client is not None:
        return _gemini_client
    if _gemini_error is not None:
        raise HTTPException(status_code=503, detail=_gemini_error)

    try:
        _gemini_client = GeminiClient()
    except Exception as exc:
        _gemini_error = f"AI service is not configured: {exc}"
        logger.error("Failed to initialise Gemini client: %s", exc)
        raise HTTPException(status_code=503, detail=_gemini_error) from exc

    return _gemini_client


# ------------------------------------------------------------------------------
# 4. Session tracking & cleanup
# ------------------------------------------------------------------------------
# { f"{session_id}:{chat_id}": last_activity_timestamp }
ACTIVE_SESSIONS: Dict[str, float] = {}


def touch_session(session_id: str, chat_id: str) -> None:
    ACTIVE_SESSIONS[f"{session_id}:{chat_id}"] = time.time()


def session_filter(session_id: str, chat_id: str) -> Dict[str, Any]:
    return {
        "$and": [
            {"session_id": {"$eq": session_id}},
            {"chat_id": {"$eq": chat_id}},
        ]
    }


def drop_session_vectors(session_id: str, chat_id: str) -> None:
    """Delete every chunk belonging to one session/chat pair.

    Reaches into `_collection` because langchain-chroma exposes no public
    metadata-filtered delete.
    """
    vector_store._collection.delete(where=session_filter(session_id, chat_id))


async def cleanup_stale_sessions() -> None:
    """Purge vectors for chats that have gone quiet, freeing RAM."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        now = time.time()

        for key, last_seen in list(ACTIVE_SESSIONS.items()):
            if now - last_seen <= SESSION_TTL_SECONDS:
                continue

            session_id, _, chat_id = key.partition(":")
            try:
                drop_session_vectors(session_id, chat_id)
            except Exception as exc:
                # Previously `except Exception: pass`, so a cleanup failure was
                # invisible while memory kept growing.
                logger.warning("Failed to purge stale session %s: %s", key, exc)
            ACTIVE_SESSIONS.pop(key, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # `@app.on_event("startup")` is deprecated in the pinned FastAPI.
    cleanup_task = asyncio.create_task(cleanup_stale_sessions())
    try:
        yield
    finally:
        cleanup_task.cancel()


# ------------------------------------------------------------------------------
# 5. App & middleware
# ------------------------------------------------------------------------------
app = FastAPI(title="RAG Document Agent", lifespan=lifespan)

# `allow_origins=["*"]` together with `allow_credentials=True` is a combination
# browsers reject outright: the spec forbids a literal `*` on
# Access-Control-Allow-Origin when credentials are included, so every preflight
# failed. Set ALLOWED_ORIGINS (comma-separated, e.g. the Vercel URL) in
# production; the permissive branch is for local development only.
_allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins or ["*"],
    allow_credentials=bool(_allowed_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    # The frontend sends top_k; accept and ignore rather than 422 on it.
    top_k: Optional[int] = None


# ------------------------------------------------------------------------------
# 6. Helpers
# ------------------------------------------------------------------------------
# Anchored to the two paginated label kinds on purpose. A looser "trailing
# digits" match turned a worksheet named "Q1" into page 1.
_PAGE_LABEL = re.compile(r"^(?:page|slide)\s+(\d+)$", re.IGNORECASE)


def page_number_from_label(label: str) -> Optional[int]:
    """Pull an integer page/slide number out of a unit label.

    Labels look like "page 3", "slide 7", "sheet Q1 Revenue", "document".
    Only the paginated ones yield a page, which is what the UI's source cards
    show; everything else is described by `location` alone.
    """
    match = _PAGE_LABEL.match(label.strip())
    return int(match.group(1)) if match else None


def safe_filename(raw: Optional[str]) -> str:
    """Reduce an uploaded filename to a bare, safe basename.

    The previous code interpolated `file.filename` straight into a path, so an
    upload named `../../etc/whatever` wrote outside the upload directory.
    """
    candidate = os.path.basename((raw or "").replace("\\", "/")).strip()
    if not candidate or candidate in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid or missing filename.")
    return candidate


async def save_upload(file: UploadFile, destination: str) -> int:
    """Stream an upload to disk, enforcing the size cap as it goes."""
    total = 0
    with open(destination, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File is larger than the "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit."
                    ),
                )
            buffer.write(chunk)
    return total


def message_to_text(response: Any) -> str:
    """Flatten a LangChain chat response into plain text.

    Current langchain-google-genai returns `content` as a list of typed blocks —
    `[{"type": "text", "text": "...", "extras": {...}}]` — not a string. Reading
    `.content` directly therefore put a JSON array in the `answer` field, and the
    UI rendered the raw structure instead of the answer.
    """
    # langchain-core exposes a `.text` convenience property that already does the
    # right thing. Test for `str` before `callable`: the backwards-compatibility
    # shim is both a string and callable, and invoking it emits a deprecation
    # warning on every answer.
    text = getattr(response, "text", None)
    if isinstance(text, str):
        if text.strip():
            return text
    elif callable(text):
        try:
            called = text()
        except Exception:  # pragma: no cover - defensive
            called = None
        if isinstance(called, str) and called.strip():
            return called

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                value = block.get("text")
                if isinstance(value, str) and value:
                    parts.append(value)
        if parts:
            return "".join(parts)

    return str(content)


# ------------------------------------------------------------------------------
# 7. Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "RAG Document Agent API is running.",
        "supported_formats": sorted(SUPPORTED_EXTENSIONS),
        "embeddings_configured": jina_embeddings is not None,
    }


@app.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(default="default_session"),
    x_chat_id: Optional[str] = Header(default="default_chat"),
):
    if jina_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service is not configured (JINA_API_KEY is missing).",
        )

    filename = safe_filename(file.filename)
    extension = os.path.splitext(filename)[1].lower()

    # Reject unreadable formats before spending time writing the file to disk.
    if extension in LEGACY_FORMAT_ADVICE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{extension} is an older binary Office format this server can't "
                f"read. {LEGACY_FORMAT_ADVICE[extension]}"
            ),
        )
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{extension or 'unknown'}'. Supported: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            ),
        )

    touch_session(x_session_id, x_chat_id)

    # A private directory per request, so two users uploading the same filename
    # at the same time can't overwrite each other's file mid-parse.
    temp_dir = tempfile.mkdtemp(prefix="docagent_")
    temp_path = os.path.join(temp_dir, filename)

    try:
        await save_upload(file, temp_path)

        try:
            doc_data = load_document(temp_path, original_filename=filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to parse %s", filename)
            raise HTTPException(
                status_code=422, detail=f"Could not read this document: {exc}"
            ) from exc

        units = doc_data.get("units") or []
        if not any(text.strip() for _, text in units):
            raise HTTPException(
                status_code=400,
                detail=(
                    "No text could be extracted from this document. If it is a "
                    "scanned PDF, it needs OCR before upload."
                ),
            )

        # Chunk each page/slide/sheet on its own so a chunk never straddles two
        # of them and every chunk keeps a location the UI can cite.
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for label, unit_text in units:
            if not unit_text.strip():
                continue
            page = page_number_from_label(label)
            for chunk in text_splitter.split_text(unit_text):
                if not chunk.strip():
                    continue
                metadata: Dict[str, Any] = {
                    "session_id": x_session_id,
                    "chat_id": x_chat_id,
                    "file_name": filename,
                    "location": label,
                    "chunk_index": len(texts),
                }
                # Chroma rejects None-valued metadata, so only set `page` when
                # the label actually carries a number.
                if page is not None:
                    metadata["page"] = page
                texts.append(chunk)
                metadatas.append(metadata)

        if not texts:
            raise HTTPException(
                status_code=400, detail="Document produced no indexable text."
            )

        if len(texts) > MAX_CHUNKS_PER_DOC:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Document is too large to index ({len(texts)} chunks, limit "
                    f"{MAX_CHUNKS_PER_DOC}). Split it into smaller files."
                ),
            )

        started = time.monotonic()
        # add_texts is synchronous and network-bound; off-thread so it doesn't
        # block the event loop for the whole upload.
        await asyncio.to_thread(vector_store.add_texts, texts, metadatas)
        elapsed = time.monotonic() - started

        logger.info(
            "Ingested %s: %s chunks from %s units in %.1fs",
            filename,
            len(texts),
            len(units),
            elapsed,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "status": "success",
        # Keyed `filename` to match what the frontend reads. It used to be
        # `file_name`, so the UI always showed `undefined`.
        "filename": filename,
        "chunks": len(texts),
        "pages": len(units),
        "message": f"Indexed {len(texts)} chunks from {filename}.",
    }


@app.post("/api/query")
async def query_document(
    payload: QueryRequest,
    x_session_id: Optional[str] = Header(default="default_session"),
    x_chat_id: Optional[str] = Header(default="default_chat"),
):
    question = payload.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    if jina_embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service is not configured (JINA_API_KEY is missing).",
        )

    touch_session(x_session_id, x_chat_id)

    try:
        # The _with_relevance_scores variant is what makes the UI's source cards
        # able to show a confidence value.
        scored = await asyncio.to_thread(
            vector_store.similarity_search_with_relevance_scores,
            question,
            RETRIEVAL_K,
            filter=session_filter(x_session_id, x_chat_id),
        )
    except Exception as exc:
        # Retrieval failing is not fatal — answer without context — but it must
        # be visible in the logs rather than silently swallowed.
        logger.warning("Retrieval failed for session %s: %s", x_session_id, exc)
        scored = []

    sources = [
        {
            "source": doc.metadata.get("file_name"),
            "page": doc.metadata.get("page"),
            "location": doc.metadata.get("location"),
            "chunk_type": "text",
            "relevance_score": round(float(score), 4),
        }
        for doc, score in scored
    ]

    context = "\n\n".join(
        f"[{doc.metadata.get('file_name', 'document')}"
        f" — {doc.metadata.get('location', 'unknown')}]\n{doc.page_content}"
        for doc, _ in scored
    )

    if not context.strip():
        context = "No relevant context found in uploaded documents for this chat."

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly using the context provided above. Cite the document and "
        "page/slide you drew from. If the context is missing or irrelevant, say so."
    )

    client = get_gemini_client()

    try:
        response = await asyncio.to_thread(client.invoke, prompt)
        answer = message_to_text(response)
    except GeminiExhaustedError as exc:
        raise HTTPException(
            status_code=503, detail=f"AI service currently unavailable: {exc}"
        ) from exc
    except Exception as exc:
        logger.exception("Answer generation failed")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate answer: {exc}"
        ) from exc

    return {"query": question, "answer": answer, "sources": sources}


@app.delete("/api/session/clear")
async def clear_chat_session(
    x_session_id: Optional[str] = Header(default="default_session"),
    x_chat_id: Optional[str] = Header(default="default_chat"),
):
    """Wipe one chat's vectors — called on New Chat and on tab close."""
    try:
        drop_session_vectors(x_session_id, x_chat_id)
    except Exception as exc:
        logger.warning(
            "Failed to clear session %s:%s: %s", x_session_id, x_chat_id, exc
        )
        raise HTTPException(
            status_code=500, detail="Could not clear this chat's context."
        ) from exc

    ACTIVE_SESSIONS.pop(f"{x_session_id}:{x_chat_id}", None)
    return {"status": "success", "message": "Chat vector context wiped successfully."}
