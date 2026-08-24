import os
import shutil
import time
import asyncio
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_chroma import Chroma
from chromadb import EphemeralClient
import requests

from .document_loaders import load_document
from .gemini_client import GeminiClient, GeminiExhaustedError

app = FastAPI(title="RAG Document Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# 1. Ephemeral ChromaDB & Session Isolation Setup
# ------------------------------------------------------------------------------
# Custom REST Jina Embeddings wrapper to keep memory light on Render free-tier
class JinaEmbeddings:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": "jina-embeddings-v2-base-en", "input": texts}
        res = requests.post(self.url, json=payload, headers=self.headers, timeout=30)
        res.raise_for_status()
        data = res.json()
        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        res = self._embed([text])
        return res[0] if res else []


JINA_API_KEY = os.getenv("JINA_API_KEY", "")
jina_embeddings = JinaEmbeddings(api_key=JINA_API_KEY) if JINA_API_KEY else None

# In-Memory (Ephemeral) Chroma Client - Zero Disk Storage
chroma_client = EphemeralClient()
vector_store = Chroma(
    client=chroma_client,
    collection_name="ephemeral_docs",
    embedding_function=jina_embeddings,
)

# Active chat session tracker for auto-cleanup (In-Memory)
# Structure: { f"{session_id}:{chat_id}": last_activity_timestamp }
ACTIVE_SESSIONS = {}
SESSION_TTL_SECONDS = 1800  # Auto-purge inactive sessions after 30 mins


def touch_session(session_id: str, chat_id: str):
    """Updates the last activity timestamp for a given session/chat pair."""
    key = f"{session_id}:{chat_id}"
    ACTIVE_SESSIONS[key] = time.time()


async def cleanup_stale_sessions():
    """Background task running every 5 minutes to purge stale session vectors."""
    while True:
        await asyncio.sleep(300)
        now = time.time()
        keys_to_delete = []

        for key, last_seen in list(ACTIVE_SESSIONS.items()):
            if now - last_seen > SESSION_TTL_SECONDS:
                session_id, chat_id = key.split(":")
                try:
                    vector_store._collection.delete(
                        where={
                            "$and": [
                                {"session_id": {"$eq": session_id}},
                                {"chat_id": {"$eq": chat_id}},
                            ]
                        }
                    )
                except Exception:
                    pass
                keys_to_delete.append(key)

        for k in keys_to_delete:
            del ACTIVE_SESSIONS[k]


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_stale_sessions())


# ------------------------------------------------------------------------------
# 2. Schemas & Gemini Setup
# ------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str


gemini_client = GeminiClient()

# ------------------------------------------------------------------------------
# 3. Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAG Document Agent API is running."}


@app.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(default="default_session"),
    x_chat_id: Optional[str] = Header(default="default_chat"),
):
    if not jina_embeddings:
        raise HTTPException(
            status_code=500, detail="JINA_API_KEY is not set in environment."
        )

    touch_session(x_session_id, x_chat_id)

    os.makedirs("/tmp/uploads", exist_ok=True)
    temp_path = f"/tmp/uploads/{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc_data = load_document(temp_path)
        content = doc_data.get("content", "")

        if not content.strip():
            raise HTTPException(
                status_code=400, detail="Could not extract text from the document."
            )

        # Simple text chunking
        chunk_size = 1000
        overlap = 200
        chunks = []
        for i in range(0, len(content), chunk_size - overlap):
            chunks.append(content[i : i + chunk_size])

        metadatas = [
            {
                "session_id": x_session_id,
                "chat_id": x_chat_id,
                "file_name": file.filename,
            }
            for _ in chunks
        ]

        vector_store.add_texts(texts=chunks, metadatas=metadatas)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "status": "success",
        "file_name": file.filename,
        "message": f"Ingested {len(chunks)} text chunks for this active chat context.",
    }


@app.post("/api/query")
async def query_document(
    payload: QueryRequest,
    x_session_id: Optional[str] = Header(default="default_session"),
    x_chat_id: Optional[str] = Header(default="default_chat"),
):
    touch_session(x_session_id, x_chat_id)

    try:
        # Retrieve context matching ONLY this specific session + chat thread
        results = vector_store.similarity_search(
            query=payload.query,
            k=4,
            filter={
                "$and": [
                    {"session_id": {"$eq": x_session_id}},
                    {"chat_id": {"$eq": x_chat_id}},
                ]
            },
        )
    except Exception as e:
        results = []

    context = "\n\n".join([doc.page_content for doc in results])

    if not context.strip():
        context = "No relevant context found in uploaded documents for this chat."

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {payload.query}\n\n"
        f"Answer clearly using the context provided above. If the context is missing or irrelevant, say so."
    )

    try:
        response = gemini_client.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
    except GeminiExhaustedError as e:
        raise HTTPException(
            status_code=503, detail=f"AI service currently unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate answer: {str(e)}"
        )

    return {"answer": answer, "sources_count": len(results)}


@app.delete("/api/session/clear")
async def clear_chat_session(
    x_session_id: Optional[str] = Header(default="default_session"),
    x_chat_id: Optional[str] = Header(default="default_chat"),
):
    """Explicit endpoint to wipe memory when user closes tab or clicks New Chat."""
    try:
        vector_store._collection.delete(
            where={
                "$and": [
                    {"session_id": {"$eq": x_session_id}},
                    {"chat_id": {"$eq": x_chat_id}},
                ]
            }
        )
        key = f"{x_session_id}:{x_chat_id}"
        ACTIVE_SESSIONS.pop(key, None)
    except Exception:
        pass
    return {"status": "success", "message": "Chat vector context wiped successfully."}