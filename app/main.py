import os
import shutil
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain & Google GenAI Imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── API Key Verification (Supports both GEMINI_API_KEY and GOOGLE_API_KEY) ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY was found in environment variables.")

# ── FastAPI App Setup ──
app = FastAPI(title="DocAgent RAG Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Embedding & LLM Setup ──
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GEMINI_API_KEY
)

PERSIST_DIR = "./chroma_db"
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2
)

# In-memory status tracker for document ingestion jobs.
# Bounded to MAX_TRACKED_JOBS entries (LRU eviction) so long-running server
# instances don't leak memory as more documents get ingested over time.
# Exposes the same dict-like surface (get/__getitem__/__setitem__/__contains__)
# used elsewhere in this file, so no other call site needs to change.
class _BoundedJobStore:
    __slots__ = ("_store", "_maxlen")

    def __init__(self, maxlen: int = 500):
        self._store: "OrderedDict[str, dict]" = OrderedDict()
        self._maxlen = maxlen

    def __setitem__(self, key: str, value: dict) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._maxlen:
            self._store.popitem(last=False)  # evict oldest (O(1))

    def __getitem__(self, key: str) -> dict:
        value = self._store[key]
        self._store.move_to_end(key)
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str, default=None):
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return default


ingestion_jobs: Dict[str, dict] = _BoundedJobStore(maxlen=500)

# ── Data Models ──
class QueryRequest(BaseModel):
    question: str

class SourceMetadata(BaseModel):
    source: str
    page: Optional[int] = 1
    chunk_type: Optional[str] = "text"

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]

# ── Background Worker ──
def process_pdf_in_background(file_path: str, filename: str):
    try:
        ingestion_jobs[filename] = {"status": "processing", "filename": filename}

        # 1. Load document (PyMuPDF is a C++-backed parser, significantly
        #    faster than pypdf for large textbook-sized PDFs)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # 2. Chunk text efficiently (larger chunks = fewer embedding API calls)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)

        # 3. Add metadata tag
        for chunk in chunks:
            chunk.metadata["source"] = filename
            if "page" in chunk.metadata:
                chunk.metadata["page"] = chunk.metadata["page"] + 1

        # 4. Embed & insert into ChromaDB in batches to avoid oversized
        #    single requests and to keep memory/latency predictable for
        #    large documents. Batches are dereferenced immediately after
        #    use so peak memory stays O(BATCH_SIZE) instead of O(n) extra
        #    copies on top of the `chunks` list itself.
        BATCH_SIZE = 100
        total_chunks = len(chunks)
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            vectorstore.add_documents(batch)
            del batch

        # 5. Update completed status
        ingestion_jobs[filename] = {
            "status": "completed",
            "filename": filename,
            "pages": len(docs),
            "chunks": total_chunks,
            "text_chunks": total_chunks,
            "visual_chunks": 0
        }
    except Exception as e:
        ingestion_jobs[filename] = {
            "status": "failed",
            "filename": filename,
            "error": str(e)
        }
    finally:
        # Single syscall on the common path instead of exists()+remove()
        # (avoids a redundant stat() call and a TOCTOU race).
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

# ── Endpoints ──

@app.get("/")
def health_check():
    return {"status": "online", "message": "FastAPI RAG Pipeline Active"}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    filename = file.filename

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"upload_{filename}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingestion_jobs[filename] = {"status": "processing", "filename": filename}

    background_tasks.add_task(process_pdf_in_background, temp_path, filename)

    return {"filename": filename, "status": "processing", "message": "Ingestion initiated."}

@app.get("/status/{filename}")
def check_status(filename: str):
    job = ingestion_jobs.get(filename)
    if not job:
        return {"status": "not_found", "filename": filename}
    return job

@app.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(payload.question)

        if not retrieved_docs:
            return QueryResponse(
                answer="No relevant content found in uploaded documents.",
                sources=[]
            )

        context_parts = []
        sources = []

        for doc in retrieved_docs:
            context_parts.append(doc.page_content)
            source_name = doc.metadata.get("source", "Document")
            page_num = doc.metadata.get("page", 1)
            
            sources.append(SourceMetadata(
                source=source_name,
                page=page_num,
                chunk_type="text"
            ))

        context_text = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant analyzing user documents. 
Answer the following question using only the provided context below. If you do not know the answer based on the context, state that clearly.

Context:
{context_text}

Question: {payload.question}

Answer:"""

        response = llm.invoke(prompt)

        return QueryResponse(
            answer=response.content,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")