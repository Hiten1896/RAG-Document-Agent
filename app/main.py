import os
import time
import tempfile
import fitz  # PyMuPDF
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI(title="DocAgent RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Models & Storage
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

CHROMA_PATH = "./chroma_db"
vectorstore = Chroma(
    collection_name="docagent_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

ingestion_status: Dict[str, Dict[str, Any]] = {}

class QueryRequest(BaseModel):
    query: str


def fast_process_pdf(file_bytes: bytes, filename: str):
    """Ultra-fast text extraction and batch embedding without slow vision calls."""
    ingestion_status[filename] = {"status": "processing", "filename": filename}
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        extracted_documents = []

        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").strip()
            if text:
                extracted_documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_num + 1}
                    )
                )
        doc.close()

        if not extracted_documents:
            ingestion_status[filename] = {"status": "failed", "error": "No text found in PDF"}
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_documents(extracted_documents)

        # Single-pass batch embedding for high speed
        vectorstore.add_documents(chunks)

        ingestion_status[filename] = {
            "status": "completed",
            "filename": filename,
            "pages": len(extracted_documents),
            "chunks": len(chunks)
        }
    except Exception as e:
        ingestion_status[filename] = {"status": "failed", "error": str(e)}


@app.get("/")
def read_root():
    return {"status": "FastAPI Backend Active", "message": "DocAgent RAG Service is running"}


@app.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_pdf_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    background_tasks.add_task(fast_process_pdf, content, file.filename)

    return {
        "status": "accepted",
        "message": "Ingestion started in background.",
        "filename": file.filename
    }


@app.get("/ingest/status/{filename}")
async def get_ingest_status(filename: str):
    return ingestion_status.get(filename, {"status": "not_found", "filename": filename})


@app.post("/upload")
async def upload_pdf_sync(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    fast_process_pdf(content, file.filename)
    res = ingestion_status.get(file.filename, {})
    
    if res.get("status") == "failed":
        raise HTTPException(status_code=400, detail=res.get("error"))

    return {"status": "success", "filename": file.filename, "chunks": res.get("chunks", 0)}


@app.post("/query")
async def query_doc(request: QueryRequest):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(request.query)

        context = "\n\n".join([d.page_content for d in relevant_docs])
        prompt = f"Context:\n{context}\n\nQuestion: {request.query}\n\nAnswer:"

        response = llm.invoke(prompt)
        return {
            "query": request.query,
            "answer": response.content,
            "sources": [d.metadata for d in relevant_docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))