import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="Multi-Modal RAG Document Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./data/uploads"
CHROMA_DIR = "./data/chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Ingestion state tracking for async BackgroundTasks
ingestion_status: Dict[str, Dict[str, Any]] = {}

# Initialize Gemini Vector Search Embeddings (Lightweight for 512MB RAM)
# Change this line:
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# Initialize Gemini LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-1.5-flash"

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY
)

RAG_PROMPT_TEMPLATE = """
You are an expert AI assistant answering questions based on the provided document context.
The context contains both extracted text and detailed visual descriptions of diagrams, tables, and images from the document.

Context:
{context}

Question:
{question}

Instructions:
1. Answer the question thoroughly using ONLY the provided context.
2. If diagrams, tables, or figures are referenced in the context, synthesize the visual descriptions clearly.
3. If the answer is not contained within the context, state clearly: "I cannot find the answer in the uploaded documents."
4. Keep the tone professional, concise, and structured.
"""

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4


def generate_vision_description(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Generate structured markdown description for an image using Gemini vision."""
    if not GOOGLE_API_KEY:
        return ""
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        prompt = (
            "You are an expert multi-modal OCR and document analysis engine. "
            "Analyze this image or diagram from a document. "
            "Provide a clear, structured markdown summary describing: "
            "1. All text, titles, headings, and labels. "
            "2. Diagrams, flowcharts, architectures, or formulas shown. "
            "3. Key concepts and relationships depicted."
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[part, prompt]
        )
        return (response.text or "").strip()
    except Exception as err:
        print(f"Gemini vision description error: {err}")
        return ""


def process_pdf_multimodal(file_path: str, filename: str) -> Dict[str, Any]:
    ingestion_status[filename] = {
        "status": "processing",
        "filename": filename,
        "started_at": time.time(),
        "chunks": 0,
        "pages": 0
    }

    try:
        doc = pymupdf.open(file_path)
        total_pages = len(doc)
        all_documents: List[Document] = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        MAX_VISION_IMAGES = 5
        vision_jobs: List[Dict[str, Any]] = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            raw_text = page.get_text().strip()
            if len(raw_text) > 20:
                text_chunks = text_splitter.split_text(raw_text)
                for chunk in text_chunks:
                    if chunk.strip():
                        all_documents.append(Document(
                            page_content=chunk.strip(),
                            metadata={
                                "source": filename,
                                "parent_doc_id": filename,
                                "page": page_idx,
                                "page_number": page_num,
                                "chunk_type": "text",
                                "image_key": "N/A"
                            }
                        ))

            image_list = page.get_images(full=True)
            extracted_images_count = 0

            if image_list and len(vision_jobs) < MAX_VISION_IMAGES:
                for img_idx, img_info in enumerate(image_list[:2]):
                    if len(vision_jobs) >= MAX_VISION_IMAGES:
                        break
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image.get("image")
                        image_ext = base_image.get("ext", "jpeg").lower()
                        mime_type = "image/png" if image_ext == "png" else "image/jpeg"

                        if image_bytes and len(image_bytes) >= 5000:
                            extracted_images_count += 1
                            image_key = f"{filename}_p{page_num}_img{img_idx + 1}"
                            vision_jobs.append({
                                "image_bytes": image_bytes,
                                "mime_type": mime_type,
                                "page_num": page_num,
                                "page_idx": page_idx,
                                "image_key": image_key,
                                "label": f"[Visual Snapshot - Page {page_num}]"
                            })
                    except Exception as img_err:
                        print(f"Error extracting image {img_idx} on page {page_num}: {img_err}")

            if len(raw_text) <= 20 and extracted_images_count == 0 and len(vision_jobs) < MAX_VISION_IMAGES:
                try:
                    pix = page.get_pixmap(dpi=150)
                    pix_bytes = pix.tobytes("jpeg")
                    if pix_bytes:
                        image_key = f"{filename}_p{page_num}_page_scan"
                        vision_jobs.append({
                            "image_bytes": pix_bytes,
                            "mime_type": "image/jpeg",
                            "page_num": page_num,
                            "page_idx": page_idx,
                            "image_key": image_key,
                            "label": f"[Page Scan Analysis - Page {page_num}]"
                        })
                except Exception as pix_err:
                    print(f"Error rendering pixmap for page {page_num}: {pix_err}")

        if vision_jobs:
            with ThreadPoolExecutor(max_workers=min(len(vision_jobs), MAX_VISION_IMAGES)) as executor:
                future_to_job = {
                    executor.submit(generate_vision_description, job["image_bytes"], job["mime_type"]): job
                    for job in vision_jobs
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        vision_desc = future.result()
                    except Exception as vision_err:
                        print(f"Vision description failed for {job['image_key']}: {vision_err}")
                        vision_desc = ""
                    if vision_desc:
                        all_documents.append(Document(
                            page_content=f"{job['label']}\n{vision_desc}",
                            metadata={
                                "source": filename,
                                "parent_doc_id": filename,
                                "page": job["page_idx"],
                                "page_number": job["page_num"],
                                "chunk_type": "visual",
                                "image_key": job["image_key"]
                            }
                        ))

        valid_chunks = [c for c in all_documents if c.page_content and c.page_content.strip()]

        if not valid_chunks:
            error_msg = "Could not extract readable text or visual content from this PDF."
            ingestion_status[filename] = {"status": "failed", "error": error_msg, "filename": filename}
            return {"status": "failed", "error": error_msg}

        vector_store.add_documents(valid_chunks)

        result_data = {
            "status": "completed",
            "filename": filename,
            "pages": total_pages,
            "chunks": len(valid_chunks),
            "text_chunks": sum(1 for c in valid_chunks if c.metadata.get("chunk_type") == "text"),
            "visual_chunks": sum(1 for c in valid_chunks if c.metadata.get("chunk_type") == "visual"),
        }
        ingestion_status[filename] = result_data
        return result_data

    except Exception as e:
        error_msg = str(e)
        ingestion_status[filename] = {"status": "failed", "error": error_msg, "filename": filename}
        print(f"Ingestion failed for {filename}: {error_msg}")
        return {"status": "failed", "error": error_msg}


@app.get("/")
async def root():
    return {"status": "FastAPI Backend Active", "message": "DocAgent Multi-Modal RAG Service is running"}


@app.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_pdf_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingestion_status[file.filename] = {
            "status": "processing",
            "filename": file.filename,
            "started_at": time.time()
        }

        background_tasks.add_task(process_pdf_multimodal, file_path, file.filename)

        return {
            "status": "accepted",
            "message": "File upload received. Multi-modal ingestion running in background.",
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate ingestion: {str(e)}")


@app.get("/ingest/status/{filename}")
async def get_ingest_status(filename: str):
    info = ingestion_status.get(filename)
    if not info:
        return {"status": "not_found", "filename": filename}
    return info


@app.post("/upload")
async def upload_pdf_sync(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_pdf_multimodal(file_path, file.filename)
        if result.get("status") == "failed":
            raise HTTPException(status_code=400, detail=result.get("error", "Processing failed"))

        return {
            "status": "success",
            "filename": file.filename,
            "pages": result.get("pages", 0),
            "chunks": result.get("chunks", 0),
            "text_chunks": result.get("text_chunks", 0),
            "visual_chunks": result.get("visual_chunks", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/query")
async def query_documents(request: QueryRequest):
    try:
        results = vector_store.similarity_search_with_score(request.query, k=request.top_k)

        if not results:
            return {"answer": "No documents found to query.", "sources": []}

        context_blocks = []
        raw_sources = []

        for doc, score in results:
            context_blocks.append(doc.page_content)
            
            raw_page = doc.metadata.get("page_number", doc.metadata.get("page", 0))
            if isinstance(raw_page, int) and raw_page == 0:
                raw_page = 1

            source_name = os.path.basename(str(doc.metadata.get("source", "Document")))
            chunk_type = doc.metadata.get("chunk_type", "text")
            image_key = doc.metadata.get("image_key", "N/A")

            raw_sources.append({
                "source": source_name,
                "page": raw_page,
                "chunk_type": chunk_type,
                "image_key": image_key,
                "relevance_score": round(float(score), 4)
            })

        deduped_sources: List[Dict[str, Any]] = []
        seen_keys = set()
        for src in raw_sources:
            dedupe_key = (src["source"], src["page"], src["chunk_type"])
            if dedupe_key not in seen_keys:
                seen_keys.add(dedupe_key)
                deduped_sources.append(src)

        formatted_context = "\n\n---\n\n".join(context_blocks)

        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        chain = prompt | llm

        response = chain.invoke({
            "context": formatted_context,
            "question": request.query
        })

        if isinstance(response.content, list):
            answer_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in response.content
            )
        else:
            answer_text = str(response.content)

        return {
            "query": request.query,
            "answer": answer_text,
            "sources": deduped_sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)