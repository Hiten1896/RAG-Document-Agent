import os
import shutil
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="Multi-Modal RAG Document Agent API", version="1.0.0")

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

# Initialize Vector Search
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# Initialize Gemini LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY
)

RAG_PROMPT_TEMPLATE = """
You are an expert AI assistant answering questions based on the provided document context.

Context:
{context}

Question:
{question}

Instructions:
1. Answer the question thoroughly using ONLY the provided context.
2. If the answer is not contained within the context, state clearly: "I cannot find the answer in the uploaded documents."
3. Keep the tone professional, concise, and structured.
"""

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4

def extract_text_with_gemini_ocr(file_path: str, filename: str) -> List[Document]:
    """Fallback OCR using Gemini vision for scanned or image-only PDFs."""
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
        prompt = (
            "You are an OCR and document extraction engine. "
            "Extract all text, diagrams, formulas, tables, and notes from this document. "
            "Preserve page sections using markers like '--- Page X ---'."
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[part, prompt]
        )

        extracted_text = response.text or ""
        if not extracted_text.strip():
            return []

        # Split into document pages if markers exist, otherwise single document
        pages = extracted_text.split("--- Page ")
        docs = []
        if len(pages) > 1:
            for idx, page_content in enumerate(pages[1:], 1):
                clean_content = page_content.strip()
                if clean_content:
                    docs.append(Document(
                        page_content=f"Page {idx}\n{clean_content}",
                        metadata={"source": filename, "page": idx - 1}
                    ))
        else:
            docs.append(Document(
                page_content=extracted_text.strip(),
                metadata={"source": filename, "page": 0}
            ))
        return docs
    except Exception as ocr_err:
        print(f"Gemini OCR fallback error: {ocr_err}")
        return []

@app.get("/")
async def root():
    return {"status": "FastAPI Backend Active", "message": "DocAgent RAG Service is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Check if PyPDFLoader extracted any meaningful text
        total_text_len = sum(len(doc.page_content.strip()) for doc in documents)
        
        # If no text extracted (e.g., scanned PDF), fall back to Gemini OCR
        if total_text_len == 0:
            print(f"No direct text found in {file.filename}. Running Gemini OCR fallback...")
            ocr_documents = extract_text_with_gemini_ocr(file_path, file.filename)
            if ocr_documents:
                documents = ocr_documents

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        # Filter out empty chunks to prevent ChromaDB empty upsert errors
        valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        
        if not valid_chunks:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract readable text from this PDF. It may be empty or unreadable."
            )
        
        vector_store.add_documents(valid_chunks)
        
        return {
            "status": "success",
            "filename": file.filename,
            "pages": len(documents),
            "chunks": len(valid_chunks)
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
        sources = []
        for doc, score in results:
            context_blocks.append(doc.page_content)
            sources.append({
                "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", 0) + 1,
                "relevance_score": round(float(score), 4)
            })
            
        formatted_context = "\n\n---\n\n".join(context_blocks)
        
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        chain = prompt | llm
        
        response = chain.invoke({
            "context": formatted_context,
            "question": request.query
        })
        
        # Format answer string properly across langchain return types
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
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")