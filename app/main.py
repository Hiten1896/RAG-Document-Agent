import os
import shutil
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

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

# Initialize Vector Search & LLM Engine
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
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
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        vector_store.add_documents(chunks)
        
        return {
            "status": "success",
            "filename": file.filename,
            "pages": len(documents),
            "chunks": len(chunks)
        }
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
                "relevance_score": float(score)
            })
            
        formatted_context = "\n\n---\n\n".join(context_blocks)
        
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        chain = prompt | llm
        
        response = chain.invoke({
            "context": formatted_context,
            "question": request.query
        })
        
        return {
            "query": request.query,
            "answer": response.content,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")