import os
import time
import tempfile
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize FastAPI App
app = FastAPI(title="DocAgent Multi-Modal RAG Backend")

# 2. Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize Gemini Models
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

# 4. Initialize Chroma Vectorstore
CHROMA_PATH = "./chroma_db"
vectorstore = Chroma(
    collection_name="docagent_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

class QueryRequest(BaseModel):
    question: str


def add_documents_with_retry(vectorstore, batch, max_retries=5):
    """Embeds a batch of documents into Chroma with automatic retry handling for 429 rate limits."""
    for attempt in range(max_retries):
        try:
            vectorstore.add_documents(batch)
            return
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = (attempt + 1) * 6  # Backoff delay: 6s, 12s, 18s...
                print(f"WARNING: Gemini Embedding Rate Limit hit (429). Retrying batch in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise HTTPException(status_code=429, detail="Gemini API rate limit exceeded. Please wait a minute and try again.")


@app.get("/")
def read_root():
    return {
        "status": "FastAPI Backend Active",
        "message": "DocAgent Multi-Modal RAG Service is running"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        # Save temporary PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Parse text using PyMuPDF
        doc = fitz.open(tmp_path)
        extracted_documents = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                extracted_documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file.filename, "page": page_num + 1}
                    )
                )

        doc.close()
        os.remove(tmp_path)

        if not extracted_documents:
            raise HTTPException(status_code=400, detail="No readable text found in document.")

        # Larger chunk size produces fewer total chunks to reduce API calls
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(extracted_documents)

        # Batch ingestion with small batch sizes and delays
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            add_documents_with_retry(vectorstore, batch)
            time.sleep(3)  # Gentle delay between batches

        return {
            "status": "success",
            "filename": file.filename,
            "total_pages": len(extracted_documents),
            "total_chunks": len(chunks),
            "message": "Document ingested successfully."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_doc(request: QueryRequest):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(request.question)

        context = "\n\n".join([d.page_content for d in relevant_docs])

        prompt = f"""You are a helpful AI document assistant. Use the provided context to answer the question accurately.

Context:
{context}

Question: {request.question}

Answer:"""

        response = llm.invoke(prompt)
        return {
            "answer": response.content,
            "sources": [d.metadata for d in relevant_docs]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))