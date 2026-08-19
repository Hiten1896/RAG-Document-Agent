FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMuPDF & C++ compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure upload & chroma directories exist
RUN mkdir -p ./data/uploads ./data/chroma_db

# Bind Uvicorn to dynamic cloud $PORT
ENV PORT=8000
EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT