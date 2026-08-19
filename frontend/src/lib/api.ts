export interface SourceMetadata {
  source?: string;
  page?: number;
  chunk_type?: 'text' | 'visual';
  image_key?: string;
  relevance_score?: number;
}

export interface IngestResponse {
  status: string;
  message: string;
  filename: string;
}

export interface IngestStatusResponse {
  status: 'processing' | 'completed' | 'failed' | 'not_found';
  filename: string;
  pages?: number;
  chunks?: number;
  text_chunks?: number;
  visual_chunks?: number;
  error?: string;
}

export interface QueryResponse {
  query: string;
  answer: string;
  sources: SourceMetadata[];
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

export async function ingestDocument(file: File): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/ingest`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to initiate document ingestion.');
  }

  return response.json();
}

export async function checkIngestStatus(filename: string): Promise<IngestStatusResponse> {
  // Properly handle spaces and special characters in document filenames
  const encodedFilename = encodeURIComponent(filename);
  const response = await fetch(`${API_BASE_URL}/ingest/status/${encodedFilename}`);

  if (!response.ok) {
    throw new Error('Failed to fetch document ingestion status.');
  }

  return response.json();
}

export async function queryDocument(query: string, top_k: number = 4): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      top_k,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to query documents.');
  }

  return response.json();
}