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

/**
 * FastAPI's `detail` field is usually a string (HTTPException(detail="...")),
 * but on 422 validation errors it's an array of {loc, msg, type} objects,
 * and sometimes it's a nested object. Normalize all of these into a single
 * human-readable string so we never end up with "[object Object]" downstream.
 */
function extractErrorMessage(errorData: unknown, fallback: string): string {
  if (!errorData || typeof errorData !== 'object') {
    return fallback;
  }

  const detail = (errorData as { detail?: unknown }).detail;

  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail)) {
    // FastAPI/Pydantic validation error array: [{loc, msg, type}, ...]
    const messages = detail
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object' && 'msg' in item) {
          const loc = 'loc' in item && Array.isArray((item as any).loc)
            ? (item as any).loc.join('.')
            : undefined;
          return loc ? `${loc}: ${(item as any).msg}` : String((item as any).msg);
        }
        return JSON.stringify(item);
      })
      .filter(Boolean);
    if (messages.length) return messages.join('; ');
  }

  if (detail && typeof detail === 'object') {
    if ('msg' in (detail as any)) return String((detail as any).msg);
    try {
      return JSON.stringify(detail);
    } catch {
      return fallback;
    }
  }

  return fallback;
}

async function parseErrorResponse(response: Response, fallback: string): Promise<Error> {
  let errorData: unknown = null;
  try {
    errorData = await response.json();
  } catch {
    // response body wasn't JSON (e.g. HTML error page, empty body) — ignore
  }
  const message = extractErrorMessage(errorData, fallback);
  return new Error(message);
}

export async function ingestDocument(file: File): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/ingest`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw await parseErrorResponse(response, 'Failed to initiate document ingestion.');
  }

  return response.json();
}

export async function checkIngestStatus(filename: string): Promise<IngestStatusResponse> {
  // Properly handle spaces and special characters in document filenames
  const encodedFilename = encodeURIComponent(filename);
  const response = await fetch(`${API_BASE_URL}/ingest/status/${encodedFilename}`);

  if (!response.ok) {
    throw await parseErrorResponse(response, 'Failed to fetch document ingestion status.');
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
    throw await parseErrorResponse(response, 'Failed to query documents.');
  }

  return response.json();
}