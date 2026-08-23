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
  // The backend echoes the question back, but treat it as optional so a
  // response without it doesn't read as a contract violation.
  query?: string;
  answer: string;
  sources: SourceMetadata[];
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Every fetch here previously had no timeout, so a hung backend left the UI
 * spinning forever with no way out. Embedding + LLM generation is slow, so the
 * query budget is generous; status polls are quick and get a short one.
 */
const INGEST_TIMEOUT_MS = 120_000;
const STATUS_TIMEOUT_MS = 10_000;
const QUERY_TIMEOUT_MS = 90_000;

function timeoutSignal(ms: number): AbortSignal | undefined {
  // AbortSignal.timeout is Baseline, but guard so a missing implementation
  // degrades to "no timeout" rather than throwing before the request is sent.
  return typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function'
    ? AbortSignal.timeout(ms)
    : undefined;
}

function describeNetworkError(err: unknown, timedOutMessage: string): never {
  if (err instanceof DOMException && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
    throw new Error(timedOutMessage);
  }
  if (err instanceof Error) throw err;
  throw new Error('Could not reach the backend. Is the FastAPI server running?');
}

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

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/ingest`, {
      method: 'POST',
      body: formData,
      signal: timeoutSignal(INGEST_TIMEOUT_MS),
    });
  } catch (err) {
    describeNetworkError(err, 'Upload timed out before the backend accepted the file.');
  }

  if (!response.ok) {
    throw await parseErrorResponse(response, 'Failed to initiate document ingestion.');
  }

  return response.json();
}

export async function checkIngestStatus(filename: string): Promise<IngestStatusResponse> {
  // Properly handle spaces and special characters in document filenames.
  // NOTE: the backend route is `/status/{filename}` — this used to request
  // `/ingest/status/...`, which 404'd on every poll, so ingestion progress was
  // never reported and the upload UI just silently stopped.
  const encodedFilename = encodeURIComponent(filename);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/status/${encodedFilename}`, {
      signal: timeoutSignal(STATUS_TIMEOUT_MS),
    });
  } catch (err) {
    describeNetworkError(err, 'Timed out while checking ingestion status.');
  }

  if (!response.ok) {
    throw await parseErrorResponse(response, 'Failed to fetch document ingestion status.');
  }

  return response.json();
}

export async function queryDocument(query: string, top_k: number = 4): Promise<QueryResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        top_k,
      }),
      signal: timeoutSignal(QUERY_TIMEOUT_MS),
    });
  } catch (err) {
    describeNetworkError(err, 'The query timed out. The document may still be indexing.');
  }

  if (!response.ok) {
    throw await parseErrorResponse(response, 'Failed to query documents.');
  }

  return response.json();
}