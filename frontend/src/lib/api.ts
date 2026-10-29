import { createNewChatId, getSessionId } from '@/lib/session';

export interface SourceMetadata {
  source?: string;
  page?: number;
  // Set for units that have no page number — "sheet Q1 Revenue", "document".
  location?: string;
  chunk_type?: 'text' | 'visual';
  image_key?: string;
  relevance_score?: number;
}

export interface IngestResponse {
  status: string;
  message: string;
  filename: string;
  chunks: number;
  pages?: number;
}

export interface QueryResponse {
  // The backend echoes the question back, but treat it as optional so a
  // response without it doesn't read as a contract violation.
  query?: string;
  answer: string;
  sources: SourceMetadata[];
  debug?: QueryDebug;
}

export interface MindMapNode {
  title: string;
  source: string | null;
  children: MindMapNode[];
}

export interface MindMapDebug {
  provider?: string | null;
  model?: string | null;
  chunks_used: number;
  files: string[];
  embed_ms: number;
  retrieve_ms: number;
  reduce_ms: number;
  total_ms: number;
  cache_hit: boolean;
}

export interface MindMapResponse {
  title: string;
  children: MindMapNode[];
  debug?: MindMapDebug;
}

export interface MindMapOptions {
  topic?: string;
  fileName?: string;
  regenerate?: boolean;
}

export interface QueryDebugChunk {
  text: string;
  source?: string;
  page?: number;
  location?: string;
  relevance_score?: number;
}

export interface QueryDebug {
  chunks: QueryDebugChunk[];
  prompt: string;
  top_k: number;
  retrieved_count: number;
  provider?: string;
  model?: string;
  attempts?: Array<{
    provider: string;
    model: string;
    status: string;
    ms?: number;
  }>;
  dropped_chunks?: number;
  embed_ms?: number;
  retrieve_ms?: number;
  total_ms?: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Every path is prefixed with /api. The frontend previously called /ingest,
 * /query and /status/{filename} while the backend served /api/ingest,
 * /api/query and /api/session/clear, so every request 404'd.
 */
const API_PREFIX = '/api';

/**
 * Each fetch here previously had no timeout, so a hung backend left the UI
 * spinning forever with no way out. Embedding plus LLM generation is slow, so
 * the query budget is generous.
 *
 * The ingest budget is deliberately large: Render's free tier spins down after
 * ~15 minutes idle and a cold start alone can take close to a minute before the
 * upload is even read.
 */
const INGEST_TIMEOUT_MS = 180_000;
const QUERY_TIMEOUT_MS = 120_000;
const MINDMAP_TIMEOUT_MS = 120_000;
const CLEAR_TIMEOUT_MS = 15_000;

function timeoutSignal(ms: number): AbortSignal | undefined {
  // AbortSignal.timeout is Baseline, but guard so a missing implementation
  // degrades to "no timeout" rather than throwing before the request is sent.
  return typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function'
    ? AbortSignal.timeout(ms)
    : undefined;
}

/**
 * Identity headers the backend filters retrieval on. Without them every client
 * shares one `default_session:default_chat` bucket and can read every other
 * client's uploaded documents.
 */
function sessionHeaders(chatId?: string): Record<string, string> {
  return {
    'X-Session-Id': getSessionId(),
    'X-Chat-Id': chatId || 'default_chat',
  };
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
          const entry = item as { loc?: unknown; msg?: unknown };
          const loc = Array.isArray(entry.loc) ? entry.loc.join('.') : undefined;
          return loc ? `${loc}: ${String(entry.msg)}` : String(entry.msg);
        }
        return JSON.stringify(item);
      })
      .filter(Boolean);
    if (messages.length) return messages.join('; ');
  }

  if (detail && typeof detail === 'object') {
    if ('msg' in detail) return String((detail as { msg?: unknown }).msg);
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

export async function ingestDocument(file: File, chatId?: string): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append('file', file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${API_PREFIX}/ingest`, {
      method: 'POST',
      // No Content-Type header: the browser must set the multipart boundary.
      headers: sessionHeaders(chatId),
      body: formData,
      signal: timeoutSignal(INGEST_TIMEOUT_MS),
    });
  } catch (err) {
    describeNetworkError(err, 'Upload timed out before the backend accepted the file.');
  }

  if (!response.ok) {
    throw await parseErrorResponse(response, 'Failed to ingest the document.');
  }

  return response.json();
}

export async function queryDocument(
  query: string,
  chatId?: string,
  top_k: number = 4,
  debug = false,
): Promise<QueryResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${API_PREFIX}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...sessionHeaders(chatId),
        ...(debug ? { 'X-Debug': 'true' } : {}),
      },
      body: JSON.stringify({ query, top_k }),
      signal: timeoutSignal(QUERY_TIMEOUT_MS),
    });
  } catch (err) {
    describeNetworkError(err, 'The query timed out. The document may still be indexing.');
  }

  if (!response.ok) {
    throw await parseErrorResponse(response, 'Failed to query documents.');
  }

  const data: QueryResponse = await response.json();
  return {
    ...data,
    sources: Array.isArray(data.sources) ? data.sources : [],
    debug:
      data.debug && Array.isArray(data.debug.chunks)
        ? {
            ...data.debug,
            chunks: data.debug.chunks,
          }
        : undefined,
  };
}

export async function generateMindMap(
  chatId?: string,
  options: MindMapOptions = {},
): Promise<MindMapResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${API_PREFIX}/mindmap`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...sessionHeaders(chatId),
        'X-Debug': 'true',
      },
      body: JSON.stringify({
        ...(options.topic ? { topic: options.topic } : {}),
        ...(options.fileName ? { file_name: options.fileName } : {}),
        ...(options.regenerate ? { regenerate: true } : {}),
      }),
      signal: timeoutSignal(MINDMAP_TIMEOUT_MS),
    });
  } catch (err) {
    describeNetworkError(err, 'Mind map generation timed out. Please try again.');
  }

  if (!response.ok) {
    throw await parseErrorResponse(response, 'Failed to generate a mind map.');
  }

  return response.json();
}

/**
 * Drop one chat's vectors on the backend. Called when starting a new chat so the
 * outgoing conversation's documents don't linger in memory — the store is
 * in-process on a 512MB instance, so leaked vectors are a real cost.
 *
 * Deliberately never throws: failing to free server-side memory must not block
 * the user from starting a new chat.
 */
export async function clearSession(chatId?: string): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}${API_PREFIX}/session/clear`, {
      method: 'DELETE',
      headers: sessionHeaders(chatId),
      signal: timeoutSignal(CLEAR_TIMEOUT_MS),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export { createNewChatId, getSessionId };
