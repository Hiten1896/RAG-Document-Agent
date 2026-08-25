// Session and chat identity for backend isolation.
//
// The backend tags every stored chunk with the X-Session-Id / X-Chat-Id pair it
// was uploaded under and filters retrieval on both. A client that omits these
// headers falls back to a shared `default_session:default_chat` bucket, which
// means it can read every other such client's documents — so api.ts must send
// them on every request.
//
// This file previously lived at the repository root under src/utils/. The `@/*`
// alias resolves to frontend/src/*, so nothing could import it and the headers
// were never sent.

const SESSION_STORAGE_KEY = 'rag_session_id';

/**
 * Stable per-tab identifier. sessionStorage (not localStorage) is deliberate:
 * it scopes documents to one browser tab, so two tabs are two isolated
 * workspaces and closing a tab drops its context.
 */
export function getSessionId(): string {
  if (typeof window === 'undefined') return 'server_side_session';

  let sessionId = sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (!sessionId) {
    sessionId = createId();
    sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }
  return sessionId;
}

export function createNewChatId(): string {
  return createId();
}

/**
 * crypto.randomUUID is unavailable on pages served over plain HTTP from a
 * non-localhost origin (it requires a secure context), where it throws rather
 * than returning undefined. Falling back to getRandomValues keeps ids unique
 * there instead of breaking session isolation entirely.
 */
function createId(): string {
  if (typeof crypto !== 'undefined') {
    if (typeof crypto.randomUUID === 'function') {
      try {
        return crypto.randomUUID();
      } catch {
        // fall through to the getRandomValues path
      }
    }
    if (typeof crypto.getRandomValues === 'function') {
      const bytes = new Uint8Array(16);
      crypto.getRandomValues(bytes);
      return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
    }
  }
  return `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
