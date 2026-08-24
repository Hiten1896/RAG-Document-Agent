// Helper utility for generating and managing session & chat IDs

export function getSessionId(): string {
  if (typeof window === 'undefined') return 'server_side_session';

  let sessionId = sessionStorage.getItem('rag_session_id');
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    sessionStorage.setItem('rag_session_id', sessionId);
  }
  return sessionId;
}

export function createNewChatId(): string {
  return crypto.randomUUID();
}