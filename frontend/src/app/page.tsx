'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { ingestDocument, checkIngestStatus, queryDocument, SourceMetadata } from '@/lib/api';
import {
  Send,
  FileText,
  Bot,
  User,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Edit3,
  X,
  Check,
  Image as ImageIcon,
  Sparkles,
  Menu,
  MessageSquarePlus,
  Zap,
  Search,
  FileUp,
  Mic,
  MicOff,
  Sun,
  Moon,
  Plus,
} from 'lucide-react';

/* ───── Types ───── */
interface Message {
  sender: 'user' | 'agent';
  text: string;
  sources?: SourceMetadata[];
}

interface IngestedFile {
  name: string;
  status: 'processing' | 'completed' | 'failed';
  pages?: number;
  chunks?: number;
  error?: string;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
}

const WELCOME_MESSAGE: Message = {
  sender: 'agent',
  text: 'Hello! Upload one or more PDFs to automatically ingest text and diagrams. Ask me anything across all of them!',
};

// Session-only: kept in memory for the tab's lifetime, not persisted to
// localStorage/sessionStorage, so a reload starts fresh by design.
function makeConversationId(): string {
  return `c_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function titleFromQuery(text: string): string {
  const trimmed = text.trim();
  return trimmed.length > 40 ? `${trimmed.slice(0, 40)}…` : trimmed || 'New Chat';
}

function getErrorMessage(err: unknown, fallback: string): string {
  if (err instanceof Error && err.message) return err.message;
  if (typeof err === 'string' && err.trim()) return err;
  if (err && typeof err === 'object') {
    const maybeDetail = (err as { detail?: unknown }).detail;
    if (typeof maybeDetail === 'string' && maybeDetail.trim()) return maybeDetail;
    const maybeMessage = (err as { message?: unknown }).message;
    if (typeof maybeMessage === 'string' && maybeMessage.trim()) return maybeMessage;
    try {
      return JSON.stringify(err);
    } catch {
      return fallback;
    }
  }
  return fallback;
}

export default function Home() {
  /* Upload state — multiple files can now be ingested and queried together */
  const [ingestedFiles, setIngestedFiles] = useState<IngestedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{
    type: 'success' | 'error' | 'info';
    message: string;
    details?: string;
  } | null>(null);

  /* Chat history — session-only (kept in memory, never persisted), so a
     reload intentionally starts clean. Each conversation is independent;
     switching conversations does not touch ingestedFiles, since documents
     stay available to every conversation in this session. */
  const [conversations, setConversations] = useState<Conversation[]>(() => [
    { id: makeConversationId(), title: 'New Chat', messages: [WELCOME_MESSAGE] },
  ]);
  const [activeConversationId, setActiveConversationId] = useState<string>(
    () => conversations[0].id
  );

  const activeConversation =
    conversations.find((c) => c.id === activeConversationId) ?? conversations[0];
  const messages = activeConversation.messages;

  // Update only the active conversation's message list, auto-titling it from
  // the first user message the same way most chat UIs do.
  const setMessagesForActive = useCallback(
    (updater: (prev: Message[]) => Message[]) => {
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeConversationId) return c;
          const nextMessages = updater(c.messages);
          const firstUserMsg = nextMessages.find((m) => m.sender === 'user');
          const nextTitle =
            c.title === 'New Chat' && firstUserMsg ? titleFromQuery(firstUserMsg.text) : c.title;
          return { ...c, messages: nextMessages, title: nextTitle };
        })
      );
    },
    [activeConversationId]
  );

  /* Chat state */
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editText, setEditText] = useState('');

  /* UI state */
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [dragOver, setDragOver] = useState(false);

  /* Theme state */
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

  const applyTheme = (targetTheme: 'dark' | 'light') => {
    const root = document.documentElement;
    if (targetTheme === 'dark') {
      root.classList.add('dark');
      root.classList.remove('light');
      root.setAttribute('data-theme', 'dark');
    } else {
      root.classList.remove('dark');
      root.classList.add('light');
      root.setAttribute('data-theme', 'light');
    }
  };

  // The inline script in layout.tsx already put the right class on <html> before
  // first paint. This only syncs React state to that, so the toggle icon matches
  // what's on screen — it must not re-derive or re-apply the theme, which is
  // what previously forced the whole app to stay `invisible` until hydration.
  useEffect(() => {
    const root = document.documentElement;
    setTheme(root.classList.contains('light') ? 'light' : 'dark');
  }, []);

  // The OS theme can change while the page is open. Follow it only while the
  // user has not pinned a choice of their own.
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;

    const media = window.matchMedia('(prefers-color-scheme: light)');
    const onChange = (event: MediaQueryListEvent) => {
      try {
        if (localStorage.getItem('docagent_theme')) return; // user pinned a theme
      } catch {
        // localStorage unavailable — fall through and follow the system
      }
      const next = event.matches ? 'light' : 'dark';
      setTheme(next);
      applyTheme(next);
    };

    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, []);

  const toggleTheme = (e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    const nextTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(nextTheme);
    applyTheme(nextTheme);

    try {
      localStorage.setItem('docagent_theme', nextTheme);
    } catch {
      // safe fallback
    }
  };

  /* Voice search setup */
  const [listening, setListening] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {
          // ignore
        }
      }
    };
  }, []);

  useEffect(() => {
    if (!micError) return;
    const id = setTimeout(() => setMicError(null), 4000);
    return () => clearTimeout(id);
  }, [micError]);

  // Auto-dismiss success/info upload toasts after a few seconds; errors stay
  // on screen until the next upload attempt since they carry a diagnostic
  // message worth reading in full.
  useEffect(() => {
    if (!uploadStatus || uploadStatus.type === 'error') return;
    const id = setTimeout(() => setUploadStatus(null), 5000);
    return () => clearTimeout(id);
  }, [uploadStatus]);

  const handleMicClick = (e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }

    if (typeof window === 'undefined') return;
    const SpeechRecognitionCtor =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognitionCtor) {
      setMicError('Voice search is not supported in this browser.');
      return;
    }

    if (listening && recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        // ignore
      }
      setListening(false);
      return;
    }

    setMicError(null);
    try {
      const recognition = new SpeechRecognitionCtor();
      recognition.continuous = false;
      recognition.lang = 'en-US';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onresult = (event: any) => {
        const transcript: string = event?.results?.[0]?.[0]?.transcript ?? '';
        if (transcript.trim()) {
          setQuery(transcript.trim());
        }
      };

      recognition.onend = () => {
        setListening(false);
      };

      recognition.onerror = (event: any) => {
        setListening(false);
        const code = event?.error;
        if (code === 'not-allowed' || code === 'service-not-allowed') {
          setMicError('Microphone access was denied. Please allow mic permission.');
        } else if (code === 'no-speech') {
          setMicError("Didn't catch that — try speaking again.");
        } else {
          setMicError('Voice search failed. Please try again.');
        }
      };

      recognition.start();
      recognitionRef.current = recognition;
      setListening(true);
    } catch {
      setListening(false);
    }
  };

  /* Refs */
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Guards against overlapping ingest poll loops, and lets us cancel the
  // pending timer on unmount so it can't setState on an unmounted tree.
  const pollTokenRef = useRef(0);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      pollTokenRef.current++;
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    };
  }, []);

  /* Auto-scroll */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  /* Backend health check */
  useEffect(() => {
    const API = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
    let mounted = true;

    const check = async () => {
      try {
        const res = await fetch(API, { signal: AbortSignal.timeout(3000) });
        if (mounted) setBackendOnline(res.ok);
      } catch {
        if (mounted) setBackendOnline(false);
      }
    };

    check();
    const id = setInterval(check, 30_000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  /* File upload — accepts one or many files. Each file gets its own poll
     loop keyed by filename, since with multiple files uploading at once a
     single shared "latest wins" token would cancel earlier ones mid-poll. */
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files ?? []);
    // Clear the input's value so re-picking the SAME file(s) fires `change`
    // again. Without this, a failed ingest could not be retried by
    // reselecting the file — the browser suppresses the event when the
    // value is unchanged.
    e.target.value = '';
    if (selectedFiles.length) {
      await ingestMultiple(selectedFiles);
    }
  };

  const handleTriggerUpload = (e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    fileInputRef.current?.click();
  };

  /* Drag & Drop */
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  }, []);

  const onDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);

    const droppedFiles = Array.from(e.dataTransfer.files ?? []);
    if (droppedFiles.length) {
      await ingestMultiple(droppedFiles);
    }
  }, []);

  /* Ingestion — fires all files off in parallel; each tracks its own status
     entry in ingestedFiles rather than sharing one uploadStatus slot, so
     one file failing doesn't hide another's progress or overwrite its result. */
  const ingestMultiple = async (files: File[]) => {
    const pdfFiles = files.filter((f) => f.name.toLowerCase().endsWith('.pdf'));
    const rejected = files.length - pdfFiles.length;

    if (rejected > 0) {
      setUploadStatus({
        type: rejected === files.length ? 'error' : 'info',
        message:
          rejected === files.length
            ? 'Only PDF documents are supported.'
            : `Skipped ${rejected} non-PDF file${rejected > 1 ? 's' : ''}.`,
      });
    }
    if (!pdfFiles.length) return;

    setSidebarOpen(false);
    setUploading(true);
    setUploadStatus({
      type: 'info',
      message:
        pdfFiles.length === 1
          ? `Ingesting ${pdfFiles[0].name}...`
          : `Ingesting ${pdfFiles.length} files...`,
      details: 'Extracting text and generating vector embeddings...',
    });

    await Promise.all(pdfFiles.map((f) => triggerAutoIngest(f)));
    setUploading(false);
  };

  const triggerAutoIngest = async (fileToIngest: File) => {
    const name = fileToIngest.name;

    // Each file's poll loop is invalidated only by a NEW upload of that same
    // filename (retry), not by other files uploading concurrently.
    const pollToken = ++pollTokenRef.current;
    const isStale = () => pollTokenRef.current !== pollToken;

    setIngestedFiles((prev) => [
      ...prev.filter((f) => f.name !== name),
      { name, status: 'processing' },
    ]);

    try {
      const res = await ingestDocument(fileToIngest);
      if (isStale()) return;

      let attempts = 0;
      const maxAttempts = 60;

      const poll = async () => {
        if (isStale()) return;
        try {
          const statusRes = await checkIngestStatus(res.filename);
          if (isStale()) return;

          if (statusRes.status === 'completed') {
            setIngestedFiles((prev) =>
              prev.map((f) =>
                f.name === name
                  ? { ...f, status: 'completed', pages: statusRes.pages, chunks: statusRes.chunks }
                  : f
              )
            );
            setUploadStatus({
              type: 'success',
              message: `Indexed ${name}`,
              details: `${statusRes.pages || 1} pages • ${statusRes.chunks || 0} chunks`,
            });
          } else if (statusRes.status === 'failed') {
            setIngestedFiles((prev) =>
              prev.map((f) =>
                f.name === name ? { ...f, status: 'failed', error: statusRes.error } : f
              )
            );
            setUploadStatus({
              type: 'error',
              message: `Ingestion failed: ${name}`,
              details: statusRes.error || 'Could not parse document.',
            });
          } else {
            // `not_found` is also treated as still-pending: the background task
            // may not have registered the job yet.
            attempts++;
            if (attempts < maxAttempts) {
              pollTimerRef.current = setTimeout(poll, 1000);
            } else {
              setIngestedFiles((prev) =>
                prev.map((f) =>
                  f.name === name
                    ? { ...f, status: 'failed', error: 'Timed out waiting for the backend to finish indexing.' }
                    : f
                )
              );
              setUploadStatus({
                type: 'info',
                message: `${name}: processing is taking longer than expected.`,
                details: 'Still indexing in the background — check back or retry.',
              });
            }
          }
        } catch (err: unknown) {
          // This used to be a bare `catch {}` that only flipped off the spinner,
          // so a failing status endpoint looked like "nothing happened at all".
          if (isStale()) return;
          const message = getErrorMessage(err, 'Could not reach the status endpoint.');
          setIngestedFiles((prev) =>
            prev.map((f) => (f.name === name ? { ...f, status: 'failed', error: message } : f))
          );
          setUploadStatus({ type: 'error', message: `Lost track of ${name}`, details: message });
        }
      };

      pollTimerRef.current = setTimeout(poll, 800);
    } catch (err: unknown) {
      if (isStale()) return;
      const message = getErrorMessage(err, 'Failed to communicate with backend.');
      setIngestedFiles((prev) =>
        prev.map((f) => (f.name === name ? { ...f, status: 'failed', error: message } : f))
      );
      setUploadStatus({ type: 'error', message: `Ingestion error: ${name}`, details: message });
    }
  };

  /* Query */
  const handleSendQuery = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || loading) return;

    const userText = query.trim();
    setQuery('');
    setMessagesForActive((prev) => [...prev, { sender: 'user', text: userText }]);
    setLoading(true);

    try {
      // No `source` filter is passed, so the backend searches across every
      // ingested document's chunks rather than restricting to one file.
      const res = await queryDocument(userText);
      setMessagesForActive((prev) => [
        ...prev,
        { sender: 'agent', text: res.answer || 'No answer returned.', sources: res.sources || [] },
      ]);
    } catch (err: unknown) {
      const message = getErrorMessage(err, 'Unknown error');
      setMessagesForActive((prev) => [
        ...prev,
        { sender: 'agent', text: `Error processing your request: ${message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  /* Edit & Regenerate */
  const handleStartEdit = (index: number, currentText: string) => {
    setEditingIndex(index);
    setEditText(currentText);
  };

  const handleCancelEdit = () => {
    setEditingIndex(null);
    setEditText('');
  };

  const handleSaveEdit = async (targetIndex: number) => {
    if (!editText.trim() || loading) return;

    const updatedQuery = editText.trim();
    setEditingIndex(null);
    setEditText('');

    setMessagesForActive((prev) => {
      const sliced = prev.slice(0, targetIndex);
      sliced.push({ sender: 'user', text: updatedQuery });
      return sliced;
    });

    setLoading(true);
    try {
      const res = await queryDocument(updatedQuery);
      setMessagesForActive((prev) => [
        ...prev,
        { sender: 'agent', text: res.answer || 'No answer returned.', sources: res.sources || [] },
      ]);
    } catch (err: unknown) {
      const message = getErrorMessage(err, 'Unknown error');
      setMessagesForActive((prev) => [
        ...prev,
        { sender: 'agent', text: `Error regenerating answer: ${message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  /* New Chat — starts a fresh conversation and adds it to the sidebar
     history list, rather than wiping the current one. History is session-only
     by design: it lives in component state and is gone on reload. */
  const handleNewChat = () => {
    const newConversation: Conversation = {
      id: makeConversationId(),
      title: 'New Chat',
      messages: [WELCOME_MESSAGE],
    };
    setConversations((prev) => [newConversation, ...prev]);
    setActiveConversationId(newConversation.id);
    setEditingIndex(null);
    setEditText('');
    setQuery('');
  };

  const handleSwitchConversation = (id: string) => {
    if (id === activeConversationId) return;
    setActiveConversationId(id);
    setEditingIndex(null);
    setEditText('');
    setQuery('');
  };

  /* Source dedup */
  const getDeduplicatedSources = (sources: SourceMetadata[] = []) => {
    const seen = new Set<string>();
    const result: SourceMetadata[] = [];
    for (const src of sources) {
      const key = `${src.source || 'Doc'}-${src.page || '1'}-${src.chunk_type || 'text'}`;
      if (!seen.has(key)) {
        seen.add(key);
        result.push(src);
      }
    }
    return result;
  };

  const isEmptyChat = messages.length <= 1;

  return (
    // `h-screen h-[100dvh]` set the same property twice via two utilities, so
    // which one won depended on Tailwind's output order rather than intent.
    // `h-dvh` alone is what was wanted: the dynamic viewport height that
    // accounts for mobile browser chrome.
    <div className="flex h-dvh bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans antialiased overflow-hidden transition-colors duration-200">
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf,.pdf"
        onChange={handleFileChange}
        className="hidden"
        id="file-auto-upload"
        multiple
      />

      {/* Mic Error Toast */}
      {micError && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-[60] px-4 py-2.5 rounded-xl bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-300 text-xs sm:text-sm font-medium shadow-lg backdrop-blur-xl flex items-center gap-2 animate-fade-in max-w-[90vw]">
          <MicOff className="w-4 h-4 shrink-0" />
          <span className="truncate">{micError}</span>
        </div>
      )}

      {/* Upload Status Toast — replaces the sidebar status card now that
          uploading only happens via the "+" icon in the message bar. Sits
          just above the input so it's visible without covering the mic toast. */}
      {uploadStatus && (
        <div
          className={`fixed bottom-24 sm:bottom-28 left-1/2 -translate-x-1/2 z-[60] px-4 py-2.5 rounded-xl text-xs sm:text-sm font-medium shadow-lg backdrop-blur-xl flex items-start gap-2 animate-fade-in max-w-[92vw] sm:max-w-md border ${
            uploadStatus.type === 'error'
              ? 'bg-red-500/10 border-red-500/30 text-red-600 dark:text-red-300'
              : uploadStatus.type === 'success'
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-300'
              : 'bg-indigo-500/10 border-indigo-500/30 text-indigo-700 dark:text-indigo-300'
          }`}
        >
          {uploadStatus.type === 'error' ? (
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          ) : uploadStatus.type === 'success' ? (
            <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
          ) : (
            <Loader2 className="w-4 h-4 animate-spin shrink-0 mt-0.5" />
          )}
          <div className="min-w-0">
            <div className="truncate">{uploadStatus.message}</div>
            {uploadStatus.details && (
              <div className="text-[11px] opacity-80 leading-relaxed">{uploadStatus.details}</div>
            )}
          </div>
        </div>
      )}

      {/* Mobile Backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-950/60 backdrop-blur-sm md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ═══════════ SIDEBAR ═══════════ */}
      <aside
        className={`
          fixed inset-y-0 left-0 z-50 w-[300px] sm:w-[320px]
          md:static md:w-80 md:z-auto
          border-r border-slate-200 dark:border-slate-800/80 bg-white dark:bg-slate-900/95 md:bg-white md:dark:bg-slate-900/60
          backdrop-blur-xl p-5 md:p-6 flex flex-col justify-between shrink-0 shadow-2xl md:shadow-none
          transition-transform duration-250 ease-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}
      >
        <div className="space-y-5 overflow-y-auto flex-1">
          {/* Logo + Actions */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-gradient-to-tr from-indigo-600 to-violet-500 text-white rounded-xl shadow-lg shadow-indigo-500/20">
                <Bot className="w-6 h-6" />
              </div>
              <div>
                <h1 className="font-bold text-slate-900 dark:text-slate-100 text-base tracking-tight">
                  DocAgent RAG
                </h1>
              </div>
            </div>

            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={toggleTheme}
                className="p-2 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700/80 text-slate-600 dark:text-slate-300 transition duration-200 cursor-pointer"
                aria-label="Toggle theme"
                title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
              >
                {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-indigo-600" />}
              </button>

              <button
                type="button"
                onClick={() => setSidebarOpen(false)}
                className="md:hidden p-2 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 transition cursor-pointer"
                aria-label="Close sidebar"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* New Chat Button */}
          <button
            type="button"
            onClick={() => {
              handleNewChat();
              setSidebarOpen(false);
            }}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700/70 hover:border-indigo-500/50 bg-slate-100 hover:bg-slate-200/80 dark:bg-slate-800/40 dark:hover:bg-slate-800/70 text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-slate-100 transition duration-200 cursor-pointer"
          >
            <MessageSquarePlus className="w-4 h-4 text-indigo-600 dark:text-indigo-400" /> New Chat
          </button>

          {/* Chat History — session-only: lives in memory for this tab and
              is intentionally lost on reload, same as the rest of the app's
              state. Not persisted to storage. */}
          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Chat History
            </label>
            <div className="space-y-1 max-h-48 overflow-y-auto pr-0.5">
              {conversations.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => {
                    handleSwitchConversation(c.id);
                    setSidebarOpen(false);
                  }}
                  className={`w-full text-left px-3 py-2 rounded-lg text-xs truncate transition cursor-pointer ${
                    c.id === activeConversationId
                      ? 'bg-indigo-100 dark:bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 font-medium'
                      : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800/60'
                  }`}
                  title={c.title}
                >
                  {c.title}
                </button>
              ))}
            </div>
          </div>

        </div>

        {/* Backend Status Footer */}
        <div className="text-xs text-slate-500 border-t border-slate-200 dark:border-slate-800/80 pt-4 mt-4 flex items-center justify-between">
          <span>Backend Pipeline:</span>
          {backendOnline === null ? (
            <span className="inline-flex items-center gap-1.5 text-slate-500 dark:text-slate-400 font-medium">
              <Loader2 className="w-3 h-3 animate-spin" /> Checking...
            </span>
          ) : backendOnline ? (
            <span className="inline-flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 dark:bg-emerald-400 animate-pulse" />
              FastAPI Active
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 text-red-600 dark:text-red-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-red-500 dark:bg-red-400" />
              Offline
            </span>
          )}
        </div>
      </aside>

      {/* ═══════════ MAIN CONTENT ═══════════ */}
      <main
        className="flex-1 flex flex-col justify-between bg-slate-50 dark:bg-slate-950 overflow-hidden relative min-w-0"
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        {/* Drag-over overlay — dropping a PDF anywhere in the chat area
            uploads it, now that the sidebar drop zone is gone. */}
        {dragOver && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-indigo-50/90 dark:bg-indigo-950/80 border-4 border-dashed border-indigo-500 rounded-none pointer-events-none">
            <div className="flex flex-col items-center gap-2 text-indigo-700 dark:text-indigo-300">
              <FileUp className="w-10 h-10" />
              <span className="font-medium text-sm">Drop PDF(s) to ingest</span>
            </div>
          </div>
        )}

        {/* Desktop Top Header Bar */}
        <header className="hidden md:flex items-center justify-between px-6 py-3.5 border-b border-slate-200 dark:border-slate-800/80 bg-white/70 dark:bg-slate-900/60 backdrop-blur-xl shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Workspace
            </span>
            <span className="text-slate-300 dark:text-slate-700">•</span>
            <span className="text-xs font-medium text-indigo-600 dark:text-indigo-400">
              {ingestedFiles.length
                ? `${ingestedFiles.length} document${ingestedFiles.length > 1 ? 's' : ''} loaded`
                : 'No active documents'}
            </span>
          </div>
        </header>

        {/* Mobile Top Bar */}
        <div className="md:hidden flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-800/80 bg-white dark:bg-slate-900/60 backdrop-blur-xl shrink-0">
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400 transition cursor-pointer"
            aria-label="Open sidebar"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-gradient-to-tr from-indigo-600 to-violet-500 text-white rounded-lg">
              <Bot className="w-4 h-4" />
            </div>
            <span className="font-semibold text-sm text-slate-900 dark:text-slate-200">DocAgent</span>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={handleNewChat}
              className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400 transition cursor-pointer"
              aria-label="New chat"
            >
              <MessageSquarePlus className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* ── Chat messages area ── */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 space-y-5 sm:space-y-6">

          {/* Empty state hero */}
          {isEmptyChat && !loading && (
            <div className="flex-1 flex flex-col items-center justify-center text-center px-4 py-12 sm:py-20">
              <div className="p-4 bg-gradient-to-tr from-indigo-600/10 to-violet-500/10 dark:from-indigo-600/20 dark:to-violet-500/20 rounded-2xl border border-indigo-200 dark:border-indigo-500/20 mb-6 shadow-sm">
                <Sparkles className="w-10 h-10 sm:w-12 sm:h-12 text-indigo-600 dark:text-indigo-400" />
              </div>
              <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                Document Intelligence
              </h2>
              <p className="text-sm text-slate-600 dark:text-slate-400 max-w-md mb-8 leading-relaxed">
                Upload a PDF document and ask questions about its text, tables, and diagrams.
                Powered by Gemini and vector search.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full max-w-lg">
                <button
                  type="button"
                  onClick={handleTriggerUpload}
                  className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white dark:bg-slate-900/60 hover:bg-slate-100 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-400 shadow-sm transition cursor-pointer"
                >
                  <FileUp className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                  <span>Upload PDF</span>
                </button>
                <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-400 shadow-sm">
                  <Zap className="w-5 h-5 text-amber-500 dark:text-amber-400" />
                  <span>Auto-Ingest</span>
                </div>
                <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-400 shadow-sm">
                  <Search className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                  <span>Ask Questions</span>
                </div>
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((msg, index) => {
            const isUser = msg.sender === 'user';
            const isEditing = editingIndex === index;
            const dedupedSources = getDeduplicatedSources(msg.sources);

            return (
              <div
                key={index}
                className={`group flex items-start gap-2.5 sm:gap-4 ${
                  isUser ? 'flex-row-reverse' : 'flex-row'
                }`}
              >
                {/* Avatar */}
                <div
                  className={`p-2 sm:p-2.5 rounded-xl shrink-0 shadow-md ${
                    isUser
                      ? 'bg-gradient-to-tr from-indigo-600 to-indigo-500 text-white'
                      : 'bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 border border-slate-200 dark:border-slate-800'
                  }`}
                >
                  {isUser ? (
                    <User className="w-4 h-4 sm:w-5 sm:h-5" />
                  ) : (
                    <Bot className="w-4 h-4 sm:w-5 sm:h-5" />
                  )}
                </div>

                {/* Bubble */}
                <div className={`max-w-[85%] sm:max-w-3xl flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
                  {isUser && isEditing ? (
                    /* ── Inline Edit Form ── */
                    <div className="w-full min-w-0 max-w-xl bg-white dark:bg-slate-900 border border-indigo-500/50 rounded-2xl p-3 shadow-xl space-y-3">
                      <textarea
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-xl p-3 text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:border-indigo-500 resize-y min-h-[70px]"
                        autoFocus
                      />
                      <div className="flex justify-end gap-2 text-xs">
                        <button
                          type="button"
                          onClick={handleCancelEdit}
                          className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 transition flex items-center gap-1 cursor-pointer"
                        >
                          <X className="w-3.5 h-3.5" /> Cancel
                        </button>
                        <button
                          type="button"
                          onClick={() => handleSaveEdit(index)}
                          className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium transition flex items-center gap-1 shadow-md shadow-indigo-600/30 cursor-pointer"
                        >
                          <Check className="w-3.5 h-3.5" /> Save & Regenerate
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* ── Normal Message Bubble ── */
                    <div
                      className={`relative rounded-2xl p-3.5 sm:p-4 text-sm leading-relaxed shadow-sm ${
                        isUser
                          ? 'bg-indigo-600 text-white rounded-tr-none shadow-indigo-600/10'
                          : 'bg-white dark:bg-slate-900/90 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200 rounded-tl-none backdrop-blur-sm'
                      }`}
                    >
                      <p className="whitespace-pre-wrap break-words">{msg.text}</p>

                      {/* Source Citations */}
                      {!isUser && dedupedSources.length > 0 && (
                        <div className="mt-3.5 pt-3 border-t border-slate-100 dark:border-slate-800/80 text-xs">
                          <div className="font-semibold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-1.5">
                            <span>Retrieved Sources:</span>
                            <span className="text-[10px] text-slate-500 dark:text-slate-400 font-normal">
                              ({dedupedSources.length} distinct{' '}
                              {dedupedSources.length === 1 ? 'citation' : 'citations'})
                            </span>
                          </div>

                          <div className="flex flex-wrap gap-1.5">
                            {dedupedSources.map((src, srcIndex) => {
                              const isVisual = src.chunk_type === 'visual';
                              const pageText = src.page ? `Page ${src.page}` : '';
                              const sourceName = src.source || 'Document';

                              return (
                                <span
                                  key={srcIndex}
                                  className={`inline-flex items-center gap-1 px-2 sm:px-2.5 py-1 rounded-lg border text-[11px] sm:text-xs font-mono transition shadow-sm ${
                                    isVisual
                                      ? 'bg-violet-50 dark:bg-violet-950/40 border-violet-200 dark:border-violet-500/40 text-violet-700 dark:text-violet-300'
                                      : 'bg-slate-100 dark:bg-slate-800/80 border-slate-200 dark:border-slate-700/80 text-indigo-700 dark:text-indigo-300'
                                  }`}
                                >
                                  {isVisual ? (
                                    <ImageIcon className="w-3 h-3 text-violet-600 dark:text-violet-400 shrink-0" />
                                  ) : (
                                    <FileText className="w-3 h-3 text-indigo-600 dark:text-indigo-400 shrink-0" />
                                  )}
                                  <span className="truncate max-w-[120px] sm:max-w-[180px]">
                                    {sourceName}
                                  </span>
                                  {pageText && <span className="opacity-75">({pageText})</span>}
                                </span>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Edit button */}
                  {isUser && !isEditing && (
                    <button
                      type="button"
                      onClick={() => handleStartEdit(index, msg.text)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 mt-1.5 text-xs text-slate-500 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-300 flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-slate-200/50 dark:hover:bg-slate-800/50 cursor-pointer"
                      title="Edit this query and regenerate response"
                    >
                      <Edit3 className="w-3 h-3" /> Edit query
                    </button>
                  )}
                </div>
              </div>
            );
          })}

          {/* Loading Indicator */}
          {loading && (
            <div className="flex items-center gap-2.5 sm:gap-3 text-slate-500 dark:text-slate-400 text-sm">
              <div className="p-2 sm:p-2.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-indigo-600 dark:text-indigo-400">
                <Bot className="w-4 h-4 sm:w-5 sm:h-5" />
              </div>
              <div className="bg-white dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 px-3.5 sm:px-4 py-2.5 sm:py-3 rounded-2xl flex items-center gap-2.5 text-xs text-slate-700 dark:text-slate-300 shadow-sm">
                <Loader2 className="w-4 h-4 animate-spin text-indigo-600 dark:text-indigo-400" />
                <span className="hidden sm:inline">
                  Retrieving document chunks & generating answer with Gemini...
                </span>
                <span className="sm:hidden">Generating answer...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* ── Chat Input ── */}
        <div className="p-3 sm:p-4 lg:p-6 border-t border-slate-200 dark:border-slate-800/80 bg-white/80 dark:bg-slate-900/40 backdrop-blur-xl shrink-0">
          <form onSubmit={handleSendQuery} className="flex gap-2 sm:gap-3 max-w-4xl mx-auto">
            {/* "+" attach button — opens the file picker with `multiple` set,
                so one or several PDFs can be added right from the message
                bar without going through the sidebar. Visible on every
                screen size (previously this was mobile-only). */}
            <button
              type="button"
              onClick={handleTriggerUpload}
              className="bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl sm:rounded-2xl p-3 text-slate-600 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:border-indigo-500/50 transition shrink-0 cursor-pointer"
              aria-label="Add PDF documents"
              title="Add PDF documents"
            >
              <Plus className="w-4 h-4 sm:w-5 sm:h-5" />
            </button>

            {/* Input field */}
            <div className="relative flex-1 flex items-center min-w-0">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about text, tables, or diagrams..."
                className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl sm:rounded-2xl pl-4 pr-11 sm:pl-5 sm:pr-12 py-3 sm:py-3.5 text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition placeholder:text-slate-400 dark:placeholder:text-slate-500 shadow-inner"
              />

              {/* Mic button */}
              <button
                type="button"
                onClick={handleMicClick}
                className={`absolute right-3 p-1.5 rounded-lg transition cursor-pointer z-10 ${
                  listening
                    ? 'bg-red-500 text-white animate-pulse'
                    : 'text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-slate-200 dark:hover:bg-slate-800'
                }`}
                title={listening ? 'Stop listening' : 'Voice input'}
              >
                <Mic className="w-4 h-4" />
              </button>
            </div>

            {/* Submit button */}
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="bg-gradient-to-tr from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 disabled:from-slate-300 dark:disabled:from-slate-800 disabled:to-slate-300 dark:disabled:to-slate-800 disabled:text-slate-400 dark:disabled:text-slate-600 text-white px-4 sm:px-6 rounded-xl sm:rounded-2xl transition duration-200 flex items-center justify-center shadow-lg shadow-indigo-600/20 font-medium text-sm shrink-0 cursor-pointer disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}