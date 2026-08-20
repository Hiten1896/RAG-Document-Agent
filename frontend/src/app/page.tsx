'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { ingestDocument, checkIngestStatus, queryDocument, SourceMetadata } from '@/lib/api';
import {
  UploadCloud,
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
  Layers,
  Menu,
  MessageSquarePlus,
  Zap,
  Search,
  FileUp,
  Mic,
  MicOff,
  Sun,
  Moon,
} from 'lucide-react';

/* ───── Types ───── */
interface Message {
  sender: 'user' | 'agent';
  text: string;
  sources?: SourceMetadata[];
}

/**
 * Safely extract a human-readable string from a caught value.
 * api.ts already normalizes backend error payloads into `Error` instances,
 * but this guards against any non-Error value (string, plain object, etc.)
 * ever bubbling up, so we never render "[object Object]" in the chat UI.
 */
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

/* ───── Component ───── */
export default function Home() {
  /* Upload state */
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{
    type: 'success' | 'error' | 'info';
    message: string;
    details?: string;
  } | null>(null);

  /* Chat state */
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editText, setEditText] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'agent',
      text: 'Hello! Upload a PDF to automatically ingest text and diagrams. Ask me anything about the document!',
    },
  ]);

  /* UI state */
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [dragOver, setDragOver] = useState(false);

  /* Theme state ('dark' | 'light'), persisted to localStorage, defaulting
     to the user's OS preference on first visit. Applied by toggling the
     `dark` class on <html> so Tailwind's `dark:` variants take effect
     (requires darkMode: 'class' in tailwind.config). */
  const [theme, setThemeState] = useState<'dark' | 'light'>('dark');
  const [themeReady, setThemeReady] = useState(false);

  const applyTheme = useCallback((next: 'dark' | 'light') => {
    if (typeof document !== 'undefined') {
      document.documentElement.classList.toggle('dark', next === 'dark');
    }
    setThemeState(next);
    try {
      localStorage.setItem('docagent_theme', next);
    } catch {
      // localStorage may be unavailable (privacy mode, SSR edge cases) — theme
      // still applies for this session via the DOM class above.
    }
  }, []);

  useEffect(() => {
    let initial: 'dark' | 'light' = 'dark';
    try {
      const saved = localStorage.getItem('docagent_theme');
      if (saved === 'dark' || saved === 'light') {
        initial = saved;
      } else if (typeof window !== 'undefined' && window.matchMedia) {
        initial = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
      }
    } catch {
      // ignore and fall back to 'dark'
    }
    applyTheme(initial);
    setThemeReady(true);
  }, [applyTheme]);

  const toggleTheme = () => {
    applyTheme(theme === 'dark' ? 'light' : 'dark');
  };

  /* Voice search: real Web Speech API, mirroring the mic behavior from the
     MyWay/Movie Hunt UI. Guards for browser support since SpeechRecognition
     is not available everywhere (e.g. Firefox desktop, most non-Chromium
     browsers) — the button hides itself gracefully when unsupported. */
  const [micSupported, setMicSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const SpeechRecognitionCtor =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognitionCtor) {
      setMicSupported(false);
      return;
    }

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
        setMicError('Microphone access was denied. Enable it in your browser settings to use voice search.');
      } else if (code === 'no-speech') {
        setMicError("Didn't catch that — try again.");
      } else {
        setMicError('Voice search failed. Please try again.');
      }
    };

    recognitionRef.current = recognition;
    setMicSupported(true);

    return () => {
      try {
        recognition.stop();
      } catch {
        // no-op: recognition may already be stopped/idle
      }
      recognitionRef.current = null;
    };
  }, []);

  // Auto-dismiss transient mic error toasts
  useEffect(() => {
    if (!micError) return;
    const id = setTimeout(() => setMicError(null), 4000);
    return () => clearTimeout(id);
  }, [micError]);

  const handleMicClick = () => {
    const recognition = recognitionRef.current;
    if (!recognition) return;

    if (listening) {
      recognition.stop();
      setListening(false);
      return;
    }

    setMicError(null);
    try {
      recognition.start();
      setListening(true);
    } catch {
      // start() throws if recognition is already active/starting; ignore,
      // the button state will resync via onend/onerror.
      setListening(false);
    }
  };

  /* Refs */
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* ───── Auto-scroll ───── */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  /* ───── Backend health check ───── */
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

  /* ───── File upload ───── */
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      await triggerAutoIngest(selectedFile);
    }
  };

  /* ───── Drag & Drop ───── */
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

    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      setFile(droppedFile);
      await triggerAutoIngest(droppedFile);
    }
  }, []);

  /* ───── Ingestion ───── */
  const triggerAutoIngest = async (fileToIngest: File) => {
    if (!fileToIngest.name.endsWith('.pdf')) {
      setUploadStatus({ type: 'error', message: 'Only PDF documents are supported.' });
      return;
    }

    setUploading(true);
    setUploadStatus({
      type: 'info',
      message: `Ingesting ${fileToIngest.name}...`,
      details: 'Extracting text and generating vector embeddings...',
    });

    // Close sidebar on mobile after selecting file
    setSidebarOpen(false);

    try {
      const res = await ingestDocument(fileToIngest);
      let attempts = 0;
      const maxAttempts = 60;

      const poll = async () => {
        try {
          const statusRes = await checkIngestStatus(res.filename);
          if (statusRes.status === 'completed') {
            const visualCount = statusRes.visual_chunks || 0;
            const textCount = statusRes.text_chunks || 0;
            setUploadStatus({
              type: 'success',
              message: `Successfully indexed ${statusRes.filename}`,
              details: `${statusRes.pages || 1} pages • ${statusRes.chunks || 0} chunks (${textCount} text, ${visualCount} visual)`,
            });
            setUploading(false);
          } else if (statusRes.status === 'failed') {
            setUploadStatus({
              type: 'error',
              message: 'Ingestion failed',
              details: statusRes.error || 'Could not parse document.',
            });
            setUploading(false);
          } else {
            attempts++;
            if (attempts < maxAttempts) {
              setTimeout(poll, 1000);
            } else {
              setUploadStatus({
                type: 'info',
                message: 'Processing is taking longer than expected.',
                details: 'Vector database is indexing in the background.',
              });
              setUploading(false);
            }
          }
        } catch {
          setUploading(false);
        }
      };

      setTimeout(poll, 800);
    } catch (err: unknown) {
      const message = getErrorMessage(err, 'Failed to communicate with backend.');
      setUploadStatus({ type: 'error', message: 'Ingestion error', details: message });
      setUploading(false);
    }
  };

  /* ───── Query ───── */
  const handleSendQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userText = query.trim();
    setQuery('');
    setMessages((prev) => [...prev, { sender: 'user', text: userText }]);
    setLoading(true);

    try {
      const res = await queryDocument(userText);
      setMessages((prev) => [
        ...prev,
        { sender: 'agent', text: res.answer || 'No answer returned.', sources: res.sources || [] },
      ]);
    } catch (err: unknown) {
      const message = getErrorMessage(err, 'Unknown error');
      setMessages((prev) => [
        ...prev,
        { sender: 'agent', text: `Error processing your request: ${message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  /* ───── Edit & Regenerate ───── */
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

    const slicedMessages = messages.slice(0, targetIndex);
    slicedMessages.push({ sender: 'user', text: updatedQuery });
    setMessages(slicedMessages);

    setLoading(true);
    try {
      const res = await queryDocument(updatedQuery);
      setMessages((prev) => [
        ...prev,
        { sender: 'agent', text: res.answer || 'No answer returned.', sources: res.sources || [] },
      ]);
    } catch (err: unknown) {
      const message = getErrorMessage(err, 'Unknown error');
      setMessages((prev) => [
        ...prev,
        { sender: 'agent', text: `Error regenerating answer: ${message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  /* ───── New Chat ───── */
  const handleNewChat = () => {
    setMessages([
      {
        sender: 'agent',
        text: 'Hello! Upload a PDF to automatically ingest text and diagrams. Ask me anything about the document!',
      },
    ]);
    setEditingIndex(null);
    setEditText('');
    setQuery('');
  };

  /* ───── Source dedup ───── */
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

  /* Is chat in "empty" state (only the welcome message)? */
  const isEmptyChat = messages.length <= 1;

  /* ─────────────────────── RENDER ─────────────────────── */
  return (
    <div
      className={`flex h-screen h-[100dvh] bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans antialiased overflow-hidden transition-colors duration-200 ${
        themeReady ? '' : 'invisible'
      }`}
    >
      {/* ── Mic permission / recognition error toast ── */}
      {micError && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-[60] px-4 py-2.5 rounded-xl bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-300 text-xs sm:text-sm font-medium shadow-lg backdrop-blur-xl flex items-center gap-2 animate-fade-in max-w-[90vw]">
          <MicOff className="w-4 h-4 shrink-0" />
          <span className="truncate">{micError}</span>
        </div>
      )}

      {/* ── Mobile Sidebar Backdrop ── */}
      {sidebarOpen && (
        <div
          className="sidebar-backdrop md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ═══════════ SIDEBAR ═══════════ */}
      <aside
        className={`
          fixed inset-y-0 left-0 z-50 w-[300px] sm:w-[320px]
          md:static md:w-80 md:z-auto
          border-r border-slate-200 dark:border-slate-800/80 bg-white dark:bg-slate-900/95 md:bg-slate-50 md:dark:bg-slate-900/60
          backdrop-blur-xl p-5 md:p-6 flex flex-col justify-between shrink-0 shadow-2xl
          transition-transform duration-250 ease-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}
      >
        <div className="space-y-5 overflow-y-auto flex-1">
          {/* Logo + close on mobile */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-gradient-to-tr from-indigo-600 to-violet-500 text-white rounded-xl shadow-lg shadow-indigo-500/20">
                <Bot className="w-6 h-6" />
              </div>
              <div>
                <h1 className="font-bold text-slate-100 text-base tracking-tight">DocAgent RAG</h1>
              </div>
            </div>
            {/* Close button (mobile only) */}
            <button
              onClick={() => setSidebarOpen(false)}
              className="md:hidden p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition"
              aria-label="Close sidebar"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* New Chat button */}
          <button
            onClick={() => { handleNewChat(); setSidebarOpen(false); }}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-slate-700/70 hover:border-indigo-500/50 bg-slate-800/40 hover:bg-slate-800/70 text-sm text-slate-300 hover:text-slate-100 transition duration-200"
          >
            <MessageSquarePlus className="w-4 h-4" /> New Chat
          </button>

          {/* Upload area */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Document Ingestion
              </label>
              <span className="text-[11px] text-indigo-400 font-medium">Auto-Ingest</span>
            </div>

            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              className={`relative border-2 border-dashed rounded-2xl p-5 text-center transition duration-200 cursor-pointer ${
                dragOver
                  ? 'border-indigo-400 bg-indigo-950/30 scale-[1.02]'
                  : uploading
                  ? 'border-indigo-500/60 bg-indigo-950/20'
                  : 'border-slate-700/70 hover:border-indigo-500/60 bg-slate-900/40 hover:bg-slate-800/40'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="hidden"
                id="file-auto-upload"
              />

              <div className="flex flex-col items-center gap-2.5">
                {uploading ? (
                  <div className="relative">
                    <Loader2 className="w-9 h-9 text-indigo-400 animate-spin" />
                    <Sparkles className="w-3.5 h-3.5 text-amber-300 absolute -top-1 -right-1 animate-pulse" />
                  </div>
                ) : (
                  <div className="p-3 bg-indigo-600/10 text-indigo-400 rounded-xl border border-indigo-500/20">
                    <UploadCloud className="w-7 h-7" />
                  </div>
                )}

                <div>
                  <div className="text-sm font-medium text-slate-200 truncate max-w-[220px]">
                    {file ? file.name : 'Select or Drop PDF'}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {uploading ? 'Auto-processing document...' : 'Drag & drop or click to browse'}
                  </div>
                </div>
              </div>
            </div>

            {/* Status card */}
            {uploadStatus && (
              <div
                className={`p-3.5 rounded-xl text-xs border transition-all duration-200 animate-fade-in ${
                  uploadStatus.type === 'error'
                    ? 'bg-red-500/10 text-red-300 border-red-500/20'
                    : uploadStatus.type === 'success'
                    ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20'
                    : 'bg-indigo-500/10 text-indigo-300 border-indigo-500/20'
                }`}
              >
                <div className="flex items-start gap-2.5">
                  {uploadStatus.type === 'error' ? (
                    <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                  ) : uploadStatus.type === 'success' ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  ) : (
                    <Loader2 className="w-4 h-4 text-indigo-400 animate-spin shrink-0 mt-0.5" />
                  )}
                  <div className="space-y-0.5 min-w-0">
                    <div className="font-medium text-slate-200 truncate">{uploadStatus.message}</div>
                    {uploadStatus.details && (
                      <div className="text-[11px] text-slate-400 leading-relaxed">
                        {uploadStatus.details}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Feature list */}
          <div className="p-3.5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-2 text-xs text-slate-400">
            <div className="font-semibold text-slate-300 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-indigo-400" /> System Features
            </div>
            <ul className="space-y-1.5 text-[11px]">
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
                Fast text chunking & vector search
              </li>
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" />
                Gemini LLM response synthesis
              </li>
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-violet-400 shrink-0" />
                Inline query editing & regeneration
              </li>
            </ul>
          </div>
        </div>

        {/* Backend status footer */}
        <div className="text-xs text-slate-500 border-t border-slate-800/80 pt-4 mt-4 flex items-center justify-between">
          <span>Backend Pipeline:</span>
          {backendOnline === null ? (
            <span className="inline-flex items-center gap-1.5 text-slate-400 font-medium">
              <Loader2 className="w-3 h-3 animate-spin" /> Checking...
            </span>
          ) : backendOnline ? (
            <span className="inline-flex items-center gap-1.5 text-emerald-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              FastAPI Active
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 text-red-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-red-400" />
              Offline
            </span>
          )}
        </div>
      </aside>

      {/* ═══════════ MAIN CONTENT ═══════════ */}
      <main className="flex-1 flex flex-col justify-between bg-slate-950 overflow-hidden relative min-w-0">

        {/* Mobile top bar */}
        <div className="md:hidden flex items-center justify-between px-4 py-3 border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-xl shrink-0">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition"
            aria-label="Open sidebar"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-gradient-to-tr from-indigo-600 to-violet-500 text-white rounded-lg">
              <Bot className="w-4 h-4" />
            </div>
            <span className="font-semibold text-sm text-slate-200">DocAgent</span>
          </div>
          <button
            onClick={handleNewChat}
            className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition"
            aria-label="New chat"
          >
            <MessageSquarePlus className="w-5 h-5" />
          </button>
        </div>

        {/* ── Chat messages area ── */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 space-y-5 sm:space-y-6">

          {/* Empty state hero */}
          {isEmptyChat && !loading && (
            <div className="flex-1 flex flex-col items-center justify-center text-center px-4 py-12 sm:py-20 animate-fade-in">
              <div className="p-4 bg-gradient-to-tr from-indigo-600/20 to-violet-500/20 rounded-2xl border border-indigo-500/20 mb-6">
                <Sparkles className="w-10 h-10 sm:w-12 sm:h-12 text-indigo-400" />
              </div>
              <h2 className="text-xl sm:text-2xl font-bold text-slate-100 mb-2">
                Document Intelligence
              </h2>
              <p className="text-sm text-slate-400 max-w-md mb-8 leading-relaxed">
                Upload a PDF document and ask questions about its text, tables, and diagrams.
                Powered by Gemini and vector search.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full max-w-lg">
                <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-400">
                  <FileUp className="w-5 h-5 text-indigo-400" />
                  <span>Upload PDF</span>
                </div>
                <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-400">
                  <Zap className="w-5 h-5 text-amber-400" />
                  <span>Auto-Ingest</span>
                </div>
                <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-400">
                  <Search className="w-5 h-5 text-emerald-400" />
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
                className={`group flex items-start gap-2.5 sm:gap-4 animate-fade-in ${
                  isUser ? 'flex-row-reverse' : 'flex-row'
                }`}
              >
                {/* Avatar */}
                <div
                  className={`p-2 sm:p-2.5 rounded-xl shrink-0 shadow-md ${
                    isUser
                      ? 'bg-gradient-to-tr from-indigo-600 to-indigo-500 text-white'
                      : 'bg-slate-900 text-indigo-400 border border-slate-800'
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
                    /* ── Inline edit form ── */
                    <div className="w-full min-w-0 max-w-xl bg-slate-900 border border-indigo-500/50 rounded-2xl p-3 shadow-xl space-y-3">
                      <textarea
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 resize-y min-h-[70px]"
                        autoFocus
                      />
                      <div className="flex justify-end gap-2 text-xs">
                        <button
                          onClick={handleCancelEdit}
                          className="px-3 py-1.5 rounded-lg border border-slate-700 hover:bg-slate-800 text-slate-300 transition flex items-center gap-1"
                        >
                          <X className="w-3.5 h-3.5" /> Cancel
                        </button>
                        <button
                          onClick={() => handleSaveEdit(index)}
                          className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium transition flex items-center gap-1 shadow-md shadow-indigo-600/30"
                        >
                          <Check className="w-3.5 h-3.5" /> Save & Regenerate
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* ── Normal message bubble ── */
                    <div
                      className={`relative rounded-2xl p-3.5 sm:p-4 text-sm leading-relaxed shadow-lg ${
                        isUser
                          ? 'bg-indigo-600 text-white rounded-tr-none'
                          : 'bg-slate-900/90 border border-slate-800 text-slate-200 rounded-tl-none backdrop-blur-sm'
                      }`}
                    >
                      <p className="whitespace-pre-wrap break-words">{msg.text}</p>

                      {/* Source citations */}
                      {!isUser && dedupedSources.length > 0 && (
                        <div className="mt-3.5 pt-3 border-t border-slate-800/80 text-xs">
                          <div className="font-semibold text-slate-300 mb-2 flex items-center gap-1.5">
                            <span>Retrieved Sources:</span>
                            <span className="text-[10px] text-slate-400 font-normal">
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
                                      ? 'bg-violet-950/40 border-violet-500/40 text-violet-300'
                                      : 'bg-slate-800/80 border-slate-700/80 text-indigo-300'
                                  }`}
                                >
                                  {isVisual ? (
                                    <ImageIcon className="w-3 h-3 text-violet-400 shrink-0" />
                                  ) : (
                                    <FileText className="w-3 h-3 text-indigo-400 shrink-0" />
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

                  {/* Edit button (user messages, desktop hover) */}
                  {isUser && !isEditing && (
                    <button
                      onClick={() => handleStartEdit(index, msg.text)}
                      className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 mt-1.5 text-xs text-slate-400 hover:text-indigo-300 flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-slate-800/50"
                      title="Edit this query and regenerate response"
                    >
                      <Edit3 className="w-3 h-3" /> Edit query
                    </button>
                  )}
                </div>
              </div>
            );
          })}

          {/* Loading indicator */}
          {loading && (
            <div className="flex items-center gap-2.5 sm:gap-3 text-slate-400 text-sm animate-pulse">
              <div className="p-2 sm:p-2.5 rounded-xl bg-slate-900 border border-slate-800 text-indigo-400">
                <Bot className="w-4 h-4 sm:w-5 sm:h-5" />
              </div>
              <div className="bg-slate-900/80 border border-slate-800 px-3.5 sm:px-4 py-2.5 sm:py-3 rounded-2xl flex items-center gap-2.5 text-xs text-slate-300">
                <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
                <span className="hidden sm:inline">
                  Retrieving document chunks & generating answer with Gemini...
                </span>
                <span className="sm:hidden">Generating answer...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* ── Chat input ── */}
        <div className="p-3 sm:p-4 lg:p-6 border-t border-slate-800/80 bg-slate-900/40 backdrop-blur-xl shrink-0">
          <form onSubmit={handleSendQuery} className="flex gap-2 sm:gap-3 max-w-4xl mx-auto">
            {/* Mobile upload button */}
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="md:hidden bg-slate-800 border border-slate-700 rounded-xl sm:rounded-2xl p-3 text-slate-400 hover:text-indigo-400 hover:border-indigo-500/50 transition shrink-0"
              aria-label="Upload document"
            >
              <FileUp className="w-4 h-4 sm:w-5 sm:h-5" />
            </button>

            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about text, tables, or diagrams..."
              className="flex-1 min-w-0 bg-slate-900 border border-slate-800 rounded-xl sm:rounded-2xl px-4 sm:px-5 py-3 sm:py-3.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition placeholder:text-slate-500 shadow-inner"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="bg-gradient-to-tr from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-600 text-white px-4 sm:px-6 rounded-xl sm:rounded-2xl transition duration-200 flex items-center justify-center shadow-lg shadow-indigo-600/20 font-medium text-sm shrink-0"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}