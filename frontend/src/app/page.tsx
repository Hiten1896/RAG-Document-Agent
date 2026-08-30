'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { Document as PdfDocument, Page as PdfPage, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import {
  ingestDocument,
  queryDocument,
  clearSession,
  SourceMetadata,
} from '@/lib/api';
import {
  Send,
  FileText,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Edit3,
  X,
  Image as ImageIcon,
  Menu,
  MessageSquarePlus,
  Search,
  FileUp,
  Mic,
  MicOff,
  Sun,
  Moon,
  Plus,
  PanelLeftClose,
  PanelLeftOpen,
  MoreHorizontal,
  Trash2,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';

// react-pdf renders PDFs itself via pdf.js instead of relying on the
// browser's built-in PDF plugin. That built-in-plugin path (a plain
// <iframe src="blob:...pdf">) is what the file viewer used before — it
// works on most desktop browsers but mobile Safari/Chrome have no such
// plugin, so on phones the iframe rendered blank or silently failed to
// open at all. Canvas-based rendering here works identically on every
// platform. pdf.js needs its worker script served from somewhere; the
// unpkg CDN build matching the installed pdfjs-dist version avoids
// bundling/copying the worker file into the Next.js build manually.
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

/* ───── Types ───── */
interface Message {
  sender: 'user' | 'agent';
  text: string;
  sources?: SourceMetadata[];
  attachedFiles?: string[];
}

interface IngestedFile {
  name: string;
  status: 'processing' | 'completed' | 'failed';
  pages?: number;
  chunks?: number;
  error?: string;
  file?: File;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  // True once the AI-generated title (via /title) has landed, so a later
  // message in the same chat doesn't re-trigger titling — only the very
  // first user message per conversation gets one.
  titleGenerated?: boolean;
}

// Session-only: kept in memory for the tab's lifetime, not persisted to
// localStorage/sessionStorage, so a reload starts fresh by design.
function makeConversationId(): string {
  return `c_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function titleFromQuery(text: string): string {
  const trimmed = text.trim();
  return trimmed.length > 40 ? `${trimmed.slice(0, 40)}…` : trimmed || 'New Chat';
}

// Formats the backend accepts. Kept next to titleFromQuery so the upload filter
// and the file input's `accept` attribute read from one list and can't drift
// apart — they already had, which is how the `accept` dialog and the drop
// handler ended up disagreeing.
const SUPPORTED_UPLOAD_EXTENSIONS = new Set([
  '.pdf',
  '.docx',
  '.pptx',
  '.xlsx',
  '.csv',
  '.txt',
  '.md',
]);

const SUPPORTED_UPLOAD_LABEL = 'PDF, DOCX, PPTX, XLSX, CSV, TXT, MD';

const SUPPORTED_UPLOAD_ACCEPT = [
  'application/pdf',
  ...SUPPORTED_UPLOAD_EXTENSIONS,
].join(',');


/* ───── Brand mark ─────
   An open document with a spark of insight rising from the page — reads at
   favicon size as a simple folded-corner sheet, and at sidebar size the
   spark/gradient reads too. Uses the same indigo→violet pair as the rest of
   the UI so it never feels like a bolted-on asset. */
function DocAgentMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 40 40" className={className} xmlns="http://www.w3.org/2000/svg" role="img" aria-label="DocAgent">
      <defs>
        <linearGradient id="docagent-mark-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#4f46e5" />
          <stop offset="100%" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>
      <rect width="40" height="40" rx="10" fill="url(#docagent-mark-grad)" />
      {/* folded-corner page */}
      <path
        d="M12 9.5c0-.83.67-1.5 1.5-1.5H21l6 6v16.5c0 .83-.67 1.5-1.5 1.5h-12A1.5 1.5 0 0 1 12 30.5v-21Z"
        fill="white"
        fillOpacity="0.95"
      />
      <path d="M21 8v4.5c0 .83.67 1.5 1.5 1.5H27" fill="none" stroke="#4f46e5" strokeOpacity="0.35" strokeWidth="1.4" />
      {/* text lines */}
      <line x1="15.5" y1="19" x2="23.5" y2="19" stroke="#4f46e5" strokeOpacity="0.55" strokeWidth="1.6" strokeLinecap="round" />
      <line x1="15.5" y1="23" x2="23.5" y2="23" stroke="#4f46e5" strokeOpacity="0.55" strokeWidth="1.6" strokeLinecap="round" />
      <line x1="15.5" y1="27" x2="20" y2="27" stroke="#4f46e5" strokeOpacity="0.35" strokeWidth="1.6" strokeLinecap="round" />
      {/* spark of insight */}
      <path
        d="M27 8.5l1.1 2.4 2.4 1.1-2.4 1.1-1.1 2.4-1.1-2.4-2.4-1.1 2.4-1.1L27 8.5Z"
        fill="white"
      />
    </svg>
  );
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

/* ───── Chat Composer ─────
   The input row + pending-attachment cards, shared between the centered
   empty-state layout and the docked bottom layout so both stay in sync
   automatically. `floating` softens the field into a pill with a visible
   ring (used when it's the sole focal element on an empty screen); the
   docked variant keeps the plainer inline style. */
function ChatComposer({
  query,
  setQuery,
  onSubmit,
  loading,
  onTriggerUpload,
  onMicClick,
  listening,
  pendingAttachments,
  setPendingAttachments,
  ingestedFiles,
  handleRetryIngest,
  floating = false,
}: {
  query: string;
  setQuery: (v: string) => void;
  onSubmit: (e?: React.FormEvent) => void;
  loading: boolean;
  onTriggerUpload: (e?: React.MouseEvent) => void;
  onMicClick: (e?: React.MouseEvent) => void;
  listening: boolean;
  pendingAttachments: string[];
  setPendingAttachments: React.Dispatch<React.SetStateAction<string[]>>;
  ingestedFiles: IngestedFile[];
  handleRetryIngest: (f: IngestedFile) => void;
  floating?: boolean;
}) {
  return (
    <div>
      {/* Pending file cards — files uploaded but not yet sent with a query,
          styled like Claude's own compose-bar attachment card (icon tile,
          truncated name, type badge). Shows loading/success/failed state
          right on the card itself instead of a separate popup toast. A
          failed card shows a Retry button. */}
      {pendingAttachments.length > 0 && (
        <div className="flex flex-wrap gap-2.5 mb-3">
          {pendingAttachments.map((fname, idx) => {
            const fileInfo = ingestedFiles.find((f) => f.name === fname);
            const status = fileInfo?.status ?? 'processing';
            const baseName = fname.replace(/\.pdf$/i, '');

            return (
              <div
                key={idx}
                className={`relative group w-[168px] rounded-xl border p-2.5 shadow-sm transition ${
                  status === 'failed'
                    ? 'bg-red-500/5 border-red-500/30'
                    : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700'
                }`}
              >
                {/* Remove button */}
                <button
                  type="button"
                  onClick={() => setPendingAttachments((prev) => prev.filter((n) => n !== fname))}
                  className="absolute -top-1.5 -right-1.5 p-0.5 rounded-full bg-slate-700 dark:bg-slate-600 text-white opacity-0 group-hover:opacity-100 transition shadow-md cursor-pointer"
                  aria-label={`Remove ${fname}`}
                  title="Remove"
                >
                  <X className="w-3 h-3" />
                </button>

                {/* Icon tile */}
                <div
                  className={`w-8 h-8 rounded-lg flex items-center justify-center mb-2 ${
                    status === 'failed'
                      ? 'bg-red-500/15 text-red-500 dark:text-red-400'
                      : 'bg-indigo-600/10 text-indigo-600 dark:text-indigo-400'
                  }`}
                >
                  {status === 'processing' ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : status === 'failed' ? (
                    <AlertCircle className="w-4 h-4" />
                  ) : (
                    <FileText className="w-4 h-4" />
                  )}
                </div>

                {/* Filename */}
                <p
                  className="text-xs font-medium text-slate-800 dark:text-slate-200 leading-snug line-clamp-2 break-words"
                  title={fname}
                >
                  {baseName}
                </p>

                {/* Footer: type badge + status/retry */}
                <div className="mt-1.5 flex items-center justify-between gap-1.5">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
                    PDF
                  </span>
                  {status === 'failed' ? (
                    <button
                      type="button"
                      onClick={() => fileInfo && handleRetryIngest(fileInfo)}
                      className="text-[10px] font-semibold text-red-600 dark:text-red-300 hover:text-red-700 dark:hover:text-red-200 underline decoration-dotted underline-offset-2 cursor-pointer"
                      title={fileInfo?.error || 'Retry ingestion'}
                    >
                      Retry
                    </button>
                  ) : status === 'completed' ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 dark:text-emerald-400" />
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      )}
      <form onSubmit={onSubmit} className="flex gap-2 sm:gap-3">
        {/* "+" attach button — opens the file picker with `multiple` set, so
            one or several PDFs can be added right from the message bar
            without going through the sidebar. */}
        <button
          type="button"
          onClick={onTriggerUpload}
          className={`bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl sm:rounded-2xl p-3 text-slate-600 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:border-indigo-500/50 transition shrink-0 cursor-pointer ${
            floating ? 'shadow-sm' : ''
          }`}
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
            className={`w-full bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl sm:rounded-2xl pl-4 pr-11 sm:pl-5 sm:pr-12 py-3 sm:py-3.5 text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition placeholder:text-slate-400 dark:placeholder:text-slate-500 ${
              floating ? 'shadow-md' : 'shadow-inner'
            }`}
            autoFocus={floating}
          />

          {/* Mic button */}
          <button
            type="button"
            onClick={onMicClick}
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
  );
}

/* ───── PDF Preview ─────
   Renders a PDF with pdf.js (via react-pdf) onto a <canvas>, instead of
   handing a blob URL to an <iframe> and hoping the browser's native PDF
   plugin picks it up. That native-plugin path is why the mobile viewer
   used to fail outright — phones generally have no such plugin — and why
   desktop quality varied: an iframe's embedded viewer doesn't scale to
   fill its container, it just renders at whatever zoom the plugin
   defaults to. Canvas rendering here is identical on every platform,
   scales crisply to the container's actual width via ResizeObserver, and
   gives real programmatic page-jump control for citations instead of the
   "#page=N" URL-hash trick some mobile browsers ignore entirely.

   Shared between the desktop side panel and the mobile bottom sheet so
   both get the same rendering, controls, and states for free. */
function PdfPreview({
  fileUrl,
  initialPage,
  fileName,
}: {
  fileUrl: string;
  initialPage: number | null;
  fileName: string;
}) {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState(initialPage && initialPage > 0 ? initialPage : 1);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [containerWidth, setContainerWidth] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Jump to the cited page whenever a new citation is opened (fileUrl
  // changing means a different open/reopen of the viewer).
  useEffect(() => {
    setPageNumber(initialPage && initialPage > 0 ? initialPage : 1);
    setLoadError(null);
    setZoom(1);
  }, [fileUrl, initialPage]);

  // Track the actual rendered width of the preview area so the page
  // scales to fill it — react-pdf renders at a fixed pixel width unless
  // told otherwise, so without this a phone screen would either show a
  // sliver of a desktop-sized render or require pinch-zooming just to
  // read a normal page.
  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) setContainerWidth(width);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const goToPage = (delta: number) => {
    setPageNumber((prev) => {
      const next = prev + delta;
      if (next < 1) return 1;
      if (numPages && next > numPages) return numPages;
      return next;
    });
  };

  const pageWidth = containerWidth > 0 ? Math.min(containerWidth - 24, 900) * zoom : undefined;

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Page canvas — the scrollable render area itself */}
      <div ref={containerRef} className="flex-1 min-h-0 overflow-auto bg-slate-100 dark:bg-slate-950 flex justify-center">
        {loadError ? (
          <div className="flex flex-col items-center justify-center gap-2 text-center px-6 py-16 text-sm text-slate-500 dark:text-slate-400">
            <AlertCircle className="w-6 h-6 text-red-500 dark:text-red-400" />
            <p>Couldn&apos;t render this PDF for preview.</p>
            <p className="text-xs text-slate-400 dark:text-slate-600">{loadError}</p>
          </div>
        ) : (
          <PdfDocument
            file={fileUrl}
            onLoadSuccess={(pdf: { numPages: number }) => setNumPages(pdf.numPages)}
            onLoadError={(error: Error) => setLoadError(error?.message || 'Unknown error')}
            loading={
              <div className="flex flex-col items-center justify-center gap-2 py-20 text-sm text-slate-400 dark:text-slate-500">
                <Loader2 className="w-5 h-5 animate-spin" />
                Loading preview…
              </div>
            }
            className="py-3"
          >
            {containerWidth > 0 && (
              <PdfPage
                pageNumber={pageNumber}
                width={pageWidth}
                className="shadow-md dark:shadow-black/40 rounded-md overflow-hidden"
                loading={
                  <div className="flex items-center justify-center py-16 text-slate-400 dark:text-slate-500">
                    <Loader2 className="w-5 h-5 animate-spin" />
                  </div>
                }
              />
            )}
          </PdfDocument>
        )}
      </div>

      {/* Controls — page navigation + zoom, theme-matched to the rest of
          the app instead of relying on whatever chrome a native PDF
          plugin would have bolted on. Hidden entirely while a document
          hasn't loaded yet or failed, since there's nothing to control. */}
      {!loadError && numPages !== null && (
        <div className="flex items-center justify-between gap-2 px-3 py-2 border-t border-slate-200 dark:border-slate-800/80 bg-white dark:bg-slate-900/80 shrink-0">
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => goToPage(-1)}
              disabled={pageNumber <= 1}
              className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition cursor-pointer"
              aria-label="Previous page"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-xs font-medium text-slate-600 dark:text-slate-300 tabular-nums px-1 min-w-[64px] text-center">
              {pageNumber} / {numPages}
            </span>
            <button
              type="button"
              onClick={() => goToPage(1)}
              disabled={numPages !== null && pageNumber >= numPages}
              className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition cursor-pointer"
              aria-label="Next page"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.2).toFixed(2)))}
              disabled={zoom <= 0.5}
              className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition cursor-pointer"
              aria-label="Zoom out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400 tabular-nums w-10 text-center">
              {Math.round(zoom * 100)}%
            </span>
            <button
              type="button"
              onClick={() => setZoom((z) => Math.min(2.5, +(z + 0.2).toFixed(2)))}
              disabled={zoom >= 2.5}
              className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition cursor-pointer"
              aria-label="Zoom in"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Home() {
  /* Upload state — multiple files can now be ingested and queried together */
  const [ingestedFiles, setIngestedFiles] = useState<IngestedFile[]>([]);
  const [uploading, setUploading] = useState(false);

  /* Chat history — session-only (kept in memory, never persisted), so a
     reload intentionally starts clean. Each conversation is independent;
     switching conversations does not touch ingestedFiles, since documents
     stay available to every conversation in this session. */
  const [conversations, setConversations] = useState<Conversation[]>(() => [
    { id: makeConversationId(), title: 'New Chat', messages: [] },
  ]);
  const [activeConversationId, setActiveConversationId] = useState<string>(
    () => conversations[0].id
  );

  const activeConversation =
    conversations.find((c) => c.id === activeConversationId) ?? conversations[0];
  const messages = activeConversation.messages;

  // Update only the active conversation's message list, titling it from the
  // first user message. There used to be a second pass here (requestAiTitle)
  // that asked the backend to summarize the question into a nicer title, but
  // it POSTed to /title, a route that does not exist — so it 404'd on every
  // send and the truncated text below was always what shipped anyway.
  const setMessagesForActive = useCallback(
    (updater: (prev: Message[]) => Message[]) => {
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeConversationId) return c;
          const nextMessages = updater(c.messages);
          const firstUserMsg = nextMessages.find((m) => m.sender === 'user');
          const nextTitle =
            c.title === 'New Chat' && firstUserMsg && !c.titleGenerated
              ? titleFromQuery(firstUserMsg.text)
              : c.title;
          return { ...c, messages: nextMessages, title: nextTitle };
        })
      );
    },
    [activeConversationId]
  );

  /* Files uploaded but not yet sent with a query — shown as chips above the
     input, then attached to the next user message once sent (mirrors how
     the Claude app shows an upload attached to the message you send after
     it, and leaves messages sent without an upload unaffected). */
  const [pendingAttachments, setPendingAttachments] = useState<string[]>([]);

  /* Right-side file viewer panel — opened by clicking an attached file chip
     or a source citation card. Renders the PDF the browser already has in
     memory (the File object kept on the matching ingestedFiles entry), via
     an object URL, so no extra backend call is needed to view it.

     Page jump: appending `#page=N` to the object URL is honored by Chrome/
     Edge/Firefox's built-in PDF viewer and scrolls straight to that page.
     True in-PDF text highlighting isn't something a plain <iframe> can be
     told to do from outside the document, so instead a small banner shows
     which page the citation came from — the closest equivalent achievable
     without shipping a full PDF.js-based custom renderer. */
  const [viewerFile, setViewerFile] = useState<string | null>(null);
  const [viewerUrl, setViewerUrl] = useState<string | null>(null);
  const [viewerPage, setViewerPage] = useState<number | null>(null);

  const openFileViewer = useCallback(
    (fname: string, page?: number) => {
      const match = ingestedFiles.find((f) => f.name === fname && f.file);
      if (!match?.file) return;
      // react-pdf can only render PDFs. Opening a DOCX/PPTX/XLSX blob in it
      // produces an "Failed to load PDF" viewer, so non-PDF cards simply don't
      // open a preview rather than opening a broken one.
      if (!fname.toLowerCase().endsWith('.pdf')) return;
      // Re-opening the file that's already showing used to still mint a new
      // blob URL every time this ran. PdfPreview resets pageNumber/zoom/
      // loadError whenever `fileUrl` changes, so a fresh URL for the same
      // file made the canvas reload and briefly go blank on every click —
      // the "flickers, then normal once touched" symptom. Only the page
      // number needs to move when it's the same file already open.
      if (fname === viewerFile) {
        setViewerPage(page ?? null);
        return;
      }
      const url = URL.createObjectURL(match.file);
      setViewerUrl((prevUrl) => {
        if (prevUrl) URL.revokeObjectURL(prevUrl);
        return url;
      });
      setViewerFile(fname);
      setViewerPage(page ?? null);
    },
    [ingestedFiles, viewerFile]
  );

  const closeFileViewer = useCallback(() => {
    setViewerFile(null);
    setViewerPage(null);
    setViewerUrl((prevUrl) => {
      if (prevUrl) URL.revokeObjectURL(prevUrl);
      return null;
    });
  }, []);

  // Release the object URL when the component unmounts so the blob isn't
  // held in memory after the panel is gone for good.
  useEffect(() => {
    return () => {
      setViewerUrl((prevUrl) => {
        if (prevUrl) URL.revokeObjectURL(prevUrl);
        return prevUrl;
      });
    };
  }, []);

  // Lock body scroll while the mobile full-screen viewer sheet is open, the
  // same way Claude's mobile app prevents the page behind a sheet from
  // scrolling.
  useEffect(() => {
    if (typeof document === 'undefined') return;
    if (!viewerFile) return;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prevOverflow;
    };
  }, [viewerFile]);

  /* Empty-state greeting — must NOT be computed during the initial render.
     `new Date().getHours()` differs between the server's clock (SSR) and
     the browser's local clock, so computing it eagerly produces two
     different strings — the server's version flashes, then snaps to the
     client's on hydration. Instead it starts empty and is set once in an
     effect, which only ever runs client-side, so there's exactly one
     render of it and no mismatch.

     Four real periods (morning / afternoon / evening / night), each with
     five headline + subtext pairs. Picked in a fixed round-robin order
     (not random) so consecutive visits within the same period never repeat
     the same line back-to-back — the next index per period is remembered
     in localStorage across reloads/sessions. */
  const GREETINGS: Record<'morning' | 'afternoon' | 'evening' | 'night', { headline: string; subtext: string }[]> = {
    morning: [
      { headline: 'Good morning.', subtext: "Drop in a PDF, or just ask a question — I'll pull it from the documents you've shared." },
      { headline: 'Morning. Ready when you are.', subtext: 'Upload a document or ask about one already shared, and I\u2019ll dig in.' },
      { headline: 'Rise and shine.', subtext: 'Bring a PDF, or ask about the ones already here — text, tables, or diagrams.' },
      { headline: 'Morning. Let\u2019s get into it.', subtext: 'Share a document to start, or ask about one you\u2019ve already uploaded.' },
      { headline: 'Good morning — first thing on the list?', subtext: 'Drop in a PDF, or ask about text, tables, or diagrams within one.' },
    ],
    afternoon: [
      { headline: 'Good afternoon.', subtext: "Drop in a PDF, or just ask a question — I'll pull it from the documents you've shared." },
      { headline: 'Afternoon. What are we digging into?', subtext: 'Share a document or pick up where an earlier one left off.' },
      { headline: 'Good afternoon — what can I help with?', subtext: 'Text, tables, diagrams — ask about anything you\u2019ve shared.' },
      { headline: 'Afternoon. Ready when you are.', subtext: 'Upload a PDF, or ask about one that\u2019s already here.' },
      { headline: 'Good afternoon — let\u2019s take a look.', subtext: 'Drop in a document and I\u2019ll help you work through it.' },
    ],
    evening: [
      { headline: 'Good evening.', subtext: "Drop in a PDF, or just ask a question — I'll pull it from the documents you've shared." },
      { headline: 'Evening. What can I help with?', subtext: 'Upload something new or ask about a document already on hand.' },
      { headline: 'Good evening — let\u2019s take a look.', subtext: 'Share a PDF, or ask about text, tables, or diagrams within one.' },
      { headline: 'Evening. Ready when you are.', subtext: 'Drop in a document to get started, or pick up an earlier one.' },
      { headline: 'Good evening — what\u2019s on the list?', subtext: 'Share a PDF and I\u2019ll help you get through it.' },
    ],
    night: [
      { headline: 'Still up? What can I help with?', subtext: "Drop in a PDF, or just ask a question — I'll pull it from the documents you've shared." },
      { headline: 'Working late?', subtext: 'Share a document and I\u2019ll help you get through it.' },
      { headline: 'Burning the midnight oil.', subtext: 'Upload a PDF or ask about one already shared — I\u2019m here.' },
      { headline: 'Late one, huh?', subtext: 'Drop in a document, or ask about one you\u2019ve already shared.' },
      { headline: 'Still up. Let\u2019s make it count.', subtext: 'Share a PDF, or ask about text, tables, or diagrams within one.' },
    ],
  };

  const [greeting, setGreeting] = useState('');
  const [greetingSubtext, setGreetingSubtext] = useState(
    "Drop in a PDF, or just ask a question — I'll pull it from the documents you've shared."
  );

  useEffect(() => {
    const hour = new Date().getHours();
    // morning 4–12, afternoon 12–16, evening 16–20, night 20–4 (wraps past midnight)
    const period: keyof typeof GREETINGS =
      hour < 4 || hour >= 20 ? 'night' : hour < 12 ? 'morning' : hour < 16 ? 'afternoon' : 'evening';
    const pool = GREETINGS[period];

    // Sequential, not random: read the last-used index for this period from
    // localStorage, advance it by one (wrapping around), and persist it —
    // so the very next visit during the same period is guaranteed to be a
    // different line, cycling through all five before any repeat.
    const storageKey = `docagent_greeting_idx_${period}`;
    let nextIndex = 0;
    try {
      const stored = window.localStorage.getItem(storageKey);
      const parsed = stored !== null ? parseInt(stored, 10) : -1;
      nextIndex = Number.isFinite(parsed) ? (parsed + 1) % pool.length : 0;
      window.localStorage.setItem(storageKey, String(nextIndex));
    } catch {
      // localStorage unavailable (private browsing, etc.) — fall back to
      // the first line in the period rather than breaking the greeting.
      nextIndex = 0;
    }

    const pick = pool[nextIndex];
    setGreeting(pick.headline);
    setGreetingSubtext(pick.subtext);
  }, []);

  /* Chat state */
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editText, setEditText] = useState('');

  /* UI state */
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Desktop sidebar is a separate toggle from the mobile drawer — collapsing
  // it on desktop removes it from layout flow (width -> 0) rather than
  // sliding an overlay over the content, matching how Claude's own desktop
  // sidebar reclaims the space instead of covering it.
  const [desktopSidebarOpen, setDesktopSidebarOpen] = useState(true);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [dragOver, setDragOver] = useState(false);

  /* Which chat-history item's overflow menu (⋯ → Delete) is open — only one
     at a time. Closed by picking an action, clicking elsewhere, or Escape. */
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const chatMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!openMenuId) return;
    const onClickAway = (e: MouseEvent) => {
      if (chatMenuRef.current && !chatMenuRef.current.contains(e.target as Node)) {
        setOpenMenuId(null);
      }
    };
    const onEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpenMenuId(null);
    };
    document.addEventListener('mousedown', onClickAway);
    document.addEventListener('keydown', onEscape);
    return () => {
      document.removeEventListener('mousedown', onClickAway);
      document.removeEventListener('keydown', onEscape);
    };
  }, [openMenuId]);

  /* File viewer panel width — user-resizable by dragging the left edge,
     same idea as Claude's own side panel. Persists only for the session.
     Dragging below MIN_VIEWER_WIDTH closes the panel outright (like
     dragging a native panel shut) instead of getting stuck at a floor. */
  const MIN_VIEWER_WIDTH = 320;
  const CLOSE_DRAG_THRESHOLD = 220; // drag narrower than this to dismiss
  const [viewerWidth, setViewerWidth] = useState(440);
  const viewerResizingRef = useRef(false);

  const startViewerResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    viewerResizingRef.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!viewerResizingRef.current) return;
      // Panel is on the right edge of the window, so its width is the
      // distance from the cursor to the right edge of the viewport.
      const next = window.innerWidth - e.clientX;
      const clamped = Math.min(Math.max(next, CLOSE_DRAG_THRESHOLD), Math.round(window.innerWidth * 0.7));
      setViewerWidth(clamped);
    };
    const onUp = (e: MouseEvent) => {
      if (!viewerResizingRef.current) return;
      viewerResizingRef.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';

      const next = window.innerWidth - e.clientX;
      if (next < MIN_VIEWER_WIDTH) {
        // Dragged shut — close the panel and reset to the default width so
        // the next open isn't stuck at whatever tiny size it was dragged to.
        closeFileViewer();
        setViewerWidth(440);
      } else {
        setViewerWidth(Math.min(next, Math.round(window.innerWidth * 0.7)));
      }
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [closeFileViewer]);

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
  // Bumped on unmount and on retry so a resolving ingest can't setState on an
  // unmounted tree or overwrite the result of a newer attempt.
  const pollTokenRef = useRef(0);

  useEffect(() => {
    return () => {
      pollTokenRef.current++;
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
     entry in ingestedFiles, shown as a file card (loading/success/failed),
     so one file failing doesn't hide another's progress or overwrite its result. */
  const ingestMultiple = async (files: File[]) => {
    // Previously filtered to `.pdf` only, so a dropped DOCX/PPTX/XLSX was
    // discarded silently — no card, no error, nothing. The backend reads all of
    // these now, and anything outside the list gets a failed card explaining why
    // rather than vanishing.
    const accepted: File[] = [];
    const rejected: File[] = [];

    for (const f of files) {
      const dot = f.name.lastIndexOf('.');
      const ext = dot === -1 ? '' : f.name.slice(dot).toLowerCase();
      (SUPPORTED_UPLOAD_EXTENSIONS.has(ext) ? accepted : rejected).push(f);
    }

    if (rejected.length) {
      setIngestedFiles((prev) => [
        ...prev.filter((f) => !rejected.some((r) => r.name === f.name)),
        ...rejected.map((f) => ({
          name: f.name,
          status: 'failed' as const,
          error: `Unsupported file type. Supported: ${SUPPORTED_UPLOAD_LABEL}.`,
        })),
      ]);
    }

    // Skip files that are already ingested or currently ingesting — retry
    // for a failed file goes through handleRetryIngest instead, which
    // explicitly re-triggers that one file.
    const newFiles = accepted.filter((f) => {
      const existing = ingestedFiles.find((ef) => ef.name === f.name);
      return !existing || existing.status === 'failed';
    });

    if (!newFiles.length) return;

    setSidebarOpen(false);
    setUploading(true);
    await Promise.all(newFiles.map((f) => triggerAutoIngest(f)));
    setUploading(false);
  };

  const triggerAutoIngest = async (fileToIngest: File) => {
    const name = fileToIngest.name;

    // Guards against a stale response overwriting a newer one for the same
    // filename — a retry started while the first request is still in flight.
    const requestToken = ++pollTokenRef.current;
    const isStale = () => pollTokenRef.current !== requestToken;

    // Captured now: the active chat can change while the upload is in flight,
    // and the document must be indexed under the chat it was dropped into.
    const targetChatId = activeConversationId;

    setIngestedFiles((prev) => [
      ...prev.filter((f) => f.name !== name),
      { name, status: 'processing', file: fileToIngest },
    ]);
    setPendingAttachments((prev) => (prev.includes(name) ? prev : [...prev, name]));

    try {
      // Ingestion is synchronous server-side: when this resolves the document is
      // fully indexed. There used to be a polling loop against /status/{file}
      // here, but no such route exists — every poll 404'd, so the card sat at
      // "processing" until it hit the 60-attempt timeout and falsely reported
      // failure for documents that had actually been indexed correctly.
      const res = await ingestDocument(fileToIngest, targetChatId);
      if (isStale()) return;

      setIngestedFiles((prev) =>
        prev.map((f) =>
          f.name === name
            ? { ...f, status: 'completed', pages: res.pages, chunks: res.chunks }
            : f
        )
      );
    } catch (err: unknown) {
      if (isStale()) return;
      const message = getErrorMessage(err, 'Failed to communicate with backend.');
      setIngestedFiles((prev) =>
        prev.map((f) => (f.name === name ? { ...f, status: 'failed', error: message } : f))
      );
    }
  };

  /* Retry — re-triggers ingestion for a failed file using the File object
     already held onto from the original upload, so the user doesn't have
     to re-pick it from disk. */
  const handleRetryIngest = (fileToRetry: IngestedFile) => {
    if (!fileToRetry.file) return;
    triggerAutoIngest(fileToRetry.file);
  };

  /* Query */
  const handleSendQuery = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || loading) return;

    const userText = query.trim();
    const attachedFiles = pendingAttachments.length ? pendingAttachments : undefined;
    // Captured before the async gap below — the active conversation could
    // change while this request is in flight, and the backend filters
    // retrieval on the chat id, so the answer must come from the vectors of
    // the conversation the message was actually sent in.
    const targetConversationId = activeConversationId;

    setQuery('');
    setPendingAttachments([]);
    setMessagesForActive((prev) => [
      ...prev,
      { sender: 'user', text: userText, ...(attachedFiles ? { attachedFiles } : {}) },
    ]);
    setLoading(true);

    try {
      // No `source` filter is passed, so the backend searches across every
      // ingested document's chunks rather than restricting to one file.
      const res = await queryDocument(userText, targetConversationId);
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
    const targetConversationId = activeConversationId;
    // Editing the very first user message changes what the chat is "about",
    // so it's worth re-titling. Editing a later message leaves the existing
    // title alone — only messages[0] being a user message identifies it as
    // the first one, since index 0 is always the welcome agent message.
    const isEditingFirstUserMessage = targetIndex === 1;
    setEditingIndex(null);
    setEditText('');

    setMessagesForActive((prev) => {
      const sliced = prev.slice(0, targetIndex);
      sliced.push({ sender: 'user', text: updatedQuery });
      return sliced;
    });

    if (isEditingFirstUserMessage) {
      // Re-title directly from the edited text. `setMessagesForActive` only
      // titles a conversation still called "New Chat", so flipping
      // `titleGenerated` alone would leave the old title in the sidebar.
      setConversations((prev) =>
        prev.map((c) =>
          c.id === targetConversationId
            ? { ...c, title: titleFromQuery(updatedQuery), titleGenerated: false }
            : c
        )
      );
    }

    setLoading(true);
    try {
      const res = await queryDocument(updatedQuery, targetConversationId);
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
      messages: [],
    };
    setConversations((prev) => [newConversation, ...prev]);
    setActiveConversationId(newConversation.id);
    setEditingIndex(null);
    setEditText('');
    setQuery('');
    // The side panel isn't scoped to a conversation, so without this the new
    // (empty) chat correctly says "no document uploaded" while the panel
    // keeps showing whichever PDF was last open in the previous chat.
    closeFileViewer();
  };

  const handleSwitchConversation = (id: string) => {
    if (id === activeConversationId) return;
    setActiveConversationId(id);
    setEditingIndex(null);
    setEditText('');
    setQuery('');
    // Same reasoning as handleNewChat: the previously open document would
    // otherwise carry over into a conversation that never had it uploaded.
    closeFileViewer();
  };

  /* Delete a conversation from the sidebar. If the deleted chat was the
     active one, falls back to whichever chat is now first in the list —
     or, if that was the last chat left, opens a fresh "New Chat" instead
     of leaving the UI with no conversation selected at all. */
  const handleDeleteConversation = (id: string) => {
    // Free the chat's vectors on the backend. The store is in-process on a
    // 512MB instance, so a deleted conversation's chunks would otherwise sit
    // there until the TTL sweep. Deliberately not awaited, and clearSession
    // never throws, so a failure here can't block the delete.
    void clearSession(id);
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (id === activeConversationId) {
        if (next.length > 0) {
          setActiveConversationId(next[0].id);
        } else {
          const fresh: Conversation = { id: makeConversationId(), title: 'New Chat', messages: [] };
          setActiveConversationId(fresh.id);
          return [fresh];
        }
      }
      return next;
    });
    setOpenMenuId(null);
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

  const isEmptyChat = messages.length === 0;

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
        accept={SUPPORTED_UPLOAD_ACCEPT}
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
          md:static md:z-auto md:shrink-0
          border-r border-slate-200 dark:border-slate-800/80 bg-white dark:bg-slate-900/95 md:bg-white md:dark:bg-slate-900/60
          backdrop-blur-xl flex flex-col justify-between shadow-2xl md:shadow-none
          transition-transform duration-250 ease-out
          md:transition-[width,margin] md:duration-200 md:ease-out md:overflow-hidden
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
          ${desktopSidebarOpen ? 'md:w-80' : 'md:w-0 md:border-r-0'}
        `}
      >
        <div className={`flex flex-col justify-between h-full p-5 md:p-6 ${desktopSidebarOpen ? '' : 'md:invisible'}`}>
        <div className="space-y-5 overflow-y-auto flex-1 min-w-[260px]">
          {/* Identity — logo and name side by side as one unit, that unit
              centered in the sidebar (not stacked/centered separately). */}
          <div className="flex items-center justify-center gap-3 pt-1">
            <DocAgentMark className="w-9 h-9 shrink-0" />
            <h1 className="font-bold text-slate-900 dark:text-slate-100 text-base tracking-tight">
              DocAgent RAG
            </h1>
          </div>

          {/* Toggles — a separate row below the identity block, with room
              above it, so they read as controls rather than part of the
              brand mark itself. */}
          <div className="flex items-center justify-center gap-1 pb-1">
            <button
              type="button"
              onClick={toggleTheme}
              className="p-2 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 transition cursor-pointer"
              aria-label="Toggle theme"
              title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>

            {/* Collapse — desktop only; mobile uses the X below via the
                overlay drawer instead. Same base style as the theme toggle
                above so both read at equal visual weight. */}
            <button
              type="button"
              onClick={() => setDesktopSidebarOpen(false)}
              className="hidden md:inline-flex p-2 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 transition cursor-pointer"
              aria-label="Collapse sidebar"
              title="Collapse sidebar"
            >
              <PanelLeftClose className="w-4 h-4" />
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

          <div className="border-t border-slate-200 dark:border-slate-800/80" />

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
              state. Not persisted to storage. Each row reveals a ⋯ menu on
              hover (always visible on touch) with a Delete action, same
              pattern as Claude's own chat list.

              No separate scroll container/height cap here: the outer sidebar
              body (line ~1330) is already `overflow-y-auto flex-1`, so this
              list just grows with it. Nesting a second `overflow-y-auto` +
              `max-h-48` here was clipping the list to ~192px regardless of
              how much vertical room the sidebar actually had, forcing its
              own cramped scrollbar and making the per-row ⋯ menu (which is
              absolutely positioned relative to each row) collide with that
              inner scrollbar. */}
          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Chat History
            </label>
            <div className="space-y-1 pr-0.5">
              {conversations.map((c) => (
                <div key={c.id} className="relative group">
                  <button
                    type="button"
                    onClick={() => {
                      handleSwitchConversation(c.id);
                      setSidebarOpen(false);
                    }}
                    className={`w-full text-left pl-3 pr-8 py-2 rounded-lg text-xs truncate transition cursor-pointer ${
                      c.id === activeConversationId
                        ? 'bg-indigo-100 dark:bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 font-medium'
                        : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800/60'
                    }`}
                    title={c.title}
                  >
                    {c.title}
                  </button>

                  {/* ⋯ trigger — quiet until the row is hovered/focused or
                      its menu is the one currently open, matching Claude's
                      own restrained per-row affordance. */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setOpenMenuId((prev) => (prev === c.id ? null : c.id));
                    }}
                    className={`absolute right-1 top-1/2 -translate-y-1/2 p-1 rounded-md text-slate-400 dark:text-slate-500 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200/70 dark:hover:bg-slate-700/60 transition cursor-pointer ${
                      openMenuId === c.id ? 'opacity-100 bg-slate-200/70 dark:bg-slate-700/60' : 'opacity-0 group-hover:opacity-100 focus:opacity-100'
                    }`}
                    aria-label={`More options for ${c.title}`}
                    aria-haspopup="menu"
                    aria-expanded={openMenuId === c.id}
                  >
                    <MoreHorizontal className="w-3.5 h-3.5" />
                  </button>

                  {openMenuId === c.id && (
                    <div
                      ref={chatMenuRef}
                      role="menu"
                      className="absolute right-0 top-[calc(100%+2px)] z-20 w-36 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-lg py-1 animate-fade-in"
                    >
                      <button
                        type="button"
                        role="menuitem"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteConversation(c.id);
                        }}
                        className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-500/10 transition cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

        </div>

        {/* Backend Status Footer */}
        <div className="text-xs text-slate-500 border-t border-slate-200 dark:border-slate-800/80 pt-4 mt-4 flex items-center justify-between min-w-[260px]">
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
        </div>
      </aside>

      {/* Floating expand control — shown only once the desktop sidebar is
          collapsed, fixed to the left edge like Claude's own collapsed-rail
          toggle rather than living inside the now-hidden sidebar. */}
      {!desktopSidebarOpen && (
        <button
          type="button"
          onClick={() => setDesktopSidebarOpen(true)}
          className="hidden md:inline-flex fixed top-5 left-5 z-40 p-2 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 shadow-sm transition cursor-pointer"
          aria-label="Expand sidebar"
          title="Expand sidebar"
        >
          <PanelLeftOpen className="w-4 h-4" />
        </button>
      )}

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
            <DocAgentMark className="w-7 h-7 shrink-0" />
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
        <div className={`flex-1 overflow-y-auto ${isEmptyChat ? 'flex flex-col' : 'p-4 sm:p-6 lg:p-8 space-y-5 sm:space-y-6'}`}>

          {/* Empty state hero — a quiet, personal welcome rather than a
              feature-card grid. The greeting varies with time of day, the
              same small touch Claude's own apps use to make a cold-start
              screen feel present rather than templated. The composer lives
              right here (not docked to the window bottom) so the whole
              greeting+input block sits centered together, the way a brand
              new Claude chat opens. */}
          {isEmptyChat && !loading && (
            <div className="flex-1 flex flex-col items-center justify-center text-center px-4 py-12">
              <DocAgentMark className="w-14 h-14 sm:w-16 sm:h-16 mb-6" />
              <h2 className="text-2xl sm:text-3xl font-semibold text-slate-900 dark:text-slate-100 mb-3 tracking-tight">
                {greeting}
              </h2>
              <p className="text-sm text-slate-500 dark:text-slate-400 max-w-sm leading-relaxed mb-8">
                {greetingSubtext}
              </p>
              <div className="w-full max-w-2xl px-4">
                <ChatComposer
                  query={query}
                  setQuery={setQuery}
                  onSubmit={handleSendQuery}
                  loading={loading}
                  onTriggerUpload={handleTriggerUpload}
                  onMicClick={handleMicClick}
                  listening={listening}
                  pendingAttachments={pendingAttachments}
                  setPendingAttachments={setPendingAttachments}
                  ingestedFiles={ingestedFiles}
                  handleRetryIngest={handleRetryIngest}
                  floating
                />
                <p className="text-center text-[11px] text-slate-400 dark:text-slate-600 mt-3">
                  Made by HS
                </p>
              </div>
            </div>
          )}

          {!isEmptyChat && (
          <>
          {/* Messages */}
          {messages.map((msg, index) => {
            const isUser = msg.sender === 'user';
            const isEditing = editingIndex === index;
            const dedupedSources = getDeduplicatedSources(msg.sources);

            return (
              <div
                key={index}
                className={`group flex items-start ${isUser ? 'justify-end' : 'justify-start'}`}
              >

                {/* Bubble */}
                <div className={`max-w-[85%] sm:max-w-3xl flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
                  {isUser && isEditing ? (
                    /* ── Inline Edit — replaces the bubble content in place,
                       same shape/width as a normal user bubble, rather than
                       a separate boxed card below it. */
                    <div className="w-full min-w-0 bg-indigo-600 rounded-2xl rounded-tr-none p-3.5 sm:p-4 shadow-sm space-y-2.5">
                      <textarea
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        className="w-full bg-white/10 border border-white/20 rounded-lg p-2.5 text-sm text-white placeholder-white/50 focus:outline-none focus:border-white/40 resize-y min-h-[60px]"
                        autoFocus
                      />
                      <div className="flex justify-end gap-2 text-xs">
                        <button
                          type="button"
                          onClick={handleCancelEdit}
                          className="px-3 py-1.5 rounded-lg text-white/80 hover:text-white hover:bg-white/10 transition cursor-pointer"
                        >
                          Cancel
                        </button>
                        <button
                          type="button"
                          onClick={() => handleSaveEdit(index)}
                          className="px-3.5 py-1.5 rounded-lg bg-white text-indigo-600 font-medium hover:bg-white/90 transition cursor-pointer"
                        >
                          Save
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
                      {isUser ? (
                        <>
                          {msg.attachedFiles && msg.attachedFiles.length > 0 && (
                            <div className="flex flex-wrap gap-1.5 mb-2">
                              {msg.attachedFiles.map((fname, fIndex) => (
                                <button
                                  key={fIndex}
                                  type="button"
                                  onClick={() => openFileViewer(fname)}
                                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/15 border border-white/20 text-[11px] sm:text-xs font-medium text-white hover:bg-white/25 transition cursor-pointer"
                                  title={`Open ${fname}`}
                                >
                                  <FileText className="w-3.5 h-3.5 shrink-0" />
                                  <span className="truncate max-w-[160px] sm:max-w-[220px]">{fname}</span>
                                </button>
                              ))}
                            </div>
                          )}
                          <p className="whitespace-pre-wrap break-words">{msg.text}</p>
                        </>
                      ) : (
                        // Gemini's answers are Markdown (headers, bold, bullet
                        // lists) — rendering that as plain text via <p> left
                        // raw "###" / "**" / "*" characters visible in the
                        // chat bubble instead of actual formatting. Only agent
                        // messages go through this: the user's own typed query
                        // should never be Markdown-interpreted (e.g. a literal
                        // "*" in their question shouldn't turn into italics).
                        <div className="markdown-body break-words text-sm leading-relaxed [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
                          <ReactMarkdown
                            components={{
                              h1: (props) => <h3 className="text-base font-bold mt-3 mb-1.5" {...props} />,
                              h2: (props) => <h3 className="text-base font-bold mt-3 mb-1.5" {...props} />,
                              h3: (props) => <h4 className="text-sm font-bold mt-3 mb-1.5" {...props} />,
                              p: (props) => <p className="mb-2 whitespace-pre-wrap" {...props} />,
                              ul: (props) => <ul className="list-disc pl-5 mb-2 space-y-0.5" {...props} />,
                              ol: (props) => <ol className="list-decimal pl-5 mb-2 space-y-0.5" {...props} />,
                              li: (props) => <li className="pl-0.5" {...props} />,
                              strong: (props) => <strong className="font-semibold" {...props} />,
                              code: (props) => (
                                <code
                                  className="bg-slate-100 dark:bg-slate-800 rounded px-1 py-0.5 text-xs font-mono"
                                  {...props}
                                />
                              ),
                              hr: () => <hr className="my-3 border-slate-200 dark:border-slate-700" />,
                              a: (props) => (
                                <a
                                  className="text-indigo-600 dark:text-indigo-400 underline"
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  {...props}
                                />
                              ),
                            }}
                          >
                            {msg.text}
                          </ReactMarkdown>
                        </div>
                      )}

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
                              const canOpen = ingestedFiles.some((f) => f.name === sourceName && f.file);

                              return (
                                <button
                                  key={srcIndex}
                                  type="button"
                                  onClick={() => canOpen && openFileViewer(sourceName, src.page)}
                                  disabled={!canOpen}
                                  className={`inline-flex items-center gap-1 px-2 sm:px-2.5 py-1 rounded-lg border text-[11px] sm:text-xs font-mono transition shadow-sm ${
                                    !canOpen ? 'cursor-default opacity-90' : 'cursor-pointer hover:shadow-md hover:-translate-y-px'
                                  } ${
                                    isVisual
                                      ? 'bg-violet-50 dark:bg-violet-950/40 border-violet-200 dark:border-violet-500/40 text-violet-700 dark:text-violet-300'
                                      : 'bg-slate-100 dark:bg-slate-800/80 border-slate-200 dark:border-slate-700/80 text-indigo-700 dark:text-indigo-300'
                                  }`}
                                  title={canOpen ? `Open ${sourceName} at ${pageText || 'page 1'}` : sourceName}
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
                                </button>
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
                      className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 mt-1 text-xs text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 flex items-center gap-1 px-1 py-0.5 cursor-pointer"
                      title="Edit this query and regenerate response"
                    >
                      <Edit3 className="w-3 h-3" /> Edit
                    </button>
                  )}
                </div>
              </div>
            );
          })}

          {/* Loading Indicator */}
          {loading && (
            <div className="flex items-center text-slate-500 dark:text-slate-400 text-sm">
              <div className="bg-white dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 px-3.5 sm:px-4 py-2.5 sm:py-3 rounded-2xl rounded-tl-none flex items-center gap-2.5 text-xs text-slate-700 dark:text-slate-300 shadow-sm">
                <Loader2 className="w-4 h-4 animate-spin text-indigo-600 dark:text-indigo-400" />
                <span className="hidden sm:inline">
                  Retrieving document chunks & generating answer with Gemini...
                </span>
                <span className="sm:hidden">Generating answer...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
          </>
          )}
        </div>

        {/* ── Chat Input — docked at the bottom once a conversation has
            started. Borderless/floating (no top divider, no bar-spanning
            background) so it reads as a composer sitting in the chat rather
            than a fixed toolbar underneath it. */}
        {!isEmptyChat && (
          <div className="px-3 sm:px-4 lg:px-6 pb-3 sm:pb-4 lg:pb-5 pt-2 shrink-0">
            <div className="max-w-4xl mx-auto">
              <ChatComposer
                query={query}
                setQuery={setQuery}
                onSubmit={handleSendQuery}
                loading={loading}
                onTriggerUpload={handleTriggerUpload}
                onMicClick={handleMicClick}
                listening={listening}
                pendingAttachments={pendingAttachments}
                setPendingAttachments={setPendingAttachments}
                ingestedFiles={ingestedFiles}
                handleRetryIngest={handleRetryIngest}
              />
            </div>
            <p className="text-center text-[11px] text-slate-400 dark:text-slate-600 mt-2.5">
              Made by HS
            </p>
          </div>
        )}
      </main>

      {/* ═══════════ FILE VIEWER PANEL ═══════════ */}
      {viewerFile && (
        <aside
          className="hidden md:flex flex-col shrink-0 relative border-l border-slate-200 dark:border-slate-800/80 bg-white dark:bg-slate-900/95 backdrop-blur-xl animate-fade-in"
          style={{ width: viewerWidth }}
        >
          {/* Drag handle — grabbing the left edge resizes the panel, same
              interaction as Claude's own side panel. A slightly wider
              invisible hit-area sits over the visible 1px border so it's
              actually easy to grab. */}
          <div
            onMouseDown={startViewerResize}
            className="absolute -left-1.5 top-0 bottom-0 w-3 cursor-col-resize z-10 group flex justify-center"
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize file viewer"
          >
            <div className="w-px h-full bg-transparent group-hover:bg-indigo-500/40 transition-colors" />
          </div>
          <div className="flex items-center justify-between gap-3 px-4 py-3.5 border-b border-slate-200 dark:border-slate-800/80 shrink-0">
            <div className="flex items-center gap-2 min-w-0">
              <FileText className="w-4 h-4 text-indigo-600 dark:text-indigo-400 shrink-0" />
              <span className="text-sm font-medium text-slate-800 dark:text-slate-200 truncate" title={viewerFile}>
                {viewerFile}
              </span>
            </div>
            <button
              type="button"
              onClick={closeFileViewer}
              className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 transition shrink-0 cursor-pointer"
              aria-label="Close file viewer"
              title="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Citation banner — a plain <iframe>'s native PDF viewer can be
              told which page to jump to (via the URL hash below) but not
              told to highlight text inside the document itself, so this
              banner names the cited page as the closest achievable stand-in. */}
          {viewerPage && (
            <div className="flex items-center gap-2 px-4 py-2 bg-amber-50 dark:bg-amber-500/10 border-b border-amber-200 dark:border-amber-500/20 text-xs text-amber-800 dark:text-amber-300 shrink-0">
              <Search className="w-3.5 h-3.5 shrink-0" />
              <span>Cited from page {viewerPage} — jumped there below.</span>
            </div>
          )}

          <div className="flex-1 min-h-0 bg-slate-100 dark:bg-slate-950">
            {viewerUrl ? (
              <PdfPreview fileUrl={viewerUrl} initialPage={viewerPage} fileName={viewerFile} />
            ) : (
              <div className="flex items-center justify-center h-full text-sm text-slate-400 dark:text-slate-500">
                Couldn&apos;t load this file for preview.
              </div>
            )}
          </div>
        </aside>
      )}

      {/* Mobile File Viewer — a bottom sheet rather than a side panel,
          matching how Claude's mobile app surfaces attachments: slides up
          over the chat, rounded top corners, drag-handle affordance, and a
          compact header instead of a full desktop chrome bar. */}
      {viewerFile && (
        <div className="md:hidden fixed inset-0 z-[70] flex flex-col justify-end">
          <style>{`
            @keyframes slide-up {
              from { transform: translateY(100%); }
              to { transform: translateY(0); }
            }
          `}</style>
          <div
            className="absolute inset-0 bg-slate-950/50 backdrop-blur-sm animate-fade-in"
            onClick={closeFileViewer}
          />
          <div className="relative flex flex-col bg-white dark:bg-slate-950 rounded-t-2xl shadow-2xl h-[92vh] overflow-hidden animate-[slide-up_0.25s_ease-out]">
            {/* Drag handle */}
            <div className="flex justify-center pt-2.5 pb-1 shrink-0">
              <div className="w-10 h-1 rounded-full bg-slate-300 dark:bg-slate-700" />
            </div>

            <div className="flex items-center justify-between gap-3 px-4 pb-3 pt-1 border-b border-slate-200 dark:border-slate-800/80 shrink-0">
              <div className="flex items-center gap-2 min-w-0">
                <FileText className="w-4 h-4 text-indigo-600 dark:text-indigo-400 shrink-0" />
                <span className="text-sm font-medium text-slate-800 dark:text-slate-200 truncate" title={viewerFile}>
                  {viewerFile}
                </span>
              </div>
              <button
                type="button"
                onClick={closeFileViewer}
                className="p-1.5 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 transition shrink-0 cursor-pointer"
                aria-label="Close file viewer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {viewerPage && (
              <div className="flex items-center gap-2 px-4 py-2 bg-amber-50 dark:bg-amber-500/10 border-b border-amber-200 dark:border-amber-500/20 text-xs text-amber-800 dark:text-amber-300 shrink-0">
                <Search className="w-3.5 h-3.5 shrink-0" />
                <span>Cited from page {viewerPage} — jumped there below.</span>
              </div>
            )}

            <div className="flex-1 min-h-0 bg-slate-100 dark:bg-slate-900">
              {viewerUrl ? (
                <PdfPreview fileUrl={viewerUrl} initialPage={viewerPage} fileName={viewerFile} />
              ) : (
                <div className="flex items-center justify-center h-full text-sm text-slate-400 dark:text-slate-500">
                  Couldn&apos;t load this file for preview.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}