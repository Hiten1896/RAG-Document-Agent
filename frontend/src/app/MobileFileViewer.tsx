'use client';

import { FileText, Search, X } from 'lucide-react';
import PdfPreview from './PdfPreview';

/* Mobile File Viewer — a bottom sheet rather than a side panel, matching how
   Claude's mobile app surfaces attachments: slides up over the chat, rounded
   top corners, drag-handle affordance, and a compact header instead of a
   full desktop chrome bar.

   Split out of page.tsx and loaded via next/dynamic (see the dynamic()
   wrapper in page.tsx) purely for load-time: this UI is CSS-hidden above the
   md: breakpoint anyway (the desktop side panel handles that case), so a
   desktop-only visitor's initial bundle has no reason to include it. It was
   previously inline in page.tsx, meaning every visitor — mobile or not —
   downloaded and parsed this JSX and its own PdfPreview reference on first
   load, contributing to the "Reduce unused JavaScript" Lighthouse flag.

   All state lives in the parent (page.tsx) and is passed down as props
   rather than owned here, since the viewer's open/closed state and which
   file/page it's showing need to stay in sync with the desktop side panel
   and the citation buttons that can open either one. */
export default function MobileFileViewer({
  viewerFile,
  viewerUrl,
  viewerPage,
  onClose,
}: {
  viewerFile: string;
  viewerUrl: string | null;
  viewerPage: number | null;
  onClose: () => void;
}) {
  return (
    <div className="md:hidden fixed inset-0 z-[70] flex flex-col justify-end">
      <style>{`
        @keyframes slide-up {
          from { transform: translateY(100%); }
          to { transform: translateY(0); }
        }
      `}</style>
      <div
        className="absolute inset-0 bg-slate-950/50 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
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
            onClick={onClose}
            className="p-2 min-w-10 min-h-10 flex items-center justify-center rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 transition shrink-0 cursor-pointer"
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
          {viewerFile && !viewerFile.toLowerCase().endsWith('.pdf') ? (
            <div className="flex flex-col items-center justify-center h-full gap-2 text-center px-6">
              <FileText className="w-8 h-8 text-slate-400 dark:text-slate-600" />
              <p className="text-sm text-slate-500 dark:text-slate-400">
                In-app preview isn&apos;t available for this file type yet.
              </p>
              <p className="text-xs text-slate-400 dark:text-slate-600">
                {viewerFile}
              </p>
            </div>
          ) : viewerUrl ? (
            <PdfPreview fileUrl={viewerUrl} initialPage={viewerPage} fileName={viewerFile} />
          ) : (
            <div className="flex items-center justify-center h-full text-sm text-slate-400 dark:text-slate-500">
              Couldn&apos;t load this file for preview.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}