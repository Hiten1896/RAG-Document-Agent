'use client';

import { useState, useRef, useEffect } from 'react';
import { Document as PdfDocument, Page as PdfPage, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import { AlertCircle, ChevronLeft, ChevronRight, Loader2, ZoomIn, ZoomOut } from 'lucide-react';

// This module is only reached via next/dynamic(() => import('./PdfPreview'),
// { ssr: false }) in page.tsx — react-pdf/pdfjs is a genuinely large
// dependency (PDF.js itself is several hundred KB), and it was previously
// imported at the top of page.tsx, so every single page load fetched and
// parsed it regardless of whether the visitor ever opened a document.
// Lighthouse flagged this directly: "Reduce unused JavaScript — Est savings
// of 1,231 KiB" and a 4.7s Total Blocking Time were both largely this.
// Splitting it into its own chunk means that cost is only paid the first
// time someone actually opens the file viewer.
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

/* Shared PDF viewer — used by both the desktop side panel and the mobile
   bottom sheet in page.tsx. Renders via react-pdf/pdfjs rather than a native
   <embed>/<iframe> so it can be styled consistently, works the same across
   browsers (some mobile browsers have no built-in PDF viewer at all), scales
   crisply to the container's actual width via ResizeObserver, and gives real
   programmatic page-jump control for citations instead of the "#page=N"
   URL-hash trick some mobile browsers ignore entirely. */
export default function PdfPreview({
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
              className="p-1.5 min-w-11 min-h-11 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition cursor-pointer"
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
              className="p-1.5 min-w-11 min-h-11 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition cursor-pointer"
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
              className="p-1.5 min-w-11 min-h-11 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition cursor-pointer"
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
              className="p-1.5 min-w-11 min-h-11 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition cursor-pointer"
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