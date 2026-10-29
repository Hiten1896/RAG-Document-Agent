'use client';

import { useEffect, useRef, useState } from 'react';
import { Markmap } from 'markmap-view';
import { Transformer } from 'markmap-lib';
import type { INode, IPureNode } from 'markmap-common';
import {
  AlertCircle,
  Download,
  Expand,
  Loader2,
  RefreshCw,
  X,
} from 'lucide-react';
import type { MindMapNode, MindMapResponse } from '@/lib/api';

interface MindMapPanelProps {
  data: MindMapResponse | null;
  loading: boolean;
  error: string | null;
  canGenerate: boolean;
  onClose: () => void;
  onRetry: () => void;
  onRegenerate: () => void;
}

function escapeHeading(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/([\\`*_{}[\]()#+.!|>])/g, '\\$1');
}

function appendHeadings(
  node: MindMapNode,
  depth: number,
  lines: string[],
): void {
  lines.push(`${'#'.repeat(depth)} ${escapeHeading(node.title)}`);
  node.children.forEach((child) => appendHeadings(child, depth + 1, lines));
}

function attachSources(
  transformed: IPureNode,
  nodes: MindMapNode[],
): void {
  transformed.children.forEach((child, index) => {
    const sourceNode = nodes[index];
    if (!sourceNode) return;
    child.payload = { ...child.payload, source: sourceNode.source };
    attachSources(child, sourceNode.children);
  });
}

function fileSlug(value: string): string {
  return value
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .toLowerCase() || 'document-mind-map';
}

function downloadBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function MindMapPanel({
  data,
  loading,
  error,
  canGenerate,
  onClose,
  onRetry,
  onRegenerate,
}: MindMapPanelProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const markmapRef = useRef<Markmap | null>(null);
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const rootElement = document.documentElement;
    const syncTheme = () => setDarkMode(rootElement.classList.contains('dark'));
    syncTheme();
    const observer = new MutationObserver(syncTheme);
    observer.observe(rootElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || !data) return;
    setSelectedSource(null);
    setExportError(null);

    const lines = [`# ${escapeHeading(data.title)}`];
    data.children.forEach((node) => appendHeadings(node, 2, lines));
    const transformed = new Transformer().transform(lines.join('\n'));
    attachSources(transformed.root, data.children);

    const markmap = Markmap.create(
      svg,
      {
        autoFit: true,
        duration: 300,
        embedGlobalCSS: true,
        maxWidth: 240,
      },
      transformed.root,
    );
    markmapRef.current = markmap;

    const handleNodeClick = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      const group = target.closest<SVGGElement>('g.markmap-node');
      if (!group) return;
      const boundNode = (group as SVGGElement & { __data__?: INode }).__data__;
      const source = boundNode?.payload?.source;
      if (typeof source !== 'string' || !source) return;
      setSelectedSource(source);
    };
    svg.addEventListener('click', handleNodeClick);

    return () => {
      svg.removeEventListener('click', handleNodeClick);
      markmap.destroy();
      markmapRef.current = null;
    };
  }, [data]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const colors = darkMode
      ? { text: '#e2e8f0', circle: '#0f172a', highlight: '#312e81' }
      : { text: '#1e293b', circle: '#ffffff', highlight: '#e0e7ff' };
    svg.style.setProperty('--markmap-text-color', colors.text);
    svg.style.setProperty('--markmap-circle-open-bg', colors.circle);
    svg.style.setProperty('--markmap-highlight-node-bg', colors.highlight);
    markmapRef.current?.updateStyle();
  }, [darkMode, data]);

  const handleFit = () => {
    void markmapRef.current?.fit().catch((cause: unknown) => {
      setExportError(cause instanceof Error ? cause.message : 'Could not fit the map to screen.');
    });
  };

  const handleDownloadSvg = () => {
    setExportError(null);
    const svg = svgRef.current;
    if (!svg || !data) return;
    try {
      const bounds = svg.getBoundingClientRect();
      const clone = svg.cloneNode(true) as SVGSVGElement;
      clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      clone.setAttribute('width', String(Math.max(1, Math.ceil(bounds.width))));
      clone.setAttribute('height', String(Math.max(1, Math.ceil(bounds.height))));
      const source = new XMLSerializer().serializeToString(clone);
      downloadBlob(
        new Blob([source], { type: 'image/svg+xml;charset=utf-8' }),
        `${fileSlug(data.title)}.svg`,
      );
    } catch (cause: unknown) {
      setExportError(cause instanceof Error ? cause.message : 'Could not export the SVG.');
    }
  };

  const handleDownloadPng = async () => {
    setExportError(null);
    const svg = svgRef.current;
    if (!svg || !data) return;

    try {
      const bounds = svg.getBoundingClientRect();
      const width = Math.max(1, Math.ceil(bounds.width));
      const height = Math.max(1, Math.ceil(bounds.height));
      const clone = svg.cloneNode(true) as SVGSVGElement;
      clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      clone.setAttribute('width', String(width));
      clone.setAttribute('height', String(height));
      const sourceLabels = svg.querySelectorAll('foreignObject');
      clone.querySelectorAll('foreignObject').forEach((foreignObject, index) => {
        const sourceLabel = sourceLabels.item(index);
        if (!sourceLabel) return;

        const label = sourceLabel.querySelector('div div') ?? sourceLabel;
        const style = getComputedStyle(label);
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        const y = Number(foreignObject.getAttribute('y') || 0);
        const fontSize = Number.parseFloat(style.fontSize) || 16;
        text.setAttribute('x', foreignObject.getAttribute('x') || '0');
        text.setAttribute('y', String(y + fontSize * 0.8));
        text.setAttribute('fill', style.color);
        text.setAttribute('font-family', style.fontFamily);
        text.setAttribute('font-size', style.fontSize);
        text.setAttribute('font-weight', style.fontWeight);
        text.setAttribute('font-style', style.fontStyle);
        text.textContent = sourceLabel.textContent || '';
        foreignObject.replaceWith(text);
      });
      const imageSource = URL.createObjectURL(
        new Blob([new XMLSerializer().serializeToString(clone)], {
          type: 'image/svg+xml;charset=utf-8',
        }),
      );

      try {
        const image = new Image();
        await new Promise<void>((resolve, reject) => {
          image.onload = () => resolve();
          image.onerror = () => reject(new Error('Could not render the mind map image.'));
          image.src = imageSource;
        });

        const scale = 2;
        const canvas = document.createElement('canvas');
        canvas.width = width * scale;
        canvas.height = height * scale;
        const context = canvas.getContext('2d');
        if (!context) throw new Error('PNG export is not supported by this browser.');
        context.fillStyle = darkMode ? '#0f172a' : '#ffffff';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.scale(scale, scale);
        context.drawImage(image, 0, 0, width, height);

        const png = await new Promise<Blob>((resolve, reject) => {
          canvas.toBlob((blob) => {
            if (blob) resolve(blob);
            else reject(new Error('Could not create the PNG file.'));
          }, 'image/png');
        });
        downloadBlob(png, `${fileSlug(data.title)}.png`);
      } finally {
        URL.revokeObjectURL(imageSource);
      }
    } catch (cause: unknown) {
      setExportError(cause instanceof Error ? cause.message : 'Could not export the PNG.');
    }
  };

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center bg-slate-950/60 p-2 sm:p-5"
      role="dialog"
      aria-modal="true"
      aria-label="Document mind map"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section className="flex h-full max-h-[900px] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-900">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-4 py-3 dark:border-slate-800 sm:px-5">
          <div className="min-w-0">
            <h2 className="truncate text-base font-semibold text-slate-900 dark:text-slate-100">
              {data?.title || 'Document mind map'}
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Built only from the documents in this chat
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-1">
            <button
              type="button"
              onClick={handleFit}
              disabled={!data || loading}
              className="inline-flex min-h-10 items-center gap-1.5 rounded-lg px-2.5 text-xs text-slate-600 hover:bg-slate-100 disabled:opacity-40 dark:text-slate-300 dark:hover:bg-slate-800"
              title="Fit to screen"
              aria-label="Fit to screen"
            >
              <Expand className="h-4 w-4" />
              <span className="hidden sm:inline">Fit</span>
            </button>
            <button
              type="button"
              onClick={handleDownloadSvg}
              disabled={!data || loading}
              className="inline-flex min-h-10 items-center gap-1.5 rounded-lg px-2.5 text-xs text-slate-600 hover:bg-slate-100 disabled:opacity-40 dark:text-slate-300 dark:hover:bg-slate-800"
              title="Download SVG"
              aria-label="Download SVG"
            >
              <Download className="h-4 w-4" />
              <span className="hidden sm:inline">SVG</span>
            </button>
            <button
              type="button"
              onClick={() => void handleDownloadPng()}
              disabled={!data || loading}
              className="inline-flex min-h-10 items-center gap-1.5 rounded-lg px-2.5 text-xs text-slate-600 hover:bg-slate-100 disabled:opacity-40 dark:text-slate-300 dark:hover:bg-slate-800"
              title="Download PNG"
              aria-label="Download PNG"
            >
              <Download className="h-4 w-4" />
              <span className="hidden sm:inline">PNG</span>
            </button>
            <button
              type="button"
              onClick={onRegenerate}
              disabled={!canGenerate || loading}
              className="inline-flex min-h-10 items-center gap-1.5 rounded-lg px-2.5 text-xs text-indigo-600 hover:bg-indigo-50 disabled:opacity-40 dark:text-indigo-300 dark:hover:bg-indigo-500/10"
              title="Regenerate"
              aria-label="Regenerate mind map"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              <span className="hidden sm:inline">Regenerate</span>
            </button>
            <button
              type="button"
              onClick={onClose}
              className="inline-flex min-h-10 min-w-10 items-center justify-center rounded-lg text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
              aria-label="Close mind map"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </header>

        {(selectedSource || exportError || (data && error)) && (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-slate-200 px-4 py-2 text-xs dark:border-slate-800">
            {selectedSource && (
              <p className="text-indigo-700 dark:text-indigo-300">
                Source location: {selectedSource}
              </p>
            )}
            {data && error && (
              <>
                <p role="alert" className="text-amber-700 dark:text-amber-300">
                  Regeneration failed: {error}
                </p>
                {canGenerate && (
                  <button
                    type="button"
                    onClick={onRetry}
                    className="rounded-md px-2 py-1 font-medium text-indigo-600 hover:bg-indigo-50 dark:text-indigo-300 dark:hover:bg-indigo-500/10"
                  >
                    Retry
                  </button>
                )}
              </>
            )}
            {exportError && (
              <p role="alert" className="text-red-600 dark:text-red-400">
                {exportError}
              </p>
            )}
          </div>
        )}

        <div className={`relative min-h-0 flex-1 bg-slate-50 dark:bg-slate-950 ${darkMode ? 'markmap-dark' : ''}`}>
          {data && (
            <svg
              ref={svgRef}
              className="h-full w-full"
              aria-label="Interactive mind map. Use the controls to zoom, pan, and collapse branches."
            />
          )}
          {!data && !loading && (
            <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
              {error ? (
                <>
                  <AlertCircle className="h-8 w-8 text-amber-500" />
                  <p className="max-w-md text-sm text-slate-600 dark:text-slate-300">
                    {error}
                  </p>
                  {canGenerate && (
                    <button
                      type="button"
                      onClick={onRetry}
                      className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500"
                    >
                      Try again
                    </button>
                  )}
                </>
              ) : (
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Upload a document to create a grounded mind map.
                </p>
              )}
            </div>
          )}
          {loading && !data && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-50/90 text-sm text-slate-600 dark:bg-slate-950/90 dark:text-slate-300">
              <Loader2 className="h-7 w-7 animate-spin text-indigo-500" />
              Building a map from your document…
            </div>
          )}
          {loading && data && (
            <div className="absolute right-3 top-3 inline-flex items-center gap-2 rounded-full bg-white/90 px-3 py-1.5 text-xs text-slate-600 shadow dark:bg-slate-900/90 dark:text-slate-300">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-500" />
              Regenerating…
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
