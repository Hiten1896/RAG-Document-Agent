'use client';

import { Download, Loader2, RefreshCw, X } from 'lucide-react';
import type { MindMapResponse, VisualChart, VisualKind, VisualResponse } from '@/lib/api';
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import ModalShell from './ModalShell';
import MindMapPanel from './MindMapPanel';

export type VisualStudioTab = 'mindmap' | VisualKind;

interface Props {
  tab: VisualStudioTab;
  visualData: Partial<Record<VisualKind, VisualResponse | null>>;
  visualLoading: boolean;
  visualError: string | null;
  mindMapData: MindMapResponse | null;
  mindMapLoading: boolean;
  mindMapError: string | null;
  canGenerateMindMap: boolean;
  onClose: () => void;
  onRetry: () => void;
  onRegenerate: () => void;
  onMindMapRetry: () => void;
  onMindMapRegenerate: () => void;
}

const colors = ['#6366f1', '#14b8a6', '#f59e0b', '#f43f5e', '#8b5cf6', '#06b6d4'];

function Source({ value }: { value: string }) {
  return <span className="text-[10px] text-slate-500 dark:text-slate-400">Source: {value}</span>;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function ChartView({ data }: { data: VisualChart }) {
  const values = data.labels.map((label, index) => ({ label, value: data.values[index] }));
  const chart = data.type === 'line' ? <LineChart data={values}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="label" /><YAxis /><Tooltip /><Line type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} /></LineChart> : data.type === 'pie' ? <PieChart><Pie data={values} dataKey="value" nameKey="label" outerRadius="75%" label>{values.map((entry, index) => <Cell key={entry.label} fill={colors[index % colors.length]} />)}</Pie><Tooltip /></PieChart> : <BarChart data={values}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="label" /><YAxis /><Tooltip /><Bar dataKey="value" fill="#6366f1" /></BarChart>;
  return <div className="min-w-0 w-full"><h3 className="mb-1 text-lg font-semibold">{data.title}</h3><div className="h-72 min-w-0 w-full"><ResponsiveContainer width="100%" height="100%" minWidth={0}>{chart}</ResponsiveContainer></div><Source value={data.source} /><div className="mt-3 flex gap-2"><button type="button" onClick={() => downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), 'visual-chart.json')} className="visual-download"><Download className="h-3.5 w-3.5" />Download data</button></div></div>;
}

export default function VisualPanel({ tab, visualData, visualLoading, visualError, mindMapData, mindMapLoading, mindMapError, canGenerateMindMap, onClose, onRetry, onRegenerate, onMindMapRetry, onMindMapRegenerate, embedded = false }: Props & { embedded?: boolean }) {
  const kind = tab === 'mindmap' ? null : tab;
  const data = kind ? visualData[kind] ?? null : null;
  const loading = tab === 'mindmap' ? mindMapLoading : visualLoading;
  const error = tab === 'mindmap' ? mindMapError : visualError;
  const content = <>
      {tab === 'mindmap' && <MindMapPanel data={mindMapData} loading={mindMapLoading} error={mindMapError} canGenerate={canGenerateMindMap} onClose={onClose} onRetry={onMindMapRetry} onRegenerate={onMindMapRegenerate} embedded />}
      {tab !== 'mindmap' && <div className="mb-4 flex justify-end"><button type="button" onClick={onRegenerate} disabled={loading} className="visual-download" title="Regenerate" aria-label="Regenerate"><RefreshCw className="h-3.5 w-3.5" /></button></div>}
      {tab !== 'mindmap' && loading && <div className="flex min-h-64 items-center justify-center gap-2 text-sm text-slate-500"><Loader2 className="h-5 w-5 animate-spin" />Preparing a grounded visual…</div>}
      {tab !== 'mindmap' && !loading && error && <div className="flex min-h-48 flex-col items-center justify-center gap-3 text-center"><p className="text-sm text-rose-600 dark:text-rose-300">{error}</p><button type="button" onClick={onRetry} className="visual-download"><RefreshCw className="h-3.5 w-3.5" />Retry</button></div>}
      {tab !== 'mindmap' && !loading && !error && data && <ChartView data={data as VisualChart} />}
  </>;
  return embedded ? (
    <aside className="flex h-full min-h-0 flex-col overflow-hidden border-l border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900" aria-label="Artifacts">
      <header className="mb-2 flex shrink-0 items-center justify-between border-b border-slate-200 pb-2 dark:border-slate-800">
        <h2 className="text-sm font-semibold">Artifacts</h2>
        <button
          type="button"
          onClick={onClose}
          className="shrink-0 rounded-lg p-1.5 text-slate-500 transition hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
          aria-label="Close visual studio"
          title="Close"
        >
          <X className="h-4 w-4" />
        </button>
      </header>
      <div className="min-h-0 min-w-0 flex-1 overflow-hidden">{content}</div>
    </aside>
  ) : <ModalShell title="Visual studio" label="Grounded visual answer" onClose={onClose}>{content}</ModalShell>;
}
