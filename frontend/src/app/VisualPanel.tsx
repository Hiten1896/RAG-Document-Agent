'use client';

import { useEffect, useRef, useState } from 'react';
import { Download, Loader2, RefreshCw } from 'lucide-react';
import type { VisualChart, VisualDiagram, VisualExplainer, VisualKind, VisualResponse } from '@/lib/api';
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import ModalShell from './ModalShell';

interface Props {
  kind: VisualKind;
  data: VisualResponse | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onRetry: () => void;
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

async function downloadSvgAsPng(svg: string, filename: string) {
  const image = new Image();
  image.src = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
  await new Promise<void>((resolve, reject) => { image.onload = () => resolve(); image.onerror = () => reject(new Error('Could not render PNG.')); });
  const canvas = document.createElement('canvas');
  canvas.width = image.width * 2;
  canvas.height = image.height * 2;
  canvas.getContext('2d')?.drawImage(image, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => { if (blob) downloadBlob(blob, filename); }, 'image/png');
}

function DiagramView({ data }: { data: VisualDiagram }) {
  const host = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState('');
  useEffect(() => {
    let cancelled = false;
    void import('mermaid/dist/mermaid.js').then(async ({ default: mermaid }) => {
      mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'base' });
      const rendered = await mermaid.render(`visual-${Date.now()}`, data.mermaid);
      if (!cancelled && host.current) {
        host.current.innerHTML = rendered.svg;
        setSvg(rendered.svg);
      }
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [data.mermaid]);
  return (
    <>
      <div ref={host} className="visual-mermaid flex min-h-64 items-center justify-center overflow-auto rounded-xl bg-white p-4 dark:bg-slate-950" aria-label="Generated diagram" />
      <div className="mt-3 flex gap-2">
        <button type="button" disabled={!svg} onClick={() => downloadBlob(new Blob([svg], { type: 'image/svg+xml' }), 'visual-diagram.svg')} className="visual-download"><Download className="h-3.5 w-3.5" />SVG</button>
        <button type="button" disabled={!svg} onClick={() => void downloadSvgAsPng(svg, 'visual-diagram.png')} className="visual-download"><Download className="h-3.5 w-3.5" />PNG</button>
      </div>
      <div className="mt-4 space-y-2 text-sm">{data.nodes.map((node) => <div key={node.id} className="flex justify-between gap-3 rounded-lg border border-slate-200 p-2 dark:border-slate-700"><span>{node.label}</span><Source value={node.source} /></div>)}</div>
    </>
  );
}

function ExplainerView({ data }: { data: VisualExplainer }) {
  return <div className="space-y-4">
    <div className="flex justify-end"><button type="button" onClick={() => downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), 'visual-explanation.json')} className="visual-download"><Download className="h-3.5 w-3.5" />Download cards</button></div>
    <div><h3 className="text-lg font-semibold">{data.title}</h3><p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{data.summary.text}</p><Source value={data.summary.source} /></div>
    <div className="grid gap-2 sm:grid-cols-2">{data.key_points.map((point, index) => <article key={`${point.text}-${index}`} className="rounded-xl border border-slate-200 p-3 dark:border-slate-700"><p className="text-sm">{point.text}</p><Source value={point.source} /></article>)}</div>
    {data.steps.length > 0 && <ol className="space-y-2">{data.steps.map((step, index) => <li key={`${step.text}-${index}`} className="flex gap-3 rounded-xl bg-indigo-50 p-3 text-sm dark:bg-indigo-500/10"><span className="font-semibold text-indigo-600">{index + 1}</span><span>{step.text}<br /><Source value={step.source} /></span></li>)}</ol>}
    {data.key_numbers.length > 0 && <div className="grid grid-cols-2 gap-2">{data.key_numbers.map((number) => <div key={`${number.label}-${number.value}`} className="rounded-xl border border-slate-200 p-3 dark:border-slate-700"><strong className="block text-xl">{number.value}</strong><span className="text-xs">{number.label}</span><br /><Source value={number.source} /></div>)}</div>}
    {data.glossary.length > 0 && <dl className="space-y-2">{data.glossary.map((entry) => <div key={entry.term} className="rounded-xl border border-slate-200 p-3 dark:border-slate-700"><dt className="font-semibold">{entry.term}</dt><dd className="text-sm text-slate-600 dark:text-slate-300">{entry.definition}<br /><Source value={entry.source} /></dd></div>)}</dl>}
  </div>;
}

function ChartView({ data }: { data: VisualChart }) {
  const values = data.labels.map((label, index) => ({ label, value: data.values[index] }));
  const chart = data.type === 'line' ? <LineChart data={values}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="label" /><YAxis /><Tooltip /><Line type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} /></LineChart> : data.type === 'pie' ? <PieChart><Pie data={values} dataKey="value" nameKey="label" outerRadius="75%" label>{values.map((entry, index) => <Cell key={entry.label} fill={colors[index % colors.length]} />)}</Pie><Tooltip /></PieChart> : <BarChart data={values}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="label" /><YAxis /><Tooltip /><Bar dataKey="value" fill="#6366f1" /></BarChart>;
  return <><h3 className="mb-1 text-lg font-semibold">{data.title}</h3><div className="h-72 w-full"><ResponsiveContainer width="100%" height="100%">{chart}</ResponsiveContainer></div><Source value={data.source} /><div className="mt-3 flex gap-2"><button type="button" onClick={() => downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), 'visual-chart.json')} className="visual-download"><Download className="h-3.5 w-3.5" />Download data</button></div></>;
}

export default function VisualPanel({ kind, data, loading, error, onClose, onRetry }: Props) {
  return <ModalShell title="Visual studio" label="Grounded visual answer" onClose={onClose}>
      <nav className="mb-4 flex gap-1 overflow-x-auto rounded-xl bg-slate-100 p-1 dark:bg-slate-800" aria-label="Visual studio tabs">
        {(['diagram', 'explainer', 'chart'] as const).map((tab) => <span key={tab} className={`rounded-lg px-3 py-2 text-xs font-semibold ${tab === kind ? 'bg-white text-indigo-700 shadow-sm dark:bg-slate-700 dark:text-indigo-200' : 'text-slate-500 dark:text-slate-400'}`}>{tab === 'explainer' ? 'Explain' : tab[0].toUpperCase() + tab.slice(1)}</span>)}
      </nav>
      {loading && <div className="flex min-h-64 items-center justify-center gap-2 text-sm text-slate-500"><Loader2 className="h-5 w-5 animate-spin" />Preparing a grounded visual…</div>}
      {!loading && error && <div className="flex min-h-48 flex-col items-center justify-center gap-3 text-center"><p className="text-sm text-rose-600 dark:text-rose-300">{error}</p><button type="button" onClick={onRetry} className="visual-download"><RefreshCw className="h-3.5 w-3.5" />Retry</button></div>}
      {!loading && !error && data && (kind === 'diagram' ? <DiagramView data={data as VisualDiagram} /> : kind === 'explainer' ? <ExplainerView data={data as VisualExplainer} /> : <ChartView data={data as VisualChart} />)}
  </ModalShell>;
}
