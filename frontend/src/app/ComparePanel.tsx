'use client';

import { useState } from 'react';
import { AlertCircle, Check, Loader2, X } from 'lucide-react';
import type { CompareResponse } from '@/lib/api';

interface ComparePanelProps {
  files: string[];
  data: CompareResponse | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onCompare: (fileA: string, fileB: string, focus: string) => void;
}

export default function ComparePanel({
  files,
  data,
  loading,
  error,
  onClose,
  onCompare,
}: ComparePanelProps) {
  const [selectedFiles, setSelectedFiles] = useState(() => files.slice(0, 2));
  const [focus, setFocus] = useState('');
  const hasTwoFiles = files.length >= 2;

  const toggleFile = (fileName: string) => {
    setSelectedFiles((previous) => {
      if (previous.includes(fileName)) {
        return previous.filter((file) => file !== fileName);
      }
      if (previous.length < 2) return [...previous, fileName];
      return [previous[1], fileName];
    });
  };

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center bg-slate-950/60 p-3 sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="compare-panel-title"
    >
      <section className="flex max-h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-900">
        <header className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800 sm:px-6">
          <div>
            <h2 id="compare-panel-title" className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Compare documents
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Compare evidence retrieved separately from each selected file.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close document comparison"
            className="rounded-lg p-2 text-slate-500 transition hover:bg-slate-100 hover:text-slate-800 dark:hover:bg-slate-800 dark:hover:text-slate-100"
          >
            <X className="h-4 w-4" />
          </button>
        </header>

        <div className="flex-1 space-y-4 overflow-y-auto p-4 sm:p-6">
          {hasTwoFiles ? (
            <>
              <div>
                <p className="mb-2 text-xs font-semibold text-slate-700 dark:text-slate-300">
                  Select two documents ({selectedFiles.length}/2)
                </p>
                <div className="flex flex-wrap gap-2">
                  {files.map((fileName) => {
                    const selected = selectedFiles.includes(fileName);
                    return (
                      <button
                        key={fileName}
                        type="button"
                        aria-pressed={selected}
                        onClick={() => toggleFile(fileName)}
                        className={`inline-flex min-h-9 max-w-full items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition ${
                          selected
                            ? 'border-indigo-400 bg-indigo-50 text-indigo-800 dark:border-indigo-500 dark:bg-indigo-950/50 dark:text-indigo-200'
                            : 'border-slate-300 bg-white text-slate-700 hover:border-indigo-300 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-300'
                        }`}
                      >
                        {selected && <Check className="h-3.5 w-3.5 shrink-0" />}
                        <span className="truncate">{fileName}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <label className="block">
                <span className="mb-1.5 block text-xs font-semibold text-slate-700 dark:text-slate-300">
                  Optional comparison focus
                </span>
                <input
                  type="text"
                  value={focus}
                  onChange={(event) => setFocus(event.target.value)}
                  maxLength={500}
                  placeholder="For example: compare the proposed budgets"
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none placeholder:text-slate-400 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-200 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:focus:border-indigo-500 dark:focus:ring-indigo-950"
                />
              </label>

              <button
                type="button"
                onClick={() => {
                  if (selectedFiles.length === 2) {
                    onCompare(selectedFiles[0], selectedFiles[1], focus);
                  }
                }}
                disabled={loading || selectedFiles.length !== 2}
                className="inline-flex min-h-10 items-center gap-2 rounded-lg bg-indigo-600 px-4 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                {loading ? 'Comparing…' : 'Compare selected files'}
              </button>
            </>
          ) : (
            <div
              role="status"
              className="rounded-xl border border-dashed border-amber-300 bg-amber-50 px-5 py-8 text-center text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-200"
            >
              Upload at least two completed documents in this chat to compare them.
            </div>
          )}

          {error && (
            <div
              role="alert"
              className="flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-900/70 dark:bg-rose-950/30 dark:text-rose-300"
            >
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {loading && (
            <div className="flex min-h-28 items-center justify-center gap-2 text-sm text-slate-500 dark:text-slate-400">
              <Loader2 className="h-5 w-5 animate-spin text-indigo-500" />
              Retrieving evidence from both documents…
            </div>
          )}

          {!loading && data && (
            <div>
              <h3 className="mb-2 text-sm font-semibold text-slate-800 dark:text-slate-100">
                Agreements and differences
              </h3>
              <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700">
                <table className="w-full min-w-[760px] border-collapse text-left text-sm">
                  <thead className="bg-slate-50 text-xs font-semibold text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                    <tr>
                      <th scope="col" className="w-36 px-3 py-2.5">Type / topic</th>
                      <th scope="col" className="px-3 py-2.5">{data.files[0]}</th>
                      <th scope="col" className="px-3 py-2.5">{data.files[1]}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.comparisons.map((row, index) => (
                      <tr
                        key={`${row.topic}-${index}`}
                        className="border-t border-slate-200 align-top dark:border-slate-700"
                      >
                        <th scope="row" className="px-3 py-3 font-medium text-slate-800 dark:text-slate-100">
                          <span
                            className={`mb-1.5 inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold capitalize ${
                              row.kind === 'agreement'
                                ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
                                : 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300'
                            }`}
                          >
                            {row.kind}
                          </span>
                          <span className="block">{row.topic}</span>
                        </th>
                        <td className="px-3 py-3 text-slate-700 dark:text-slate-300">
                          <p className="whitespace-pre-wrap">{row.document_a}</p>
                          <p className="mt-2 text-xs text-indigo-700 dark:text-indigo-300">
                            {row.source_a}
                          </p>
                        </td>
                        <td className="px-3 py-3 text-slate-700 dark:text-slate-300">
                          <p className="whitespace-pre-wrap">{row.document_b}</p>
                          <p className="mt-2 text-xs text-indigo-700 dark:text-indigo-300">
                            {row.source_b}
                          </p>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
