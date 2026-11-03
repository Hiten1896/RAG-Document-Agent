'use client';

import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

interface ModalShellProps {
  title: string;
  label?: string;
  onClose: () => void;
  children: React.ReactNode;
  className?: string;
}

export default function ModalShell({ title, label, onClose, children, className = '' }: ModalShellProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.removeEventListener('keydown', onKeyDown); document.body.style.overflow = previousOverflow; };
  }, [onClose]);
  return <div className="fixed inset-0 z-[80] flex items-end justify-center bg-slate-950/60 p-0 sm:items-center sm:p-5" role="dialog" aria-modal="true" aria-labelledby="modal-shell-title" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className={`flex max-h-[90vh] w-full max-w-5xl flex-col overflow-hidden rounded-t-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-900 sm:rounded-2xl ${className}`}>
      <header className="flex shrink-0 items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800 sm:px-6"><div>{label && <p className="text-[10px] font-semibold uppercase tracking-widest text-indigo-600 dark:text-indigo-400">{label}</p>}<h2 id="modal-shell-title" className="text-base font-semibold text-slate-900 dark:text-slate-100">{title}</h2></div><button ref={closeRef} type="button" onClick={onClose} aria-label={`Close ${title}`} className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"><X className="h-4 w-4" /></button></header>
      <div className="min-h-0 flex-1 overflow-y-auto p-4 sm:p-6">{children}</div>
    </section>
  </div>;
}
