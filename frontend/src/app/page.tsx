'use client';

import { useState, useRef, useEffect } from 'react';
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
  Layers
} from 'lucide-react';

interface Message {
  sender: 'user' | 'agent';
  text: string;
  sources?: SourceMetadata[];
  isEditing?: boolean;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ type: 'success' | 'error' | 'info'; message: string; details?: string } | null>(null);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editText, setEditText] = useState('');

  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'agent',
      text: 'Hello! Upload a PDF to automatically ingest text, diagrams, and figures. Ask me anything about the document!',
    },
  ]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      await triggerAutoIngest(selectedFile);
    }
  };

  const triggerAutoIngest = async (fileToIngest: File) => {
    if (!fileToIngest.name.endsWith('.pdf')) {
      setUploadStatus({
        type: 'error',
        message: 'Only PDF documents are supported.',
      });
      return;
    }

    setUploading(true);
    setUploadStatus({
      type: 'info',
      message: `Ingesting ${fileToIngest.name}...`,
      details: 'Extracting text, analyzing charts & diagrams with Gemini Vision...',
    });

    try {
      const res = await ingestDocument(fileToIngest);

      let attempts = 0;
      const maxAttempts = 180; // Extended polling duration to 3 minutes
      const pollInterval = 1000;

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
              setTimeout(poll, pollInterval);
            } else {
              setUploadStatus({
                type: 'info',
                message: 'Processing is taking longer than expected.',
                details: 'Vector database is indexing in the background.',
              });
              setUploading(false);
            }
          }
        } catch (pollErr) {
          console.error('Polling error:', pollErr);
          setUploading(false);
        }
      };

      setTimeout(poll, 1000);

    } catch (err: any) {
      setUploadStatus({
        type: 'error',
        message: 'Ingestion error',
        details: err.message || 'Failed to communicate with backend.',
      });
      setUploading(false);
    }
  };

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
        {
          sender: 'agent',
          text: res.answer || 'No answer returned.',
          sources: res.sources || [],
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          sender: 'agent',
          text: `Error processing your request: ${err.message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

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
        {
          sender: 'agent',
          text: res.answer || 'No answer returned.',
          sources: res.sources || [],
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          sender: 'agent',
          text: `Error regenerating answer: ${err.message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

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

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans antialiased overflow-hidden">
      <aside className="w-84 border-r border-slate-800/80 bg-slate-900/60 backdrop-blur-xl p-6 flex flex-col justify-between shrink-0 shadow-2xl">
        <div className="space-y-6">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-tr from-indigo-600 to-violet-500 text-white rounded-xl shadow-lg shadow-indigo-500/20">
              <Bot className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <h1 className="font-bold text-slate-100 text-base tracking-tight">DocAgent RAG</h1>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  v2.0
                </span>
              </div>
              <p className="text-xs text-slate-400 flex items-center gap-1 mt-0.5">
                <Sparkles className="w-3 h-3 text-amber-400" /> Multi-Modal Gemini + Chroma
              </p>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Document Ingestion
              </label>
              <span className="text-[11px] text-indigo-400 font-medium">Auto-Ingest</span>
            </div>

            <div
              onClick={() => fileInputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-2xl p-5 text-center transition duration-200 cursor-pointer ${
                uploading
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
                    {uploading ? 'Auto-processing document...' : 'Ingestion starts instantly'}
                  </div>
                </div>
              </div>
            </div>

            {uploadStatus && (
              <div
                className={`p-3.5 rounded-xl text-xs border transition-all duration-200 animate-fadeIn ${
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
                  <div className="space-y-0.5">
                    <div className="font-medium text-slate-200">{uploadStatus.message}</div>
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

          <div className="p-3.5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-2 text-xs text-slate-400">
            <div className="font-semibold text-slate-300 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-indigo-400" /> Multi-Modal Features
            </div>
            <ul className="space-y-1.5 text-[11px]">
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                Text chunking & vector search
              </li>
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                Gemini Vision diagram analysis
              </li>
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-violet-400" />
                Inline query editing & regeneration
              </li>
            </ul>
          </div>
        </div>

        <div className="text-xs text-slate-500 border-t border-slate-800/80 pt-4 flex items-center justify-between">
          <span>Backend Pipeline:</span>
          <span className="inline-flex items-center gap-1.5 text-emerald-400 font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            FastAPI Active
          </span>
        </div>
      </aside>

      <main className="flex-1 flex flex-col justify-between bg-slate-950 overflow-hidden relative">
        <div className="flex-1 overflow-y-auto p-6 lg:p-8 space-y-6">
          {messages.map((msg, index) => {
            const isUser = msg.sender === 'user';
            const isEditing = editingIndex === index;
            const dedupedSources = getDeduplicatedSources(msg.sources);

            return (
              <div
                key={index}
                className={`group flex items-start gap-4 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
              >
                <div
                  className={`p-2.5 rounded-xl shrink-0 shadow-md ${
                    isUser
                      ? 'bg-gradient-to-tr from-indigo-600 to-indigo-500 text-white'
                      : 'bg-slate-900 text-indigo-400 border border-slate-800'
                  }`}
                >
                  {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                </div>

                <div className={`max-w-3xl flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
                  {isUser && isEditing ? (
                    <div className="w-full min-w-[320px] max-w-xl bg-slate-900 border border-indigo-500/50 rounded-2xl p-3 shadow-xl space-y-3">
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
                    <div
                      className={`relative rounded-2xl p-4 text-sm leading-relaxed shadow-lg ${
                        isUser
                          ? 'bg-indigo-600 text-white rounded-tr-none'
                          : 'bg-slate-900/90 border border-slate-800 text-slate-200 rounded-tl-none backdrop-blur-sm'
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{msg.text}</p>

                      {!isUser && dedupedSources.length > 0 && (
                        <div className="mt-3.5 pt-3 border-t border-slate-800/80 text-xs">
                          <div className="font-semibold text-slate-300 mb-2 flex items-center gap-1.5">
                            <span>Retrieved Sources:</span>
                            <span className="text-[10px] text-slate-400 font-normal">
                              ({dedupedSources.length} distinct {dedupedSources.length === 1 ? 'citation' : 'citations'})
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
                                  className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border text-xs font-mono transition shadow-sm ${
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
                                  <span className="truncate max-w-[180px]">{sourceName}</span>
                                  {pageText && (
                                    <span className="opacity-75">({pageText})</span>
                                  )}
                                </span>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

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

          {loading && (
            <div className="flex items-center gap-3 text-slate-400 text-sm animate-pulse">
              <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800 text-indigo-400">
                <Bot className="w-5 h-5" />
              </div>
              <div className="bg-slate-900/80 border border-slate-800 px-4 py-3 rounded-2xl flex items-center gap-2.5 text-xs text-slate-300">
                <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
                <span>Retrieving multi-modal document chunks & generating answer with Gemini...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 lg:p-6 border-t border-slate-800/80 bg-slate-900/40 backdrop-blur-xl">
          <form onSubmit={handleSendQuery} className="flex gap-3 max-w-4xl mx-auto">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about document text, tables, or diagrams..."
              className="flex-1 bg-slate-900 border border-slate-800 rounded-2xl px-5 py-3.5 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition placeholder:text-slate-500 shadow-inner"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="bg-gradient-to-tr from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-600 text-white px-6 rounded-2xl transition duration-200 flex items-center justify-center shadow-lg shadow-indigo-600/20 font-medium text-sm"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}