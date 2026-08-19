'use client';

import { useState } from 'react';
import { uploadDocument, queryDocument } from '@/lib/api';
import { Upload, Send, FileText, Bot, User, Loader2, CheckCircle2 } from 'lucide-react';

interface Message {
  sender: 'user' | 'agent';
  text: string;
  sources?: string[];
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'agent',
      text: 'Hello! Upload a PDF document using the sidebar, then ask me any question about its content.',
    },
  ]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadStatus(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadStatus(null);

    try {
      const res = await uploadDocument(file);
      setUploadStatus(`Successfully processed ${res.filename} (${res.chunks} chunks indexed)`);
    } catch (err: any) {
      setUploadStatus(`Error: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleSendQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userText = query;
    setQuery('');
    setMessages((prev) => [...prev, { sender: 'user', text: userText }]);
    setLoading(true);

    try {
      const res = await queryDocument(userText);
      setMessages((prev) => [
        ...prev,
        {
          sender: 'agent',
          text: res.answer || res.response || 'No answer returned.',
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

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans">
      {/* Sidebar - Document Ingestion */}
      <aside className="w-80 border-r border-slate-800 bg-slate-900/50 p-6 flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2 bg-indigo-600/20 text-indigo-400 rounded-lg">
              <Bot className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-semibold text-slate-100">DocAgent RAG</h1>
              <p className="text-xs text-slate-400">Gemini 1.5 + ChromaDB</p>
            </div>
          </div>

          <div className="space-y-4">
            <label className="block text-xs font-medium uppercase tracking-wider text-slate-400">
              Document Upload
            </label>
            <div className="border-2 border-dashed border-slate-700 hover:border-indigo-500/50 rounded-xl p-4 text-center transition cursor-pointer bg-slate-800/30">
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="hidden"
                id="file-input"
              />
              <label htmlFor="file-input" className="cursor-pointer flex flex-col items-center gap-2">
                <FileText className="w-8 h-8 text-indigo-400" />
                <span className="text-sm font-medium text-slate-300">
                  {file ? file.name : 'Select PDF Document'}
                </span>
                <span className="text-xs text-slate-500">PDF up to 20MB</span>
              </label>
            </div>

            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-medium text-sm rounded-lg transition flex items-center justify-center gap-2"
            >
              {uploading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Ingesting...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" /> Ingest PDF
                </>
              )}
            </button>

            {uploadStatus && (
              <div
                className={`p-3 rounded-lg text-xs flex items-start gap-2 ${
                  uploadStatus.startsWith('Error')
                    ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                    : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                }`}
              >
                <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{uploadStatus}</span>
              </div>
            )}
          </div>
        </div>

        <div className="text-xs text-slate-500 border-t border-slate-800 pt-4">
          Status: <span className="text-emerald-400 font-medium">FastAPI Backend Active</span>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col justify-between bg-slate-950">
        {/* Messages Scroll Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex items-start gap-4 ${
                msg.sender === 'user' ? 'flex-row-reverse' : ''
              }`}
            >
              <div
                className={`p-2 rounded-lg shrink-0 ${
                  msg.sender === 'user'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-800 text-indigo-400 border border-slate-700'
                }`}
              >
                {msg.sender === 'user' ? (
                  <User className="w-5 h-5" />
                ) : (
                  <Bot className="w-5 h-5" />
                )}
              </div>

              <div
                className={`max-w-2xl rounded-2xl p-4 text-sm leading-relaxed ${
                  msg.sender === 'user'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-900 border border-slate-800 text-slate-200'
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.text}</p>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-800 text-xs text-slate-400">
                    <span className="font-semibold text-slate-300">Sources:</span>{' '}
                    {msg.sources.join(', ')}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex items-center gap-3 text-slate-400 text-sm">
              <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
              <span>Analyzing document context with Gemini...</span>
            </div>
          )}
        </div>

        {/* Query Input Form */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/40">
          <form onSubmit={handleSendQuery} className="flex gap-3 max-w-4xl mx-auto">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your uploaded document..."
              className="flex-1 bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 transition placeholder:text-slate-500"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 text-white px-5 rounded-xl transition flex items-center justify-center"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}