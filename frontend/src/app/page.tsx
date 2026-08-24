'use client';

import React, { useState, useEffect, useRef } from 'react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

function getSessionId(): string {
  if (typeof window === 'undefined') return 'server_side_session';
  let sessionId = sessionStorage.getItem('rag_session_id');
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    sessionStorage.setItem('rag_session_id', sessionId);
  }
  return sessionId;
}

function createNewChatId(): string {
  return crypto.randomUUID();
}

interface UploadedFile {
  id: string;
  name: string;
  isUploading: boolean;
  fileObj?: File;
}

interface Message {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  files?: string[];
  sourcesCount?: number;
}

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string>('');
  const [chatId, setChatId] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<UploadedFile[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);
  const [activePdfUrl, setActivePdfUrl] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setSessionId(getSessionId());
    setChatId(createNewChatId());
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const hasUploadingFile = attachedFiles.some((f) => f.isUploading);

  const handleNewChat = async () => {
    if (sessionId && chatId) {
      try {
        await fetch(`${API_BASE_URL}/api/session/clear`, {
          method: 'DELETE',
          headers: { 'X-Session-ID': sessionId, 'X-Chat-ID': chatId },
        });
      } catch (err) {
        console.error('Failed to wipe session:', err);
      }
    }
    setChatId(createNewChatId());
    setMessages([]);
    setAttachedFiles([]);
    setActivePdfUrl(null);
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const fileId = crypto.randomUUID();
    const newFileBadge: UploadedFile = {
      id: fileId,
      name: file.name,
      isUploading: true,
      fileObj: file,
    };

    setAttachedFiles((prev) => [...prev, newFileBadge]);

    if (file.type === 'application/pdf') {
      const pdfBlobUrl = URL.createObjectURL(file);
      setActivePdfUrl(pdfBlobUrl);
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/ingest`, {
        method: 'POST',
        headers: {
          'X-Session-ID': sessionId,
          'X-Chat-ID': chatId,
        },
        body: formData,
      });

      if (response.ok) {
        setAttachedFiles((prev) =>
          prev.map((f) => (f.id === fileId ? { ...f, isUploading: false } : f))
        );
      } else {
        const data = await response.json();
        alert(`Ingestion failed: ${data.detail || 'Error uploading file'}`);
        removeFile(fileId);
      }
    } catch (err) {
      alert('Network error while processing file.');
      removeFile(fileId);
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const removeFile = (fileId: string) => {
    setAttachedFiles((prev) => prev.filter((f) => f.id !== fileId));
  };

  const handleSendQuery = async (e: React.FormEvent) => {
    e.preventDefault();

    // Lock query submission if any document is currently ingesting
    if (hasUploadingFile) {
      alert('Please wait for document ingestion to finish or remove loading files before sending.');
      return;
    }

    if (!inputQuery.trim() || isQuerying) return;

    const currentFileNames = attachedFiles.map((f) => f.name);
    const userMsg: Message = {
      id: crypto.randomUUID(),
      sender: 'user',
      text: inputQuery,
      files: currentFileNames.length > 0 ? currentFileNames : undefined,
    };

    setMessages((prev) => [...prev, userMsg]);
    const query = inputQuery;
    setInputQuery('');
    setIsQuerying(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': sessionId,
          'X-Chat-ID': chatId,
        },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            sender: 'agent',
            text: data.answer || data.response || 'No answer returned.',
            sourcesCount: data.sources_count,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            sender: 'agent',
            text: `Error processing your request: ${data.detail || 'Not Found'}`,
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          sender: 'agent',
          text: 'Error processing your request: Network Connection Failed',
        },
      ]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#f3f4f6] text-gray-800 font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 p-4 flex flex-col justify-between shrink-0 select-none">
        <div className="space-y-6">
          <div className="flex items-center space-x-3 px-2">
            <div className="w-8 h-8 bg-indigo-600 rounded-xl flex items-center justify-center text-white font-bold text-lg shadow-md">
              📄
            </div>
            <span className="font-bold text-gray-800 text-base">DocAgent RAG</span>
          </div>

          <button
            onClick={handleNewChat}
            className="w-full bg-indigo-50/80 hover:bg-indigo-100 text-indigo-600 border border-indigo-200/60 py-2.5 px-4 rounded-xl text-sm font-medium flex items-center justify-center space-x-2 transition"
          >
            <span>+</span>
            <span>New Chat</span>
          </button>

          <div>
            <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider px-2 mb-2">
              Chat History
            </p>
            <div className="bg-indigo-50/50 text-indigo-700 text-xs py-2 px-3 rounded-lg font-medium truncate">
              {messages.length > 0 ? messages[0].text : 'New Chat'}
            </div>
          </div>
        </div>

        <div className="border-t border-gray-100 pt-3 px-1 text-[11px] text-gray-400 flex items-center justify-between">
          <span>Backend Pipeline:</span>
          <span className="flex items-center space-x-1.5 text-emerald-600 font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>FastAPI Active</span>
          </span>
        </div>
      </aside>

      {/* Main Area */}
      <main className="flex-1 flex flex-col relative bg-[#f8fafc] overflow-hidden">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto space-y-3">
              <div className="w-16 h-16 bg-indigo-600/10 text-indigo-600 rounded-3xl flex items-center justify-center text-3xl mb-2">
                📄
              </div>
              <h2 className="text-2xl font-bold text-gray-800">Late one, huh?</h2>
              <p className="text-xs text-gray-500">
                Drop in a document, or ask about one you’ve already shared.
              </p>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${
                  msg.sender === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-xl rounded-2xl px-5 py-4 text-sm leading-relaxed shadow-sm ${
                    msg.sender === 'user'
                      ? 'bg-indigo-600 text-white rounded-tr-none'
                      : 'bg-white text-gray-800 border border-gray-100 rounded-tl-none'
                  }`}
                >
                  {msg.files && msg.files.length > 0 && (
                    <div className="flex flex-wrap gap-2 mb-2">
                      {msg.files.map((fname, idx) => (
                        <div
                          key={idx}
                          className="bg-indigo-500/30 text-white text-xs px-2.5 py-1 rounded-lg flex items-center space-x-1.5"
                        >
                          <span>📄</span>
                          <span className="font-medium">{fname}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* File Cards Attachment Area above Input Pill */}
        {attachedFiles.length > 0 && (
          <div className="flex flex-wrap justify-center gap-3 px-4 mb-2">
            {attachedFiles.map((file) => (
              <div
                key={file.id}
                className="relative bg-white border border-gray-200 shadow-sm rounded-xl px-4 py-2.5 flex items-center space-x-3 text-xs"
              >
                {file.isUploading ? (
                  <div className="w-4 h-4 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                ) : (
                  <span className="text-indigo-600">📄</span>
                )}
                <div className="flex flex-col max-w-[140px]">
                  <span className="font-semibold text-gray-700 truncate">
                    {file.name}
                  </span>
                  <span className="text-[10px] text-gray-400 uppercase">
                    {file.isUploading ? 'Ingesting...' : 'Ready'}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => removeFile(file.id)}
                  className="w-5 h-5 bg-gray-600 text-white rounded-full flex items-center justify-center text-[10px] font-bold hover:bg-red-500 transition ml-2"
                  title="Remove file"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Input Bar Pill */}
        <footer className="p-4 flex flex-col items-center">
          <form
            onSubmit={handleSendQuery}
            className="w-full max-w-2xl bg-white border border-gray-200/80 rounded-full px-4 py-2 flex items-center shadow-lg shadow-gray-200/50 focus-within:border-indigo-400 transition"
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".pdf,.png,.jpg,.jpeg,.txt,.md"
              className="hidden"
            />

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="w-8 h-8 rounded-full hover:bg-gray-100 flex items-center justify-center text-gray-500 text-base transition"
              title="Add file"
            >
              +
            </button>

            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask about text, tables, or diagrams..."
              className="flex-1 bg-transparent px-3 text-sm text-gray-800 placeholder-gray-400 focus:outline-none"
            />

            <button
              type="submit"
              disabled={!inputQuery.trim() || isQuerying || hasUploadingFile}
              className="w-10 h-10 bg-indigo-100 hover:bg-indigo-600 text-indigo-600 hover:text-white disabled:bg-gray-100 disabled:text-gray-300 rounded-full flex items-center justify-center transition"
            >
              ➔
            </button>
          </form>
          <span className="text-[10px] text-gray-400 mt-2">Made by HS</span>
        </footer>
      </main>

      {/* PDF Viewer Panel (Right Side) */}
      {activePdfUrl && (
        <section className="w-[450px] bg-white border-l border-gray-200 flex flex-col h-full shrink-0">
          <div className="p-3 border-b border-gray-100 flex items-center justify-between text-xs font-semibold text-gray-600">
            <span>Document Preview</span>
            <button
              onClick={() => setActivePdfUrl(null)}
              className="text-gray-400 hover:text-gray-600 text-sm p-1"
            >
              ✕
            </button>
          </div>
          <iframe
            src={activePdfUrl}
            className="w-full flex-1 border-0"
            title="PDF Preview"
          />
        </section>
      )}
    </div>
  );
}