'use client';

import React, { useState, useEffect, useRef } from 'react';

// Helper functions for Session and Chat management
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

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface UploadedFile {
  id: string;
  name: string;
  extension: string;
  isUploading: boolean;
}

interface Message {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  sourcesCount?: number;
}

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string>('');
  const [chatId, setChatId] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<UploadedFile[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setSessionId(getSessionId());
    setChatId(createNewChatId());
  }, []);

  useEffect(() => {
    if (!sessionId || !chatId) return;
    const handleBeforeUnload = () => {
      navigator.sendBeacon(
        `${API_BASE_URL}/api/session/clear?x_session_id=${sessionId}&x_chat_id=${chatId}`
      );
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [sessionId, chatId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleNewChat = async () => {
    if (sessionId && chatId) {
      try {
        await fetch(`${API_BASE_URL}/api/session/clear`, {
          method: 'DELETE',
          headers: { 'X-Session-ID': sessionId, 'X-Chat-ID': chatId },
        });
      } catch (err) {
        console.error('Failed to wipe old chat context:', err);
      }
    }
    setChatId(createNewChatId());
    setMessages([]);
    setAttachedFiles([]);
    setIsSidebarOpen(false);
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const fileId = crypto.randomUUID();
    const ext = file.name.split('.').pop()?.toUpperCase() || 'FILE';

    const newFileBadge: UploadedFile = {
      id: fileId,
      name: file.name,
      extension: ext,
      isUploading: true,
    };

    setAttachedFiles((prev) => [...prev, newFileBadge]);

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
        alert(`Ingestion failed: ${data.detail || 'Unknown error'}`);
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
    if (!inputQuery.trim() || isQuerying) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      sender: 'user',
      text: inputQuery,
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
            text: data.answer,
            sourcesCount: data.sources_count,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            sender: 'agent',
            text: `⚠️ ${data.detail || 'An error occurred.'}`,
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          sender: 'agent',
          text: '⚠️ Unable to reach backend service.',
        },
      ]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="flex h-[100dvh] bg-[#18181b] text-gray-100 font-sans overflow-hidden">
      {/* Mobile Overlay */}
      {isSidebarOpen && (
        <div
          onClick={() => setIsSidebarOpen(false)}
          className="fixed inset-0 bg-black/60 z-40 sm:hidden backdrop-blur-sm"
        />
      )}

      {/* Sidebar Drawer */}
      <aside
        className={`fixed sm:static inset-y-0 left-0 z-50 w-72 bg-[#0f0f11] border-r border-gray-800 p-4 flex flex-col justify-between transform transition-transform duration-200 ease-in-out ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full sm:translate-x-0'
        }`}
      >
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h1 className="text-lg font-bold text-indigo-400">DocAgent RAG</h1>
            <button
              onClick={() => setIsSidebarOpen(false)}
              className="sm:hidden text-gray-400 p-1"
            >
              ✕
            </button>
          </div>

          <button
            onClick={handleNewChat}
            className="w-full bg-[#27272a] hover:bg-[#3f3f46] text-white py-2.5 px-4 rounded-xl text-sm font-medium flex items-center justify-center space-x-2 transition border border-gray-700/50"
          >
            <span>+</span>
            <span>New Chat</span>
          </button>
        </div>

        <div className="text-[11px] text-gray-500 bg-[#18181b] p-3 rounded-xl border border-gray-800 space-y-1">
          <p className="truncate">Session: {sessionId || 'Loading...'}</p>
          <p className="truncate">Chat: {chatId || 'Loading...'}</p>
        </div>
      </aside>

      {/* Main Container */}
      <main className="flex-1 flex flex-col relative min-w-0">
        {/* Mobile Header */}
        <header className="sm:hidden flex items-center justify-between p-4 border-b border-gray-800 bg-[#0f0f11]">
          <button
            onClick={() => setIsSidebarOpen(true)}
            className="text-gray-300 p-2 rounded-lg bg-gray-800"
          >
            ☰
          </button>
          <span className="font-semibold text-sm text-indigo-400">DocAgent</span>
          <button
            onClick={handleNewChat}
            className="text-xs bg-indigo-600 px-3 py-1.5 rounded-lg text-white"
          >
            + New
          </button>
        </header>

        {/* Messages View */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
              <div className="w-14 h-14 bg-indigo-600/20 text-indigo-400 rounded-2xl flex items-center justify-center text-2xl font-bold">
                📄
              </div>
              <h2 className="text-xl sm:text-2xl font-semibold text-gray-200">
                What can I help with today?
              </h2>
              <p className="text-xs sm:text-sm text-gray-400 max-w-sm">
                Attach a PDF, PNG, or TXT file using the + button below to start asking questions.
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
                  className={`max-w-[85%] sm:max-w-xl rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    msg.sender === 'user'
                      ? 'bg-indigo-600 text-white rounded-br-sm'
                      : 'bg-[#27272a] text-gray-200 border border-gray-700/60 rounded-bl-sm'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                  {msg.sourcesCount !== undefined && (
                    <span className="block text-[10px] text-gray-400 mt-2 border-t border-gray-700/60 pt-1">
                      Retrieved from {msg.sourcesCount} document chunk(s)
                    </span>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar (Claude Mobile Design) */}
        <footer className="p-3 sm:p-4 bg-[#0f0f11] border-t border-gray-800/80">
          <div className="max-w-3xl mx-auto">
            {/* Attached Files Badges Container */}
            {attachedFiles.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-2 px-1">
                {attachedFiles.map((f) => (
                  <div
                    key={f.id}
                    className="flex items-center space-x-2 bg-[#27272a] border border-gray-700/80 rounded-xl px-3 py-1.5 text-xs shadow-sm"
                  >
                    {f.isUploading ? (
                      <span className="w-3.5 h-3.5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <span className="text-indigo-400 text-xs">📄</span>
                    )}
                    <span className="max-w-[120px] truncate text-gray-200 font-medium">
                      {f.name}
                    </span>
                    <button
                      type="button"
                      onClick={() => removeFile(f.id)}
                      className="ml-1 p-1 hover:bg-gray-700 rounded-full text-gray-400 hover:text-red-400 transition"
                      aria-label="Remove document"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Input Form Pill */}
            <form
              onSubmit={handleSendQuery}
              className="flex items-center bg-[#18181b] border border-gray-700/80 rounded-2xl px-3 py-2 focus-within:border-indigo-500 transition shadow-inner"
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
                className="p-2 text-gray-400 hover:text-indigo-400 hover:bg-gray-800 rounded-xl transition"
                title="Attach Document"
              >
                📎
              </button>

              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                placeholder="Ask about text, tables, or diagrams..."
                className="flex-1 bg-transparent text-gray-100 placeholder-gray-500 px-3 py-1.5 text-sm focus:outline-none"
              />

              <button
                type="submit"
                disabled={!inputQuery.trim() || isQuerying}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-600 text-white p-2.5 rounded-xl transition flex items-center justify-center min-w-[38px]"
              >
                {isQuerying ? '...' : '➔'}
              </button>
            </form>
          </div>
        </footer>
      </main>
    </div>
  );
}