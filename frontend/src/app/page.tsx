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

// Time-of-day based greetings (4 time buckets, 5 lines each)
const GREETINGS = {
  morning: [
    "Good morning! Ready to analyze your docs?",
    "Rise and shine! Let's dive into your papers.",
    "Morning! Drop your files and ask away.",
    "Early start! How can I assist you today?",
    "Good morning! Let's get through these documents."
  ],
  afternoon: [
    "Good afternoon! What are we reviewing today?",
    "Hope your day is going well! Ready for queries.",
    "Afternoon! Let's process some documents.",
    "Good afternoon! Upload a file or ask a question.",
    "Ready when you are. What's on your agenda?"
  ],
  evening: [
    "Good evening! Let's finish up today's research.",
    "Evening! Ready to explore your documents.",
    "Winding down or starting up? I'm ready to help.",
    "Good evening! Drop your PDFs here.",
    "Evening! Ask me anything about your files."
  ],
  night: [
    "Burning the midnight oil.",
    "Late one, huh? Let's analyze your docs.",
    "Working late? I'm right here with you.",
    "Night owl mode activated. How can I help?",
    "Still up? Let's crack these papers open."
  ]
};

function getRandomGreeting(): string {
  const hour = new Date().getHours();
  let category: keyof typeof GREETINGS = 'afternoon';

  if (hour >= 5 && hour < 12) {
    category = 'morning';
  } else if (hour >= 12 && hour < 17) {
    category = 'afternoon';
  } else if (hour >= 17 && hour < 22) {
    category = 'evening';
  } else {
    category = 'night';
  }

  const list = GREETINGS[category];
  return list[Math.floor(Math.random() * list.length)];
}

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string>('');
  const [chatId, setChatId] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<UploadedFile[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);
  const [activePdfUrl, setActivePdfUrl] = useState<string | null>(null);
  const [greeting, setGreeting] = useState<string>('Burning the midnight oil.');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setSessionId(getSessionId());
    setChatId(createNewChatId());
    setGreeting(getRandomGreeting());
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
        console.error('Failed to clear session:', err);
      }
    }
    setChatId(createNewChatId());
    setMessages([]);
    setAttachedFiles([]);
    setActivePdfUrl(null);
    setGreeting(getRandomGreeting());
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

    // Prevent submission if a file is currently ingesting
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
    <div className={`flex h-screen w-screen font-sans overflow-hidden ${isDarkMode ? 'bg-[#0f172a] text-white' : 'bg-[#f8fafc] text-gray-800'}`}>
      
      {/* Sidebar - Fixed width with no flex transitions to eliminate shaking */}
      {isSidebarOpen && (
        <aside className={`w-64 border-r p-5 flex flex-col justify-between shrink-0 select-none ${isDarkMode ? 'bg-[#1e293b] border-gray-700' : 'bg-white border-gray-100'}`}>
          <div className="space-y-6">
            {/* Logo */}
            <div className="flex items-center space-x-3 px-1">
              <div className="w-9 h-9 bg-[#6366f1] rounded-xl flex items-center justify-center text-white shadow-md shadow-indigo-200">
                <svg className="w-5 h-5 fill-current" viewBox="0 0 24 24">
                  <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" />
                </svg>
              </div>
              <span className={`font-bold text-base tracking-tight ${isDarkMode ? 'text-white' : 'text-slate-800'}`}>DocAgent RAG</span>
            </div>

            {/* Icons Bar: Dark Mode & Sidebar Toggle */}
            <div className="flex items-center space-x-4 px-2 text-gray-400">
              <button
                onClick={() => setIsDarkMode(!isDarkMode)}
                className="hover:text-gray-600 transition"
                title="Toggle Theme"
              >
                {isDarkMode ? '☀️' : '🌙'}
              </button>
              <button
                onClick={() => setIsSidebarOpen(false)}
                className="hover:text-gray-600 transition"
                title="Collapse Sidebar"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                </svg>
              </button>
            </div>

            {/* New Chat Button */}
            <button
              onClick={handleNewChat}
              className={`w-full py-2.5 px-4 rounded-xl text-sm font-medium flex items-center justify-center space-x-2 border transition shadow-sm ${
                isDarkMode 
                  ? 'bg-indigo-950/40 text-indigo-300 border-indigo-800/60 hover:bg-indigo-900/60' 
                  : 'bg-[#f5f3ff] text-[#6366f1] border-indigo-100 hover:bg-indigo-100/70'
              }`}
            >
              <span>+</span>
              <span>New Chat</span>
            </button>

            {/* Chat History */}
            <div>
              <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider px-1 mb-2">
                CHAT HISTORY
              </p>
              <div className={`py-2 px-3 rounded-lg text-xs font-medium truncate ${
                isDarkMode ? 'bg-indigo-900/40 text-indigo-200' : 'bg-[#eef2ff] text-[#4f46e5]'
              }`}>
                {messages.length > 0 ? messages[0].text : 'New Chat'}
              </div>
            </div>
          </div>

          {/* Sidebar Footer Status */}
          <div className="border-t border-gray-100 dark:border-gray-800 pt-3 text-[11px] text-gray-400 flex items-center justify-between">
            <span>Backend Pipeline:</span>
            <span className="flex items-center space-x-1.5 text-emerald-500 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>FastAPI Active</span>
            </span>
          </div>
        </aside>
      )}

      {/* Main Area */}
      <main className="flex-1 flex flex-col relative overflow-hidden">
        {/* Toggle Sidebar Button if collapsed */}
        {!isSidebarOpen && (
          <button
            onClick={() => setIsSidebarOpen(true)}
            className="absolute top-4 left-4 z-10 p-2 bg-white dark:bg-slate-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 text-gray-500 hover:text-gray-700"
            title="Expand Sidebar"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
            </svg>
          </button>
        )}

        {/* Chat Messages / Center Empty State */}
        <div className="flex-1 overflow-y-auto p-6 flex flex-col">
          {messages.length === 0 ? (
            <div className="m-auto flex flex-col items-center justify-center text-center max-w-lg space-y-4">
              <div className="w-16 h-16 bg-[#6366f1] rounded-2xl flex items-center justify-center text-white shadow-xl shadow-indigo-200/50">
                <svg className="w-9 h-9 fill-current" viewBox="0 0 24 24">
                  <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" />
                </svg>
              </div>

              <h1 className={`text-2xl font-bold tracking-tight ${isDarkMode ? 'text-white' : 'text-slate-900'}`}>
                {greeting}
              </h1>

              <p className="text-xs text-gray-400 font-normal">
                Upload a PDF or ask about one already shared — I'm here.
              </p>
            </div>
          ) : (
            <div className="space-y-6 max-w-3xl w-full mx-auto">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${
                    msg.sender === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`max-w-xl rounded-2xl px-5 py-4 text-sm leading-relaxed shadow-sm ${
                      msg.sender === 'user'
                        ? 'bg-[#6366f1] text-white rounded-tr-none'
                        : isDarkMode
                        ? 'bg-[#1e293b] text-gray-200 border border-gray-700 rounded-tl-none'
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
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Dock Area */}
        <footer className="p-6 flex flex-col items-center">
          {/* File Badges above Input Pill */}
          {attachedFiles.length > 0 && (
            <div className="flex flex-wrap justify-center gap-2 mb-3">
              {attachedFiles.map((file) => (
                <div
                  key={file.id}
                  className={`border shadow-sm rounded-xl px-3 py-1.5 flex items-center space-x-2 text-xs ${
                    isDarkMode ? 'bg-[#1e293b] border-gray-700 text-gray-200' : 'bg-white border-gray-200 text-gray-700'
                  }`}
                >
                  {file.isUploading ? (
                    <div className="w-3.5 h-3.5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <span className="text-indigo-500">📄</span>
                  )}
                  <span className="font-medium truncate max-w-[120px]">
                    {file.name}
                  </span>
                  <button
                    type="button"
                    onClick={() => removeFile(file.id)}
                    className="text-gray-400 hover:text-red-500 font-bold ml-1"
                    title="Remove file"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Bottom Bar Controls */}
          <div className="w-full max-w-2xl flex items-center space-x-3">
            {/* File Add (+) Button */}
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
              className={`w-11 h-11 rounded-2xl flex items-center justify-center border shadow-sm transition text-gray-500 hover:text-indigo-600 ${
                isDarkMode ? 'bg-[#1e293b] border-gray-700 hover:bg-gray-800' : 'bg-white border-gray-200/80 hover:bg-gray-50'
              }`}
              title="Add File"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
              </svg>
            </button>

            {/* Input Pill */}
            <form
              onSubmit={handleSendQuery}
              className={`flex-1 rounded-2xl border px-4 py-2.5 flex items-center shadow-sm transition ${
                isDarkMode
                  ? 'bg-[#1e293b] border-gray-700 focus-within:border-indigo-500'
                  : 'bg-white border-gray-200/80 focus-within:border-indigo-400'
              }`}
            >
              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                placeholder="Ask about text, tables, or diagrams..."
                className="flex-1 bg-transparent text-sm focus:outline-none placeholder-gray-400"
              />

              {/* Mic Icon */}
              <button
                type="button"
                className="text-gray-400 hover:text-gray-600 px-1"
                title="Voice Input"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              </button>
            </form>

            {/* Send Paperplane Button */}
            <button
              type="button"
              onClick={handleSendQuery}
              disabled={!inputQuery.trim() || isQuerying || hasUploadingFile}
              className={`w-11 h-11 rounded-2xl flex items-center justify-center transition shadow-md ${
                !inputQuery.trim() || isQuerying || hasUploadingFile
                  ? 'bg-indigo-100 text-indigo-300 dark:bg-slate-800 dark:text-gray-600 cursor-not-allowed'
                  : 'bg-[#818cf8] hover:bg-[#6366f1] text-white'
              }`}
            >
              <svg className="w-5 h-5 transform rotate-90" fill="currentColor" viewBox="0 0 24 24">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>

          <span className="text-[10px] text-gray-400 mt-3 font-normal">
            Made by HS
          </span>
        </footer>
      </main>

      {/* PDF Viewer Panel (Right Side) */}
      {activePdfUrl && (
        <section className={`w-[450px] border-l flex flex-col h-full shrink-0 ${isDarkMode ? 'bg-[#1e293b] border-gray-700' : 'bg-white border-gray-200'}`}>
          <div className="p-3 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between text-xs font-semibold text-gray-500">
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