'use client';

import React, { useState, useEffect, useRef } from 'react';
import { getSessionId, createNewChatId } from '@/utils/session';

// Set your backend URL (Render endpoint or local backend)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize Session and Chat IDs on component mount
  useEffect(() => {
    const sId = getSessionId();
    const cId = createNewChatId();
    setSessionId(sId);
    setChatId(cId);
  }, []);

  // Send beacon signal to wipe vectors in ChromaDB when the user closes the tab
  useEffect(() => {
    if (!sessionId || !chatId) return;

    const handleBeforeUnload = () => {
      const endpoint = `${API_BASE_URL}/api/session/clear`;
      // Send beacon request carrying headers isn't directly supported by standard beacon,
      // so we pass headers as query params or basic payload if needed.
      navigator.sendBeacon(
        `${endpoint}?x_session_id=${sessionId}&x_chat_id=${chatId}`
      );
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [sessionId, chatId]);

  // Auto scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handler to start a fresh chat session
  const handleNewChat = async () => {
    // Optionally wipe previous chat context on the backend
    if (sessionId && chatId) {
      try {
        await fetch(`${API_BASE_URL}/api/session/clear`, {
          method: 'DELETE',
          headers: {
            'X-Session-ID': sessionId,
            'X-Chat-ID': chatId,
          },
        });
      } catch (err) {
        console.error('Failed to wipe old chat context:', err);
      }
    }

    // Generate a fresh chat ID and reset local chat UI
    setChatId(createNewChatId());
    setMessages([]);
    setFile(null);
    setUploadStatus('');
  };

  // Upload & Ingest File
  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setIsUploading(true);
    setUploadStatus('Processing document...');

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

      const data = await response.json();

      if (response.ok) {
        setUploadStatus(`Uploaded successfully: ${data.file_name}`);
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            sender: 'agent',
            text: `📄 Document "${file.name}" ingested into this chat session! Ask me anything about it.`,
          },
        ]);
      } else {
        setUploadStatus(`Error: ${data.detail || 'Failed to upload document'}`);
      }
    } catch (err) {
      setUploadStatus('Error: Could not connect to backend server.');
    } finally {
      setIsUploading(false);
    }
  };

  // Submit Query
  const handleSendQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || isQuerying) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      sender: 'user',
      text: inputQuery,
    };

    setMessages((prev) => [...prev, userMsg]);
    const currentQuery = inputQuery;
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
        body: JSON.stringify({ query: currentQuery }),
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
            text: `⚠️ ${data.detail || 'An error occurred while answering.'}`,
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          sender: 'agent',
          text: '⚠️ Unable to connect to backend server.',
        },
      ]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100 font-sans">
      {/* Sidebar Controls */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 p-4 flex flex-col justify-between">
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-bold text-indigo-400">RAG Doc Agent</h1>
            <button
              onClick={handleNewChat}
              className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-3 py-2 rounded-lg transition font-medium"
            >
              + New Chat
            </button>
          </div>

          {/* Document Ingestion Form */}
          <div className="bg-gray-700/50 p-4 rounded-xl border border-gray-600 space-y-3">
            <h2 className="text-sm font-semibold text-gray-300">
              Ingest Document
            </h2>
            <form onSubmit={handleFileUpload} className="space-y-3">
              <input
                type="file"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                accept=".pdf,.png,.jpg,.jpeg,.txt,.md"
                className="block w-full text-xs text-gray-400 file:mr-2 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer"
              />
              <button
                type="submit"
                disabled={!file || isUploading}
                className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-600 text-white text-xs py-2 rounded-lg font-medium transition"
              >
                {isUploading ? 'Ingesting...' : 'Upload & Ingest'}
              </button>
            </form>
            {uploadStatus && (
              <p className="text-xs text-gray-400 mt-2 break-words">
                {uploadStatus}
              </p>
            )}
          </div>
        </div>

        {/* Active Session Info */}
        <div className="text-[10px] text-gray-500 space-y-1 bg-gray-900/40 p-3 rounded-lg border border-gray-800">
          <p className="truncate">Session ID: {sessionId}</p>
          <p className="truncate">Chat ID: {chatId}</p>
        </div>
      </aside>

      {/* Main Chat View */}
      <main className="flex-1 flex flex-col">
        {/* Messages Window */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-500 space-y-2">
              <p className="text-lg">No messages in this chat thread.</p>
              <p className="text-sm">
                Upload a document in the sidebar to begin asking questions!
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
                  className={`max-w-xl rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    msg.sender === 'user'
                      ? 'bg-indigo-600 text-white rounded-br-none'
                      : 'bg-gray-800 text-gray-200 border border-gray-700 rounded-bl-none'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                  {msg.sourcesCount !== undefined && (
                    <span className="block text-[10px] text-gray-400 mt-2 border-t border-gray-700/60 pt-1">
                      Retrieved from {msg.sourcesCount} relevant document chunk(s)
                    </span>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Query Input Bar */}
        <footer className="p-4 bg-gray-800 border-t border-gray-700">
          <form
            onSubmit={handleSendQuery}
            className="flex items-center space-x-3 max-w-4xl mx-auto"
          >
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask a question about your ingested document..."
              className="flex-1 bg-gray-900 text-gray-100 placeholder-gray-500 border border-gray-700 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-indigo-500"
            />
            <button
              type="submit"
              disabled={!inputQuery.trim() || isQuerying}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 text-white text-sm px-5 py-3 rounded-xl font-medium transition"
            >
              {isQuerying ? 'Thinking...' : 'Send'}
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
}