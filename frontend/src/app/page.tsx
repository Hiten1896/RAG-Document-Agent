"use client";

import { useState, useEffect } from "react";
import { Sun, Moon, Plus, Upload, Sparkles, Send, Mic, Search, Zap } from "lucide-react";

export default function Home() {
  const [darkMode, setDarkMode] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello! Upload a PDF to automatically ingest text and diagrams. Ask me anything about the document!",
    },
  ]);
  const [input, setInput] = useState("");

  // Sync theme with html root element
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  const toggleTheme = () => setDarkMode(!darkMode);

  return (
    <div className={`min-h-screen flex ${darkMode ? "dark bg-slate-950 text-slate-100" : "bg-slate-50 text-slate-800"}`}>
      {/* Sidebar */}
      <aside className={`w-80 border-r p-6 flex flex-col justify-between ${darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200"}`}>
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-600 rounded-xl text-white">
                <Sparkles className="w-5 h-5" />
              </div>
              <h1 className="font-semibold text-lg tracking-tight">DocAgent RAG</h1>
            </div>
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg border transition-colors ${
                darkMode ? "bg-slate-800 border-slate-700 text-amber-400 hover:bg-slate-700" : "bg-slate-100 border-slate-200 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>

          <button className={`w-full py-2.5 px-4 rounded-xl border flex items-center justify-center gap-2 font-medium transition-all ${
            darkMode ? "border-slate-700 bg-slate-800/50 hover:bg-slate-800 text-slate-200" : "border-slate-200 bg-white hover:bg-slate-50 text-slate-700 shadow-sm"
          }`}>
            <Plus className="w-4 h-4" /> New Chat
          </button>

          {/* Document Ingestion Dropzone */}
          <div className="space-y-3">
            <div className="flex justify-between items-center text-xs font-semibold uppercase tracking-wider text-slate-400">
              <span>Document Ingestion</span>
              <span className="text-indigo-500">Auto-Ingest</span>
            </div>
            <div className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all ${
              darkMode ? "border-slate-800 hover:border-indigo-500/50 bg-slate-900/50" : "border-slate-200 hover:border-indigo-400 bg-slate-50/50"
            }`}>
              <div className="w-10 h-10 mx-auto rounded-full bg-indigo-500/10 flex items-center justify-center text-indigo-500 mb-2">
                <Upload className="w-5 h-5" />
              </div>
              <p className="text-xs font-medium">Select or Drop PDF</p>
              <p className="text-[10px] text-slate-400 mt-1">Drag & drop or click to browse</p>
            </div>
          </div>

          {/* System Features */}
          <div className={`p-4 rounded-xl border space-y-2 text-xs ${
            darkMode ? "bg-slate-900/60 border-slate-800 text-slate-300" : "bg-slate-50 border-slate-200 text-slate-600"
          }`}>
            <p className="font-semibold text-slate-400 flex items-center gap-1.5 mb-2">
              <Zap className="w-3.5 h-3.5 text-indigo-500" /> System Features
            </p>
            <p className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Fast text chunking & vector search</p>
            <p className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span> Gemini LLM response synthesis</p>
            <p className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span> Inline query editing & regeneration</p>
          </div>
        </div>

        <div className="text-xs text-slate-400 flex items-center justify-between border-t border-slate-200 dark:border-slate-800 pt-4">
          <span>Backend Pipeline:</span>
          <span className="flex items-center gap-1.5 text-emerald-500 font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> FastAPI Active
          </span>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="flex-1 flex flex-col justify-between p-8 max-w-5xl mx-auto">
        {/* Main Banner */}
        <div className="flex-1 flex flex-col items-center justify-center text-center my-8 space-y-6">
          <div className="p-4 rounded-3xl bg-indigo-500/10 text-indigo-500">
            <Sparkles className="w-10 h-10" />
          </div>
          <div className="space-y-2">
            <h2 className="text-3xl font-bold tracking-tight">Document Intelligence</h2>
            <p className="text-slate-400 max-w-md text-sm">
              Upload a PDF document and ask questions about its text, tables, and diagrams. Powered by Gemini and vector search.
            </p>
          </div>

          <div className="grid grid-cols-3 gap-4 w-full max-w-xl text-xs pt-4">
            <div className={`p-4 rounded-2xl border flex flex-col items-center gap-2 ${darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-sm"}`}>
              <Upload className="w-5 h-5 text-indigo-500" />
              <span>Upload PDF</span>
            </div>
            <div className={`p-4 rounded-2xl border flex flex-col items-center gap-2 ${darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-sm"}`}>
              <Zap className="w-5 h-5 text-amber-500" />
              <span>Auto-Ingest</span>
            </div>
            <div className={`p-4 rounded-2xl border flex flex-col items-center gap-2 ${darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-sm"}`}>
              <Search className="w-5 h-5 text-emerald-500" />
              <span>Ask Questions</span>
            </div>
          </div>
        </div>

        {/* Message Feed & Input area */}
        <div className="space-y-4">
          {messages.map((msg, idx) => (
            <div key={idx} className="flex gap-3 max-w-2xl">
              <div className="w-8 h-8 rounded-xl bg-indigo-500/10 flex items-center justify-center text-indigo-500 shrink-0">
                <Sparkles className="w-4 h-4" />
              </div>
              <div className={`p-4 rounded-2xl text-sm border ${
                darkMode ? "bg-slate-900 border-slate-800 text-slate-200" : "bg-white border-slate-200 text-slate-700 shadow-sm"
              }`}>
                {msg.content}
              </div>
            </div>
          ))}

          {/* Prompt Bar */}
          <div className={`p-2 rounded-2xl border flex items-center gap-2 ${
            darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-sm"
          }`}>
            <button className="p-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200">
              <Mic className="w-5 h-5" />
            </button>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about text, tables, or diagrams..."
              className="flex-1 bg-transparent border-none outline-none text-sm px-2 placeholder:text-slate-400"
            />
            <button className="p-2.5 bg-indigo-600 text-white rounded-xl hover:bg-indigo-500 transition-colors">
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}