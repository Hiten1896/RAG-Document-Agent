import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DocAgent RAG — Multi-Modal Document Intelligence",
  description:
    "Upload PDFs and ask questions. Powered by Gemini Vision, ChromaDB vector search, and FastAPI for enterprise-grade document Q&A with page-level citations.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // `maximumScale: 1` / `userScalable: false` used to be set here, which
  // blocks pinch-zoom and fails WCAG 2.1 SC 1.4.4 (Resize Text).
  // Renders <meta name="color-scheme"> so the browser knows both schemes are
  // supported before first paint, which keeps native UI from flashing white.
  colorScheme: "light dark",
};

// Runs before first paint, so the pinned theme is already on <html> by the time
// anything renders. Previously the theme was only applied in a useEffect and the
// entire app was held at `invisible` until then, so every load showed a blank
// screen and then flashed into the wrong theme. Keep the class names in sync
// with applyTheme() in page.tsx.
const themeInitScript = `
(function () {
  try {
    var saved = localStorage.getItem('docagent_theme');
    var theme =
      saved === 'dark' || saved === 'light'
        ? saved
        : window.matchMedia('(prefers-color-scheme: light)').matches
          ? 'light'
          : 'dark';
    var root = document.documentElement;
    root.classList.remove('dark', 'light');
    root.classList.add(theme);
    root.setAttribute('data-theme', theme);
  } catch (e) {}
})();
`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      // The inline script above mutates <html>'s class list before React
      // hydrates, which would otherwise be reported as a hydration mismatch.
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
