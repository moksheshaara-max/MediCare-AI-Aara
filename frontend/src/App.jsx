import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import mermaid from 'mermaid';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Stethoscope, Send, AlertTriangle, BookOpen, Microscope, ShieldAlert,
  RefreshCw, ExternalLink, Sparkles, Info, Activity, HeartPulse,
  Award, FileText, UploadCloud, CheckCircle2, FileCheck,
  TrendingUp, TrendingDown, Minus, ClipboardList,
  X, Trash2, Download, Filter, UserPlus, Leaf, TestTube, 
  TestTubes, User, Calendar, Building, ChevronDown, ChevronUp, 
  BarChart2, Layers, GitBranch, Moon, Sun, Copy, Check, RotateCcw, 
  BrainCircuit, Dna, FileSearch, MessageCircle, Brain, Target, Zap
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || "https://medicare-ai-aara-backend.onrender.com";

mermaid.initialize({
  startOnLoad: false,
  theme: 'neutral',
  securityLevel: 'loose',
  fontFamily: 'ui-sans-serif, system-ui, sans-serif',
  themeVariables: {
    fontSize: '11px',
    primaryTextColor: '#0f172a',
    lineColor: '#64748b',
  },
  flowchart: { curve: 'linear', padding: 5, useMaxWidth: false, nodeSpacing: 14, rankSpacing: 18 },
});

let __mermaidIdCounter = 0;
function nextMermaidId() {
  __mermaidIdCounter += 1;
  return `mermaid-diagram-${__mermaidIdCounter}`;
}

const SUGGESTED_PROMPTS = [
  { icon: '🩺', text: "What is the step-by-step diagnostic and management algorithm for Type 2 Diabetes?" },
  { icon: '🔬', text: "Evaluate recent clinical trial evidence for Tirzepatide versus Semaglutide in NASH." },
  { icon: '💊', text: "What is the first-line and second-line pharmacotherapy for Pneumonia?" },
  { icon: '🚨', text: "I have severe crushing chest pain radiating to the jaw with shortness of breath." },
];

/* ═══════════════════════════════════════════
   PREMIUM ICON CONTAINER
   ═══════════════════════════════════════════ */
function PremiumIcon({ Icon, gradient = "from-sky-500 to-indigo-600", size = "lg", glow = "sky" }) {
  const sizes = {
    sm: "w-8 h-8 rounded-lg",
    md: "w-10 h-10 rounded-xl",
    lg: "w-14 h-14 rounded-2xl",
    xl: "w-16 h-16 md:w-20 md:h-20 rounded-3xl"
  };
  const iconSizes = {
    sm: "w-4 h-4",
    md: "w-5 h-5",
    lg: "w-7 h-7",
    xl: "w-8 h-8 md:w-10 md:h-10"
  };
  const glows = {
    sky: "shadow-[0_8px_30px_rgba(14,165,233,0.35)]",
    teal: "shadow-[0_8px_30px_rgba(20,184,166,0.35)]",
    rose: "shadow-[0_8px_30px_rgba(244,63,94,0.35)]",
    indigo: "shadow-[0_8px_30px_rgba(99,102,241,0.35)]"
  };
  return (
    <div className={`${sizes[size]} bg-gradient-to-br ${gradient} ${glows[glow]} flex items-center justify-center shrink-0`}>
      <Icon className={`${iconSizes[size]} text-white`} strokeWidth={2.5} />
    </div>
  );
}

function MermaidDiagram({ chart }) {
  const [svgContent, setSvgContent] = useState('');
  const [renderError, setRenderError] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const idRef = useRef(null);
  if (idRef.current === null) idRef.current = nextMermaidId();

  useEffect(() => {
    if (!chart) return;
    let isMounted = true;
    try {
      mermaid.render(idRef.current, chart.trim())
        .then(({ svg }) => {
          if (isMounted) {
            const responsiveSvg = svg.replace(
              /<svg /,
              '<svg style="width:auto;max-width:100%;height:auto;display:block;margin:auto;" '
            );
            setSvgContent(responsiveSvg);
            setRenderError(false);
          }
        })
        .catch(() => { if (isMounted) setRenderError(true); });
    } catch { if (isMounted) setRenderError(true); }
    return () => { isMounted = false; };
  }, [chart]);

  if (renderError || !svgContent) {
    return <MermaidFallbackBlock chart={chart} />;
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="my-5 w-full max-w-full min-w-0 overflow-hidden rounded-2xl md:rounded-3xl border border-sky-200/70 dark:border-slate-700 bg-gradient-to-br from-sky-50/70 via-white to-indigo-50/50 dark:from-slate-800/70 dark:via-[#111827] dark:to-indigo-950/20 shadow-sm"
    >
      <div className="flex items-center justify-between gap-3 px-4 py-3.5 md:px-5 border-b border-sky-100 dark:border-slate-700">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 text-white flex items-center justify-center shrink-0 shadow-sm">
            <GitBranch className="w-4 h-4" strokeWidth={2.5} />
          </div>
          <div className="min-w-0">
            <div className="text-[10px] md:text-xs font-extrabold uppercase tracking-[0.14em] text-sky-700 dark:text-sky-300">
              Quick Clinical Flow
            </div>
            <div className="text-[10px] md:text-xs text-slate-500 dark:text-slate-400">
              Fast-scan diagnostic and management pathway
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <span className="hidden sm:inline-flex px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 text-[9px] font-extrabold uppercase tracking-wider">
            At a glance
          </span>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-[10px] font-bold text-slate-500 hover:text-sky-600 dark:text-slate-400 dark:hover:text-sky-300 hover:bg-white/70 dark:hover:bg-slate-800 transition-colors"
          >
            {isExpanded ? 'Collapse' : 'Open full view'}
            <ChevronDown className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
          </button>
        </div>
      </div>

      <div
        className={`w-full max-w-full overflow-x-auto overscroll-x-contain px-2 sm:px-3 md:px-4 py-3 transition-all ${
          isExpanded ? '' : 'max-h-[280px] sm:max-h-[320px] md:max-h-[340px] overflow-y-auto'
        }`}
      >
        <div className="mx-auto w-full max-w-2xl min-w-0 flex justify-center">
          <div
            className="w-auto max-w-full min-w-0 flex justify-center [&_svg]:w-auto [&_svg]:max-w-full [&_svg]:h-auto"
            dangerouslySetInnerHTML={{ __html: svgContent }}
          />
        </div>
      </div>
    </motion.section>
  );
}


function MermaidFallbackBlock({ chart }) {
  const quotedLabels = [...String(chart || '').matchAll(/["']([^"']{2,120})["']/g)]
    .map((m) => m[1].trim())
    .filter((label) => !/^(yes|no|true|false)$/i.test(label));

  const bracketLabels = String(chart || '')
    .split('\n')
    .map((line) => {
      const match = line.match(/\w+\s*[\[\(\{]+\s*([^"\]\)\}][^\]\)\}]*)[\]\)\}]+/);
      return match?.[1]?.trim() || null;
    })
    .filter(Boolean);

  const steps = [...new Set([...quotedLabels, ...bracketLabels])].slice(0, 10);

  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="my-5 w-full max-w-full min-w-0 overflow-hidden rounded-2xl border border-sky-200/70 dark:border-slate-700 bg-gradient-to-br from-sky-50/70 via-white to-indigo-50/40 dark:from-slate-800/70 dark:via-[#111827] dark:to-indigo-950/20 shadow-sm"
    >
      <div className="flex items-center gap-3 px-4 py-3.5 border-b border-sky-100 dark:border-slate-700">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 text-white flex items-center justify-center shrink-0">
          <GitBranch className="w-4 h-4" strokeWidth={2.5} />
        </div>
        <div>
          <div className="text-[10px] md:text-xs font-extrabold uppercase tracking-[0.14em] text-sky-700 dark:text-sky-300">
            Quick Clinical Flow
          </div>
          <div className="text-[10px] md:text-xs text-slate-500 dark:text-slate-400">
            Simplified view because the diagram renderer could not load this chart
          </div>
        </div>
      </div>

      <div className="p-4 md:p-5 space-y-2.5">
        {(steps.length ? steps : ['Review the detailed clinical pathway below']).map((step, index) => (
          <div key={`${step}-${index}`} className="flex items-start gap-3 min-w-0">
            <div className="mt-0.5 w-7 h-7 rounded-full bg-sky-500/10 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800 flex items-center justify-center shrink-0 text-[10px] font-extrabold">
              {index + 1}
            </div>
            <div className="min-w-0 flex-1 rounded-xl border border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/60 px-3.5 py-2.5 text-xs md:text-sm leading-6 text-slate-700 dark:text-slate-300 whitespace-normal break-words [overflow-wrap:anywhere]">
              {step}
            </div>
          </div>
        ))}
      </div>
    </motion.section>
  );
}


function cleanAssistantMarkdown(text) {
  if (!text) return '';
  return String(text)
    .replace(
      /^\s*(?:📚🔬|📚|🔬|🤖)?\s*\[(?:Grounded in|General Clinical Knowledge)[^\]]*\]\s*/i,
      ''
    )
    .trim();
}

function evidenceModeLabel(mode) {
  const labels = {
    books_and_research: 'Guidelines · Textbooks · PubMed',
    books: 'Guidelines · Textbooks',
    research: 'PubMed',
    fallback: 'General knowledge',
    non_medical: 'Non-medical',
    error: 'Unavailable',
  };
  return labels[mode] || String(mode || '').replaceAll('_', ' ');
}

function ClinicalPathwayBlock({ chart }) {
  const lines = String(chart || '')
    .split('\n')
    .map((raw) => {
      if (!raw.trim() || !/[A-Za-z0-9[]/.test(raw)) return null;
      const firstContent = raw.search(/[A-Za-z0-9[]/);
      const level = Math.min(4, Math.max(0, Math.floor(firstContent / 6)));
      const value = raw
        .replace(/^[\s│├└─►→]+/, '')
        .replace(/^\[/, '')
        .replace(/\]$/, '')
        .trim();
      return value ? { level, value } : null;
    })
    .filter(Boolean);

  if (!lines.length) return null;

  const root = lines[0]?.value || 'Clinical pathway';
  const stages = [];
  let current = null;
  const preface = [];

  lines.slice(1).forEach((line) => {
    const isStage = /\b(?:grade|stage)\s*\d+\b/i.test(line.value);

    if (isStage) {
      current = { title: line.value, items: [] };
      stages.push(current);
      return;
    }

    if (current) current.items.push(line.value);
    else preface.push(line.value);
  });

  const stageThemes = [
    {
      badge: 'bg-sky-500/10 text-sky-700 dark:text-sky-300 border-sky-200 dark:border-sky-800',
      line: 'bg-sky-500',
      surface: 'border-sky-200/80 dark:border-sky-900/80',
    },
    {
      badge: 'bg-indigo-500/10 text-indigo-700 dark:text-indigo-300 border-indigo-200 dark:border-indigo-800',
      line: 'bg-indigo-500',
      surface: 'border-indigo-200/80 dark:border-indigo-900/80',
    },
    {
      badge: 'bg-rose-500/10 text-rose-700 dark:text-rose-300 border-rose-200 dark:border-rose-900',
      line: 'bg-rose-500',
      surface: 'border-rose-200/80 dark:border-rose-900/80',
    },
  ];

  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="my-6 w-full max-w-full min-w-0 overflow-hidden rounded-2xl md:rounded-3xl border border-slate-200 dark:border-slate-700 bg-gradient-to-br from-white via-sky-50/40 to-indigo-50/60 dark:from-slate-900 dark:via-slate-900 dark:to-indigo-950/20 shadow-sm font-sans whitespace-normal"
    >
      <div className="flex items-start gap-3 px-4 py-4 md:px-5 md:py-5 border-b border-slate-200/80 dark:border-slate-700">
        <div className="w-9 h-9 md:w-10 md:h-10 rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 text-white flex items-center justify-center shrink-0 shadow-sm">
          <GitBranch className="w-4 h-4" strokeWidth={2.5} />
        </div>
        <div className="min-w-0">
          <div className="text-[10px] md:text-xs font-extrabold uppercase tracking-[0.16em] text-sky-700 dark:text-sky-300">
            Clinical management pathway
          </div>
          <div className="mt-0.5 text-xs md:text-sm font-semibold text-slate-900 dark:text-white break-words">
            {root}
          </div>
        </div>
      </div>

      {preface.length > 0 && (
        <div className="px-4 pt-4 md:px-5">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-800/60 p-3">
            {preface.map((item, idx) => (
              <p key={idx} className="text-xs md:text-sm leading-6 text-slate-700 dark:text-slate-300 break-words whitespace-normal">
                {item}
              </p>
            ))}
          </div>
        </div>
      )}

      {stages.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 md:gap-4 p-4 md:p-5">
          {stages.map((stage, index) => {
            const theme = stageThemes[Math.min(index, stageThemes.length - 1)];
            return (
              <div
                key={`${stage.title}-${index}`}
                className={`relative min-w-0 overflow-hidden rounded-2xl border ${theme.surface} bg-white/90 dark:bg-slate-900/75 shadow-sm`}
              >
                <div className={`h-1 w-full ${theme.line}`} />
                <div className="p-4">
                  <div className={`inline-flex max-w-full rounded-full border px-2.5 py-1 text-[10px] md:text-xs font-extrabold ${theme.badge}`}>
                    <span className="break-words whitespace-normal">{stage.title}</span>
                  </div>

                  <div className="mt-3 space-y-2.5">
                    {stage.items.length > 0 ? stage.items.map((item, itemIndex) => (
                      <div key={itemIndex} className="flex items-start gap-2.5 min-w-0">
                        <span className={`mt-2 w-1.5 h-1.5 rounded-full shrink-0 ${theme.line}`} />
                        <p className="min-w-0 text-xs md:text-sm leading-6 text-slate-700 dark:text-slate-300 whitespace-normal break-words [overflow-wrap:anywhere]">
                          {item}
                        </p>
                      </div>
                    )) : (
                      <p className="text-xs md:text-sm text-slate-500 dark:text-slate-400">
                        Follow the applicable guideline pathway for this stage.
                      </p>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="p-4 md:p-5 space-y-2.5">
          {lines.slice(1).map((line, index) => (
            <div
              key={`${index}-${line.value}`}
              className="flex items-start gap-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-800/60 px-3.5 py-3 min-w-0"
            >
              <span className="mt-2 w-2 h-2 rounded-full bg-sky-500 shrink-0" />
              <p className="min-w-0 text-xs md:text-sm leading-6 text-slate-700 dark:text-slate-300 whitespace-normal break-words [overflow-wrap:anywhere]">
                {line.value}
              </p>
            </div>
          ))}
        </div>
      )}
    </motion.section>
  );
}


function StatusBadge({ status }) {
  const s = (status || "UNKNOWN").toUpperCase();
  const config = {
    HIGH: { bg: "bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-300/60 dark:border-rose-700/60", icon: TrendingUp },
    LOW: { bg: "bg-amber-500/10 text-amber-800 dark:text-amber-400 border-amber-300/60 dark:border-amber-700/60", icon: TrendingDown },
    NORMAL: { bg: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-300/60 dark:border-emerald-700/60", icon: Minus }
  };
  const c = config[s] || { bg: "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-700", icon: Minus };
  const Icon = c.icon;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border ${c.bg}`}>
      <Icon className="w-2.5 h-2.5" strokeWidth={2.75} /> {s}
    </span>
  );
}

function PrevalenceBadge({ prevalence }) {
  const p = (prevalence || "COMMON").toUpperCase();
  const config = {
    "COMMON": { bg: "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700/60", label: "COMMON" },
    "LESS COMMON": { bg: "bg-amber-100 dark:bg-amber-500/20 text-amber-900 dark:text-amber-300 border-amber-300 dark:border-amber-700/60", label: "LESS COMMON" },
    "RARE": { bg: "bg-purple-100 dark:bg-purple-500/20 text-purple-900 dark:text-purple-300 border-purple-300 dark:border-purple-500/30", label: "RARE / ATYPICAL" }
  };
  const c = config[p] || config["COMMON"];
  return (
    <span className={`text-[9px] font-extrabold px-2 py-0.5 rounded-md border tracking-wider uppercase ${c.bg}`}>
      {c.label}
    </span>
  );
}

function RiskBanner({ risk }) {
  const r = (risk || "MODERATE").toUpperCase();
  const styles = {
    HIGH: "bg-gradient-to-r from-rose-50 to-red-100 dark:from-rose-950/40 dark:to-red-900/40 border-rose-300 dark:border-rose-800 text-rose-900 dark:text-rose-200",
    MODERATE: "bg-gradient-to-r from-amber-50 to-orange-100 dark:from-amber-950/40 dark:to-orange-900/40 border-amber-300 dark:border-amber-800 text-amber-950 dark:text-amber-200",
    LOW: "bg-gradient-to-r from-emerald-50 to-teal-100 dark:from-emerald-950/40 dark:to-teal-900/40 border-emerald-300 dark:border-emerald-800 text-emerald-950 dark:text-emerald-200",
  };
  return (
    <div className={`rounded-xl border px-3.5 py-2 shadow-soft ${styles[r] || styles.MODERATE}`}>
      <div className="flex items-center gap-1.5 font-bold text-xs md:text-sm">
        <ShieldAlert className="w-4 h-4 shrink-0 text-current" strokeWidth={2.5} /> Risk Triage: {r}
      </div>
    </div>
  );
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000); } catch {}
  };
  return (
    <button onClick={handleCopy} className="p-1.5 rounded-lg text-slate-400 hover:text-sky-600 dark:hover:text-sky-400 transition-colors">
      {copied ? <Check className="w-4 h-4 text-emerald-500" strokeWidth={2.5} /> : <Copy className="w-4 h-4" strokeWidth={2.25} />}
    </button>
  );
}

const exportClinicalPDF = async (reportData) => {
  try {
    const response = await fetch(`${API_BASE}/api/export-pdf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reportData)
    });
    if (!response.ok) throw new Error("Failed to generate PDF on server.");
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `MediCare_AI_Report_${(reportData.patient_name || 'Patient').replace(/\s+/g, '_')}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert("Error downloading PDF: " + err.message);
  }
};

/* ── MAIN APP ── */
export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem('medicare_theme');
      if (saved) return saved === 'dark';
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });

  const [messages, setMessages] = useState(() => {
    try { const saved = localStorage.getItem('medicare_chat_history'); return saved ? JSON.parse(saved) : []; } catch { return []; }
  });

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [assistantMode, setAssistantMode] = useState(() => {
    try {
      const saved = localStorage.getItem('medicare_assistant_mode');
      return saved === 'pg_student' ? 'pg_student' : 'doctor';
    } catch {
      return 'doctor';
    }
  });
  const [showTransparency, setShowTransparency] = useState(false);
  const [openEvalId, setOpenEvalId] = useState(null);

  const [reportFile, setReportFile] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportResult, setReportResult] = useState(null);
  const [reportError, setReportError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [labFilter, setLabFilter] = useState('ALL');

  const [serverHealth, setServerHealth] = useState({ online: false, chunks: 0, docs: 0 });
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => { localStorage.setItem('medicare_chat_history', JSON.stringify(messages)); }, [messages]);
  useEffect(() => { localStorage.setItem('medicare_assistant_mode', assistantMode); }, [assistantMode]);

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
      root.style.colorScheme = 'dark';
    } else {
      root.classList.remove('dark');
      root.style.colorScheme = 'light';
    }
    localStorage.setItem('medicare_theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const scrollToBottom = useCallback(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, []);
  useEffect(() => { if (activeTab === 'chat') scrollToBottom(); }, [messages, loading, activeTab, scrollToBottom]);

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setServerHealth({ online: true, chunks: data.total_chunks_indexed || 0, docs: data.total_documents || 0 });
      } else { setServerHealth({ online: false, chunks: 0, docs: 0 }); }
    } catch { setServerHealth({ online: false, chunks: 0, docs: 0 }); }
  }, []);

  useEffect(() => { checkHealth(); const id = setInterval(checkHealth, 60000); return () => clearInterval(id); }, [checkHealth]);

  const clearChatHistory = () => {
    if (window.confirm("Clear all chat history?")) {
      setMessages([]);
      localStorage.removeItem('medicare_chat_history');
    }
  };

  const handleSend = async (textToSend) => {
    const questionText = textToSend || input;
    if (!questionText.trim() || loading) return;

    const userMsg = { id: Date.now(), sender: 'user', text: questionText, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const chatHistory = messages.slice(-6).map(m => ({ role: m.sender === 'user' ? 'user' : 'assistant', content: (m.text || '').substring(0, 500) }));
      const response = await fetch(`${API_BASE}/api/ask`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: questionText,
          chat_history: chatHistory,
          assistant_mode: assistantMode
        })
      });
      if (!response.ok) throw new Error(`Server Error`);
      const data = await response.json();

      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'ai',
        text: data.answer,
        assistantMode: data.assistant_mode || assistantMode,
        evidenceMode: data.evidence_mode || data.mode,
        retrievalConfidence: data.retrieval_confidence ?? data.accuracy_score ?? null,
        evaluation: data.evaluation,
        isEmergency: data.is_emergency,
        emergencyMessage: data.emergency_message,
        pubmedSources: data.pubmed_sources || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    } catch {
      setMessages(prev => [...prev, {
        id: Date.now() + 1, sender: 'ai', text: `⚠️ **Server Error:** Could not reach the backend.`,
        assistantMode, evidenceMode: 'error',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    } finally { setLoading(false); }
  };

  const handleReportUpload = async (e) => {
    e.preventDefault();
    if (!reportFile || reportLoading) return;
    setReportLoading(true); setReportError(''); setReportResult(null);
    const formData = new FormData();
    formData.append('file', reportFile);
    try {
      const response = await fetch(`${API_BASE}/api/analyze-report`, { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      setReportResult(data);
    } catch (err) { setReportError("Analysis failed. Please check the PDF."); }
    finally { setReportLoading(false); }
  };

  const groupedLabs = useMemo(() => {
    if (!reportResult?.lab_values) return {};
    const filtered = reportResult.lab_values.filter(l => {
      if (labFilter === 'HIGH') return l.status === 'HIGH';
      if (labFilter === 'LOW') return l.status === 'LOW';
      if (labFilter === 'NORMAL') return l.status === 'NORMAL';
      if (labFilter === 'ABNORMAL') return l.status === 'HIGH' || l.status === 'LOW';
      return true;
    });
    
    return filtered.reduce((acc, lab) => {
      const sys = lab.system || 'General';
      if (!acc[sys]) acc[sys] = [];
      acc[sys].push(lab);
      return acc;
    }, {});
  }, [reportResult, labFilter]);

  const labCounts = useMemo(() => ({
    total: reportResult?.lab_values?.length || 0,
    high: reportResult?.lab_values?.filter(l => l.status === 'HIGH').length || 0,
    low: reportResult?.lab_values?.filter(l => l.status === 'LOW').length || 0,
    normal: reportResult?.lab_values?.filter(l => l.status === 'NORMAL').length || 0,
  }), [reportResult]);

  useEffect(() => {
    const handler = (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); inputRef.current?.focus(); } };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // Auto-focus the input when the chat tab opens with no messages
  useEffect(() => {
    if (activeTab === 'chat' && messages.length === 0) {
      const t = setTimeout(() => inputRef.current?.focus(), 300);
      return () => clearTimeout(t);
    }
  }, [activeTab, messages.length]);

  return (
    <div className="flex flex-col min-h-screen relative overflow-x-hidden transition-colors">
      
      {/* Decorative Background Orbs */}
      <div aria-hidden className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        <div className="absolute -top-32 -right-32 w-96 h-96 bg-gradient-to-br from-sky-200/40 to-indigo-200/30 dark:from-sky-900/30 dark:to-indigo-900/20 rounded-full blur-3xl" />
        <div className="absolute top-1/2 -left-32 w-96 h-96 bg-gradient-to-br from-teal-200/30 to-emerald-200/20 dark:from-teal-900/30 dark:to-emerald-900/20 rounded-full blur-3xl" />
      </div>

      {/* ── HEADER ── */}
      <header className="sticky top-0 z-50 bg-white/90 dark:bg-slate-900/90 backdrop-blur-xl border-b border-slate-200 dark:border-slate-800 shadow-sm w-full">
        <div className="max-w-6xl mx-auto px-4 py-3 flex flex-col md:flex-row items-center justify-between gap-3 md:gap-6">
          
          <div className="flex items-center justify-between w-full md:w-auto">
            <div className="flex items-center gap-3">
              <motion.div whileHover={{ scale: 1.05, rotate: 5 }} className="bg-gradient-to-tr from-sky-500 via-teal-500 to-indigo-500 text-white p-2.5 rounded-xl shadow-glow-sky shrink-0">
                <Stethoscope className="w-5 h-5 md:w-6 md:h-6" strokeWidth={2.5} />
              </motion.div>
              <div>
                <div className="flex items-center gap-1.5">
                  <h1 className="text-lg md:text-xl font-extrabold gradient-text truncate">MediCare AI</h1>
                  <span className="text-[9px] md:text-[10px] font-bold px-2 py-0.5 rounded-full bg-sky-50 dark:bg-sky-900/30 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800 hidden xs:inline-block">Clinical Suite</span>
                </div>
                <p className="text-[10px] md:text-xs text-slate-500 dark:text-slate-400 truncate">Guidelines · Textbooks · PubMed · Lab Analysis</p>
              </div>
            </div>
            
            <div className="flex md:hidden items-center gap-1.5">
              <button onClick={() => setShowTransparency(true)} className="p-2 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                <Info className="w-4 h-4" strokeWidth={2.25} />
              </button>
              <button onClick={() => setDarkMode(!darkMode)} className="p-2 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                {darkMode ? <Sun className="w-4 h-4" strokeWidth={2.25} /> : <Moon className="w-4 h-4" strokeWidth={2.25} />}
              </button>
            </div>
          </div>

          <div className="flex bg-slate-100/80 dark:bg-slate-800 p-1.5 rounded-xl w-full md:w-auto shadow-inner border border-slate-200 dark:border-slate-700">
            <button onClick={() => setActiveTab('chat')} className={`flex-1 md:flex-none flex justify-center items-center gap-2 px-6 py-2 rounded-lg text-xs md:text-sm font-bold transition-all ${activeTab === 'chat' ? 'bg-white dark:bg-slate-700 text-sky-600 dark:text-sky-400 shadow-sm' : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'}`}>
              <MessageCircle className="w-4 h-4" strokeWidth={2.25} /> Clinical Assistant
            </button>
            <button onClick={() => setActiveTab('report')} className={`flex-1 md:flex-none flex justify-center items-center gap-2 px-6 py-2 rounded-lg text-xs md:text-sm font-bold transition-all ${activeTab === 'report' ? 'bg-white dark:bg-slate-700 text-teal-600 dark:text-teal-400 shadow-sm' : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'}`}>
              <FileSearch className="w-4 h-4" strokeWidth={2.25} /> Lab Analyzer
            </button>
          </div>

          <div className="hidden md:flex items-center gap-2.5">
            <button onClick={() => setShowTransparency(true)} className="flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-bold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition">
              <Info className="w-4 h-4 text-sky-600 dark:text-sky-400" strokeWidth={2.25} /> Transparency
            </button>
            <button onClick={() => setDarkMode(!darkMode)} className="p-2 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 transition hover:bg-slate-200 dark:hover:bg-slate-700">
              {darkMode ? <Sun className="w-4 h-4" strokeWidth={2.25} /> : <Moon className="w-4 h-4" strokeWidth={2.25} />}
            </button>
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold border ${serverHealth.online ? 'bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800' : 'bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800'}`}>
              <span className={`w-2 h-2 rounded-full ${serverHealth.online ? 'bg-emerald-500 pulse-ring' : 'bg-amber-500'}`} />
              <span>{serverHealth.online ? 'Online' : 'Offline'}</span>
            </div>
          </div>
        </div>
      </header>

      {/* ── TRANSPARENCY MODAL ── */}
      <AnimatePresence>
        {showTransparency && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setShowTransparency(false)}>
            <motion.div initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, y: 20 }} className="bg-white dark:bg-[#131a2c] rounded-3xl max-w-lg w-full p-6 md:p-8 shadow-2xl border border-slate-200 dark:border-slate-800" onClick={e => e.stopPropagation()}>
              <h3 className="font-extrabold text-xl mb-5 text-slate-900 dark:text-white flex items-center gap-3">
                <PremiumIcon Icon={Info} gradient="from-sky-500 to-indigo-600" size="md" glow="sky" />
                System Transparency
              </h3>
              <div className="text-sm text-slate-600 dark:text-slate-400 space-y-4">
                <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded-xl text-sky-900 dark:text-sky-300 font-medium border border-sky-100 dark:border-sky-800/50">
                  <strong>{serverHealth.chunks.toLocaleString()}</strong> indexed clinical knowledge chunks + Live PubMed.
                </div>
                <div className="space-y-3">
                  <p className="flex gap-3 items-start"><Target className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" strokeWidth={2.5} /><span><strong>Retrieval Confidence:</strong> Vector similarity is a grounding signal, not a measure of clinical accuracy.</span></p>
                  <p className="flex gap-3 items-start"><BarChart2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" strokeWidth={2.5} /><span><strong>Evidence Routing:</strong> Current guidelines are prioritized for management; textbooks add background and mechanisms.</span></p>
                  <p className="flex gap-3 items-start"><Microscope className="w-5 h-5 text-purple-500 shrink-0 mt-0.5" strokeWidth={2.5} /><span><strong>Evaluation Metrics:</strong> BLEU, ROUGE, groundedness, and semantic similarity are internal quality signals, not clinical validation.</span></p>
                  <p className="flex gap-3 items-start"><ShieldAlert className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" strokeWidth={2.5} /><span><strong>Safety:</strong> Educational decision-support only.</span></p>
                </div>
              </div>
              <button onClick={() => setShowTransparency(false)} className="w-full mt-8 py-3 bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-700 hover:to-indigo-700 text-white rounded-xl font-bold transition-colors shadow-glow-sky">
                Acknowledge
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── CHAT TAB ── */}
      {activeTab === 'chat' && (
        <main className="flex-1 max-w-5xl mx-auto w-full flex flex-col min-h-0" style={{ height: 'calc(100dvh - 4.5rem)' }}>
          
          <div className="flex-1 overflow-y-auto px-4 pt-4 pb-3 space-y-4 min-h-0">
            
            {messages.length > 0 && (
              <div className="flex justify-end">
                <button onClick={clearChatHistory} className="flex items-center gap-1.5 text-xs font-bold text-slate-400 hover:text-rose-600 bg-white dark:bg-slate-900 px-3 py-1.5 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm transition-colors">
                  <Trash2 className="w-3.5 h-3.5" strokeWidth={2.25} /> Clear History
                </button>
              </div>
            )}

            {messages.length === 0 && (
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="bg-white dark:bg-slate-900 rounded-3xl p-5 md:p-6 border border-slate-200 dark:border-slate-800 shadow-soft text-center">
                <div className="mx-auto mb-3 flex justify-center">
                  <PremiumIcon Icon={HeartPulse} gradient="from-sky-500 via-indigo-500 to-teal-500" size="lg" glow="sky" />
                </div>
                <h2 className="text-xl md:text-2xl font-extrabold text-slate-900 dark:text-white mb-1.5 tracking-tight">Clinical Assistant</h2>
                <p className="text-xs md:text-sm text-slate-600 dark:text-slate-400 max-w-md mx-auto mb-4">
                  <span className="font-bold text-sky-700 dark:text-sky-400">Guidelines + Textbooks</span> · <span className="font-bold text-indigo-700 dark:text-indigo-400">Live PubMed</span>
                </p>

                <div className="mb-4 inline-flex rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-100/80 dark:bg-slate-800 p-1.5 shadow-inner">
                  <button type="button" onClick={() => setAssistantMode('doctor')} disabled={loading}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${assistantMode === 'doctor' ? 'bg-white dark:bg-slate-700 text-sky-700 dark:text-sky-300 shadow-sm' : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'}`}>
                    <Stethoscope className="w-4 h-4" strokeWidth={2.5} /> Doctor Mode
                  </button>
                  <button type="button" onClick={() => setAssistantMode('pg_student')} disabled={loading}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${assistantMode === 'pg_student' ? 'bg-white dark:bg-slate-700 text-indigo-700 dark:text-indigo-300 shadow-sm' : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'}`}>
                    <BookOpen className="w-4 h-4" strokeWidth={2.5} /> PG Student Mode
                  </button>
                </div>

                <p className="text-[10px] md:text-xs text-slate-500 dark:text-slate-400 max-w-lg mx-auto mb-4">
                  {assistantMode === 'doctor'
                    ? 'Concise, guideline-heavy clinical decision support for practicing clinicians.'
                    : 'Guideline-based management with deeper mechanisms, rationale, and exam-focused explanation.'}
                </p>

                <div className="mb-4 flex flex-col items-center gap-1">
                  <span className="text-[11px] font-bold text-sky-600 dark:text-sky-400 animate-pulse">
                    ↓ Type any medical question in the search bar below
                  </span>
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-left">
                  {SUGGESTED_PROMPTS.map((p, idx) => (
                    <motion.button whileHover={{ scale: 1.02, y: -1 }} whileTap={{ scale: 0.98 }} key={idx} onClick={() => handleSend(p.text)} 
                      className="p-3 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-xl flex items-start justify-between gap-2 hover:border-sky-400 dark:hover:border-sky-500 hover:shadow-md transition-all group shadow-sm">
                      <span className="flex items-start gap-2">
                        <span className="text-base shrink-0">{p.icon}</span>
                        <span className="text-[11px] md:text-xs font-medium text-slate-700 dark:text-slate-200 leading-snug">{p.text}</span>
                      </span>
                      <Send className="w-3.5 h-3.5 text-slate-400 group-hover:text-sky-500 shrink-0 mt-0.5" strokeWidth={2.25} />
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            )}

            <AnimatePresence>
              {messages.map((msg) => (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} key={msg.id} className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'} space-y-1.5`}>
                  {msg.sender === 'user' ? (
                    <div className="bg-gradient-to-r from-sky-600 to-indigo-600 text-white rounded-3xl rounded-tr-md px-5 py-3.5 max-w-[92%] sm:max-w-[86%] md:max-w-3xl text-sm md:text-base leading-relaxed break-words font-medium shadow-soft">
                      {msg.text}
                    </div>
                  ) : (
                    <div className="bg-white dark:bg-slate-900 rounded-2xl md:rounded-3xl border border-slate-200 dark:border-slate-800 p-4 sm:p-5 md:p-7 w-full min-w-0 overflow-hidden shadow-soft space-y-5">
                      {msg.isEmergency && (
                        <div className="bg-rose-50 dark:bg-rose-500/10 border-2 border-rose-500 dark:border-rose-500/30 rounded-2xl p-4 text-rose-800 dark:text-rose-300 text-xs md:text-sm font-bold flex items-start gap-3 shadow-sm">
                          <PremiumIcon Icon={AlertTriangle} gradient="from-rose-500 to-red-600" size="sm" glow="rose" />
                          <p className="leading-relaxed whitespace-pre-line pt-1">{msg.emergencyMessage}</p>
                        </div>
                      )}
                      
                      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center border-b border-slate-100 dark:border-slate-800 pb-3 gap-3">
                        <div className="flex items-center gap-3">
                          <PremiumIcon Icon={BrainCircuit} gradient="from-sky-500 to-indigo-600" size="sm" glow="sky" />
                          <div>
                            <div className="font-bold text-sm text-slate-900 dark:text-white">Clinical Synthesis</div>
                            <div className="flex flex-wrap items-center gap-1.5 mt-0.5">
                              <span className={`text-[9px] font-extrabold uppercase tracking-wider px-2 py-0.5 rounded-full border ${msg.assistantMode === 'pg_student' ? 'bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 border-indigo-200 dark:border-indigo-800' : 'bg-sky-50 dark:bg-sky-900/30 text-sky-700 dark:text-sky-300 border-sky-200 dark:border-sky-800'}`}>
                                {msg.assistantMode === 'pg_student' ? 'PG Student Mode' : 'Doctor Mode'}
                              </span>
                              {msg.evidenceMode && msg.evidenceMode !== 'non_medical' && msg.evidenceMode !== 'error' && (
                                <span className="text-[9px] font-bold text-slate-400 dark:text-slate-500">
                                  {evidenceModeLabel(msg.evidenceMode)}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          {typeof msg.retrievalConfidence === 'number' && msg.retrievalConfidence > 0 && (
                            <span title="Vector retrieval similarity / grounding confidence. This is not clinical accuracy."
                              className="hidden md:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-300 text-[10px] font-bold border border-slate-200 dark:border-slate-700">
                              <Target className="w-3.5 h-3.5 text-sky-500" strokeWidth={2.5} />
                              Retrieval {msg.retrievalConfidence.toFixed(1)}
                            </span>
                          )}
                          {msg.evaluation && (
                            <button onClick={() => setOpenEvalId(openEvalId === msg.id ? null : msg.id)}
                              title="Internal automated evaluation signal. Not a measure of clinical accuracy or validation."
                              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400 text-[10px] md:text-xs font-bold border border-indigo-200 dark:border-indigo-800 hover:bg-indigo-100 dark:hover:bg-indigo-900/50 transition-colors">
                              <BarChart2 className="w-3.5 h-3.5" strokeWidth={2.5} /> Quality signal {msg.evaluation.composite_score}/100
                            </button>
                          )}
                          <CopyButton text={msg.text} />
                        </div>
                      </div>

                      {msg.evaluation && openEvalId === msg.id && (
                        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="bg-indigo-50 dark:bg-slate-800/80 border border-indigo-100 dark:border-slate-700 rounded-2xl p-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                          {[
                            { label: "Grounded", value: msg.evaluation.groundedness },
                            { label: "Similarity", value: msg.evaluation.semantic_similarity },
                            { label: "BLEU-4", value: msg.evaluation.bleu_score },
                            { label: "ROUGE-L", value: msg.evaluation.rouge_score }
                          ].map((metric, i) => (
                            <div key={i} className="bg-white dark:bg-slate-900 p-3 rounded-xl border border-indigo-50 dark:border-slate-700 text-center shadow-sm">
                              <span className="text-[10px] text-slate-500 dark:text-slate-400 font-bold uppercase tracking-wider block mb-1">{metric.label}</span>
                              <span className="text-lg font-extrabold text-indigo-700 dark:text-indigo-400">{typeof metric.value === 'number' ? metric.value.toFixed(0) : metric.value}%</span>
                            </div>
                          ))}
                        </motion.div>
                      )}

                      <div className="markdown-body min-w-0 max-w-full overflow-hidden">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            h1({ children }) {
                              return <h1 className="mt-7 mb-3 text-xl md:text-2xl font-extrabold tracking-tight text-slate-950 dark:text-white break-words">{children}</h1>;
                            },
                            h2({ children }) {
                              return <h2 className="mt-7 mb-3 text-lg md:text-xl font-extrabold tracking-tight text-slate-950 dark:text-white break-words">{children}</h2>;
                            },
                            h3({ children }) {
                              return (
                                <h3 className="mt-6 mb-3 flex items-start gap-2.5 text-base md:text-lg font-extrabold text-slate-900 dark:text-white break-words">
                                  <span className="mt-2 w-1.5 h-5 rounded-full bg-gradient-to-b from-sky-500 to-indigo-600 shrink-0" />
                                  <span className="min-w-0">{children}</span>
                                </h3>
                              );
                            },
                            h4({ children }) {
                              return <h4 className="mt-5 mb-2 text-sm md:text-base font-bold text-slate-900 dark:text-slate-100 break-words">{children}</h4>;
                            },
                            p({ children }) {
                              return <p className="my-3 text-[13px] sm:text-sm md:text-[15px] leading-6 md:leading-7 text-slate-700 dark:text-slate-300 whitespace-normal break-words [overflow-wrap:anywhere]">{children}</p>;
                            },
                            ul({ children }) {
                              return <ul className="my-3 space-y-2 pl-1">{children}</ul>;
                            },
                            ol({ children }) {
                              return <ol className="my-3 space-y-2 pl-1">{children}</ol>;
                            },
                            li({ children }) {
                              return (
                                <li className="flex items-start gap-2.5 text-[13px] sm:text-sm md:text-[15px] leading-6 md:leading-7 text-slate-700 dark:text-slate-300 min-w-0">
                                  <span className="mt-[0.62rem] w-1.5 h-1.5 rounded-full bg-sky-500 shrink-0" />
                                  <span className="min-w-0 whitespace-normal break-words [overflow-wrap:anywhere]">{children}</span>
                                </li>
                              );
                            },
                            strong({ children }) {
                              return <strong className="font-extrabold text-slate-950 dark:text-white">{children}</strong>;
                            },
                            em({ children }) {
                              return <em className="text-slate-700 dark:text-slate-300">{children}</em>;
                            },
                            hr() {
                              return <hr className="my-6 border-slate-200 dark:border-slate-700" />;
                            },
                            blockquote({ children }) {
                              return (
                                <blockquote className="my-4 rounded-r-xl border-l-4 border-sky-500 bg-sky-50/70 dark:bg-sky-950/20 px-4 py-2 text-slate-700 dark:text-slate-300">
                                  {children}
                                </blockquote>
                              );
                            },
                            table({ children }) {
                              return (
                                <div className="my-5 w-full max-w-full overflow-hidden rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-sm">
                                  <div className="md:hidden flex items-center justify-between gap-3 px-3 py-2 bg-slate-50 dark:bg-slate-800/80 border-b border-slate-200 dark:border-slate-700">
                                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">Clinical table</span>
                                    <span className="text-[10px] text-slate-400">Swipe horizontally →</span>
                                  </div>
                                  <div className="w-full overflow-x-auto overscroll-x-contain">
                                    <table className="min-w-[760px] md:min-w-full w-full text-xs md:text-sm leading-relaxed">
                                      {children}
                                    </table>
                                  </div>
                                </div>
                              );
                            },
                            thead({ children }) {
                              return <thead className="bg-slate-100 dark:bg-slate-800/95">{children}</thead>;
                            },
                            tbody({ children }) {
                              return <tbody className="divide-y divide-slate-200 dark:divide-slate-800">{children}</tbody>;
                            },
                            tr({ children }) {
                              return <tr className="even:bg-slate-50/60 dark:even:bg-slate-800/20">{children}</tr>;
                            },
                            th({ children }) {
                              return <th className="px-3.5 md:px-4 py-3.5 text-left align-bottom font-extrabold text-slate-950 dark:text-white border-b border-slate-200 dark:border-slate-700 whitespace-normal">{children}</th>;
                            },
                            td({ children }) {
                              return <td className="px-3.5 md:px-4 py-3.5 align-top text-slate-700 dark:text-slate-300 whitespace-normal break-words [overflow-wrap:anywhere]">{children}</td>;
                            },
                            pre({ children }) {
                              return <>{children}</>;
                            },
                            code({ inline, className, children, ...props }) {
                              const str = String(children).replace(/\n$/, '').trim();
                              const isMermaid = !inline && (
                                className?.includes('language-mermaid') ||
                                className?.includes('mermaid') ||
                                str.startsWith('flowchart ') ||
                                str.startsWith('graph ')
                              );
                              const isAsciiClinicalPathway = !inline
                                && /[│├└►→]/.test(str)
                                && /(stage|grade|therapy|treatment|management|blood pressure|diagnosis|initiate|emergency)/i.test(str);

                              if (isMermaid) return <MermaidDiagram chart={str} />;
                              if (isAsciiClinicalPathway) return <ClinicalPathwayBlock chart={str} />;

                              if (inline) {
                                return (
                                  <code
                                    className="px-1.5 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-sky-700 dark:text-sky-300 text-[0.92em] whitespace-normal break-words"
                                    {...props}
                                  >
                                    {children}
                                  </code>
                                );
                              }

                              return (
                                <pre className="my-4 max-w-full overflow-x-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-950 text-slate-100 p-4 text-xs md:text-sm leading-6 whitespace-pre">
                                  <code className={className} {...props}>{children}</code>
                                </pre>
                              );
                            }
                          }}
                        >
                          {cleanAssistantMarkdown(msg.text)}
                        </ReactMarkdown>
                      </div>

                      {msg.pubmedSources?.length > 0 && (
                        <div className="pt-4 border-t border-slate-100 dark:border-slate-800 space-y-2.5">
                          <div className="text-xs font-bold text-indigo-800 dark:text-indigo-400 uppercase tracking-wider flex items-center gap-1.5">
                            <Microscope className="w-4 h-4" strokeWidth={2.25} /> Live PubMed Citations
                          </div>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {msg.pubmedSources.map((p, i) => (
                              <a key={i} href={p.url} target="_blank" rel="noopener noreferrer" className="min-w-0 p-3.5 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-200 dark:border-slate-700 hover:border-sky-400 dark:hover:border-sky-500 hover:shadow-sm transition-all group">
                                <div className="flex justify-between items-start gap-2 mb-1.5">
                                  <strong className="min-w-0 text-slate-900 dark:text-white text-xs md:text-sm line-clamp-2 leading-snug break-words group-hover:text-sky-600 dark:group-hover:text-sky-400 transition-colors">{p.title}</strong>
                                  <ExternalLink className="w-4 h-4 text-slate-400 shrink-0" strokeWidth={2.25} />
                                </div>
                                <div className="text-[10px] md:text-xs text-slate-500 dark:text-slate-400">{p.journal} ({p.year}) • <span className="font-mono font-bold text-indigo-600 dark:text-indigo-400">PMID: {p.pmid}</span></div>
                              </a>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>

            {loading && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 max-w-sm shadow-soft flex items-center gap-4">
                <PremiumIcon Icon={Activity} gradient="from-sky-500 to-indigo-600" size="sm" glow="sky" />
                <div className="space-y-1 w-full">
                  <span className="text-xs font-bold text-slate-700 dark:text-slate-300">Synthesizing response...</span>
                  <div className="h-1.5 shimmer rounded-full w-full" />
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* ALWAYS-VISIBLE HIGHLIGHTED SEARCH BAR */}
          <div className="shrink-0 px-4 pb-4 pt-1">
            {messages.length > 0 && (
              <div className="mb-2 flex items-center justify-between gap-3">
                <div className="inline-flex rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-1 shadow-sm">
                  <button type="button" onClick={() => setAssistantMode('doctor')} disabled={loading}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[10px] md:text-xs font-bold transition-all ${assistantMode === 'doctor' ? 'bg-sky-50 dark:bg-sky-900/30 text-sky-700 dark:text-sky-300' : 'text-slate-500 dark:text-slate-400'}`}>
                    <Stethoscope className="w-3.5 h-3.5" strokeWidth={2.5} /> Doctor
                  </button>
                  <button type="button" onClick={() => setAssistantMode('pg_student')} disabled={loading}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[10px] md:text-xs font-bold transition-all ${assistantMode === 'pg_student' ? 'bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300' : 'text-slate-500 dark:text-slate-400'}`}>
                    <BookOpen className="w-3.5 h-3.5" strokeWidth={2.5} /> PG Student
                  </button>
                </div>
                <span className="hidden sm:inline text-[10px] text-slate-400 dark:text-slate-500">Applies to the next response</span>
              </div>
            )}

            {messages.length === 0 && (
              <p className="text-center text-[10px] md:text-xs font-bold text-sky-600 dark:text-sky-400 mb-2 flex items-center justify-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5" strokeWidth={2.5} />
                Ask anything medical — type below and press Send
              </p>
            )}
            <footer
              className={`bg-white dark:bg-slate-900 border-2 p-2.5 rounded-2xl shadow-elevated transition-all duration-500 ${
                messages.length === 0
                  ? 'border-sky-400 dark:border-sky-500 shadow-[0_0_0_4px_rgba(14,165,233,0.15),0_8px_30px_rgba(14,165,233,0.2)] ring-2 ring-sky-400/30 dark:ring-sky-500/40 animate-pulse'
                  : 'border-slate-200 dark:border-slate-700'
              }`}
              style={messages.length === 0 ? { animationDuration: '2.5s' } : undefined}
            >
              <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="flex items-stretch gap-2 min-w-0">
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onFocus={(e) => e.target.parentElement?.parentElement?.classList.remove('animate-pulse')}
                  placeholder="Type your clinical question here..."
                  disabled={loading}
                  autoFocus
                  className="min-w-0 flex-1 bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white rounded-xl px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-sky-500/50 transition-all placeholder:text-slate-400 placeholder:font-medium"
                />
                <button
                  type="submit"
                  disabled={loading || !input.trim()}
                  className="bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-700 hover:to-indigo-700 text-white font-bold px-5 md:px-6 rounded-xl shadow-glow-sky transition-all disabled:opacity-50 flex items-center gap-1.5"
                >
                  <Send className="w-4 h-4" strokeWidth={2.5} />
                  <span className="hidden sm:inline text-sm">Send</span>
                </button>
              </form>
            </footer>
          </div>
        </main>
      )}

      {/* ── REPORT ANALYZER TAB ── */}
      {activeTab === 'report' && (
        <main className="flex-1 max-w-6xl mx-auto w-full p-4 md:p-6 space-y-6">
          
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="relative overflow-hidden bg-gradient-to-br from-teal-700 via-sky-800 to-indigo-900 rounded-3xl p-6 md:p-10 text-white shadow-elevated">
            <div className="absolute -top-24 -right-24 w-64 h-64 bg-white/10 rounded-full blur-3xl" />
            <div className="absolute -bottom-16 -left-16 w-48 h-48 bg-teal-300/20 rounded-full blur-3xl" />
            <div className="relative z-10 flex flex-col md:flex-row justify-between items-center gap-6">
              <div className="space-y-3 text-center md:text-left">
                <div className="inline-flex items-center gap-1.5 bg-white/20 backdrop-blur-md px-3 py-1.5 rounded-full text-xs font-bold">
                  <BrainCircuit className="w-4 h-4" strokeWidth={2.5} /> Two-Brain RAG Architecture
                </div>
                <h2 className="text-2xl md:text-4xl font-extrabold tracking-tight">Blood Test & Lab Report Analyzer</h2>
                <p className="text-sky-100 text-xs md:text-sm max-w-xl">Extracts 100+ parameters, groups values by organ system, arranges differentials by prevalence, and exports PDF summaries.</p>
              </div>
              <div className="hidden md:block">
                <PremiumIcon Icon={Dna} gradient="from-white/30 to-white/10" size="xl" glow="teal" />
              </div>
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 shadow-soft p-6 md:p-8 text-center">
            <form onSubmit={handleReportUpload} className="max-w-lg mx-auto">
              <label className="cursor-pointer block border-2 border-dashed border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 rounded-2xl p-8 hover:border-teal-500 dark:hover:border-teal-400 hover:bg-teal-50/30 dark:hover:bg-teal-900/10 transition-all group">
                <div className="flex justify-center mb-4">
                  <motion.div whileHover={{ scale: 1.1, rotate: -5 }} transition={{ type: "spring", stiffness: 300 }}>
                    <PremiumIcon Icon={UploadCloud} gradient="from-teal-500 to-emerald-600" size="lg" glow="teal" />
                  </motion.div>
                </div>
                <h3 className="text-base font-bold text-slate-900 dark:text-white mb-1">Upload Clinical PDF</h3>
                <span className="text-xs text-slate-500">Supports CBC, Metabolic, Lipid, Renal, Thyroid profiles</span>
                <input type="file" accept=".pdf" onChange={(e) => setReportFile(e.target.files?.[0])} className="hidden" />
              </label>
              {reportFile && <div className="mt-4 text-sm font-bold text-teal-600 dark:text-teal-400 flex items-center justify-center gap-2"><FileCheck className="w-4 h-4" strokeWidth={2.5} /> {reportFile.name}</div>}
              {reportError && <div className="mt-4 text-xs font-bold text-rose-600 bg-rose-50 dark:bg-rose-900/30 p-3 rounded-xl flex justify-center gap-2"><AlertTriangle className="w-4 h-4" strokeWidth={2.5} /> {reportError}</div>}
              
              <button type="submit" disabled={!reportFile || reportLoading} className="mt-5 w-full px-8 py-3.5 bg-gradient-to-r from-teal-600 to-sky-600 hover:from-teal-700 hover:to-sky-700 text-white rounded-xl font-bold text-sm hover:shadow-glow-sky disabled:opacity-50 transition-all flex items-center justify-center gap-2">
                {reportLoading ? <><Activity className="w-5 h-5 animate-spin" strokeWidth={2.5} /> Performing RAG Clinical Analysis...</> : <><Sparkles className="w-5 h-5" strokeWidth={2.5} /> Analyze Report</>}
              </button>
            </form>
          </motion.div>

          {reportResult && !reportLoading && (
            <div className="space-y-6">
              
              {reportResult.critical_alerts?.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-rose-50 dark:bg-rose-900/30 border-2 border-rose-500 dark:border-rose-700 rounded-3xl p-5 text-rose-900 dark:text-rose-200 shadow-soft">
                  <div className="flex items-center gap-3 font-bold text-base text-rose-700 dark:text-rose-400 mb-3">
                    <PremiumIcon Icon={AlertTriangle} gradient="from-rose-500 to-red-600" size="sm" glow="rose" />
                    CRITICAL LAB ALERTS
                  </div>
                  <ul className="list-disc pl-6 text-sm font-semibold space-y-1.5">{reportResult.critical_alerts.map((a, i) => <li key={i}>{a}</li>)}</ul>
                </motion.div>
              )}

              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-white dark:bg-[#131a2c] rounded-3xl border border-slate-200 dark:border-slate-800 shadow-soft p-5 md:p-8 space-y-5">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-4">
                  <h3 className="font-extrabold text-lg md:text-xl text-slate-900 dark:text-white flex items-center gap-3">
                    <PremiumIcon Icon={CheckCircle2} gradient="from-emerald-500 to-teal-600" size="sm" glow="teal" />
                    <span className="truncate">{reportResult.file_name}</span>
                  </h3>
                  <div className="flex items-center gap-3">
                    <RiskBanner risk={reportResult.risk_level} />
                    <button onClick={() => exportClinicalPDF(reportResult)} className="bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-700 hover:to-indigo-700 text-white text-xs px-4 py-2 rounded-xl font-bold flex items-center gap-1.5 shadow-glow-sky transition-all">
                      <Download className="w-4 h-4" strokeWidth={2.5} /> Export PDF
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 bg-slate-50 dark:bg-slate-800/50 rounded-2xl p-4 text-xs md:text-sm">
                  {[
                    { l: 'Patient', v: reportResult.patient_name, i: User, c: 'text-sky-600' },
                    { l: 'Age/Sex', v: reportResult.patient_age_gender, i: Activity, c: 'text-indigo-600' },
                    { l: 'Lab', v: reportResult.lab_name, i: Building, c: 'text-purple-600' },
                    { l: 'Date', v: reportResult.report_date, i: Calendar, c: 'text-emerald-600' }
                  ].map((x, i) => (
                    <div key={i}><span className="text-slate-400 text-[10px] font-bold uppercase block mb-1">{x.l}</span><strong className="text-slate-900 dark:text-white flex items-center gap-1.5 truncate"><x.i className={`w-4 h-4 ${x.c}`} strokeWidth={2.5} />{x.v}</strong></div>
                  ))}
                </div>

                {reportResult.summary && (
                  <div className="bg-sky-50 dark:bg-sky-900/20 border border-sky-100 dark:border-sky-800/50 rounded-2xl p-5">
                    <div className="text-xs font-bold text-sky-800 dark:text-sky-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                      <ClipboardList className="w-4 h-4" strokeWidth={2.5} /> Executive Summary
                    </div>
                    <p className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed font-medium">{reportResult.summary}</p>
                  </div>
                )}
              </motion.div>

              {reportResult.lab_values?.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-white dark:bg-[#131a2c] rounded-3xl border border-slate-200 dark:border-slate-800 shadow-soft p-5 md:p-8">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-4 mb-5">
                    <h4 className="font-bold text-base md:text-lg text-slate-900 dark:text-white flex items-center gap-3">
                      <PremiumIcon Icon={TestTubes} gradient="from-teal-500 to-cyan-600" size="sm" glow="teal" />
                      Extracted Parameters
                    </h4>
                    <div className="flex gap-2 overflow-x-auto pb-2 md:pb-0">
                      {['ALL', 'HIGH', 'LOW', 'NORMAL'].map(f => (
                        <button key={f} onClick={() => setLabFilter(f)} className={`px-4 py-1.5 text-xs font-bold rounded-xl border transition-colors ${labFilter === f ? 'bg-teal-600 text-white border-teal-600 shadow-sm' : 'bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-700'}`}>{f}</button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-6">
                    {Object.entries(groupedLabs).map(([sys, labs], sIdx) => (
                      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: sIdx * 0.08 }} key={sys} className="space-y-3">
                        <h5 className="text-[11px] font-bold uppercase tracking-wider text-teal-800 dark:text-teal-300 bg-teal-50 dark:bg-teal-900/30 border border-teal-200 dark:border-teal-800/50 px-3 py-1.5 rounded-xl inline-flex items-center gap-1.5">
                          <Layers className="w-4 h-4 text-teal-600 dark:text-teal-400" strokeWidth={2.5} /> {sys} ({labs.length})
                        </h5>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                          {labs.map((lab, i) => {
                            const st = (lab.status || "").toUpperCase();
                            const bClass = st === "HIGH" ? "border-rose-200 bg-rose-50/50 dark:border-rose-800/50 dark:bg-rose-900/10" : st === "LOW" ? "border-amber-200 bg-amber-50/50 dark:border-amber-800/50 dark:bg-amber-900/10" : st === "NORMAL" ? "border-emerald-200 bg-emerald-50/30 dark:border-emerald-800/50 dark:bg-emerald-900/10" : "border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800/50";
                            return (
                              <motion.div whileHover={{ y: -3 }} transition={{ type: "spring", stiffness: 300 }} key={i} className={`p-4 rounded-2xl border ${bClass} shadow-sm hover:shadow-md transition-all`}>
                                <div className="flex justify-between items-start mb-2">
                                  <span className="text-xs font-bold text-slate-800 dark:text-slate-200 pr-2 leading-tight">{lab.test}</span>
                                  <StatusBadge status={st} />
                                </div>
                                <div className="text-xl font-extrabold text-slate-900 dark:text-white">{lab.result} <span className="text-[10px] font-medium text-slate-500">{lab.unit}</span></div>
                                <div className="text-[10px] text-slate-500 mt-1">Ref: {lab.reference_range}</div>
                              </motion.div>
                            );
                          })}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}

              {reportResult.differential_considerations?.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-5 md:p-8 shadow-soft">
                  <div className="border-b border-slate-100 dark:border-slate-800 pb-3 mb-5">
                    <h4 className="font-bold text-base md:text-lg text-slate-900 dark:text-white flex items-center gap-3">
                      <PremiumIcon Icon={Target} gradient="from-indigo-500 to-purple-600" size="sm" glow="indigo" />
                      Differential Considerations
                    </h4>
                    <p className="text-xs text-slate-500 mt-2 ml-11">Ranked from highest epidemiological likelihood to rare atypical patterns.</p>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {reportResult.differential_considerations.map((diff, idx) => (
                      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: idx * 0.1 }} whileHover={{ y: -3 }} key={idx} 
                        className="p-5 bg-gradient-to-br from-slate-50 to-white dark:from-slate-800 dark:to-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl shadow-sm hover:border-indigo-300 dark:hover:border-indigo-700 hover:shadow-md transition-all">
                        <div className="flex justify-between items-start mb-3 gap-2">
                          <strong className="text-sm text-slate-900 dark:text-white flex items-center gap-2">
                            <span className="w-7 h-7 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white flex items-center justify-center text-xs font-extrabold shadow-sm">{idx+1}</span>
                            {diff.title}
                          </strong>
                          <PrevalenceBadge prevalence={diff.prevalence} />
                        </div>
                        <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed pl-9">{diff.rationale}</p>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                {reportResult.pathophysiology && (
                  <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-6 shadow-soft">
                    <h4 className="font-bold text-sm text-slate-900 dark:text-white mb-3 flex items-center gap-3">
                      <PremiumIcon Icon={Brain} gradient="from-indigo-500 to-purple-600" size="sm" glow="indigo" />
                      Pathophysiology
                    </h4>
                    <p className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed">{reportResult.pathophysiology}</p>
                  </motion.div>
                )}

                {reportResult.recommendations && (
                  <motion.div initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-6 shadow-soft space-y-4">
                    <h4 className="font-bold text-sm text-slate-900 dark:text-white border-b border-slate-100 dark:border-slate-800 pb-2 flex items-center gap-3">
                      <PremiumIcon Icon={ClipboardList} gradient="from-teal-500 to-emerald-600" size="sm" glow="teal" />
                      Action Plan
                    </h4>
                    <div className="space-y-3 text-xs">
                      {reportResult.recommendations.urgent_actions?.length > 0 && (
                        <div className="bg-rose-50 dark:bg-rose-900/20 p-4 rounded-2xl border border-rose-200 dark:border-rose-800/50">
                          <strong className="text-rose-800 dark:text-rose-300 flex items-center gap-1.5 mb-2"><AlertTriangle className="w-4 h-4 text-rose-500" strokeWidth={2.5} /> Urgent Actions</strong>
                          <ul className="list-disc pl-5 text-rose-900 dark:text-rose-200 space-y-1">{reportResult.recommendations.urgent_actions.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                      {reportResult.recommendations.further_tests?.length > 0 && (
                        <div className="bg-sky-50 dark:bg-sky-900/20 p-4 rounded-2xl border border-sky-200 dark:border-sky-800/50">
                          <strong className="text-sky-800 dark:text-sky-300 flex items-center gap-1.5 mb-2"><TestTube className="w-4 h-4 text-sky-500" strokeWidth={2.5} /> Further Tests</strong>
                          <ul className="list-disc pl-5 text-sky-900 dark:text-sky-200 space-y-1">{reportResult.recommendations.further_tests.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                      {reportResult.recommendations.specialty_consultation?.length > 0 && (
                        <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-2xl border border-indigo-200 dark:border-indigo-800/50">
                          <strong className="text-indigo-800 dark:text-indigo-300 flex items-center gap-1.5 mb-2"><UserPlus className="w-4 h-4 text-indigo-500" strokeWidth={2.5} /> Specialist Consultations</strong>
                          <ul className="list-disc pl-5 text-indigo-900 dark:text-indigo-200 space-y-1">{reportResult.recommendations.specialty_consultation.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                      {reportResult.recommendations.lifestyle_modifications?.length > 0 && (
                        <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-2xl border border-emerald-200 dark:border-emerald-800/50">
                          <strong className="text-emerald-800 dark:text-emerald-300 flex items-center gap-1.5 mb-2"><Leaf className="w-4 h-4 text-emerald-500" strokeWidth={2.5} /> Lifestyle</strong>
                          <ul className="list-disc pl-5 text-emerald-900 dark:text-emerald-200 space-y-1">{reportResult.recommendations.lifestyle_modifications.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </div>
            </div>
          )}
        </main>
      )}
    </div>
  );
}