import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { 
  Stethoscope, Send, AlertTriangle, BookOpen, Microscope, ShieldAlert, 
  RefreshCw, ExternalLink, Sparkles, Info, Activity, HeartPulse, 
  Award, FileText, UploadCloud, CheckCircle2, FileCheck, MessageSquare, 
  FlaskConical, TrendingUp, TrendingDown, Minus, ClipboardList, 
  Stethoscope as ScopeIcon, HelpCircle, X, Trash2, Download, Filter, 
  Compass, AlertOctagon, UserPlus, Leaf, TestTube, User, Calendar, 
  Building, ChevronDown, ChevronUp, BarChart2
} from 'lucide-react';

const API_BASE = "http://127.0.0.1:5000";

const SUGGESTED_PROMPTS = [
  "What are the diagnostic criteria, risk factors, and treatment protocols for Type 2 Diabetes?",
  "Evaluate recent clinical trial evidence for Tirzepatide versus Semaglutide in NASH and kidney disease.",
  "I have severe chest pain and difficulty breathing.",
  "How do I bake chocolate chip cookies?"
];

function StatusBadge({ status }) {
  const s = (status || "UNKNOWN").toUpperCase();
  const config = {
    HIGH: { bg: "bg-rose-100", text: "text-rose-700", border: "border-rose-200", icon: TrendingUp },
    LOW: { bg: "bg-amber-100", text: "text-amber-800", border: "border-amber-200", icon: TrendingDown },
    NORMAL: { bg: "bg-emerald-100", text: "text-emerald-700", border: "border-emerald-200", icon: Minus }
  };
  const c = config[s] || { bg: "bg-slate-100", text: "text-slate-600", border: "border-slate-200", icon: Minus };
  const Icon = c.icon;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold ${c.bg} ${c.text} border ${c.border}`}>
      <Icon className="w-3 h-3" /> {s}
    </span>
  );
}

function RiskBanner({ risk }) {
  const r = (risk || "MODERATE").toUpperCase();
  const styles = {
    HIGH: "bg-rose-50 border-rose-300 text-rose-800",
    MODERATE: "bg-amber-50 border-amber-300 text-amber-900",
    LOW: "bg-emerald-50 border-emerald-300 text-emerald-800"
  };
  return (
    <div className={`rounded-2xl border px-4 py-3 ${styles[r] || styles.MODERATE}`}>
      <div className="flex items-center gap-2 font-bold text-sm">
        <ShieldAlert className="w-4 h-4" /> Overall Risk Level: {r}
      </div>
    </div>
  );
}

// Server-side PDF export function
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

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  
  // Chat state with localStorage persistence
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem('medicare_chat_history');
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showTransparencyModal, setShowTransparencyModal] = useState(false);
  const [openEvalId, setOpenEvalId] = useState(null);

  // Report state
  const [reportFile, setReportFile] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportResult, setReportResult] = useState(null);
  const [reportError, setReportError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [labFilter, setLabFilter] = useState('ALL');

  // Server health
  const [serverHealth, setServerHealth] = useState({ online: false, chunks: 0, docs: 0 });
  const messagesEndRef = useRef(null);

  // Persist chat to localStorage
  useEffect(() => {
    try { localStorage.setItem('medicare_chat_history', JSON.stringify(messages)); } catch {}
  }, [messages]);

  // Auto-scroll
  const scrollToBottom = () => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); };
  useEffect(() => { if (activeTab === 'chat') scrollToBottom(); }, [messages, loading, activeTab]);

  // Health check on mount
  useEffect(() => { checkHealth(); }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setServerHealth({ online: true, chunks: data.total_chunks_indexed || 24520, docs: data.total_documents || 125 });
      } else { setServerHealth({ online: false, chunks: 0, docs: 0 }); }
    } catch { setServerHealth({ online: false, chunks: 0, docs: 0 }); }
  };

  const clearChatHistory = () => { setMessages([]); localStorage.removeItem('medicare_chat_history'); };

  // Chat handler WITH conversation memory
  const handleSend = async (textToSend) => {
    const questionText = textToSend || input;
    if (!questionText.trim() || loading) return;

    const userMessage = {
      id: Date.now(), sender: 'user', text: questionText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage]);
    if (!textToSend) setInput('');
    setLoading(true);

    try {
      // Build conversation history from last 6 messages
      const allMsgs = [...messages, userMessage];
      const chatHistory = allMsgs.slice(-6).map(m => ({
        role: m.sender === 'user' ? 'user' : 'assistant',
        content: (m.text || '').substring(0, 500)
      }));

      const response = await fetch(`${API_BASE}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: questionText, chat_history: chatHistory })
      });

      if (!response.ok) throw new Error(`Server status ${response.status}`);
      const data = await response.json();

      setMessages(prev => [...prev, {
        id: Date.now() + 1, sender: 'ai', text: data.answer, mode: data.mode,
        accuracyScore: data.accuracy_score || 0, evaluation: data.evaluation || null,
        isMedical: data.is_medical, isEmergency: data.is_emergency,
        emergencyMessage: data.emergency_message, pubmedSources: data.pubmed_sources || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    } catch {
      setMessages(prev => [...prev, {
        id: Date.now() + 1, sender: 'ai',
        text: `⚠️ **Connection Error:** Could not connect to MediCare AI Backend.`,
        mode: 'error', accuracyScore: 0, evaluation: null, isMedical: true, isEmergency: false,
        pubmedSources: [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    } finally { setLoading(false); }
  };

  // Report upload handler
  const handleReportUpload = async (e) => {
    e.preventDefault();
    if (!reportFile || reportLoading) return;
    setReportLoading(true); setReportError(''); setReportResult(null);
    const formData = new FormData();
    formData.append('file', reportFile);
    try {
      const response = await fetch(`${API_BASE}/api/analyze-report`, { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Failed to analyze report.');
      setReportResult(data);
    } catch (err) { setReportError(err.message || 'An error occurred.'); }
    finally { setReportLoading(false); }
  };

  const onDrop = (e) => { e.preventDefault(); setDragActive(false); const file = e.dataTransfer.files?.[0]; if (file) setReportFile(file); };
  const toggleEval = (id) => { setOpenEvalId(prev => (prev === id ? null : id)); };

  // Lab filter helpers
  const getFilteredLabs = () => {
    if (!reportResult?.lab_values) return [];
    const labs = reportResult.lab_values;
    if (labFilter === 'HIGH') return labs.filter(l => l.status === 'HIGH');
    if (labFilter === 'LOW') return labs.filter(l => l.status === 'LOW');
    if (labFilter === 'NORMAL') return labs.filter(l => l.status === 'NORMAL');
    if (labFilter === 'ABNORMAL') return labs.filter(l => l.status === 'HIGH' || l.status === 'LOW');
    return labs;
  };

  const getGroupedLabs = () => {
    const filtered = getFilteredLabs();
    const groups = {};
    filtered.forEach(lab => { const sys = lab.system || 'General / Other'; if (!groups[sys]) groups[sys] = []; groups[sys].push(lab); });
    return groups;
  };

  const labCounts = {
    total: reportResult?.lab_values?.length || 0,
    high: reportResult?.lab_values?.filter(l => l.status === 'HIGH').length || 0,
    low: reportResult?.lab_values?.filter(l => l.status === 'LOW').length || 0,
    normal: reportResult?.lab_values?.filter(l => l.status === 'NORMAL').length || 0,
  };

  // ═══════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════

  return (
    <div className="flex flex-col h-screen bg-slate-50 text-slate-800">
      
      {/* ═══════════ HEADER BAR ═══════════ */}
      <header className="bg-white border-b border-slate-200 px-4 md:px-6 py-3 flex flex-wrap items-center justify-between gap-3 shadow-sm sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <div className="bg-sky-600 text-white p-2 rounded-xl shadow-sm"><Stethoscope className="w-6 h-6" /></div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg md:text-xl font-bold bg-gradient-to-r from-sky-700 to-indigo-700 bg-clip-text text-transparent">MediCare AI</h1>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-sky-100 text-sky-800 border border-sky-200">SaaS v2.0</span>
            </div>
            <p className="text-[10px] md:text-xs text-slate-500">Evidence-Based Clinical Knowledge & Lab Report Analysis</p>
          </div>
        </div>

        <div className="flex items-center bg-slate-100 p-1 rounded-xl border border-slate-200">
          <button onClick={() => setActiveTab('chat')} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${activeTab === 'chat' ? 'bg-white text-sky-700 shadow-sm' : 'text-slate-600 hover:text-slate-900'}`}>
            <MessageSquare className="w-3.5 h-3.5" /> AI Clinical Chat
          </button>
          <button onClick={() => setActiveTab('report')} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${activeTab === 'report' ? 'bg-white text-sky-700 shadow-sm' : 'text-slate-600 hover:text-slate-900'}`}>
            <FlaskConical className="w-3.5 h-3.5" /> Lab Report Analyzer
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button onClick={() => setShowTransparencyModal(true)} className="flex items-center gap-1 px-2.5 py-1.5 rounded-full text-[10px] font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition-colors">
            <HelpCircle className="w-3 h-3 text-sky-600" /> Transparency
          </button>
          <div className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-[10px] font-medium border ${serverHealth.online ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-amber-50 text-amber-700 border-amber-200'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${serverHealth.online ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`}></span>
            {serverHealth.online ? `Online (${serverHealth.chunks.toLocaleString()})` : 'Offline'}
            <button onClick={checkHealth} className="ml-0.5 hover:rotate-180 transition-transform duration-300"><RefreshCw className="w-2.5 h-2.5" /></button>
          </div>
        </div>
      </header>

      {/* ═══════════ TRANSPARENCY MODAL ═══════════ */}
      {showTransparencyModal && (
        <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-3xl max-w-xl w-full p-6 space-y-5 shadow-2xl border border-slate-200">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2 text-sky-800 font-bold text-lg"><Info className="w-5 h-5 text-sky-600" /> System Transparency</div>
              <button onClick={() => setShowTransparencyModal(false)} className="p-1 text-slate-400"><X className="w-5 h-5" /></button>
            </div>
            <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
              <div className="p-3 bg-sky-50 border border-sky-200 rounded-xl text-sky-900">
                <strong>What MediCare AI Is:</strong> A cloud-native decision support platform that retrieves context from 24,520 medical textbook paragraphs and live NIH PubMed papers.
              </div>
              <p><strong>Vector Retrieval Similarity:</strong> Measures mathematical cosine similarity between query and database. It is NOT doctor-verified accuracy.</p>
              <p><strong>BLEU & ROUGE Scores:</strong> Measure word-level overlap between retrieved textbook context and AI output. Low BLEU is expected in generative RAG (AI rephrases rather than copies).</p>
              <p><strong>Evidence Groundedness:</strong> Measures what percentage of medical terms in the AI answer exist directly in retrieved textbook chunks.</p>
              <p><strong>Preliminary Guidance Only:</strong> MediCare AI cannot issue medical diagnoses, prescriptions, or replace clinical judgment.</p>
            </div>
            <div className="text-right">
              <button onClick={() => setShowTransparencyModal(false)} className="bg-sky-600 text-white font-bold text-xs px-5 py-2.5 rounded-xl">I Understand</button>
            </div>
          </div>
        </div>
      )}

      {/* ═══════════ CHAT TAB ═══════════ */}
      {activeTab === 'chat' && (
        <>
          <main className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 max-w-5xl mx-auto w-full">
            
            {messages.length > 0 && (
              <div className="flex justify-end">
                <button onClick={clearChatHistory} className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-rose-600 bg-white px-3 py-1.5 rounded-lg border shadow-2xs">
                  <Trash2 className="w-3.5 h-3.5" /> Clear Chat
                </button>
              </div>
            )}

            {/* Welcome Screen */}
            {messages.length === 0 && (
              <div className="bg-white rounded-2xl p-6 md:p-8 border border-slate-200 shadow-sm text-center my-6 space-y-6">
                <div className="w-16 h-16 bg-sky-100 text-sky-600 rounded-2xl flex items-center justify-center mx-auto shadow-inner">
                  <HeartPulse className="w-9 h-9" />
                </div>
                <div className="max-w-xl mx-auto space-y-2">
                  <h2 className="text-2xl font-bold text-slate-900">Welcome to MediCare AI Assistant</h2>
                  <p className="text-sm text-slate-600 leading-relaxed">
                    Query over <span className="font-semibold text-sky-700">24,520+ medical textbook chunks</span> combined with <span className="font-semibold text-indigo-700">Live NIH PubMed Research</span>.
                  </p>
                </div>

                {/* Feature Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-left pt-2">
                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                    <div className="flex items-center gap-2 text-sky-700 font-semibold text-sm"><BookOpen className="w-4 h-4" /> Grounded Textbooks</div>
                    <p className="text-xs text-slate-500">Every response is grounded in verified medical textbooks with exact page references.</p>
                  </div>
                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                    <div className="flex items-center gap-2 text-indigo-700 font-semibold text-sm"><Microscope className="w-4 h-4" /> Live PubMed Research</div>
                    <p className="text-xs text-slate-500">Fetches 2024-2025 peer-reviewed trial abstracts directly from NIH PubMed.</p>
                  </div>
                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                    <div className="flex items-center gap-2 text-rose-700 font-semibold text-sm"><ShieldAlert className="w-4 h-4" /> Emergency Protection</div>
                    <p className="text-xs text-slate-500">Automatic detection of critical symptoms with regional helpline protocols.</p>
                  </div>
                </div>

                {/* Sample Prompts */}
                <div className="pt-4 text-left">
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-sky-600" /> Click a sample clinical prompt:
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                    {SUGGESTED_PROMPTS.map((prompt, idx) => (
                      <button key={idx} onClick={() => handleSend(prompt)} className="p-3 text-xs text-slate-700 bg-white hover:bg-sky-50 hover:border-sky-300 border border-slate-200 rounded-xl transition-all text-left shadow-2xs hover:shadow flex items-start justify-between group">
                        <span>{prompt}</span>
                        <Send className="w-3.5 h-3.5 text-slate-400 group-hover:text-sky-600 transition-colors shrink-0 ml-2 mt-0.5" />
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Chat Messages */}
            {messages.map((msg) => (
              <div key={msg.id} className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'} space-y-2`}>
                {msg.sender === 'user' ? (
                  <div className="bg-sky-600 text-white rounded-2xl rounded-tr-none px-5 py-3.5 max-w-2xl shadow-sm text-sm leading-relaxed">
                    {msg.text}
                    <div className="text-[10px] text-sky-100 mt-1 text-right">{msg.timestamp}</div>
                  </div>
                ) : (
                  <div className="bg-white rounded-2xl border border-slate-200 p-5 md:p-6 max-w-4xl w-full shadow-sm space-y-4">
                    
                    {/* Emergency Banner */}
                    {msg.isEmergency && (
                      <div className="bg-rose-50 border-2 border-rose-500 rounded-xl p-4 text-rose-900 space-y-2">
                        <div className="flex items-center gap-2 font-bold text-base">
                          <AlertTriangle className="w-5 h-5 text-rose-600 animate-bounce" /> MEDICAL EMERGENCY ALERT DETECTED
                        </div>
                        <p className="text-xs text-rose-800 leading-relaxed whitespace-pre-line font-medium">{msg.emergencyMessage}</p>
                      </div>
                    )}

                    {/* AI Header with Evaluation Score */}
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
                      <div className="flex items-center gap-2">
                        <div className="w-7 h-7 bg-sky-100 text-sky-700 rounded-lg flex items-center justify-center font-bold text-xs">AI</div>
                        <span className="font-semibold text-sm text-slate-900">MediCare AI Clinical Assistant</span>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        {/* Evaluation Quality Score Badge */}
                        {msg.evaluation && (
                          <button onClick={() => toggleEval(msg.id)}
                            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold bg-indigo-50 text-indigo-700 border border-indigo-200 hover:bg-indigo-100 transition-colors shadow-2xs">
                            <BarChart2 className="w-3.5 h-3.5 text-indigo-600" />
                            <span>Quality: {msg.evaluation.composite_score}/100</span>
                            {openEvalId === msg.id ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Expandable Evaluation Metrics Card */}
                    {msg.evaluation && openEvalId === msg.id && (
                      <div className="bg-indigo-50/70 border border-indigo-200 rounded-2xl p-4 text-xs space-y-3">
                        <div className="flex items-center justify-between font-bold text-indigo-950">
                          <span className="flex items-center gap-1.5"><Award className="w-4 h-4 text-indigo-600" /> RAG Quality Evaluation (NLP Benchmarks)</span>
                          <span className="px-2 py-0.5 rounded bg-indigo-200 text-indigo-900 text-[10px]">{msg.evaluation.grade}</span>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                          {[
                            { label: "Evidence Groundedness", value: msg.evaluation.groundedness },
                            { label: "Semantic Similarity", value: msg.evaluation.semantic_similarity },
                            { label: "BLEU-4 Score", value: msg.evaluation.bleu_score },
                            { label: "ROUGE-L Score", value: msg.evaluation.rouge_score }
                          ].map((metric, i) => (
                            <div key={i} className="bg-white p-2.5 rounded-xl border border-indigo-100 text-center">
                              <span className="text-[10px] text-slate-500 font-semibold uppercase block">{metric.label}</span>
                              <span className="text-base font-extrabold text-indigo-700">{typeof metric.value === 'number' ? metric.value.toFixed(1) : metric.value}%</span>
                            </div>
                          ))}
                        </div>
                        <p className="text-[10px] text-indigo-800/80 italic">
                          Calculated using string overlap, n-gram recall, and vector similarity between retrieved textbook chunks and AI output.
                        </p>
                      </div>
                    )}

                    {/* Answer Body (Markdown) */}
                    <div className="prose prose-slate prose-sm max-w-none text-slate-800 leading-relaxed">
                      <ReactMarkdown>{msg.text}</ReactMarkdown>
                    </div>

                    {/* PubMed Sources (kept in chat since these are live research) */}
                    {msg.pubmedSources?.length > 0 && (
                      <div className="pt-3 border-t border-slate-100 space-y-2">
                        <div className="flex items-center gap-2 text-xs font-bold text-indigo-800 uppercase tracking-wider">
                          <Microscope className="w-4 h-4 text-indigo-600" /> Live NIH PubMed Research ({msg.pubmedSources.length})
                        </div>
                        <div className="space-y-2">
                          {msg.pubmedSources.map((paper, idx) => (
                            <div key={idx} className="bg-indigo-50/50 p-3 rounded-lg border border-indigo-100 text-xs space-y-1">
                              <a href={paper.url} target="_blank" rel="noopener noreferrer" className="font-bold text-indigo-900 hover:text-indigo-600 flex items-center gap-1">
                                <span>{paper.title}</span>
                                <ExternalLink className="w-3 h-3 text-indigo-400 shrink-0" />
                              </a>
                              <div className="text-[11px] text-indigo-700">
                                Journal: <strong>{paper.journal} ({paper.year})</strong> | PMID: <strong className="font-mono bg-indigo-100 px-1 rounded">{paper.pmid}</strong>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="text-[10px] text-slate-400 text-right pt-1">{msg.timestamp}</div>
                  </div>
                )}
              </div>
            ))}

            {/* Loading */}
            {loading && (
              <div className="bg-white rounded-2xl border border-slate-200 p-5 max-w-md shadow-sm space-y-3">
                <div className="flex items-center gap-3 text-sky-700 font-semibold text-sm">
                  <Activity className="w-5 h-5 animate-spin text-sky-600" /> Searching Knowledge Base & NIH PubMed...
                </div>
                <div className="space-y-2">
                  <div className="h-2.5 bg-slate-100 rounded-full w-3/4 animate-pulse"></div>
                  <div className="h-2.5 bg-slate-100 rounded-full w-1/2 animate-pulse"></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </main>

          {/* Input Bar */}
          <footer className="bg-white border-t border-slate-200 p-4 sticky bottom-0 z-20">
            <div className="max-w-5xl mx-auto space-y-1.5">
              <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="flex items-center gap-2">
                <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask a medical question (follow-ups work too)..." disabled={loading}
                  className="flex-1 bg-slate-50 border border-slate-300 focus:border-sky-500 focus:bg-white text-slate-900 placeholder-slate-400 rounded-xl px-4 py-3 text-sm outline-none transition-all disabled:opacity-50" />
                <button type="submit" disabled={loading || !input.trim()}
                  className="bg-sky-600 hover:bg-sky-700 active:bg-sky-800 text-white font-medium px-5 py-3 rounded-xl transition-all shadow-sm flex items-center gap-2 disabled:opacity-50 text-sm">
                  Send <Send className="w-4 h-4" />
                </button>
              </form>
              <div className="flex items-center justify-between text-[10px] text-slate-400 px-1">
                <span>MediCare AI provides preliminary guidance only. Not a substitute for clinical diagnosis.</span>
                <span className="font-semibold text-slate-500">24,520 Chunks + Live PubMed + Conversation Memory</span>
              </div>
            </div>
          </footer>
        </>
      )}

      {/* ═══════════ REPORT ANALYZER TAB ═══════════ */}
      {activeTab === 'report' && (
        <main className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 max-w-6xl mx-auto w-full">
          {/* Hero */}
          <div className="bg-gradient-to-br from-sky-600 via-sky-700 to-indigo-700 rounded-3xl p-6 md:p-8 text-white shadow-lg">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="space-y-2">
                <div className="inline-flex items-center gap-2 bg-white/15 px-3 py-1 rounded-full text-xs font-semibold">
                  <FlaskConical className="w-3.5 h-3.5" /> Phase 2 Module
                </div>
                <h2 className="text-2xl md:text-3xl font-bold">Blood Test & Lab Report Analyzer</h2>
                <p className="text-sky-100 text-sm max-w-2xl">Upload a digital or scanned lab PDF. MediCare AI categorizes values by organ systems, flags abnormalities, and synthesizes clinical pathophysiology.</p>
              </div>
              <div className="bg-white/10 backdrop-blur rounded-2xl px-4 py-3 text-xs space-y-1 border border-white/20">
                <div className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4" /> Color-coded lab cards</div>
                <div className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4" /> Differential diagnoses</div>
                <div className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4" /> 4-tier recommendations</div>
                <div className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4" /> PDF report export</div>
              </div>
            </div>
          </div>

          {/* Upload Card */}
          <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 space-y-5">
            <form onSubmit={handleReportUpload} className="space-y-4">
              <div onDragOver={(e) => { e.preventDefault(); setDragActive(true); }} onDragLeave={() => setDragActive(false)} onDrop={onDrop}
                className={`border-2 border-dashed rounded-3xl p-10 text-center transition-all ${dragActive ? 'border-sky-500 bg-sky-50' : 'border-slate-300 bg-slate-50 hover:border-sky-400'}`}>
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex items-center justify-center">
                  <UploadCloud className="w-8 h-8 text-sky-600" />
                </div>
                <h3 className="text-lg font-bold text-slate-900 mb-1">Drop your lab PDF here</h3>
                <p className="text-sm text-slate-500 mb-4">or click below to browse files</p>
                <label className="cursor-pointer inline-block">
                  <span className="bg-sky-600 hover:bg-sky-700 text-white text-sm font-bold px-5 py-2.5 rounded-xl transition-all inline-flex items-center gap-2 shadow-sm">
                    <FileText className="w-4 h-4" /> Select Lab Report PDF
                  </span>
                  <input type="file" accept=".pdf" onChange={(e) => setReportFile(e.target.files?.[0] || null)} className="hidden" />
                </label>
                <div className="mt-4 text-xs text-slate-500">
                  {reportFile ? (
                    <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-sky-100 text-sky-800 font-semibold border border-sky-200">
                      <FileCheck className="w-4 h-4" /> {reportFile.name} ({(reportFile.size / 1024).toFixed(1)} KB)
                    </span>
                  ) : "Supports digital text & scanned photo lab PDFs"}
                </div>
              </div>

              {reportError && (
                <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-xl flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" /> {reportError}
                </div>
              )}

              <button type="submit" disabled={!reportFile || reportLoading}
                className="w-full bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-700 hover:to-indigo-700 disabled:opacity-50 text-white font-bold text-sm py-3.5 rounded-2xl transition-all shadow-sm flex items-center justify-center gap-2">
                {reportLoading ? (<><Activity className="w-4 h-4 animate-spin" /> Extracting Lab Values & Analyzing Pathophysiology...</>) : (<><Sparkles className="w-4 h-4" /> Analyze Report with RAG Engine</>)}
              </button>
            </form>
          </div>

          {/* ═══════════ ANALYSIS DASHBOARD ═══════════ */}
          {reportResult && !reportLoading && (
            <div className="space-y-6">
              
              {/* Critical Alerts */}
              {reportResult.critical_alerts?.length > 0 && (
                <div className="bg-rose-50 border-2 border-rose-500 rounded-2xl p-5 text-rose-900 space-y-2 shadow-sm">
                  <div className="flex items-center gap-2 font-bold text-base text-rose-700">
                    <AlertOctagon className="w-6 h-6 text-rose-600 animate-bounce" /> CRITICAL LAB ALERTS DETECTED
                  </div>
                  <ul className="list-disc pl-5 text-xs font-medium space-y-1 text-rose-800">
                    {reportResult.critical_alerts.map((alert, idx) => <li key={idx}>{alert}</li>)}
                  </ul>
                </div>
              )}

              {/* Summary + Patient + Export */}
              <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-4">
                  <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                    <CheckCircle2 className="text-emerald-600" /> {reportResult.file_name}
                  </h3>
                  <div className="flex items-center gap-2">
                    <RiskBanner risk={reportResult.risk_level} />
                    <button onClick={() => exportClinicalPDF(reportResult)}
                      className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl text-xs font-bold bg-sky-600 hover:bg-sky-700 text-white transition-all shadow-md active:scale-95">
                      <Download className="w-4 h-4" /> Export PDF Report
                    </button>
                  </div>
                </div>

                {/* Patient Metadata */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-50 border border-slate-200 rounded-2xl p-4 text-xs">
                  <div><span className="text-slate-400 font-semibold block text-[10px] uppercase">Patient</span><span className="font-bold text-slate-800 text-sm flex items-center gap-1"><User className="w-3.5 h-3.5 text-sky-600" />{reportResult.patient_name || 'Patient'}</span></div>
                  <div><span className="text-slate-400 font-semibold block text-[10px] uppercase">Age / Sex</span><span className="font-bold text-slate-800 text-sm">{reportResult.patient_age_gender || 'N/A'}</span></div>
                  <div><span className="text-slate-400 font-semibold block text-[10px] uppercase">Laboratory</span><span className="font-bold text-slate-800 text-sm truncate flex items-center gap-1"><Building className="w-3.5 h-3.5 text-indigo-600" />{reportResult.lab_name || 'Lab'}</span></div>
                  <div><span className="text-slate-400 font-semibold block text-[10px] uppercase">Date</span><span className="font-bold text-slate-800 text-sm flex items-center gap-1"><Calendar className="w-3.5 h-3.5 text-emerald-600" />{reportResult.report_date || 'N/A'}</span></div>
                </div>

                {reportResult.summary && (
                  <div className="bg-slate-50 border rounded-2xl p-4">
                    <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1 flex items-center gap-1"><ClipboardList className="w-3.5 h-3.5" /> Clinical Summary</div>
                    <p className="text-sm font-medium text-slate-800 leading-relaxed">{reportResult.summary}</p>
                  </div>
                )}
              </div>

              {/* Lab Values with Filters & System Grouping */}
              {reportResult.lab_values?.length > 0 && (
                <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 space-y-5">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-4">
                    <div>
                      <h4 className="text-base font-bold text-slate-900 flex items-center gap-2"><FlaskConical className="w-5 h-5 text-sky-600" /> Extracted Lab Values ({labCounts.total})</h4>
                      <p className="text-xs text-slate-500">Grouped by organ systems with real-time status filtering.</p>
                    </div>
                    <div className="flex items-center gap-2 text-xs font-bold">
                      <span className="px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 border">Total: {labCounts.total}</span>
                      <span className="px-2.5 py-1 rounded-full bg-rose-100 text-rose-700 border border-rose-200">🔴 {labCounts.high}</span>
                      <span className="px-2.5 py-1 rounded-full bg-amber-100 text-amber-800 border border-amber-200">🟡 {labCounts.low}</span>
                      <span className="px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-700 border border-emerald-200">🟢 {labCounts.normal}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 overflow-x-auto pb-1 text-xs font-bold">
                    <span className="text-slate-400 flex items-center gap-1"><Filter className="w-3.5 h-3.5"/> Filter:</span>
                    {['ALL', 'ABNORMAL', 'HIGH', 'LOW', 'NORMAL'].map((f) => (
                      <button key={f} onClick={() => setLabFilter(f)}
                        className={`px-3 py-1.5 rounded-xl border transition-all ${labFilter === f ? 'bg-sky-600 text-white border-sky-600' : 'bg-slate-50 text-slate-600 hover:bg-slate-100 border-slate-200'}`}>
                        {f === 'ALL' ? 'All Tests' : f === 'ABNORMAL' ? '⚠️ Abnormal' : f}
                      </button>
                    ))}
                  </div>

                  <div className="space-y-6 pt-2">
                    {Object.keys(getGroupedLabs()).length === 0 ? (
                      <div className="text-center py-8 text-xs text-slate-400">No lab values match filter ({labFilter}).</div>
                    ) : (
                      Object.entries(getGroupedLabs()).map(([systemName, labs]) => (
                        <div key={systemName} className="space-y-3">
                          <h5 className="text-xs font-bold uppercase tracking-wider text-sky-800 bg-sky-50/70 border border-sky-100 px-3 py-1.5 rounded-xl inline-block">
                            🫀 {systemName} ({labs.length})
                          </h5>
                          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                            {labs.map((lab, idx) => {
                              const status = (lab.status || "UNKNOWN").toUpperCase();
                              const border = status === "HIGH" ? "border-rose-200 bg-rose-50/40" : status === "LOW" ? "border-amber-200 bg-amber-50/40" : status === "NORMAL" ? "border-emerald-200 bg-emerald-50/30" : "border-slate-200 bg-white";
                              return (
                                <div key={idx} className={`rounded-xl border p-3.5 shadow-2xs ${border}`}>
                                  <div className="flex items-start justify-between gap-1 mb-1">
                                    <div className="text-xs font-bold text-slate-800 leading-snug">{lab.test}</div>
                                    <StatusBadge status={status} />
                                  </div>
                                  <div className="text-xl font-extrabold text-slate-900">{lab.result} <span className="text-xs font-semibold text-slate-500">{lab.unit}</span></div>
                                  <div className="text-[10px] text-slate-500 mt-1">Ref: {lab.reference_range || "N/A"}</div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}

              {/* Differentials */}
              {reportResult.differential_considerations?.length > 0 && (
                <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 space-y-3">
                  <div className="border-b pb-3">
                    <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2"><Compass className="w-4 h-4 text-indigo-600" /> Possible Clinical Considerations (Differentials)</h4>
                    <p className="text-[11px] text-slate-500">Possible disease pattern correlations — not a final diagnosis.</p>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                    {reportResult.differential_considerations.map((item, idx) => (
                      <div key={idx} className="rounded-2xl border border-indigo-100 bg-indigo-50/30 p-4 space-y-1">
                        <div className="font-bold text-sm text-indigo-950 flex items-center gap-2">
                          <span className="w-5 h-5 rounded-full bg-indigo-100 text-indigo-800 text-[11px] flex items-center justify-center font-extrabold">{idx + 1}</span>
                          {item.title}
                        </div>
                        <p className="text-xs text-indigo-900/80 leading-relaxed pl-7">{item.rationale}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Pathophysiology & Recommendations */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {reportResult.pathophysiology && (
                  <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 space-y-3">
                    <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2"><ScopeIcon className="text-indigo-600 w-4 h-4"/> Pathophysiology & Inter-Organ Connections</h4>
                    <p className="text-xs text-slate-700 leading-relaxed whitespace-pre-line">{reportResult.pathophysiology}</p>
                  </div>
                )}

                {reportResult.recommendations && (
                  <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 space-y-4">
                    <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2 border-b pb-3"><ClipboardList className="text-emerald-600 w-4 h-4"/> Recommended Next Steps</h4>
                    <div className="space-y-3 text-xs">
                      {reportResult.recommendations.urgent_actions?.length > 0 && (
                        <div className="bg-rose-50 p-3 rounded-xl border border-rose-200">
                          <div className="font-bold text-rose-800 flex items-center gap-1.5 mb-1"><AlertTriangle className="w-3.5 h-3.5 text-rose-600" /> Urgent Actions</div>
                          <ul className="list-disc pl-5 text-rose-900 space-y-0.5">{reportResult.recommendations.urgent_actions.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                      {reportResult.recommendations.further_tests?.length > 0 && (
                        <div className="bg-sky-50 p-3 rounded-xl border border-sky-200">
                          <div className="font-bold text-sky-800 flex items-center gap-1.5 mb-1"><TestTube className="w-3.5 h-3.5 text-sky-600" /> Further Tests</div>
                          <ul className="list-disc pl-5 text-sky-900 space-y-0.5">{reportResult.recommendations.further_tests.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                      {reportResult.recommendations.specialty_consultation?.length > 0 && (
                        <div className="bg-indigo-50 p-3 rounded-xl border border-indigo-200">
                          <div className="font-bold text-indigo-800 flex items-center gap-1.5 mb-1"><UserPlus className="w-3.5 h-3.5 text-indigo-600" /> Specialty Consultations</div>
                          <ul className="list-disc pl-5 text-indigo-900 space-y-0.5">{reportResult.recommendations.specialty_consultation.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                      {reportResult.recommendations.lifestyle_modifications?.length > 0 && (
                        <div className="bg-emerald-50 p-3 rounded-xl border border-emerald-200">
                          <div className="font-bold text-emerald-800 flex items-center gap-1.5 mb-1"><Leaf className="w-3.5 h-3.5 text-emerald-600" /> Lifestyle Modifications</div>
                          <ul className="list-disc pl-5 text-emerald-900 space-y-0.5">{reportResult.recommendations.lifestyle_modifications.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Disclaimer */}
              <div className="text-[11px] text-slate-500 bg-slate-100 border border-slate-200 rounded-2xl px-4 py-3 text-center">
                <strong>Disclaimer:</strong> This AI report analysis is for decision-support and educational purposes only. All parameters must be interpreted by a qualified physician alongside patient history.
              </div>
            </div>
          )}
        </main>
      )}
    </div>
  );
}