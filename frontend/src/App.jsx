import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import mermaid from 'mermaid';
import { 
  Stethoscope, Send, AlertTriangle, BookOpen, Microscope, ShieldAlert, 
  RefreshCw, ExternalLink, Sparkles, Info, Activity, HeartPulse, 
  Award, FileText, UploadCloud, CheckCircle2, FileCheck, MessageSquare, 
  FlaskConical, TrendingUp, TrendingDown, Minus, ClipboardList, 
  Stethoscope as ScopeIcon, HelpCircle, X, Trash2, Download, Filter, 
  Compass, AlertOctagon, UserPlus, Leaf, TestTube, User, Calendar, 
  Building, ChevronDown, ChevronUp, BarChart2, Layers, GitBranch
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || "https://medicare-ai-aara-backend.onrender.com";

mermaid.initialize({
  startOnLoad: false,
  theme: 'neutral',
  securityLevel: 'loose',
  fontFamily: 'ui-sans-serif, system-ui, sans-serif'
});

function MermaidDiagram({ chart }) {
  const [svgContent, setSvgContent] = useState('');
  const [renderError, setRenderError] = useState(false);
  const containerRef = useRef(null);
  const uniqueId = useRef(`mermaid-${Math.random().toString(36).substring(2, 11)}`);

  useEffect(() => {
    if (!chart) return;
    let isMounted = true;

    try {
      mermaid.render(uniqueId.current, chart.trim())
        .then(({ svg }) => {
          if (isMounted) {
            setSvgContent(svg);
            setRenderError(false);
          }
        })
        .catch(() => {
          if (isMounted) setRenderError(true);
        });
    } catch {
      if (isMounted) setRenderError(true);
    }

    return () => { isMounted = false; };
  }, [chart]);

  if (renderError || !svgContent) {
    return (
      <pre className="my-2 p-3 bg-slate-900 text-slate-100 rounded-xl text-xs overflow-x-auto font-mono">
        {chart}
      </pre>
    );
  }

  return (
    <div className="my-3 p-3 md:p-4 bg-gradient-to-b from-sky-50/40 to-slate-50 border border-sky-100 rounded-2xl shadow-2xs overflow-x-auto flex flex-col items-center">
      <div className="w-full flex items-center justify-between text-[10px] font-bold text-sky-800 uppercase tracking-wider mb-2 border-b border-sky-100 pb-1.5">
        <span className="flex items-center gap-1"><GitBranch className="w-3.5 h-3.5 text-sky-600" /> Clinical Decision Pathway Algorithm</span>
        <span className="text-slate-400 font-normal">Interactive Flowchart</span>
      </div>
      <div 
        ref={containerRef}
        className="w-full overflow-x-auto flex justify-center py-1"
        dangerouslySetInnerHTML={{ __html: svgContent }} 
      />
    </div>
  );
}

const SUGGESTED_PROMPTS = [
  "What is the step-by-step diagnostic and management algorithm for Type 2 Diabetes?",
  "Evaluate recent clinical trial evidence for Tirzepatide versus Semaglutide in NASH and CKD.",
  "What is the first-line and second-line pharmacotherapy for Community-Acquired Pneumonia?",
  "I have severe crushing chest pain radiating to the jaw with shortness of breath."
];

function StatusBadge({ status }) {
  const s = (status || "UNKNOWN").toUpperCase();
  const config = {
    HIGH: { bg: "bg-rose-500/10 text-rose-700 border-rose-300/80", icon: TrendingUp },
    LOW: { bg: "bg-amber-500/10 text-amber-800 border-amber-300/80", icon: TrendingDown },
    NORMAL: { bg: "bg-emerald-500/10 text-emerald-700 border-emerald-300/80", icon: Minus }
  };
  const c = config[s] || { bg: "bg-slate-100 text-slate-600 border-slate-200", icon: Minus };
  const Icon = c.icon;
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border class-config">
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border ${c.bg}`}>
        <Icon className="w-3 h-3" /> {s}
      </span>
    </span>
  );
}

function PrevalenceBadge({ prevalence }) {
  const p = (prevalence || "COMMON").toUpperCase();
  const config = {
    "COMMON": { bg: "bg-emerald-100 text-emerald-800 border-emerald-300", label: "COMMON (MOST LIKELY)" },
    "LESS COMMON": { bg: "bg-amber-100 text-amber-900 border-amber-300", label: "LESS COMMON" },
    "RARE": { bg: "bg-purple-100 text-purple-900 border-purple-300", label: "RARE / ATYPICAL" }
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
    HIGH: "bg-gradient-to-r from-rose-50 to-red-100 border-rose-300 text-rose-900",
    MODERATE: "bg-gradient-to-r from-amber-50 to-orange-100 border-amber-300 text-amber-950",
    LOW: "bg-gradient-to-r from-emerald-50 to-teal-100 border-emerald-300 text-emerald-950"
  };
  return (
    <div className={`rounded-xl border px-3.5 py-2 shadow-xs ${styles[r] || styles.MODERATE}`}>
      <div className="flex items-center gap-1.5 font-bold text-xs md:text-sm">
        <ShieldAlert className="w-4 h-4 shrink-0 text-current" /> Risk Triage: {r}
      </div>
    </div>
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

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  
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

  const [reportFile, setReportFile] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportResult, setReportResult] = useState(null);
  const [reportError, setReportError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [labFilter, setLabFilter] = useState('ALL');

  const [serverHealth, setServerHealth] = useState({ online: false, chunks: 0, docs: 0 });
  const messagesEndRef = useRef(null);

  useEffect(() => {
    try { localStorage.setItem('medicare_chat_history', JSON.stringify(messages)); } catch {}
  }, [messages]);

  const scrollToBottom = () => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); };
  useEffect(() => { if (activeTab === 'chat') scrollToBottom(); }, [messages, loading, activeTab]);

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
        text: `⚠️ **Connection Notice:** Could not connect to MediCare AI Backend. Please verify server status.`,
        mode: 'error', accuracyScore: 0, evaluation: null, isMedical: true, isEmergency: false,
        pubmedSources: [],
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
      if (!response.ok) throw new Error(data.error || 'Failed to analyze report.');
      setReportResult(data);
    } catch (err) { setReportError(err.message || 'An error occurred.'); }
    finally { setReportLoading(false); }
  };

  const onDrop = (e) => { e.preventDefault(); setDragActive(false); const file = e.dataTransfer.files?.[0]; if (file) setReportFile(file); };
  const toggleEval = (id) => { setOpenEvalId(prev => (prev === id ? null : id)); };

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

  return (
    <div className="flex flex-col min-h-screen bg-slate-50 text-slate-800 font-sans antialiased">
      
      {/* ═══════════ HEADER ═══════════ */}
      <header className="bg-white/95 backdrop-blur-md border-b border-slate-200/80 px-3 md:px-6 py-2.5 flex flex-wrap items-center justify-between gap-2 shadow-xs sticky top-0 z-30">
        <div className="flex items-center gap-2.5">
          <div className="bg-gradient-to-tr from-sky-600 via-teal-600 to-indigo-600 text-white p-2 rounded-xl shadow-sm shadow-sky-500/20 animate-in zoom-in duration-300">
            <Stethoscope className="w-5 h-5 md:w-6 md:h-6" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <h1 className="text-base md:text-lg font-extrabold bg-gradient-to-r from-sky-700 via-indigo-700 to-teal-700 bg-clip-text text-transparent">MediCare AI</h1>
              <span className="text-[9px] md:text-[10px] font-bold px-2 py-0.5 rounded-full bg-gradient-to-r from-sky-50 to-indigo-50 text-sky-800 border border-sky-200">Clinical Suite</span>
            </div>
            <p className="text-[9px] md:text-[11px] text-slate-500 hidden sm:block">Textbook RAG Grounding + PubMed Research + Lab Pathophysiology</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center bg-slate-100 p-1 rounded-xl border border-slate-200/70 order-3 sm:order-2 w-full sm:w-auto justify-center">
          <button onClick={() => setActiveTab('chat')} 
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${activeTab === 'chat' ? 'bg-white text-sky-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'}`}>
            <MessageSquare className="w-3.5 h-3.5 text-sky-600" /> Clinical Assistant
          </button>
          <button onClick={() => setActiveTab('report')} 
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${activeTab === 'report' ? 'bg-white text-teal-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'}`}>
            <FlaskConical className="w-3.5 h-3.5 text-teal-600" /> Lab Report Analyzer
          </button>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-1.5 order-2 sm:order-3">
          <button onClick={() => setShowTransparencyModal(true)} 
            className="flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition-colors">
            <Info className="w-3 h-3 text-sky-600" /> <span className="hidden xs:inline">Transparency</span>
          </button>
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium border ${serverHealth.online ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-amber-50 text-amber-700 border-amber-200'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${serverHealth.online ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`}></span>
            <span>{serverHealth.online ? `Online (${serverHealth.chunks.toLocaleString()})` : 'Connecting...'}</span>
            <button onClick={checkHealth} className="hover:rotate-180 transition-transform duration-300"><RefreshCw className="w-2.5 h-2.5" /></button>
          </div>
        </div>
      </header>

      {/* ═══════════ TRANSPARENCY MODAL ═══════════ */}
      {showTransparencyModal && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs z-50 flex items-center justify-center p-3 md:p-4">
          <div className="bg-white rounded-3xl max-w-lg w-full p-5 md:p-6 space-y-4 shadow-2xl border border-slate-200 animate-in fade-in duration-200">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2 text-sky-900 font-bold text-base md:text-lg">
                <Info className="w-5 h-5 text-sky-600" /> Clinical System Transparency
              </div>
              <button onClick={() => setShowTransparencyModal(false)} className="p-1 text-slate-400 hover:text-slate-700"><X className="w-5 h-5" /></button>
            </div>
            <div className="space-y-2.5 text-xs text-slate-600 leading-relaxed max-h-[70vh] overflow-y-auto pr-1">
              <div className="p-3 bg-gradient-to-r from-sky-50 to-indigo-50 border border-sky-200 rounded-2xl text-sky-950 font-medium">
                <strong>Knowledge Base:</strong> Over 24,520 verified medical textbook chunks (Harrison's, Davidson's, Hutchison's, KD Tripathi, WHO, ICMR guidelines) combined with real-time NIH PubMed API querying.
              </div>
              <p>• <strong>Vector Similarity Score:</strong> Mathematical cosine similarity across 768 dimensions between clinical query and indexed textbook chunks.</p>
              <p>• <strong>Prevalence Ranking:</strong> Differential considerations are sorted from most common clinical conditions to rare presentations.</p>
              <p>• <strong>NLP Composite Quality:</strong> Automated groundedness, semantic similarity, BLEU-4, and ROUGE-L verification against source texts.</p>
              <p>• <strong>Safety Disclaimers:</strong> Designed for clinical educational decision-support. Not a substitute for licensed clinical judgment.</p>
            </div>
            <div className="text-right pt-2 border-t border-slate-100">
              <button onClick={() => setShowTransparencyModal(false)} className="bg-sky-600 hover:bg-sky-700 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow-xs">
                Understood
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ═══════════ CHAT TAB ═══════════ */}
      {activeTab === 'chat' && (
        <div className="flex-1 flex flex-col justify-between max-w-5xl mx-auto w-full p-3 md:p-6">
          <main className="space-y-4 md:space-y-6 pb-4">
            
            {messages.length > 0 && (
              <div className="flex justify-end animate-in fade-in duration-300">
                <button onClick={clearChatHistory} className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-rose-600 bg-white px-3 py-1.5 rounded-xl border border-slate-200 shadow-2xs transition-colors">
                  <Trash2 className="w-3.5 h-3.5" /> Clear History
                </button>
              </div>
            )}

            {/* Welcome Screen */}
            {messages.length === 0 && (
              <div className="bg-gradient-to-b from-white to-slate-50 rounded-3xl p-5 md:p-8 border border-slate-200/90 shadow-sm text-center my-2 md:my-4 space-y-5 animate-in slide-in-from-bottom duration-500">
                <div className="w-14 h-14 md:w-16 md:h-16 bg-gradient-to-tr from-sky-500 via-indigo-500 to-teal-500 text-white rounded-2xl flex items-center justify-center mx-auto shadow-md shadow-sky-500/20">
                  <HeartPulse className="w-8 h-8 md:w-9 md:h-9" />
                </div>
                <div className="max-w-xl mx-auto space-y-1.5">
                  <h2 className="text-xl md:text-2xl font-extrabold text-slate-900">Clinical Assistant & Medical RAG</h2>
                  <p className="text-xs md:text-sm text-slate-600 leading-relaxed">
                    Grounded in <span className="font-bold text-sky-700">24,520+ medical textbook chunks</span> and <span className="font-bold text-indigo-700">Live NIH PubMed Clinical Trials</span>.
                  </p>
                </div>

                {/* Feature Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 md:gap-3 text-left pt-2">
                  <div className="p-3.5 rounded-2xl bg-white border border-sky-100 shadow-2xs space-y-1 hover:border-sky-300 transition-colors">
                    <div className="flex items-center gap-1.5 text-sky-800 font-bold text-xs"><BookOpen className="w-4 h-4 text-sky-600" /> Textbook Grounding</div>
                    <p className="text-[11px] text-slate-500 leading-normal">Harrison's, Davidson's, Hutchison's, and MoHFW protocols with citations.</p>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-white border border-indigo-100 shadow-2xs space-y-1 hover:border-indigo-300 transition-colors">
                    <div className="flex items-center gap-1.5 text-indigo-800 font-bold text-xs"><Microscope className="w-4 h-4 text-indigo-600" /> Live PubMed Trials</div>
                    <p className="text-[11px] text-slate-500 leading-normal">Fetches 2024–2025 peer-reviewed trial abstracts with verified PMIDs.</p>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-white border border-rose-100 shadow-2xs space-y-1 hover:border-rose-300 transition-colors">
                    <div className="flex items-center gap-1.5 text-rose-800 font-bold text-xs"><ShieldAlert className="w-4 h-4 text-rose-600" /> Clinical Safety Guardrails</div>
                    <p className="text-[11px] text-slate-500 leading-normal">Immediate detection of critical emergencies with emergency dispatch alerts.</p>
                  </div>
                </div>

                {/* Suggested Prompts */}
                <div className="pt-2 text-left animate-in fade-in duration-500 delay-300">
                  <p className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2.5 flex items-center gap-1">
                    <Sparkles className="w-3.5 h-3.5 text-amber-500" /> Suggested Clinical Inquiries:
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {SUGGESTED_PROMPTS.map((prompt, idx) => (
                      <button key={idx} onClick={() => handleSend(prompt)} 
                        className="p-3 text-xs text-slate-700 bg-white hover:bg-sky-50/70 hover:border-sky-300 border border-slate-200/90 rounded-2xl transition-all text-left shadow-2xs flex items-start justify-between group">
                        <span className="font-medium leading-snug">{prompt}</span>
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
                  <div className="bg-gradient-to-r from-sky-600 to-indigo-600 text-white rounded-3xl rounded-tr-xs px-4 md:px-5 py-3 max-w-xl md:max-w-2xl shadow-sm text-xs md:text-sm leading-relaxed animate-in slide-in-from-right-5 duration-350">
                    {msg.text}
                    <div className="text-[9px] text-sky-100/80 mt-1 text-right">{msg.timestamp}</div>
                  </div>
                ) : (
                  <div className="bg-white rounded-3xl border border-slate-200 p-4 md:p-6 max-w-4xl w-full shadow-xs space-y-4 animate-in slide-in-from-left-5 duration-350">
                    
                    {/* Emergency Alert Banner */}
                    {msg.isEmergency && (
                      <div className="bg-rose-50 border-2 border-rose-500 rounded-2xl p-4 text-rose-900 space-y-2 animate-pulse">
                        <div className="flex items-center gap-2 font-bold text-sm md:text-base text-rose-700">
                          <AlertTriangle className="w-5 h-5 text-rose-600" /> CRITICAL MEDICAL EMERGENCY DETECTED
                        </div>
                        <p className="text-xs text-rose-800 leading-relaxed whitespace-pre-line font-medium">{msg.emergencyMessage}</p>
                      </div>
                    )}

                    {/* AI Response Header */}
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
                      <div className="flex items-center gap-2">
                        <div className="w-7 h-7 bg-gradient-to-tr from-sky-600 to-teal-600 text-white rounded-xl flex items-center justify-center font-bold text-xs shadow-2xs">
                          AI
                        </div>
                        <span className="font-bold text-xs md:text-sm text-slate-900">MediCare AI Clinical Synthesis</span>
                      </div>
                      
                      {msg.evaluation && (
                        <button onClick={() => toggleEval(msg.id)}
                          className="flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-bold bg-indigo-50 text-indigo-800 border border-indigo-200 hover:bg-indigo-100 transition-all shadow-2xs">
                          <BarChart2 className="w-3.5 h-3.5 text-indigo-600" />
                          <span>Quality: {msg.evaluation.composite_score}/100</span>
                          {openEvalId === msg.id ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                        </button>
                      )}
                    </div>

                    {/* Evaluation Metrics Card */}
                    {msg.evaluation && openEvalId === msg.id && (
                      <div className="bg-gradient-to-br from-indigo-50/80 via-slate-50 to-sky-50/80 border border-indigo-200 rounded-2xl p-3.5 md:p-4 text-xs space-y-3 animate-in slide-in-from-top-3 duration-250">
                        <div className="flex items-center justify-between font-bold text-indigo-950">
                          <span className="flex items-center gap-1.5"><Award className="w-4 h-4 text-indigo-600" /> RAG Quality Benchmarks</span>
                          <span className="px-2 py-0.5 rounded-md bg-indigo-200 text-indigo-900 text-[10px] font-bold">{msg.evaluation.grade}</span>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                          {[
                            { label: "Groundedness", value: msg.evaluation.groundedness },
                            { label: "Similarity", value: msg.evaluation.semantic_similarity },
                            { label: "BLEU-4", value: msg.evaluation.bleu_score },
                            { label: "ROUGE-L", value: msg.evaluation.rouge_score }
                          ].map((metric, i) => (
                            <div key={i} className="bg-white p-2.5 rounded-xl border border-indigo-100/80 text-center shadow-2xs animate-in zoom-in duration-300">
                              <span className="text-[9px] text-slate-500 font-bold uppercase block">{metric.label}</span>
                              <span className="text-sm md:text-base font-extrabold text-indigo-700">{typeof metric.value === 'number' ? metric.value.toFixed(1) : metric.value}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Markdown Body with Interactive Mermaid Flowchart Parser */}
                    <div className="prose prose-slate prose-sm md:prose-base max-w-none text-slate-800 leading-relaxed font-normal">
                      <ReactMarkdown
                        components={{
                          code({ node, inline, className, children, ...props }) {
                            const codeContent = String(children).replace(/\n$/, '');
                            const isMermaid = !inline && (
                              className?.includes('mermaid') || 
                              codeContent.trim().startsWith('graph ') || 
                              codeContent.trim().startsWith('flowchart ')
                            );
                            if (isMermaid) {
                              return <MermaidDiagram chart={codeContent} />;
                            }
                            return (
                              <code className={className} {...props}>
                                {children}
                              </code>
                            );
                          }
                        }}
                      >
                        {msg.text}
                      </ReactMarkdown>
                    </div>

                    {/* PubMed Citations */}
                    {msg.pubmedSources?.length > 0 && (
                      <div className="pt-3 border-t border-slate-100 space-y-2">
                        <div className="flex items-center gap-1.5 text-xs font-bold text-indigo-900 uppercase tracking-wider">
                          <Microscope className="w-4 h-4 text-indigo-600" /> Live NIH PubMed Citations ({msg.pubmedSources.length})
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {msg.pubmedSources.map((paper, idx) => (
                            <div key={idx} className="bg-indigo-50/40 p-3 rounded-xl border border-indigo-100/90 text-xs space-y-1 hover:border-indigo-300 transition-colors">
                              <a href={paper.url} target="_blank" rel="noopener noreferrer" 
                                className="font-bold text-indigo-900 hover:text-indigo-600 flex items-center justify-between gap-1 leading-snug">
                                <span className="line-clamp-2">{paper.title}</span>
                                <ExternalLink className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
                              </a>
                              <div className="text-[10px] text-indigo-700 font-medium">
                                {paper.journal} ({paper.year}) • <strong className="font-mono bg-indigo-100 px-1 py-0.5 rounded text-indigo-900">PMID:{paper.pmid}</strong>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="text-[9px] text-slate-400 text-right pt-1">{msg.timestamp}</div>
                  </div>
                )}
              </div>
            ))}

            {/* Loading Indicator */}
            {loading && (
              <div className="bg-white rounded-2xl border border-slate-200 p-4 max-w-md shadow-xs space-y-2.5 animate-pulse">
                <div className="flex items-center gap-2.5 text-sky-800 font-bold text-xs md:text-sm">
                  <Activity className="w-4 h-4 animate-spin text-teal-600" /> Retrieving Textbooks & Live PubMed Papers...
                </div>
                <div className="space-y-1.5">
                  <div className="h-2 bg-slate-100 rounded-full w-4/5"></div>
                  <div className="h-2 bg-slate-100 rounded-full w-3/5"></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </main>

          {/* Sticky Input Bar */}
          <footer className="bg-white/95 backdrop-blur-md border border-slate-200 p-2.5 md:p-3 rounded-2xl shadow-sm sticky bottom-2 z-20 mt-2">
            <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="flex items-center gap-2">
              <input type="text" value={input} onChange={(e) => setInput(e.target.value)} 
                placeholder="Ask a clinical question (e.g. step-by-step algorithms, dosages, trials)..." disabled={loading}
                className="flex-1 bg-slate-50 border border-slate-200 focus:border-sky-500 focus:bg-white text-slate-900 placeholder-slate-400 rounded-xl px-3.5 py-2.5 text-xs md:text-sm outline-none transition-all disabled:opacity-50" />
              <button type="submit" disabled={loading || !input.trim()}
                className="bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-700 hover:to-indigo-700 text-white font-bold px-4 py-2.5 rounded-xl transition-all shadow-xs flex items-center gap-1.5 disabled:opacity-50 text-xs md:text-sm shrink-0">
                Send <Send className="w-3.5 h-3.5" />
              </button>
            </form>
          </footer>
        </div>
      )}

      {/* ═══════════ REPORT ANALYZER TAB ═══════════ */}
      {activeTab === 'report' && (
        <main className="flex-1 p-3 md:p-6 space-y-5 max-w-6xl mx-auto w-full animate-in fade-in duration-400">
          
          {/* Header Banner */}
          <div className="bg-gradient-to-br from-teal-700 via-sky-800 to-indigo-900 rounded-3xl p-5 md:p-8 text-white shadow-md">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="space-y-1.5">
                <div className="inline-flex items-center gap-1.5 bg-white/15 px-3 py-1 rounded-full text-[11px] font-bold">
                  <FlaskConical className="w-3.5 h-3.5" /> Phase 2 Module • Two-Brain Clinical Architecture
                </div>
                <h2 className="text-xl md:text-3xl font-extrabold">Blood Test & Lab Report Analyzer</h2>
                <p className="text-sky-100 text-xs md:text-sm max-w-2xl leading-relaxed">
                  Extracts 100+ parameters, groups values by organ system, arranges differential diagnoses by clinical prevalence (Common → Rare), and exports physician-grade PDF summaries.
                </p>
              </div>
            </div>
          </div>

          {/* Upload Card */}
          <div className="bg-white rounded-3xl border border-slate-200 shadow-xs p-4 md:p-6 space-y-4">
            <form onSubmit={handleReportUpload} className="space-y-4">
              <div onDragOver={(e) => { e.preventDefault(); setDragActive(true); }} onDragLeave={() => setDragActive(false)} onDrop={onDrop}
                className={`border-2 border-dashed rounded-2xl md:rounded-3xl p-6 md:p-8 text-center transition-all ${dragActive ? 'border-teal-500 bg-teal-50/50' : 'border-slate-300 bg-slate-50/60 hover:border-teal-400'}`}>
                <div className="w-12 h-12 md:w-14 md:h-14 mx-auto mb-3 rounded-2xl bg-white border border-slate-200 shadow-2xs flex items-center justify-center">
                  <UploadCloud className="w-7 h-7 text-teal-600 animate-bounce" />
                </div>
                <h3 className="text-sm md:text-base font-bold text-slate-900 mb-0.5">Upload Clinical Lab PDF</h3>
                <p className="text-xs text-slate-500 mb-3">Drop file here or click below</p>
                <label className="cursor-pointer inline-block">
                  <span className="bg-teal-600 hover:bg-teal-700 text-white text-xs md:text-sm font-bold px-4 md:px-5 py-2.5 rounded-xl transition-all inline-flex items-center gap-2 shadow-xs">
                    <FileText className="w-4 h-4" /> Select PDF File
                  </span>
                  <input type="file" accept=".pdf" onChange={(e) => setReportFile(e.target.files?.[0] || null)} className="hidden" />
                </label>
                <div className="mt-3 text-[11px] text-slate-500">
                  {reportFile ? (
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-teal-50 text-teal-800 font-bold border border-teal-200">
                      <FileCheck className="w-3.5 h-3.5 text-teal-600 animate-pulse" /> {reportFile.name} ({(reportFile.size / 1024).toFixed(1)} KB)
                    </span>
                  ) : "Supports complete blood panels, metabolic, renal, liver, lipid, and endocrine profiles."}
                </div>
              </div>

              {reportError && (
                <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-xl flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" /> {reportError}
                </div>
              )}

              <button type="submit" disabled={!reportFile || reportLoading}
                className="w-full bg-gradient-to-r from-teal-600 via-sky-600 to-indigo-600 hover:opacity-95 disabled:opacity-50 text-white font-bold text-xs md:text-sm py-3 md:py-3.5 rounded-2xl transition-all shadow-xs flex items-center justify-center gap-2">
                {reportLoading ? (
                  <><Activity className="w-4 h-4 animate-spin" /> Extracting All Lab Parameters & Performing Clinical Reasoning...</>
                ) : (
                  <><Sparkles className="w-4 h-4 animate-pulse" /> Analyze Lab Report with Two-Brain RAG</>
                )}
              </button>
            </form>
          </div>

          {/* ═══════════ ANALYSIS RESULTS DASHBOARD ═══════════ */}
          {reportResult && !reportLoading && (
            <div className="space-y-5 animate-in fade-in duration-300">
              
              {/* Critical Alerts */}
              {reportResult.critical_alerts?.length > 0 && (
                <div className="bg-rose-50 border-2 border-rose-500 rounded-3xl p-4 md:p-5 text-rose-900 space-y-2 shadow-xs">
                  <div className="flex items-center gap-2 font-bold text-sm md:text-base text-rose-700">
                    <AlertOctagon className="w-5 h-5 text-rose-600 animate-bounce" /> CRITICAL LABORATORY ALERTS DETECTED
                  </div>
                  <ul className="list-disc pl-5 text-xs font-semibold space-y-1 text-rose-800">
                    {reportResult.critical_alerts.map((alert, idx) => <li key={idx}>{alert}</li>)}
                  </ul>
                </div>
              )}

              {/* Patient Metadata & Actions */}
              <div className="bg-white rounded-3xl border border-slate-200 shadow-xs p-4 md:p-6 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3.5">
                  <h3 className="text-base md:text-lg font-bold text-slate-900 flex items-center gap-2">
                    <CheckCircle2 className="text-teal-600 w-5 h-5" /> {reportResult.file_name}
                  </h3>
                  <div className="flex items-center gap-2">
                    <RiskBanner risk={reportResult.risk_level} />
                    <button onClick={() => exportClinicalPDF(reportResult)}
                      className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-bold bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-700 hover:to-indigo-700 text-white transition-all shadow-xs active:scale-95">
                      <Download className="w-3.5 h-3.5" /> Export PDF
                    </button>
                  </div>
                </div>

                {/* Patient Info Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 bg-slate-50 border border-slate-200/80 rounded-2xl p-3 md:p-4 text-xs">
                  <div><span className="text-slate-400 font-bold block text-[9px] uppercase">Patient</span><span className="font-bold text-slate-800 text-xs md:text-sm flex items-center gap-1"><User className="w-3.5 h-3.5 text-sky-600" />{reportResult.patient_name || 'Patient'}</span></div>
                  <div><span className="text-slate-400 font-bold block text-[9px] uppercase">Age / Sex</span><span className="font-bold text-slate-800 text-xs md:text-sm">{reportResult.patient_age_gender || 'N/A'}</span></div>
                  <div><span className="text-slate-400 font-bold block text-[9px] uppercase">Laboratory</span><span className="font-bold text-slate-800 text-xs md:text-sm truncate flex items-center gap-1"><Building className="w-3.5 h-3.5 text-indigo-600" />{reportResult.lab_name || 'Diagnostic Lab'}</span></div>
                  <div><span className="text-slate-400 font-bold block text-[9px] uppercase">Report Date</span><span className="font-bold text-slate-800 text-xs md:text-sm flex items-center gap-1"><Calendar className="w-3.5 h-3.5 text-emerald-600" />{reportResult.report_date || 'N/A'}</span></div>
                </div>

                {reportResult.summary && (
                  <div className="bg-gradient-to-br from-slate-50 to-sky-50/50 border border-slate-200 rounded-2xl p-4 shadow-2xs">
                    <div className="text-[11px] font-bold text-sky-900 uppercase tracking-wider mb-1 flex items-center gap-1">
                      <ClipboardList className="w-3.5 h-3.5 text-sky-600" /> Executive Clinical Summary
                    </div>
                    <p className="text-xs md:text-sm font-medium text-slate-800 leading-relaxed">{reportResult.summary}</p>
                  </div>
                )}
              </div>

              {/* Lab Values Grid with System Grouping */}
              {reportResult.lab_values?.length > 0 && (
                <div className="bg-white rounded-3xl border border-slate-200 shadow-xs p-4 md:p-6 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3.5">
                    <div>
                      <h4 className="text-sm md:text-base font-bold text-slate-900 flex items-center gap-1.5">
                        <FlaskConical className="w-4 h-4 text-teal-600" /> Extracted Laboratory Parameters ({labCounts.total})
                      </h4>
                      <p className="text-[11px] text-slate-500">Grouped by organ systems with status filters.</p>
                    </div>
                    <div className="flex items-center gap-1.5 text-[11px] font-bold">
                      <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border">Total: {labCounts.total}</span>
                      <span className="px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200 animate-pulse">🔴 {labCounts.high}</span>
                      <span className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200">🟡 {labCounts.low}</span>
                      <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">🟢 {labCounts.normal}</span>
                    </div>
                  </div>

                  {/* Filter Pills */}
                  <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs font-bold">
                    <span className="text-slate-400 flex items-center gap-1 text-[11px]"><Filter className="w-3 h-3"/> Filter:</span>
                    {['ALL', 'ABNORMAL', 'HIGH', 'LOW', 'NORMAL'].map((f) => (
                      <button key={f} onClick={() => setLabFilter(f)}
                        className={`px-3 py-1 rounded-xl border text-xs transition-all ${labFilter === f ? 'bg-teal-600 text-white border-teal-600 shadow-2xs' : 'bg-slate-50 text-slate-600 hover:bg-slate-100 border-slate-200'}`}>
                        {f === 'ALL' ? 'All' : f === 'ABNORMAL' ? '⚠️ Abnormal' : f}
                      </button>
                    ))}
                  </div>

                  {/* Grouped Organ System Cards */}
                  <div className="space-y-5 pt-1">
                    {Object.keys(getGroupedLabs()).length === 0 ? (
                      <div className="text-center py-6 text-xs text-slate-400">No lab parameters match filter ({labFilter}).</div>
                    ) : (
                      Object.entries(getGroupedLabs()).map(([systemName, labs]) => (
                        <div key={systemName} className="space-y-2.5">
                          <h5 className="text-[11px] font-bold uppercase tracking-wider text-teal-900 bg-teal-50 border border-teal-200/80 px-3 py-1 rounded-xl inline-flex items-center gap-1.5 animate-in slide-in-from-left duration-250">
                            <Layers className="w-3.5 h-3.5 text-teal-600" /> {systemName} ({labs.length})
                          </h5>
                          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                            {labs.map((lab, idx) => {
                              const status = (lab.status || "UNKNOWN").toUpperCase();
                              const border = status === "HIGH" ? "border-rose-200 bg-rose-50/30" : status === "LOW" ? "border-amber-200 bg-amber-50/30" : status === "NORMAL" ? "border-emerald-200 bg-emerald-50/20" : "border-slate-200 bg-white";
                              return (
                                <div key={idx} className={`rounded-2xl border p-3 shadow-2xs ${border} hover:scale-[1.01] transition-transform duration-200`}>
                                  <div className="flex items-start justify-between gap-1 mb-1">
                                    <div className="text-xs font-bold text-slate-800 leading-snug">{lab.test}</div>
                                    <StatusBadge status={status} />
                                  </div>
                                  <div className="text-lg font-extrabold text-slate-900">{lab.result} <span className="text-[11px] font-semibold text-slate-500">{lab.unit}</span></div>
                                  <div className="text-[10px] text-slate-500 mt-0.5">Ref Range: {lab.reference_range || "N/A"}</div>
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

              {/* Prevalence-Ordered Differentials */}
              {reportResult.differential_considerations?.length > 0 && (
                <div className="bg-white rounded-3xl border border-slate-200 shadow-xs p-4 md:p-6 space-y-3">
                  <div className="border-b border-slate-100 pb-3">
                    <h4 className="text-sm md:text-base font-bold text-slate-900 flex items-center gap-2">
                      <Compass className="w-4 h-4 text-indigo-600" /> Differential Considerations (Ordered by Prevalence)
                    </h4>
                    <p className="text-[11px] text-slate-500">Ranked from highest epidemiological likelihood to rare atypical patterns.</p>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1 animate-in zoom-in-95 duration-350">
                    {reportResult.differential_considerations.map((item, idx) => (
                      <div key={idx} className="rounded-2xl border border-slate-200 bg-gradient-to-br from-white to-slate-50/50 p-4 space-y-2 shadow-2xs hover:border-indigo-300 transition-all hover:shadow duration-200">
                        <div className="flex items-center justify-between gap-2">
                          <div className="font-bold text-xs md:text-sm text-slate-900 flex items-center gap-1.5">
                            <span className="w-5 h-5 rounded-lg bg-indigo-100 text-indigo-800 text-[10px] flex items-center justify-center font-extrabold">{idx + 1}</span>
                            {item.title}
                          </div>
                          <PrevalenceBadge prevalence={item.prevalence} />
                        </div>
                        <p className="text-xs text-slate-600 leading-relaxed pl-6.5">{item.rationale}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Pathophysiology & Recommendations */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {reportResult.pathophysiology && (
                  <div className="bg-white rounded-3xl border border-slate-200 shadow-xs p-4 md:p-6 space-y-2.5">
                    <h4 className="text-xs md:text-sm font-bold text-slate-900 flex items-center gap-1.5">
                      <ScopeIcon className="text-indigo-600 w-4 h-4"/> Inter-Organ Pathophysiological Assessment
                    </h4>
                    <p className="text-xs text-slate-700 leading-relaxed whitespace-pre-line">{reportResult.pathophysiology}</p>
                  </div>
                )}

                {reportResult.recommendations && (
                  <div className="bg-white rounded-3xl border border-slate-200 shadow-xs p-4 md:p-6 space-y-3">
                    <h4 className="text-xs md:text-sm font-bold text-slate-900 flex items-center gap-1.5 border-b border-slate-100 pb-2.5">
                      <ClipboardList className="text-teal-600 w-4 h-4"/> Evidence-Based Action Plan
                    </h4>
                    <div className="space-y-2.5 text-xs animate-in zoom-in-95 duration-250">
                      {reportResult.recommendations.urgent_actions?.length > 0 && (
                        <div className="bg-rose-50 p-3 rounded-2xl border border-rose-200/80">
                          <div className="font-bold text-rose-800 flex items-center gap-1.5 mb-1"><AlertTriangle className="w-3.5 h-3.5 text-rose-600 animate-pulse" /> Urgent Triage Actions</div>
                          <ul className="list-disc pl-5 text-rose-900 space-y-0.5">{reportResult.recommendations.urgent_actions.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                      {reportResult.recommendations.further_tests?.length > 0 && (
                        <div className="bg-sky-50 p-3 rounded-2xl border border-sky-200/80">
                          <div className="font-bold text-sky-800 flex items-center gap-1.5 mb-1"><TestTube className="w-3.5 h-3.5 text-sky-600" /> Recommended Follow-Up Investigations</div>
                          <ul className="list-disc pl-5 text-sky-900 space-y-0.5">{reportResult.recommendations.further_tests.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                      {reportResult.recommendations.specialty_consultation?.length > 0 && (
                        <div className="bg-indigo-50 p-3 rounded-2xl border border-indigo-200/80">
                          <div className="font-bold text-indigo-800 flex items-center gap-1.5 mb-1"><UserPlus className="w-3.5 h-3.5 text-indigo-600" /> Specialty Consultations</div>
                          <ul className="list-disc pl-5 text-indigo-900 space-y-0.5">{reportResult.recommendations.specialty_consultation.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                      {reportResult.recommendations.lifestyle_modifications?.length > 0 && (
                        <div className="bg-emerald-50 p-3 rounded-2xl border border-emerald-200/80">
                          <div className="font-bold text-emerald-800 flex items-center gap-1.5 mb-1"><Leaf className="w-3.5 h-3.5 text-emerald-600" /> Lifestyle & Dietary Optimization</div>
                          <ul className="list-disc pl-5 text-emerald-900 space-y-0.5">{reportResult.recommendations.lifestyle_modifications.map((x, i) => <li key={i}>{x}</li>)}</ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Standard Medical Disclaimer */}
              <div className="text-[11px] text-slate-500 bg-slate-100 border border-slate-200/80 rounded-2xl px-4 py-3 text-center">
                <strong>Clinical Decision-Support Disclaimer:</strong> This automated laboratory analysis is synthesized for educational and preliminary triage evaluation. All parameters must be interpreted by a licensed clinical physician.
              </div>
            </div>
          )}
        </main>
      )}
    </div>
  );
}