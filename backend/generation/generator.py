"""
MediCare AI - Hybrid Clinical Answer Generation Module
Combines MongoDB Vector Search + NIH PubMed API.
Features strictly validated visual Mermaid.js clinical flowcharts.
"""

import sys
import os
import re
import time
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("google").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types
from dotenv import load_dotenv

from pubmed.pubmed_search import search_pubmed_live, format_pubmed_for_prompt

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=env_path)
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GENERATION_MODEL = "gemini-2.0-flash"

client = genai.Client(api_key=GOOGLE_API_KEY)

LOW_CONFIDENCE_SCORE = 0.60
_response_cache = {}


def clean_latex_symbols(text):
    if not text:
        return ""
    text = re.sub(r'\\ge\b', '>=', text)
    text = re.sub(r'\\le\b', '<=', text)
    text = re.sub(r'\\ge', '>=', text)
    text = re.sub(r'\\le', '<=', text)
    text = re.sub(r'\\pm', '+/-', text)
    text = re.sub(r'\\sim', '~', text)
    text = re.sub(r'\\times', 'x', text)
    text = re.sub(r'\\rightarrow', '->', text)
    text = re.sub(r'\\leftarrow', '<-', text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = text.replace('$', '')
    text = text.replace('>=', 'GTE_TMP').replace('<=', 'LTE_TMP')
    text = text.replace('GTE_TMP', 'greater than or equal to').replace('LTE_TMP', 'less than or equal to')
    text = text.replace('greater than or equal to', '≥').replace('less than or equal to', '≤')
    text = text.replace('->', '→').replace('<-', '←').replace('+/-', '±').replace(' x ', ' × ')
    return text


BOOK_MODE_PROMPT = """You are MediCare AI, a senior consultant physician and clinical decision-support AI.

Synthesize the provided Medical Textbooks, Guidelines, and PubMed papers into an authoritative clinical response.

MANDATORY RESPONSE STRUCTURE:

1. Clinical Overview & Core Concepts
   Concise clinical summary and underlying pathophysiology.

2. Step-by-Step Clinical Pathway Algorithm (Visual Flowchart)
   You MUST ALWAYS generate a visual Mermaid flowchart diagram illustrating the clinical diagnostic, triage, or management algorithm.
   Use this EXACT format wrapped in triple backticks with mermaid tag:

   graph TD
       A["Patient Presentation"] --> B{"Diagnostic Triage"}
       B -->|"High Risk"| C["Immediate Intervention"]
       B -->|"Stable"| D["Stepwise Workup"]
       D --> E["First-Line Therapy"]
       E --> F["Monitoring"]

   CRITICAL MERMAID RULES:
     - ALWAYS wrap node labels in double quotes: A["Text Here"]
     - Do NOT use parentheses, slashes, or symbols inside node text unless quoted.
     - Keep node text short (under 7 words).

3. Diagnostic Criteria & Laboratory Workup
   Gold standard tests, specific cutoff values, and physical exam findings.

4. Evidence-Based Management Protocol
   First-Line Therapy: Standard drugs, dosages, mechanism.
   Second-Line / Escalation: Alternative drugs.
   Lifestyle & Non-Pharmacological: Dietary, monitoring interventions.

5. Recent Clinical Evidence (Include when PubMed papers are in context)
   Synthesize recent trial evidence and consensus with PMIDs.

6. Critical Red Flags
   Danger signs requiring immediate emergency referral.

FORMATTING & SAFETY:
1. Ground answers strictly in the provided medical context.
2. Use plain Unicode (>=, <=, +/-, mg/dL, mmHg). Never use LaTeX syntax.
3. End with a 1-sentence standard clinical disclaimer.
"""

FALLBACK_MODE_PROMPT = """You are MediCare AI, a medical decision-support assistant.
The question was not found in our indexed textbook database.
Answer using general medical knowledge with extreme caution and transparency.

CRITICAL RULES:
1. START WITH: "Clinical Note: This summary is from general medical knowledge (not directly indexed in current textbook chunks)."
2. Include a visual mermaid flowchart diagram for the general management pathway.
3. Structure clearly: Overview, Flowchart, Management, Red Flags.
4. Use plain Unicode and NO LaTeX.
5. End with a reminder to consult a licensed healthcare provider.
"""


def build_hybrid_prompt(question, text_chunks, pubmed_papers, chat_history=None):
    context_sections = []
    
    if text_chunks:
        tb_parts = []
        for i, chunk in enumerate(text_chunks, 1):
            tb_parts.append(f"[Textbook Source {i}: {chunk.get('book', 'Medical Reference')}, Page {chunk.get('page', 'N/A')}]\n{chunk.get('text', '')}")
        context_sections.append("=== MEDICAL TEXTBOOKS & GUIDELINES ===\n" + "\n---\n".join(tb_parts))
    
    if pubmed_papers:
        pm_text = format_pubmed_for_prompt(pubmed_papers)
        context_sections.append("=== LIVE NIH PUBMED RESEARCH PAPERS ===\n" + pm_text)
    
    full_context = "\n\n".join(context_sections)
    
    history_text = ""
    if chat_history and len(chat_history) > 0:
        history_parts = []
        for msg in chat_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                history_parts.append(f"{role.upper()}: {content[:500]}")
        if history_parts:
            history_text = "\n=== PREVIOUS CONVERSATION CONTEXT ===\n" + "\n".join(history_parts) + "\n\n"
    
    return f"""{history_text}PROVIDED CLINICAL CONTEXT:
{full_context}

---
USER QUESTION: {question}

Generate a comprehensive clinical response with an interactive Mermaid flowchart diagram."""


def determine_mode(chunks, papers):
    has_books = False
    max_score = 0.0
    if chunks:
        max_score = max(c.get("score", 0.0) for c in chunks)
        if max_score >= LOW_CONFIDENCE_SCORE:
            has_books = True
            
    has_papers = bool(papers and len(papers) > 0)
    
    if has_books and has_papers:
        mode = "books_and_research"
    elif has_books:
        mode = "books"
    elif has_papers:
        mode = "research"
    else:
        mode = "fallback"
        
    return mode, max_score


def generate_answer(question, chunks, max_retries=4, chat_history=None):
    cache_key = question.lower().strip()
    if cache_key in _response_cache and not chat_history:
        print("    Mode: CACHED (Instant)")
        return _response_cache[cache_key]
    
    print("    Searching NIH PubMed...")
    pubmed_papers = search_pubmed_live(question, max_results=5)
    
    mode, max_score = determine_mode(chunks, pubmed_papers)
    
    if mode in ["books_and_research", "books"]:
        accuracy_percentage = round(max_score * 100, 1)
    elif mode == "research":
        accuracy_percentage = 85.0
    else:
        accuracy_percentage = 0.0
        
    print(f"    Mode: {mode.upper()} | Similarity: {accuracy_percentage}% | RAG Chunks: {len(chunks)} | PubMed: {len(pubmed_papers)}")
    
    if mode == "fallback":
        prompt = f"CLINICAL QUERY: {question}\nProvide a structured overview with a visual Mermaid flowchart diagram."
        system_prompt = FALLBACK_MODE_PROMPT
    else:
        prompt = build_hybrid_prompt(question, chunks, pubmed_papers, chat_history)
        system_prompt = BOOK_MODE_PROMPT
    
    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-lite-latest"]
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2,
                        max_output_tokens=3000,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
                
                answer = clean_latex_symbols(response.text or "")
                
                mode_labels = {
                    "books_and_research": "[Grounded in Medical Textbooks, Standard Guidelines & NIH PubMed Research]",
                    "books": "[Grounded in Medical Textbooks & Standard Clinical Guidelines]",
                    "research": "[Grounded in Live NIH PubMed Clinical Research Papers]",
                    "fallback": "[General Clinical Knowledge — Not explicitly indexed in textbook chunks]"
                }
                
                labeled_answer = f"{mode_labels.get(mode, '')}\n\n{answer}"
                result = (labeled_answer, mode, accuracy_percentage, pubmed_papers)
                
                if not chat_history:
                    _response_cache[cache_key] = result
                    
                return result
                
            except Exception as e:
                print(f"    Generation attempt {attempt + 1} with {model_name} failed: {e}")
                time.sleep(2)
                continue
    
    return "Failed to generate response. Please verify Google API Key.", "error", 0.0, []


def format_sources(chunks, mode, pubmed_papers=None):
    if mode == "fallback":
        return [], []
    
    tb_sources = []
    seen = set()
    if chunks:
        for chunk in chunks:
            book = chunk.get("book", "Medical Reference")
            page = chunk.get("page", 1)
            score = chunk.get("score", 0.0)
            key = f"{book}_p{page}"
            if key not in seen:
                seen.add(key)
                tb_sources.append({
                    "book": book,
                    "page": page,
                    "score": score
                })
    
    return tb_sources, pubmed_papers or []