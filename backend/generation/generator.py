"""
MediCare AI - Hybrid Answer Generation Module
Combines MongoDB Vector Search (Textbooks) + PubMed API (Live Research).
Includes conversation memory, LaTeX cleaning, accuracy scoring, and response caching.
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

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GENERATION_MODEL = "gemini-flash-lite-latest"

client = genai.Client(api_key=GOOGLE_API_KEY)

HIGH_CONFIDENCE_SCORE = 0.75
LOW_CONFIDENCE_SCORE = 0.60

_response_cache = {}


def clean_latex_symbols(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\\ge\b', '≥', text)
    text = re.sub(r'\\le\b', '≤', text)
    text = re.sub(r'\\ge', '≥', text)
    text = re.sub(r'\\le', '≤', text)
    text = re.sub(r'\\pm', '±', text)
    text = re.sub(r'\\sim', '~', text)
    text = re.sub(r'\\times', '×', text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = text.replace('$', '')
    return text


BOOK_MODE_PROMPT = """You are MediCare AI, a medical knowledge assistant for healthcare professionals.

CRITICAL RULES:
1. Answer using ONLY the medical context provided (Textbooks, Guidelines, and Research Papers).
2. Do NOT invent information not present in the context.
3. Structure your response clearly using headings and bullet points.
4. When both Textbooks/Guidelines AND PubMed Research Papers are present in the context:
   - Synthesize textbook principles for core clinical background.
   - Include a dedicated section titled '### Recent Research & Clinical Evidence' highlighting trial results or consensus from PubMed.
5. If previous conversation messages are provided, use them for continuity. Answer follow-up questions in context of the conversation.

FORMATTING RULES:
1. Do NOT use LaTeX or TeX notation. Use plain Unicode (≥, ≤, %, mg/dL).
2. Do NOT include a manual 'Sources' or 'References' section at the bottom.
3. End with a brief medical disclaimer.
"""

FALLBACK_MODE_PROMPT = """You are MediCare AI, a medical knowledge assistant.

The question was not found in our medical knowledge base or PubMed.
Answer using general medical knowledge with extreme caution and transparency.

CRITICAL RULES:
1. START WITH: "NOTE: This information is NOT from your medical knowledge base or PubMed."
2. Provide a safe, conservative medical overview.
3. Do NOT use LaTeX notation.
4. Recommend consulting a healthcare provider.
5. Do NOT add a "Sources" section.
"""

def get_cache_key(question):
    return question.lower().strip()

def build_hybrid_prompt(question, text_chunks, pubmed_papers, chat_history=None):
    context_sections = []
    
    if text_chunks:
        tb_parts = []
        for i, chunk in enumerate(text_chunks, 1):
            tb_parts.append(f"[Textbook/Guideline Source {i}: {chunk['book']}, Page {chunk['page']}]\n{chunk['text']}")
        context_sections.append("=== MEDICAL TEXTBOOKS & GUIDELINES ===\n" + "\n---\n".join(tb_parts))
    
    if pubmed_papers:
        pm_text = format_pubmed_for_prompt(pubmed_papers)
        context_sections.append("=== LIVE PUBMED RESEARCH PAPERS ===\n" + pm_text)
    
    full_context = "\n\n".join(context_sections)
    
    # Build conversation history context
    history_text = ""
    if chat_history and len(chat_history) > 0:
        history_parts = []
        for msg in chat_history[-6:]:  # Last 6 messages for context window
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                history_parts.append(f"{role.upper()}: {content[:500]}")
        if history_parts:
            history_text = "\n=== PREVIOUS CONVERSATION ===\n" + "\n".join(history_parts) + "\n"
    
    return f"""{history_text}PROVIDED MEDICAL KNOWLEDGE:
{full_context}

---
USER QUESTION: {question}

Based on the provided medical context above, generate a structured, clinical answer.
If this is a follow-up question, answer in context of the previous conversation.
Do NOT add a manual source list at the bottom."""

def determine_mode(chunks, papers):
    has_books = False
    max_score = 0.0
    if chunks:
        max_score = max(c["score"] for c in chunks)
        if max_score >= LOW_CONFIDENCE_SCORE:
            has_books = True
            
    has_papers = bool(papers)
    
    if has_books and has_papers:
        mode = "books_and_research"
    elif has_books:
        mode = "books"
    elif has_papers:
        mode = "research"
    else:
        mode = "fallback"
        
    return mode, max_score

def generate_answer(question, chunks, max_retries=5, chat_history=None):
    cache_key = get_cache_key(question)
    if cache_key in _response_cache and not chat_history:
        print(f"    Mode: CACHED (Instant!)")
        return _response_cache[cache_key]
    
    print("    Fetching live PubMed research papers...")
    pubmed_papers = search_pubmed_live(question, max_results=7)
    
    mode, max_score = determine_mode(chunks, pubmed_papers)
    
    if mode in ["books_and_research", "books"]:
        accuracy_percentage = round(max_score * 100, 1)
    elif mode == "research":
        accuracy_percentage = 85.0
    else:
        accuracy_percentage = 0.0
        
    print(f"    Mode: {mode.upper()} | Accuracy: {accuracy_percentage}% (Retrieved {len(chunks)} chunks, {len(pubmed_papers)} research papers)")
    
    if mode == "fallback":
        prompt = f"USER QUESTION: {question}\nProvide general knowledge overview."
        system_prompt = FALLBACK_MODE_PROMPT
    else:
        prompt = build_hybrid_prompt(question, chunks, pubmed_papers, chat_history)
        system_prompt = BOOK_MODE_PROMPT
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=2048,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            )
            
            answer = clean_latex_symbols(response.text)
            
            mode_labels = {
                "books_and_research": "📚🔬 [Grounded in Medical Textbooks, Guidelines & Live PubMed Research]",
                "books": "📚 [Grounded in Medical Textbooks & Clinical Guidelines]",
                "research": "🔬 [Grounded in Live PubMed Research Papers]",
                "fallback": "🤖 [General Knowledge - Not found in Knowledge Base]"
            }
            
            labeled_answer = f"{mode_labels.get(mode, '')}\n\n{answer}"
            
            result = (labeled_answer, mode, accuracy_percentage, pubmed_papers)
            if not chat_history:
                _response_cache[cache_key] = result
            return result
            
        except Exception as e:
            error_msg = str(e)
            if any(err in error_msg for err in ["503", "UNAVAILABLE", "SSL", "EOF", "socket", "Connection"]):
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                else:
                    return "Server temporarily unavailable. Please try again.", "error", 0.0, []
            
            if "429" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                    
            return f"Error generating answer: {e}", "error", 0.0, []
    
    return "Failed to generate answer.", "error", 0.0, []

def format_sources(chunks, mode, pubmed_papers=None):
    if mode == "fallback":
        return [], []
    
    tb_sources = []
    seen = set()
    if chunks:
        for chunk in chunks:
            key = f"{chunk['book']}_p{chunk['page']}"
            if key not in seen:
                seen.add(key)
                tb_sources.append({
                    "book": chunk["book"],
                    "page": chunk["page"],
                    "score": chunk["score"]
                })
    
    return tb_sources, pubmed_papers or []