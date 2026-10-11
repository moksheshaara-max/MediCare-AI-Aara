"""Run AFTER copying v3 modules into the existing MediCare AI backend.

Connects to live MongoDB Atlas and PubMed; DOES NOT invoke Gemini generation.
Uses current .env and may take time. There is no clinical validation claim.
    python .\tests\validate_modes_live.py
or, if left inside extracted upgrade directory, copy it to backend root first.
"""
from collections import Counter
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from retrieval.search import search_medical_chunks, pg_retrieval_budget, classify_pg_query_depth
from pubmed.pubmed_search import search_pubmed_live

CASES = [
    ("doctor", "What is the current guideline-based first-line pharmacological treatment for HFrEF?"),
    ("pg_student", "Explain hypertension in detail for postgraduate medicine including pathophysiology, diagnosis, management and pharmacology."),
    ("pg_student", "Teach chronic kidney disease comprehensively: renal physiology, pathology, diagnosis, staging, anemia, pharmacology, complications, management, monitoring and dialysis."),
    ("pg_student", "Explain diagnosis, insulin management and monitoring of type 1 diabetes in a 10-year-old child."),
]

for mode, question in CASES:
    print("=" * 100)
    print("MODE:", mode)
    print("QUESTION:", question)
    depth = classify_pg_query_depth(question) if mode == "pg_student" else "doctor"
    chunk_k = pg_retrieval_budget(question) if mode == "pg_student" else 36
    pubmed_k = {"doctor":5, "simple":5, "detailed":7, "comprehensive":9}[depth]
    print("DEPTH:", depth, "CHUNK BUDGET:", chunk_k, "PUBMED LIMIT:", pubmed_k)
    try:
        chunks = search_medical_chunks(question, top_k=chunk_k, assistant_mode=mode)
        counts = Counter(c.get("source_id", "UNKNOWN") for c in chunks)
        kinds = Counter(str(c.get("source_type", "other")) for c in chunks)
        print("CHUNKS RETURNED:",len(chunks),"SOURCE TYPE COUNTS:",dict(kinds))
        print("TOP SOURCES:", counts.most_common(10))
        if not chunks:
            print("FAIL: zero retrieved chunks; investigate connectivity / source availability")
    except Exception as exc:
        print("RETRIEVAL ERROR:", type(exc).__name__, str(exc))
        continue
    papers = search_pubmed_live(question, max_results=pubmed_k, assistant_mode=mode, question_depth=depth)
    print("PUBMED PAPERS RETURNED:",len(papers))
    for p in papers:
        print("  PMID",p.get("pmid"),"|",p.get("year"),"|",p.get("title",""))
    if not papers:
        print("NOTE: no papers passed topic/population verification; not automatically a failure")

print("LIVE EVIDENCE SMOKE TEST FINISHED. Inspect topic identity and source contents manually.")
