"""Post-fix clinical evidence LIVE check: four cases (PubMed + Gemini + Atlas).

Run from backend root: python -u .\\verify_answer_evidence_live.py
Writes clinical_master_live_results.json after EACH request. No DB writes.
The automatic flags are targeted smoke checks, NOT medical validation.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CASES = (
    ("DS_TB_PG", "pg_student",
     "Describe the diagnostic workup and management of newly diagnosed drug-susceptible pulmonary tuberculosis in India under NTEP, including standard treatment regimen, monitoring and adherence support."),
    ("PPH_DOCTOR", "doctor",
     "A patient develops postpartum hemorrhage after vaginal delivery. Explain recognition, initial resuscitation, uterotonic medications, tranexamic acid and escalation of care."),
    ("DR_TB_DOCTOR", "doctor",
     "Explain evaluation and management of rifampicin-resistant or multidrug-resistant pulmonary tuberculosis in India, including susceptibility testing, regimen durations, principles and adverse-effect monitoring."),
    ("MALARIA_PG", "pg_student",
     "Explain severe falciparum malaria in an adult in India: diagnosis, organ dysfunction, parenteral treatment, monitoring and complications."),
)
OUT = Path("clinical_master_live_results.json")


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(data):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Overwrite prior results and rerun all 4")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N NEW cases; results checkpointed")
    args = parser.parse_args()
    results = {}
    if OUT.exists() and not args.fresh:
        try:
            existing = json.loads(OUT.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                results.update(existing)
        except (OSError, ValueError):
            pass

    import app as api
    from generation.generator import _clinical_answer_blockers
    from pubmed.pubmed_search import _is_relevant_paper

    live_search = api.search_medical_chunks
    new_runs = 0
    with api.app.test_client() as client:
        for case_id, mode, question in CASES:
            if case_id in results:
                print("SKIP", case_id, "already checkpointed (--fresh to rerun)", flush=True)
                continue
            if args.limit and new_runs >= args.limit:
                break

            chunks_observed = []
            def observe(*a, **kw):
                chunks = live_search(*a, **kw)
                if isinstance(chunks, list):
                    chunks_observed.extend(chunks)
                return chunks

            api.search_medical_chunks = observe
            print("\n" + "=" * 78 + "\nRUN", case_id, mode, flush=True)
            start = time.monotonic()
            try:
                response = client.post(
                    "/api/ask",
                    json={"question": question, "assistant_mode": mode, "chat_history": []},
                )
                data = response.get_json(silent=True)
                if not isinstance(data, dict):
                    data = {"error": "Non-JSON API response"}
                answer = data.get("answer") if isinstance(data.get("answer"), str) else ""
                papers = data.get("pubmed_sources") if isinstance(data.get("pubmed_sources"), list) else []
                counts = dict(Counter(str(c.get("source_id") or c.get("book") or "unknown")
                                      for c in chunks_observed if isinstance(c, dict)))

                flags = []
                if response.status_code != 200: flags.append(f"HTTP {response.status_code}")
                if data.get("assistant_mode") != mode: flags.append("Wrong assistant mode")
                if data.get("evidence_mode") == "error": flags.append("Generator withheld/failed answer")
                if not answer.strip(): flags.append("Empty generated answer")
                if not chunks_observed: flags.append("No MongoDB evidence")
                flags.extend(_clinical_answer_blockers(question, answer, papers, chunks_observed))
                off_topic_pmids = [str(p.get("pmid") or "?") for p in papers
                                    if isinstance(p, dict) and not _is_relevant_paper(question, p)]
                if off_topic_pmids:
                    flags.append("Off-topic PubMed evidence passed final filter: " + ", ".join(off_topic_pmids))
                if case_id == "DS_TB_PG":
                    if "who_tb_module4_treatment_care_2026_2ed" not in counts:
                        flags.append("WHO active TB treatment guideline missing")
                    if "ntep_dr_tb_2025" in counts:
                        flags.append("DR-TB guidance selected for DS-TB question")
                if case_id == "PPH_DOCTOR" and "who_pph_2025" not in counts:
                    flags.append("WHO PPH guideline missing")
                if case_id == "DR_TB_DOCTOR" and "ntep_dr_tb_2025" not in counts:
                    flags.append("NTEP DR-TB guideline missing")
                if case_id == "MALARIA_PG" and "who_malaria_guidelines_2025_08_13" not in counts:
                    flags.append("WHO clinical malaria guidance missing")
                rec = {
                    "case": case_id, "mode": mode, "tested_utc": utc(),
                    "http_status": response.status_code,
                    "returned_mode": data.get("assistant_mode"),
                    "evidence_mode": data.get("evidence_mode"),
                    "retrieval_confidence_is_similarity_not_clinical_accuracy": data.get("retrieval_confidence"),
                    "kb_chunk_count": len(chunks_observed), "kb_source_counts": counts,
                    "pubmed_paper_count": len(papers), "pubmed_papers": papers,
                    "off_topic_pubmed_pmids": off_topic_pmids,
                    "kb_inline_citation_count": answer.count("[KB:"),
                    "pubmed_inline_citation_count": answer.count("[PMID:"),
                    "answer": answer, "automatic_flags": flags,
                    "test_status": "NEEDS_REVIEW" if flags else "STRUCTURAL_PASS_NOT_CLINICALLY_VALIDATED",
                    "elapsed_seconds": round(time.monotonic() - start, 2),
                }
                if response.status_code != 200:
                    rec["error"] = data.get("error")
                    rec["details"] = data.get("details")
                results[case_id] = rec
                save(results)
                print("HTTP", response.status_code, "chunks", len(chunks_observed),
                      "PubMed", len(papers), "answer_chars", len(answer), flush=True)
                print("KB:", counts, flush=True)
                for p in papers:
                    print(" PAPER", p.get("pmid"), p.get("title"), flush=True)
                print("FLAGS:", flags if flags else "none (manual clinical review required)", flush=True)
                print("Saved:", OUT, flush=True)
            except Exception as exc:
                results[case_id] = {"case": case_id, "mode": mode, "tested_utc": utc(),
                                    "test_status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
                save(results)
                print("ERROR:", type(exc).__name__, str(exc), flush=True)
            finally:
                api.search_medical_chunks = live_search
            new_runs += 1

    print("\nCompleted records:", len(results), "/", len(CASES), "->", OUT, flush=True)
    print("Test flags cannot replace independent clinical review.", flush=True)


if __name__ == "__main__":
    main()
