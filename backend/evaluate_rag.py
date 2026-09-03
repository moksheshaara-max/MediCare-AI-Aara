"""
MediCare AI - Standalone Research Evaluation Script
Uses 12 chunks and 7 PubMed papers per question.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval.search import search_medical_chunks
from generation.generator import generate_answer
from evaluator.rag_evaluator import evaluate_response_quality

TEST_DATASET = [
    "What are the diagnostic criteria for Type 2 Diabetes mellitus?",
    "How is severe acute pancreatitis treated according to guidelines?",
    "What is the first-line treatment for essential hypertension in pregnancy?",
    "What are the clinical manifestations of diabetic nephropathy?",
    "How is severe falciparum malaria treated under ICMR protocols?",
    "What are the diagnostic criteria for systemic lupus erythematosus?",
    "What is the management protocol for organophosphate poisoning?",
    "What are the main causes of severe hyperkalemia?",
    "How is Community-Acquired Pneumonia managed in outpatient settings?",
    "What are the complications of untreated primary hypothyroidism?"
]

def run_benchmark():
    print("=" * 80)
    print("MediCare AI - Automated RAG Scientific Benchmark Evaluation")
    print("=" * 80)
    
    results = []
    
    for idx, question in enumerate(TEST_DATASET, 1):
        print(f"\n[{idx}/10] Evaluating: '{question}'...")
        
        chunks = search_medical_chunks(question, top_k=12)
        answer, mode, accuracy_score, papers = generate_answer(question, chunks)
        metrics = evaluate_response_quality(question, chunks, answer, mode)
        
        results.append({
            "id": idx, "question": question, "mode": mode,
            "composite": metrics["composite_score"],
            "bleu": metrics["bleu_score"],
            "rouge": metrics["rouge_score"],
            "semantic": metrics["semantic_similarity"],
            "groundedness": metrics["groundedness"]
        })
        
        print(f"    ✓ Composite: {metrics['composite_score']:.1f}/100 | BLEU-4: {metrics['bleu_score']:.1f}% | ROUGE-L: {metrics['rouge_score']:.1f}% | Groundedness: {metrics['groundedness']:.1f}%")

    print("\n" + "=" * 80)
    print("FINAL ACADEMIC RESEARCH BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    
    print("\n| ID | Clinical Question | Mode | BLEU-4 | ROUGE-L | Semantic Sim | Groundedness | Composite |")
    print("|---|---|---|---|---|---|---|---|")

    for r in results:
        q_short = r["question"][:35] + "..." if len(r["question"]) > 35 else r["question"]
        print(f"| {r['id']} | {q_short} | {r['mode']} | {r['bleu']:.1f}% | {r['rouge']:.1f}% | {r['semantic']:.1f}% | {r['groundedness']:.1f}% | **{r['composite']:.1f}/100** |")

    avg = lambda key: sum(r[key] for r in results) / len(results)
    print(f"| **AVG** | **Overall Mean** | **-** | **{avg('bleu'):.1f}%** | **{avg('rouge'):.1f}%** | **{avg('semantic'):.1f}%** | **{avg('groundedness'):.1f}%** | **{avg('composite'):.1f}/100** |")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()