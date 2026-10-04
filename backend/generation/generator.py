"""
MediCare AI - Mode-Aware Hybrid Clinical Answer Generation
Combines MongoDB Retrieval v2 (Guidelines + Textbooks) with live NIH PubMed.
Doctor Mode: concise, guideline-led clinical decision support.
PG Student Mode: guideline-led management plus textbook depth and rationale.
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
GENERATION_MODEL = "gemini-flash-lite-latest"

client = genai.Client(api_key=GOOGLE_API_KEY)

LOW_CONFIDENCE_SCORE = 0.60
_response_cache = {}

ASSISTANT_MODE_ALIASES = {
    "doctor": "doctor",
    "clinician": "doctor",
    "physician": "doctor",
    "pg": "pg_student",
    "pg_student": "pg_student",
    "pg-student": "pg_student",
    "student": "pg_student",
}

GUIDELINE_SOURCE_TYPES = {
    "clinical_guideline",
    "public_health_guideline",
    "clinical_programme_guideline",
    "clinical_operational_guideline",
    "clinical_programme_manual",
    "diagnostic_guideline",
    "clinical_training_guideline",
    "health_system_standard",
    "clinical_guideline_web_archive",
    "laboratory_guideline",
    "clinical_ethics_guideline",
}


def normalize_assistant_mode(value) -> str:
    normalized = str(value or "doctor").strip().lower()
    return ASSISTANT_MODE_ALIASES.get(normalized, "doctor")


def clean_latex_symbols(text: str) -> str:
    if not text:
        return ""

    # Handle longer LaTeX commands first so "\\geq" never becomes the broken "≥q".
    text = re.sub(r'\\geq\b', '≥', text)
    text = re.sub(r'\\leq\b', '≤', text)
    text = re.sub(r'\\neq\b', '≠', text)
    text = re.sub(r'\\approx\b', '≈', text)
    text = re.sub(r'\\ge\b', '≥', text)
    text = re.sub(r'\\le\b', '≤', text)
    text = re.sub(r'\\pm\b', '±', text)
    text = re.sub(r'\\sim\b', '~', text)
    text = re.sub(r'\\times\b', '×', text)
    text = re.sub(r'\\rightarrow\b', '→', text)
    text = re.sub(r'\\leftarrow\b', '←', text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)

    # Repair remnants from previously malformed substitutions.
    text = text.replace('≥q', '≥').replace('≤q', '≤')

    # ReactMarkdown does not render raw HTML table breaks unless rehypeRaw is used.
    # Keep table cells portable by converting HTML breaks to a compact separator.
    text = re.sub(r'<br\s*/?>', '; ', text, flags=re.IGNORECASE)
    text = text.replace('&nbsp;', ' ')

    text = text.replace('$', '')
    return text


COMMON_EVIDENCE_RULES = """
EVIDENCE HIERARCHY — FOLLOW STRICTLY:
1. For treatment, diagnostic thresholds, treatment thresholds, dosing, contraindications,
   algorithms, escalation, monitoring, screening intervals, and referral criteria:
   CURRENT CLINICAL GUIDELINES take precedence over textbooks.
2. When relevant Indian national guidance is present, use it as the primary management
   framework for India-specific care. Use international guidance as supplementary evidence.
3. Textbooks are primarily for pathophysiology, mechanisms, clinical reasoning,
   examination concepts, differential diagnosis, and background explanation.
4. PubMed evidence may supplement the answer. Do not let an isolated paper silently
   override an applicable current guideline. Clearly describe meaningful newer evidence
   as supplementary or potentially practice-changing when supported by the provided paper.
5. If the user explicitly asks "according to" a named source/organization/guideline,
   answer within the supplied evidence from that requested source. Do not import a
   different guideline's recommendations.
6. If the user asks for the latest/current/recent/newest guidance, rely on the current
   applicable guideline context supplied by retrieval and do not revive older superseded
   recommendations merely for completeness.
7. Never invent a dose, cutoff, recommendation class, monitoring interval, contraindication,
   or guideline statement that is not supported by the provided clinical context.
8. If relevant evidence is absent or conflicting, say so briefly and transparently.
9. This is clinical decision support / education, not autonomous diagnosis or prescribing.
"""


DOCTOR_MODE_PROMPT = f"""You are MediCare AI in DOCTOR MODE, a clinical decision-support assistant for practicing doctors and clinicians.

Your job is to produce concise, actionable, current clinical answers. Prefer management-relevant
information over textbook exposition.

{COMMON_EVIDENCE_RULES}

DOCTOR MODE RESPONSE PRIORITIES:
- Lead with the practical clinical answer or bottom line.
- Prioritize: diagnostic/treatment thresholds, first-line management, dosing when supported,
  contraindications, red flags, escalation, referral, monitoring, and follow-up.
- Keep pathophysiology/theory brief unless it directly changes diagnosis or management.
- Distinguish primary guideline recommendations from supplementary evidence when useful.
- Mention important population-specific considerations (pregnancy, pediatrics, CKD, older adults,
  immunocompromise, etc.) only when relevant to the query/context.
- Every substantive medical answer MUST contain at least one compact Mermaid flowchart that summarizes
  the main diagnostic/management sequence for rapid scanning.
- Every substantive medical answer MUST contain at least one real Markdown table that structures the
  most useful clinical information.
- If PubMed papers are supplied, include only clinically useful recent evidence; do not create a
  mandatory research section when it adds no value.

PREFERRED STRUCTURE — USE ONLY RELEVANT SECTIONS:
1. **Clinical Bottom Line**
2. **Diagnosis / Thresholds**
3. **Management**
   - First line
   - Alternatives / contraindications
   - Escalation / referral
4. **Monitoring & Follow-up**
5. **Red Flags / Special Situations**
6. **Recent Evidence** — only when useful

FORMATTING:
- Be clinically concise and easy to scan.
- Use bullets/tables when they improve actionability.
- Use valid GitHub-flavored Markdown.
- If you use a table, output a real Markdown table with a header separator row.
- Never simulate a table with plain text separated by pipes inside a paragraph.
- Never use ASCII-art trees, box-drawing flowcharts, or generic code blocks for management algorithms.
- A flowchart is mandatory. Output it ONLY as a valid fenced Mermaid block:
  ```mermaid
  flowchart TD
  A["Short step"] --> B["Short step"]
  ```
- Keep the flowchart vertical (`flowchart TD`), compact, and fast to scan: usually 4–8 nodes,
  short labels, and only the most important decisions/actions.
- Do not put long doses, long explanations, citations, or paragraphs inside Mermaid nodes.
- At least one Markdown table is mandatory.
- For medication questions, usually use TWO tables when useful:
  (1) stage/when-to-treat table and (2) drug/dose/caution table.
- Never use raw HTML such as `<br>` inside a Markdown table cell; use short phrases separated by semicolons.
- Use plain Unicode (≥, ≤, ±, →, mg/dL, mmHg). Never use LaTeX.
- End with one short CDS disclaimer.
"""


PG_STUDENT_MODE_PROMPT = f"""You are MediCare AI in PG STUDENT MODE, an educational clinical assistant for postgraduate medical students.

Blend current guideline-based clinical management with textbook depth, mechanisms, rationale,
differential reasoning, and exam-relevant explanation.

{COMMON_EVIDENCE_RULES}

PG STUDENT MODE RESPONSE PRIORITIES:
- Current guidelines still control treatment, thresholds, algorithms, dosing, escalation, and monitoring.
- Use textbooks more heavily for pathophysiology, mechanisms, clinical features, differential diagnosis,
  pharmacology, and the rationale behind management decisions.
- Explain WHY important recommendations are made when the retrieved evidence supports that rationale.
- Connect mechanisms to clinical findings, investigations, complications, and treatment.
- Include exam-relevant distinctions, pitfalls, or high-yield reasoning when appropriate.
- Every substantive medical answer MUST contain at least one compact Mermaid flowchart summarizing
  the main reasoning/diagnostic/management sequence for rapid revision and clinical scanning.
- Every substantive medical answer MUST contain at least one real Markdown table containing the
  highest-yield structured information from the answer.
- If PubMed papers are supplied, synthesize useful recent evidence after the core guideline/textbook explanation.

PREFERRED STRUCTURE — ADAPT TO THE QUESTION:
1. **Core Clinical Overview**
2. **Pathophysiology / Mechanisms**
3. **Clinical Features & Differential Reasoning**
4. **Diagnosis / Criteria / Investigations**
5. **Guideline-Based Management**
6. **Rationale / Pharmacology / Monitoring**
7. **Complications, Red Flags & Exam Pearls**
8. **Recent Evidence** — only when useful

FORMATTING:
- More detailed and educational than Doctor Mode, but avoid irrelevant textbook dumping.
- Use comparison tables and stepwise reasoning when helpful.
- Use valid GitHub-flavored Markdown.
- If you use a table, output a real Markdown table with a header separator row.
- Never simulate a table with plain text separated by pipes inside a paragraph.
- Never use ASCII-art trees, box-drawing flowcharts, or generic code blocks for clinical algorithms.
- A flowchart is mandatory. Output it ONLY as a valid fenced Mermaid block:
  ```mermaid
  flowchart TD
  A["Short step"] --> B["Short step"]
  ```
- Keep the flowchart vertical (`flowchart TD`), compact, and fast to scan: usually 4–9 nodes.
  It should capture the diagnostic/clinical reasoning/management sequence without becoming a textbook paragraph.
- Do not put long doses, long explanations, citations, or paragraphs inside Mermaid nodes.
- At least one Markdown table is mandatory.
- Use additional tables when the question contains distinct structured domains such as:
  diagnostic pattern + differential, staging + management, or drug choice + dosing.
- Never use raw HTML such as `<br>` inside a Markdown table cell; use short phrases separated by semicolons.
- Use plain Unicode (≥, ≤, ±, →, mg/dL, mmHg). Never use LaTeX.
- End with one short educational/CDS disclaimer.
"""


DOCTOR_FALLBACK_PROMPT = """You are MediCare AI in DOCTOR MODE.
The retrieved knowledge base and PubMed search did not provide sufficiently grounded evidence for this query.

Answer from general medical knowledge only with extreme caution:
- Start with: "ℹ️ **Clinical Note:** *This answer is not directly grounded in the indexed clinical knowledge base for this query.*"
- Keep the response concise and management-focused.
- Do not invent guideline-specific thresholds, doses, recommendation classes, or claims of currency.
- Explicitly identify uncertainty where it matters.
- Include one compact Mermaid flowchart and one cautious Markdown summary table using only information you are confident about.
- End with a short clinical decision-support disclaimer.
"""


PG_STUDENT_FALLBACK_PROMPT = """You are MediCare AI in PG STUDENT MODE.
The retrieved knowledge base and PubMed search did not provide sufficiently grounded evidence for this query.

Answer from general medical knowledge only with extreme caution:
- Start with: "ℹ️ **Clinical Note:** *This answer is not directly grounded in the indexed clinical knowledge base for this query.*"
- Provide an educational overview with mechanisms and clinical reasoning, but do not claim guideline currency.
- Clearly distinguish general principles from uncertain/current-treatment details.
- Include one compact Mermaid flowchart and one cautious Markdown summary table using only information you are confident about.
- End with a short educational/CDS disclaimer.
"""


def _chunk_source_kind(chunk: dict) -> str:
    source_type = str(chunk.get("source_type") or "").strip().lower()
    if source_type == "textbook":
        return "Textbook"
    if source_type in GUIDELINE_SOURCE_TYPES or "guideline" in source_type or "standard" in source_type or "programme" in source_type:
        return "Guideline"
    return "Clinical Reference"


def _format_chunk_for_prompt(chunk: dict, index: int) -> str:
    kind = _chunk_source_kind(chunk)
    title = chunk.get("official_title") or chunk.get("book") or "Medical Reference"
    organization = chunk.get("organization") or ""
    year = chunk.get("publication_year")
    country = chunk.get("country") or ""
    page = chunk.get("page", "N/A")
    source_id = chunk.get("source_id") or ""
    is_current = chunk.get("is_current")

    metadata = [f"{kind} {index}", f"Title: {title}"]
    if organization:
        metadata.append(f"Organization: {organization}")
    if year not in (None, ""):
        metadata.append(f"Year: {year}")
    if country:
        metadata.append(f"Country: {country}")
    if is_current is not None:
        metadata.append(f"Current: {bool(is_current)}")
    if source_id:
        metadata.append(f"Source ID: {source_id}")
    metadata.append(f"Page: {page}")

    return f"[{' | '.join(metadata)}]\n{chunk.get('text', '')}"


def build_hybrid_prompt(question, text_chunks, pubmed_papers, chat_history=None, assistant_mode="doctor"):
    assistant_mode = normalize_assistant_mode(assistant_mode)
    context_sections = []

    if text_chunks:
        guideline_parts = []
        textbook_parts = []
        other_parts = []

        for i, chunk in enumerate(text_chunks, 1):
            formatted = _format_chunk_for_prompt(chunk, i)
            kind = _chunk_source_kind(chunk)
            if kind == "Guideline":
                guideline_parts.append(formatted)
            elif kind == "Textbook":
                textbook_parts.append(formatted)
            else:
                other_parts.append(formatted)

        if guideline_parts:
            context_sections.append(
                "=== RETRIEVED CLINICAL GUIDELINES / STANDARDS ===\n"
                + "\n---\n".join(guideline_parts)
            )
        if textbook_parts:
            context_sections.append(
                "=== RETRIEVED MEDICAL TEXTBOOKS ===\n"
                + "\n---\n".join(textbook_parts)
            )
        if other_parts:
            context_sections.append(
                "=== OTHER RETRIEVED CLINICAL REFERENCES ===\n"
                + "\n---\n".join(other_parts)
            )

    if pubmed_papers:
        pm_text = format_pubmed_for_prompt(pubmed_papers)
        context_sections.append("=== LIVE NIH PUBMED RESEARCH PAPERS ===\n" + pm_text)

    full_context = "\n\n".join(context_sections)

    history_text = ""
    if chat_history:
        history_parts = []
        for msg in chat_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                history_parts.append(f"{role.upper()}: {content[:500]}")
        if history_parts:
            history_text = (
                "\n=== PREVIOUS CONVERSATION CONTEXT ===\n"
                + "\n".join(history_parts)
                + "\n\n"
            )

    answer_instruction = (
        "Answer for a practicing clinician. Be concise and action-oriented."
        if assistant_mode == "doctor"
        else "Answer for a postgraduate medical student. Add mechanisms, rationale, and exam-relevant depth."
    )

    return f"""{history_text}PROVIDED CLINICAL KNOWLEDGE BASE:
{full_context}

---
CLINICAL QUERY: {question}

MODE-SPECIFIC INSTRUCTION: {answer_instruction}

Use the evidence hierarchy in the system instructions. Ground the answer in the supplied context and adapt the structure to the actual question rather than mechanically including every possible section."""


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


def _system_prompt_for(assistant_mode: str, fallback: bool = False) -> str:
    assistant_mode = normalize_assistant_mode(assistant_mode)
    if fallback:
        return PG_STUDENT_FALLBACK_PROMPT if assistant_mode == "pg_student" else DOCTOR_FALLBACK_PROMPT
    return PG_STUDENT_MODE_PROMPT if assistant_mode == "pg_student" else DOCTOR_MODE_PROMPT



def _explicitly_requests_flowchart(question: str) -> bool:
    q = (question or "").lower()
    terms = (
        "flowchart", "flow chart", "decision tree", "decision-tree",
        "clinical pathway", "management pathway", "algorithm diagram",
        "visual algorithm", "mermaid"
    )
    return any(term in q for term in terms)


def _is_medication_dose_query(question: str) -> bool:
    q = (question or "").lower()
    medication_terms = ("medication", "medications", "drug", "drugs", "pharmacotherapy", "medicine")
    dose_terms = ("dose", "doses", "dosage", "dosages", "titrate", "titration")
    stage_terms = ("stage", "stages", "grade", "grades", "when to start", "first line", "first-line")
    return (
        any(x in q for x in medication_terms)
        and (any(x in q for x in dose_terms) or any(x in q for x in stage_terms))
    )


def _is_case_based_query(question: str) -> bool:
    q = (question or "").lower()
    markers = (
        "year-old", "year old", "presents with", "patient presents", "cbc shows",
        "ct shows", "mri shows", "x-ray shows", "xray shows", "serum ferritin",
        "clinical significance", "differential diagnosis", "next diagnostic",
        "tissue diagnosis", "staging", "case"
    )
    return any(marker in q for marker in markers)


def _presentation_instruction(question: str) -> str:
    q = (question or "").lower()

    base = """
FINAL PRESENTATION CONTRACT — MANDATORY FOR EVERY SUBSTANTIVE MEDICAL ANSWER:

A. FAST-SCAN FLOWCHART — REQUIRED
- Include a section titled exactly: `### Quick Clinical Flow`
- Immediately under that heading, output a VALID fenced Mermaid diagram.
- Use `flowchart TD` only. Never use `LR`.
- Usually use 4–8 nodes (maximum about 10 unless the question truly needs more).
- Node labels must be short, clinically meaningful, and easy to read on a phone.
- Use quoted node labels such as `A["Confirm diagnosis"]`.
- Keep detailed doses, long criteria, and long explanations OUT of the flowchart.
- The flowchart should contain the main matter a clinician/student needs when scanning quickly.
- Never output an ASCII tree or box-drawing flowchart.

B. STRUCTURED TABLE — REQUIRED
- Include at least ONE real GitHub-flavored Markdown table.
- The table must summarize the most useful structured clinical information for the question.
- Keep cells concise and readable.
- Never use raw HTML such as `<br>` inside table cells.
- Use additional tables when the question contains multiple structured domains.

C. PLACEMENT
- Put the Quick Clinical Flow and first high-value table near the top of the answer,
  after only a brief clinical overview/bottom line.
- Do not bury both visual summaries at the end.

D. RESPONSIVENESS / READABILITY
- Keep Mermaid labels short enough to remain legible on mobile.
- Prefer 2–5 table columns. Use more columns only when clinically necessary.
- Do not duplicate entire paragraphs inside the table or flowchart.
- The prose should explain; the flowchart should navigate; the table should organize.
"""

    if _is_medication_dose_query(question):
        return base + """
QUESTION-SPECIFIC VISUAL PLAN:
- Flowchart: confirmation/severity → stage/grade → treatment intensity → escalation/emergency gate.
- Table 1: stage/grade and when to start treatment.
- Table 2: medication class, example agent, starting dose, titration/range when supported, key cautions.
- Keep pathophysiology secondary unless specifically requested.
"""

    if _is_case_based_query(question):
        return base + """
QUESTION-SPECIFIC VISUAL PLAN:
- Flowchart: presentation/key findings → leading interpretation → confirmatory diagnostic step →
  severity/staging step → management decision.
- Use at least one table for the highest-yield structured domain such as:
  key findings/interpretation, differential diagnosis, diagnostic tests, staging, or treatment implications.
- If the case asks both differential/work-up AND management/staging, two tables are usually appropriate.
"""

    return base + """
QUESTION-SPECIFIC VISUAL PLAN:
- Flowchart: convert the central concept into a short clinical sequence
  (cause/risk → mechanism/assessment → diagnosis/severity → action/monitoring).
- Table: summarize the core diagnostic, therapeutic, comparative, or high-yield facts.
"""


def _looks_like_ascii_pathway(body: str) -> bool:
    if not body:
        return False
    tree_chars = sum(body.count(c) for c in ("│", "├", "└", "►", "→"))
    clinical_words = re.search(
        r"\b(stage|grade|therapy|treatment|management|blood pressure|diagnosis|start|initiate|referral|emergency)\b",
        body,
        flags=re.IGNORECASE,
    )
    return tree_chars >= 3 and bool(clinical_words)


def _ascii_pathway_to_markdown(body: str) -> str:
    """Convert a stray ASCII clinical pathway into readable Markdown bullets."""
    rows = []
    for raw in body.splitlines():
        if not raw.strip() or not re.search(r"[A-Za-z0-9\[]", raw):
            continue

        match = re.search(r"[A-Za-z0-9\[]", raw)
        start = match.start() if match else 0
        level = min(3, max(0, start // 6))
        cleaned = re.sub(r"^[\s│├└─►→]+", "", raw).strip().strip("[] ")

        if cleaned:
            rows.append((level, cleaned))

    if not rows:
        return body.strip()

    out = ["**Clinical pathway**"]
    for level, item in rows:
        out.append(f"{'  ' * level}- {item}")
    return "\n".join(out)


def _ascii_pathway_to_stage_table(body: str) -> str:
    """Convert a stage/grade ASCII pathway into a compact Markdown table."""
    cleaned_lines = []
    for raw in body.splitlines():
        if not raw.strip() or not re.search(r"[A-Za-z0-9\[]", raw):
            continue
        cleaned = re.sub(r"^[\s│├└─►→]+", "", raw).strip().strip("[] ")
        if cleaned:
            cleaned_lines.append(cleaned)

    stages = []
    current = None

    for line in cleaned_lines:
        if re.search(r"\b(?:grade|stage)\s*\d+\b", line, flags=re.IGNORECASE):
            current = {"stage": line, "details": []}
            stages.append(current)
        elif current is not None:
            current["details"].append(line)

    if not stages:
        return _ascii_pathway_to_markdown(body)

    rows = [
        "| Stage / BP range | Recommended approach |",
        "| --- | --- |",
    ]

    for stage in stages:
        details = "; ".join(stage["details"]) if stage["details"] else "Follow the applicable guideline pathway."
        stage_text = stage["stage"].replace("|", "/")
        details = details.replace("|", "/")
        rows.append(f"| **{stage_text}** | {details} |")

    return "\n".join(rows)


def _sanitize_generated_markdown(text: str, allow_mermaid: bool, medication_dose_query: bool = False) -> str:
    """
    Preserve valid Mermaid only when explicitly requested.
    Convert accidental ASCII clinical code blocks into readable Markdown.
    Remove generic fences from ordinary prose-like blocks.
    """
    if not text:
        return ""

    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

    def repl(match):
        lang = (match.group(1) or "").strip().lower()
        body = (match.group(2) or "").strip()

        if lang == "mermaid" and allow_mermaid:
            return f"```mermaid\n{body}\n```"

        if _looks_like_ascii_pathway(body):
            if medication_dose_query and not allow_mermaid:
                return _ascii_pathway_to_stage_table(body)
            return _ascii_pathway_to_markdown(body)

        # Do not leave clinical prose trapped in an ugly generic code box.
        if lang not in ("python", "javascript", "js", "json", "bash", "sh", "powershell"):
            return body

        return match.group(0)

    return pattern.sub(repl, text)



def _has_markdown_table(text: str) -> bool:
    if not text:
        return False
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        header = lines[i].strip()
        sep = lines[i + 1].strip()
        if header.startswith("|") and header.endswith("|") and sep.startswith("|") and sep.endswith("|"):
            if re.search(r"\|\s*:?-{3,}:?\s*(?=\|)", sep):
                return True
    return False


def _has_mermaid_flowchart(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(
            r"```mermaid\s*\n\s*flowchart\s+(?:TD|TB)\b[\s\S]*?```",
            text,
            flags=re.IGNORECASE,
        )
    )


def _repair_missing_visuals(
    question: str,
    answer: str,
    assistant_mode: str,
    need_table: bool,
    need_flowchart: bool,
    max_output_tokens: int,
) -> str:
    """One formatting-repair pass. It must preserve the original answer's medical content."""
    if not need_table and not need_flowchart:
        return answer

    missing = []
    if need_flowchart:
        missing.append("a compact valid Mermaid flowchart")
    if need_table:
        missing.append("a real Markdown table")

    repair_system = """You are a medical answer presentation editor.
Your ONLY job is to improve structure and visual scannability of an already-generated clinical answer.

STRICT RULES:
- Preserve the medical meaning, facts, doses, thresholds, caveats, uncertainty, and disclaimer already present.
- Do NOT introduce new medical claims, doses, tests, diagnoses, staging facts, or treatment recommendations.
- Do NOT silently correct or replace the clinical content with outside knowledge.
- Return the COMPLETE revised answer only.
- Every revised answer must contain at least one valid Mermaid flowchart and one real Markdown table.
- Mermaid must use `flowchart TD`, short quoted labels, and no ASCII-art.
- Tables must be GitHub-flavored Markdown and must not contain raw HTML such as <br>.
"""

    repair_prompt = f"""ORIGINAL QUESTION:
{question}

MISSING REQUIRED VISUAL ELEMENTS:
{", ".join(missing)}

ORIGINAL ANSWER:
{answer}

REVISE THE ORIGINAL ANSWER FOR PRESENTATION ONLY.

Required placement:
1. Keep only a brief opening overview.
2. Add `### Quick Clinical Flow` near the top with a compact `flowchart TD` Mermaid diagram.
3. Add at least one high-value Markdown table near the top.
4. Preserve useful detailed sections below.
5. Avoid repetition.

Return only the complete revised answer."""

    try:
        repaired = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=repair_prompt,
            config=types.GenerateContentConfig(
                system_instruction=repair_system,
                temperature=0.1,
                max_output_tokens=max_output_tokens,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        repaired_text = clean_latex_symbols(repaired.text or "")
        repaired_text = _sanitize_generated_markdown(
            repaired_text,
            allow_mermaid=True,
            medication_dose_query=_is_medication_dose_query(question),
        )
        return repaired_text or answer
    except Exception as e:
        print(f"    Visual-format repair notice: {e}")
        return answer


def generate_answer(
    question: str,
    chunks: list,
    max_retries: int = 3,
    chat_history: list = None,
    assistant_mode: str = "doctor",
):
    assistant_mode = normalize_assistant_mode(assistant_mode)

    # IMPORTANT: assistant_mode must be part of the cache key or Doctor/PG answers
    # for the same question can collide.
    cache_key = (assistant_mode, question.lower().strip())
    if cache_key in _response_cache and not chat_history:
        print(f"    Mode: CACHED ({assistant_mode})")
        return _response_cache[cache_key]

    print("    Searching NIH PubMed for relevant peer-reviewed trials...")
    pubmed_papers = search_pubmed_live(question, max_results=5)

    mode, max_score = determine_mode(chunks, pubmed_papers)

    if mode in ["books_and_research", "books"]:
        accuracy_percentage = round(max_score * 100, 1)
    elif mode == "research":
        accuracy_percentage = 85.0
    else:
        accuracy_percentage = 0.0

    print(
        f"    Mode: {mode.upper()} | Assistant: {assistant_mode} | "
        f"Similarity: {accuracy_percentage}% "
        f"({len(chunks)} chunks, {len(pubmed_papers)} papers)"
    )

    if mode == "fallback":
        prompt = (
            f"CLINICAL QUERY: {question}\n"
            "Provide a cautious response appropriate to the selected assistant mode."
        )
        system_prompt = _system_prompt_for(assistant_mode, fallback=True)
    else:
        prompt = build_hybrid_prompt(
            question,
            chunks,
            pubmed_papers,
            chat_history=chat_history,
            assistant_mode=assistant_mode,
        )
        prompt += "\n\n" + _presentation_instruction(question)
        system_prompt = _system_prompt_for(assistant_mode, fallback=False)

    max_output_tokens = 2200 if assistant_mode == "doctor" else 3600

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    max_output_tokens=max_output_tokens,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )

            raw_text = response.text or ""
            answer = clean_latex_symbols(raw_text)
            answer = _sanitize_generated_markdown(
                answer,
                allow_mermaid=True,
                medication_dose_query=_is_medication_dose_query(question),
            )

            need_flowchart = not _has_mermaid_flowchart(answer)
            need_table = not _has_markdown_table(answer)

            if need_flowchart or need_table:
                print(
                    "    Presentation repair:",
                    f"flowchart_missing={need_flowchart}",
                    f"table_missing={need_table}"
                )
                answer = _repair_missing_visuals(
                    question=question,
                    answer=answer,
                    assistant_mode=assistant_mode,
                    need_table=need_table,
                    need_flowchart=need_flowchart,
                    max_output_tokens=max_output_tokens,
                )

            mode_labels = {
                "books_and_research": "📚🔬 [Grounded in Clinical Guidelines, Medical Textbooks & NIH PubMed Research]",
                "books": "📚 [Grounded in Clinical Guidelines & Medical Textbooks]",
                "research": "🔬 [Grounded in Live NIH PubMed Clinical Research Papers]",
                "fallback": "🤖 [General Clinical Knowledge — Not directly grounded in indexed clinical sources]",
            }

            labeled_answer = f"{mode_labels.get(mode, '')}\n\n{answer}"
            result = (labeled_answer, mode, accuracy_percentage, pubmed_papers)

            if not chat_history:
                _response_cache[cache_key] = result

            return result

        except Exception as e:
            print(f"    Generation attempt {attempt + 1} notice: {e}")
            time.sleep(2)
            continue

    return "Failed to generate response. Please try again.", "error", 0.0, []


def format_sources(chunks: list, mode: str, pubmed_papers: list = None):
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
                    "score": score,
                })

    return tb_sources, pubmed_papers or []
