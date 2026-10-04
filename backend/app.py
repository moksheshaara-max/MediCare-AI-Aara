"""
MediCare AI - Backend REST API (Flask)
Exposes RAG Pipeline + PubMed Live Search + Two-Step Lab Report Analyzer (PyMuPDF + RAG).
Uses gemini-flash-lite-latest ONLY.
"""

import sys
import os
import io
import re
import json
import time
import pymupdf
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval.search import (
    search_medical_chunks,
    is_medical_question,
    get_non_medical_response,
    check_emergency,
    get_emergency_message
)
from generation.generator import generate_answer, format_sources, clean_latex_symbols, client, GENERATION_MODEL
from database.mongo_store import get_stats
from evaluator.rag_evaluator import evaluate_response_quality
from google.genai import types

app = Flask(__name__)
CORS(app)

ASSISTANT_MODE_ALIASES = {
    "doctor": "doctor",
    "clinician": "doctor",
    "physician": "doctor",
    "pg": "pg_student",
    "pg_student": "pg_student",
    "pg-student": "pg_student",
    "student": "pg_student",
}


def normalize_assistant_mode_input(value):
    """Normalize the product-facing Clinical Assistant mode."""
    normalized = str(value or "doctor").strip().lower()
    return ASSISTANT_MODE_ALIASES.get(normalized)


BOOK_TITLES = {
    "book1.pdf": "Hutchison's Clinical Methods (25th Ed)",
    "book2.pdf": "Manual of Practical Medicine (R Alagappan)",
    "harrisons.pdf": "Harrison's Principles of Internal Medicine (2022)",
    "davidson.pdf": "Davidson's Principles of Medicine",
    "kd_tripathi.pdf": "KD Tripathi Essentials of Medical Pharmacology",
    "icmr_diabetes.pdf": "ICMR Guidelines for Diabetes Management",
    "who_essential_meds.pdf": "WHO Model List of Essential Medicines",
    "icmr_antimicrobial.pdf": "ICMR Antimicrobial Stewardship Guidelines",
    "mohfw_stw.pdf": "MoHFW Standard Treatment Workflows (India)",
    "who_hypertension.pdf": "WHO Clinical Guidelines for Hypertension",
    "Type2Diabetes.pdf": "Clinical Guidelines for Type 2 Diabetes"
}


def get_book_display_name(filename):
    if filename in BOOK_TITLES:
        return BOOK_TITLES[filename]
    return filename.replace(".pdf", "").replace("_", " ").title()


def extract_text_from_pdf_stream(pdf_bytes):
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        extracted_text = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text:
                extracted_text.append(text)
        doc.close()
        return "\n".join(extracted_text).strip()
    except Exception as e:
        print(f"PDF Extraction Error: {e}")
        return ""


def parse_json_safely(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    for i in range(len(cleaned), 0, -1):
        if cleaned[i-1] in [',', '}', ']']:
            candidate = cleaned[:i-1]
            ob = candidate.count('{') - candidate.count('}')
            oa = candidate.count('[') - candidate.count(']')
            candidate += ']' * max(0, oa)
            candidate += '}' * max(0, ob)
            try:
                return json.loads(candidate)
            except Exception:
                continue
    return None


@app.route("/", methods=["GET"])
def home():
    return jsonify({"project": "MediCare AI SaaS API", "status": "Online", "model": GENERATION_MODEL})


@app.route("/api/health", methods=["GET"])
def health():
    try:
        stats = get_stats()
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "total_chunks_indexed": stats["total_chunks"],
            "total_documents": len(stats["by_book"]),
            "generation_model": GENERATION_MODEL
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        data = request.json or {}
        question = data.get("question", "").strip()
        chat_history = data.get("chat_history", [])
        assistant_mode = normalize_assistant_mode_input(data.get("assistant_mode", "doctor"))

        if not question:
            return jsonify({"error": "Question is required."}), 400

        if assistant_mode is None:
            return jsonify({
                "error": "Invalid assistant_mode.",
                "allowed_modes": ["doctor", "pg_student"]
            }), 400

        if is_medical_question(question) == "non_medical":
            return jsonify({
                "question": question,
                "answer": get_non_medical_response(),
                "mode": "non_medical",  # legacy alias; prefer evidence_mode
                "evidence_mode": "non_medical",
                "assistant_mode": assistant_mode,
                "accuracy_score": 0.0,  # legacy alias; prefer retrieval_confidence
                "retrieval_confidence": 0.0,
                "evaluation": None,
                "is_medical": False,
                "is_emergency": False,
                "emergency_message": None,
                "textbook_sources": [],
                "pubmed_sources": []
            })

        is_emergency = check_emergency(question)
        emergency_msg = get_emergency_message() if is_emergency else None

        chunks = search_medical_chunks(
            question,
            top_k=36,
            assistant_mode=assistant_mode
        )
        answer, mode, retrieval_confidence, pubmed_papers = generate_answer(
            question,
            chunks,
            chat_history=chat_history,
            assistant_mode=assistant_mode
        )

        eval_metrics = evaluate_response_quality(question, chunks, answer, mode)
        tb_sources_raw, pm_sources = format_sources(chunks, mode, pubmed_papers)

        tb_sources = [
            {"file_name": s["book"], "display_name": get_book_display_name(s["book"]), "page": s["page"], "relevance_score": round(s["score"], 3)}
            for s in tb_sources_raw
        ]

        return jsonify({
            "question": question,
            "answer": answer,
            "mode": mode,  # legacy alias; prefer evidence_mode
            "evidence_mode": mode,
            "assistant_mode": assistant_mode,
            "accuracy_score": retrieval_confidence,  # legacy alias; prefer retrieval_confidence
            "retrieval_confidence": retrieval_confidence,
            "evaluation": eval_metrics,
            "is_medical": True,
            "is_emergency": is_emergency,
            "emergency_message": emergency_msg,
            "textbook_sources": tb_sources,
            "pubmed_sources": pm_sources
        })
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500


@app.route("/api/analyze-report", methods=["POST"])
def analyze_report():
    """
    Two-Step RAG Lab Report Analyzer:
    Step 1: Exhaustive Data Extraction (PyMuPDF -> Gemini Text JSON)
    Step 2: RAG Search + Clinical Reasoning (Prevalence Ordered)
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files['file']
        if file.filename == '' or not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Please select a valid PDF file."}), 400

        pdf_bytes = file.read()
        pdf_text = extract_text_from_pdf_stream(pdf_bytes)

        if not pdf_text or len(pdf_text) < 30:
            return jsonify({"error": "Could not extract readable text from this PDF."}), 400

        print(f"\n[Report Analyzer] Analyzing: {file.filename} ({len(pdf_text)} characters)")

        # STEP 1: EXHAUSTIVE STRUCTURED CLINICAL DATA EXTRACTION
        # Brain 1 extracts every structured clinical measurement/finding in the PDF,
        # not only conventional blood laboratory values.
        print("  Step 1: Extracting all structured clinical parameters...")

        extract_sys_prompt = """You are MediCare AI Brain 1, a meticulous clinical report data-extraction engine.

YOUR ONLY TASK:
Extract EVERY structured clinical parameter, measurement, laboratory result, test result,
quantitative observation, and clinically meaningful structured finding explicitly present
in the supplied report text.

This is NOT limited to blood laboratory tests.

INCLUDE ALL APPLICABLE CONTENT FROM:
- Vitals & anthropometry: BP, pulse, respiratory rate, temperature, SpO2, BMI, waist/hip, etc.
- Hematology / CBC / ESR
- Glycemic / metabolic / insulin resistance
- Lipid profile
- Kidney function / renal markers / electrolytes / minerals
- Liver function
- Thyroid
- Cardiac biomarkers / inflammatory / cardiovascular risk markers
- Vitamins / hormones
- Tumor markers
- Urine studies / urine microscopy / albuminuria / ACR
- Stool examination
- Coagulation, serology, immunology, microbiology, cultures when present
- ECG measurements and structured ECG findings
- Echocardiography measurements and valve findings
- Treadmill / stress-test parameters and structured findings
- Pulmonary function / spirometry
- DEXA / bone mineral density
- Ultrasound measurements and structured organ findings
- X-ray / CT / MRI / radiology structured findings when explicitly stated
- Any OTHER structured diagnostic parameter or measured value present in the report

STRICT EXTRACTION RULES:
1. DO NOT summarize.
2. DO NOT skip rows because they are normal.
3. DO NOT skip qualitative rows such as Trace, Negative, Mild regurgitation,
   No effusion, Sinus, or Inversion in aVL when they are structured results.
4. Preserve report wording and numeric precision. Do not recalculate values.
5. Never invent a unit or reference range. If absent, use an empty string for unit and N/A for ref.
6. If the same parameter occurs in different sections/contexts, preserve BOTH by making the test name context-specific, e.g.:
   Vitals - Blood Pressure (Right Arm)
   Treadmill - Baseline Blood Pressure
   Treadmill - Peak Exercise BP
7. Categorize each item into a useful system/category. Preferred categories:
   Vitals & Anthropometry, Hematology, Glycemic & Metabolic, Lipid,
   Renal & Electrolytes, Liver, Thyroid, Cardiac Biomarkers,
   Vitamins & Hormones, Tumor Markers, Urine, Stool, ECG,
   Echocardiography, Stress Test, Pulmonary Function, DEXA,
   Ultrasound, Radiology, Clinical Examination, Other.
8. Status must be one of HIGH, LOW, NORMAL, ABNORMAL, UNKNOWN.
   - Use HIGH/LOW only when direction is clear from the printed reference range.
   - Use NORMAL when the result clearly satisfies the printed reference/expected finding.
   - Use ABNORMAL for qualitative abnormalities that are not naturally HIGH/LOW.
   - Use UNKNOWN when no reliable status can be determined from the report itself.
9. Extract ONLY information explicitly present in the supplied text.
10. Return ONLY a JSON array. No markdown. No explanation.

JSON OBJECT SCHEMA:
[
  {
    "system": "Hematology",
    "test": "Hemoglobin",
    "result": "14.8",
    "unit": "g/dL",
    "ref": "13.0 - 17.0",
    "status": "NORMAL"
  }
]
"""

        # Overlapping chunks remove the old hard 15,000-character truncation and
        # improve recall on dense multi-page reports.
        extraction_chunk_size = 6000
        extraction_overlap = 700
        extraction_chunks = []

        cursor = 0
        text_length = len(pdf_text)
        while cursor < text_length:
            chunk_end = min(text_length, cursor + extraction_chunk_size)
            extraction_chunks.append(pdf_text[cursor:chunk_end])
            if chunk_end >= text_length:
                break
            cursor = max(cursor + 1, chunk_end - extraction_overlap)

        print(
            f"  Step 1: Processing {len(extraction_chunks)} extraction chunk(s) "
            f"across {text_length} characters..."
        )

        raw_extracted_items = []

        for chunk_index, report_chunk in enumerate(extraction_chunks, start=1):
            chunk_items = []

            for attempt in range(3):
                try:
                    extract_response = client.models.generate_content(
                        model=GENERATION_MODEL,
                        contents=(
                            f"REPORT CHUNK {chunk_index} OF {len(extraction_chunks)}\n"
                            "Extract EVERY structured clinical parameter/finding from this chunk.\n\n"
                            f"{report_chunk}"
                        ),
                        config=types.GenerateContentConfig(
                            system_instruction=extract_sys_prompt,
                            temperature=0.0,
                            max_output_tokens=8192,
                            response_mime_type="application/json",
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                        )
                    )

                    if extract_response and extract_response.text:
                        parsed_chunk = parse_json_safely(extract_response.text)
                        if isinstance(parsed_chunk, list):
                            chunk_items = [item for item in parsed_chunk if isinstance(item, dict)]
                            break

                except Exception as e:
                    print(
                        f"  Step 1 chunk {chunk_index} attempt "
                        f"{attempt + 1} notice: {e}"
                    )
                    time.sleep(2)

            raw_extracted_items.extend(chunk_items)
            print(
                f"    ✓ Chunk {chunk_index}/{len(extraction_chunks)}: "
                f"{len(chunk_items)} structured parameters"
            )

        # Deterministic de-duplication is required because chunks overlap.
        # Genuine repeated measurements survive when their context/result differs.
        extracted_labs = []
        seen_extracted = set()

        for item in raw_extracted_items:
            system = str(item.get("system", "Other")).strip() or "Other"
            test = str(item.get("test", "Clinical Parameter")).strip() or "Clinical Parameter"
            result = str(item.get("result", "-")).strip() or "-"
            unit = str(item.get("unit", "")).strip()
            ref_value = str(item.get("ref", item.get("reference_range", "N/A"))).strip() or "N/A"
            status = str(item.get("status", "UNKNOWN")).upper().strip()

            if status not in ["HIGH", "LOW", "NORMAL", "ABNORMAL", "UNKNOWN"]:
                status = "UNKNOWN"

            dedupe_key = (
                re.sub(r"\s+", " ", system.lower()),
                re.sub(r"\s+", " ", test.lower()),
                re.sub(r"\s+", " ", result.lower()),
                re.sub(r"\s+", " ", unit.lower()),
                re.sub(r"\s+", " ", ref_value.lower()),
            )

            if dedupe_key in seen_extracted:
                continue

            seen_extracted.add(dedupe_key)
            extracted_labs.append({
                "system": system,
                "test": test,
                "result": result,
                "unit": unit,
                "ref": ref_value,
                "status": status,
            })

        extraction_counts = {}
        for item in extracted_labs:
            category = item.get("system", "Other")
            extraction_counts[category] = extraction_counts.get(category, 0) + 1

        print(
            f"  ✓ Step 1 Complete: Extracted "
            f"{len(extracted_labs)} structured clinical parameters."
        )
        for category, count in sorted(
            extraction_counts.items(),
            key=lambda x: (-x[1], x[0].lower())
        ):
            print(f"      - {category}: {count}")

        # STEP 2: CLINICAL REASONING (RAG)
        print("  Step 2: Performing Clinical Reasoning with RAG...")
        abnormal_labs = [
            l for l in extracted_labs
            if isinstance(l, dict) and l.get("status") in ["HIGH", "LOW", "ABNORMAL"]
        ]
        search_query = ", ".join(
            [f"{l.get('test')} {l.get('result')} {l.get('status')}" for l in abnormal_labs[:20]]
        )
        if not search_query:
            search_query = "Normal complete blood count metabolic panel"

        chunks = search_medical_chunks(search_query, top_k=15)
        tb_context = "\n---\n".join([f"[Source {i}: {c['book']}, Page {c['page']}]\n{c['text']}" for i, c in enumerate(chunks, 1)])

        reasoning_sys_prompt = """You are MediCare AI, a senior consultant pathologist and physician.
Analyze the patient lab data and return a JSON object:

{
  "patient_name": "Name or Not Specified",
  "patient_age_gender": "Age / Sex or N/A",
  "report_date": "Date or N/A",
  "lab_name": "Diagnostic Lab Name",
  "risk_level": "HIGH",
  "critical_alerts": ["Alert 1"],
  "summary": "Clinical executive summary...",
  "pathophysiology": "Detailed inter-organ pathophysiological connection assessment...",
  "differential_considerations": [
    {"prevalence": "COMMON", "title": "Condition", "rationale": "Reason"},
    {"prevalence": "LESS COMMON", "title": "Condition", "rationale": "Reason"},
    {"prevalence": "RARE", "title": "Condition", "rationale": "Reason"}
  ],
  "recommendations": {
    "urgent_actions": [],
    "further_tests": [],
    "specialty_consultation": [],
    "lifestyle_modifications": []
  }
}

RULES:
1. ORDER DIFFERENTIALS STRICTLY BY EPIDEMIOLOGICAL PREVALENCE: COMMON first, LESS COMMON middle, RARE last.
2. risk_level MUST be LOW, MODERATE, or HIGH.
3. Use plain Unicode (≥, ≤, %). NO LaTeX.
"""
        user_prompt = f"""REFERENCE TEXTBOOK CONTEXT:
{tb_context}
---
RAW REPORT TEXT (Metadata & Content):
{pdf_text[:3000]}
---
EXTRACTED LAB VALUES:
{json.dumps(extracted_labs)}
---
Output the Clinical JSON Object with Differentials sorted by Prevalence."""

        parsed = {}
        for attempt in range(3):
            try:
                reason_response = client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=reasoning_sys_prompt,
                        temperature=0.1,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
                if reason_response and reason_response.text:
                    parsed = parse_json_safely(reason_response.text) or {}
                    if parsed:
                        break
            except Exception as e:
                print(f"  Step 2 attempt {attempt + 1} notice: {e}")
                time.sleep(2)

        print("  ✓ Step 2 Complete: Clinical reasoning finished.")

        patient_name = clean_latex_symbols(str(parsed.get("patient_name", "Patient")))
        patient_age_gender = clean_latex_symbols(str(parsed.get("patient_age_gender", "N/A")))
        report_date = clean_latex_symbols(str(parsed.get("report_date", "N/A")))
        lab_name = clean_latex_symbols(str(parsed.get("lab_name", "Diagnostic Laboratory")))
        risk = str(parsed.get("risk_level", "MODERATE")).upper()
        if risk not in ["LOW", "MODERATE", "HIGH"]:
            risk = "MODERATE"

        summary = clean_latex_symbols(str(parsed.get("summary", "")))
        critical_alerts = [clean_latex_symbols(str(x)) for x in parsed.get("critical_alerts", []) if str(x).strip()]
        pathophysiology = clean_latex_symbols(str(parsed.get("pathophysiology", "")))

        raw_diffs = parsed.get("differential_considerations", [])
        normalized_diffs = []
        if isinstance(raw_diffs, list):
            for item in raw_diffs:
                if isinstance(item, dict):
                    prev_tag = str(item.get("prevalence", "COMMON")).upper()
                    if prev_tag not in ["COMMON", "LESS COMMON", "RARE"]:
                        prev_tag = "COMMON"
                    normalized_diffs.append({
                        "prevalence": prev_tag,
                        "title": clean_latex_symbols(str(item.get("title", "Clinical Consideration"))),
                        "rationale": clean_latex_symbols(str(item.get("rationale", "")))
                    })

        raw_recs = parsed.get("recommendations", {})
        if not isinstance(raw_recs, dict):
            raw_recs = {}

        recommendations = {
            "urgent_actions": [clean_latex_symbols(str(x)) for x in raw_recs.get("urgent_actions", []) if str(x).strip()],
            "further_tests": [clean_latex_symbols(str(x)) for x in raw_recs.get("further_tests", []) if str(x).strip()],
            "specialty_consultation": [clean_latex_symbols(str(x)) for x in raw_recs.get("specialty_consultation", []) if str(x).strip()],
            "lifestyle_modifications": [clean_latex_symbols(str(x)) for x in raw_recs.get("lifestyle_modifications", []) if str(x).strip()]
        }

        normalized_labs = []
        for lab in extracted_labs:
            if isinstance(lab, dict):
                status = str(lab.get("status", "UNKNOWN")).upper()
                if status not in ["LOW", "NORMAL", "HIGH", "ABNORMAL", "UNKNOWN"]:
                    status = "UNKNOWN"
                normalized_labs.append({
                    "system": str(lab.get("system", "Other")),
                    "test": str(lab.get("test", "Lab Test")),
                    "result": str(lab.get("result", "-")),
                    "unit": str(lab.get("unit", "")),
                    "reference_range": str(lab.get("ref", lab.get("reference_range", "N/A"))),
                    "status": status
                })

        return jsonify({
            "file_name": file.filename,
            "patient_name": patient_name,
            "patient_age_gender": patient_age_gender,
            "report_date": report_date,
            "lab_name": lab_name,
            "risk_level": risk,
            "critical_alerts": critical_alerts,
            "summary": summary,
            "lab_values": normalized_labs,
            "differential_considerations": normalized_diffs,
            "pathophysiology": pathophysiology,
            "recommendations": recommendations,
            "extraction_method": "pymupdf+gemini_chunked_exhaustive"
        })

    except Exception as e:
        print(f"Report Analysis Error: {e}")
        return jsonify({"error": "An error occurred during report analysis.", "details": str(e)}), 500


@app.route("/api/export-pdf", methods=["POST"])
def export_pdf():
    try:
        data = request.json or {}

        patient_name = data.get("patient_name", "Patient")
        patient_age = data.get("patient_age_gender", "N/A")
        lab_name = data.get("lab_name", "Diagnostic Laboratory")
        report_date = data.get("report_date", "N/A")
        file_name = data.get("file_name", "Lab_Report.pdf")
        risk_level = data.get("risk_level", "MODERATE")
        summary = data.get("summary", "")
        pathophysiology = data.get("pathophysiology", "")
        critical_alerts = data.get("critical_alerts", [])
        lab_values = data.get("lab_values", [])
        differentials = data.get("differential_considerations", [])
        recs = data.get("recommendations", {})

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=20, textColor=colors.HexColor('#0369a1'), spaceAfter=2)
        subtitle_style = ParagraphStyle('DocSubTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor=colors.HexColor('#0284c7'), spaceAfter=15)
        section_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=11, textColor=colors.white, backColor=colors.HexColor('#0f172a'), borderPadding=(5, 8, 5, 8), spaceBefore=12, spaceAfter=8)
        body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14, textColor=colors.HexColor('#1e293b'), spaceAfter=8)
        alert_style = ParagraphStyle('AlertText', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=colors.HexColor('#9f1239'), spaceAfter=4)

        story = []
        story.append(Paragraph("MediCare AI", title_style))
        story.append(Paragraph("AUTOMATED CLINICAL LABORATORY REPORT ANALYSIS", subtitle_style))

        meta_data = [
            [Paragraph(f"<b>Patient Name:</b> {patient_name}", body_style), Paragraph(f"<b>Age / Sex:</b> {patient_age}", body_style)],
            [Paragraph(f"<b>Laboratory:</b> {lab_name}", body_style), Paragraph(f"<b>Report Date:</b> {report_date}", body_style)],
            [Paragraph(f"<b>Source File:</b> {file_name}", body_style), Paragraph(f"<b>Triage Risk:</b> <font color='{'#dc2626' if risk_level=='HIGH' else '#d97706'}'><b>{risk_level}</b></font>", body_style)]
        ]
        meta_table = Table(meta_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        if critical_alerts:
            alert_content = [Paragraph("<b>CRITICAL LAB ALERTS:</b>", alert_style)]
            for alert in critical_alerts:
                alert_content.append(Paragraph(f"- {alert}", alert_style))
            alert_table = Table([[alert_content]], colWidths=[540])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef2f2')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#fecdd3')),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(alert_table)
            story.append(Spacer(1, 10))

        if summary:
            story.append(Paragraph("CLINICAL SUMMARY", section_style))
            story.append(Paragraph(summary, body_style))

        if lab_values:
            story.append(Paragraph("EXTRACTED LABORATORY VALUES", section_style))
            grouped_labs = {}
            for lab in lab_values:
                sys_name = lab.get("system", "General / Other")
                if sys_name not in grouped_labs:
                    grouped_labs[sys_name] = []
                grouped_labs[sys_name].append(lab)

            for sys_name, labs in grouped_labs.items():
                story.append(Spacer(1, 6))
                sys_header_style = ParagraphStyle('SysHeader', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor=colors.HexColor('#0369a1'))
                story.append(Paragraph(f"<b>{sys_name.upper()} ({len(labs)})</b>", sys_header_style))
                story.append(Spacer(1, 4))

                table_data = [["TEST", "RESULT", "UNITS", "REF. RANGE", "STATUS"]]
                for lab in labs:
                    st = (lab.get("status") or "UNKNOWN").upper()
                    st_color = "#059669"
                    if st == "HIGH": st_color = "#dc2626"
                    elif st == "LOW": st_color = "#d97706"

                    status_p = Paragraph(f"<font color='{st_color}'><b>{st}</b></font>", body_style)
                    result_p = Paragraph(f"<b>{lab.get('result', '-')}</b>", body_style) if st in ["HIGH", "LOW"] else Paragraph(str(lab.get('result', '-')), body_style)

                    table_data.append([
                        Paragraph(str(lab.get("test", "")), body_style), result_p,
                        Paragraph(str(lab.get("unit", "")), body_style), Paragraph(str(lab.get("reference_range", lab.get("ref", "-"))), body_style),
                        status_p
                    ])

                lab_table = Table(table_data, colWidths=[160, 80, 70, 150, 80], repeatRows=1)
                lab_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#334155')),
                    ('FONTNAME', (0, 0), (-1, 0), "Helvetica-Bold"),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ]))
                story.append(lab_table)

        if pathophysiology:
            story.append(Paragraph("PATHOPHYSIOLOGY ASSESSMENT", section_style))
            story.append(Paragraph(pathophysiology, body_style))

        if differentials:
            story.append(Paragraph("POSSIBLE CLINICAL CONSIDERATIONS (ORDERED BY PREVALENCE)", section_style))
            for idx, diff in enumerate(differentials, 1):
                if isinstance(diff, dict):
                    prev = diff.get("prevalence", "COMMON")
                    story.append(Paragraph(f"<b>{idx}. [{prev}] {diff.get('title', 'Consideration')}:</b> {diff.get('rationale', '')}", body_style))

        if recs and isinstance(recs, dict):
            story.append(Paragraph("RECOMMENDED NEXT STEPS", section_style))
            if recs.get("urgent_actions"):
                story.append(Paragraph("<b>Urgent Actions:</b>", body_style))
                for x in recs["urgent_actions"]: story.append(Paragraph(f"- {x}", body_style))
            if recs.get("further_tests"):
                story.append(Paragraph("<b>Further Tests:</b>", body_style))
                for x in recs["further_tests"]: story.append(Paragraph(f"- {x}", body_style))
            if recs.get("specialty_consultation"):
                story.append(Paragraph("<b>Specialty Consultations:</b>", body_style))
                for x in recs["specialty_consultation"]: story.append(Paragraph(f"- {x}", body_style))
            if recs.get("lifestyle_modifications"):
                story.append(Paragraph("<b>Lifestyle Modifications:</b>", body_style))
                for x in recs["lifestyle_modifications"]: story.append(Paragraph(f"- {x}", body_style))

        story.append(Spacer(1, 15))
        disc_style = ParagraphStyle('DiscStyle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, leading=11, textColor=colors.HexColor('#64748b'))
        story.append(Paragraph("<b>MEDICAL DISCLAIMER:</b> This document is a preliminary clinical decision-support summary generated by MediCare AI. It does not constitute a formal diagnosis or prescription.", disc_style))

        doc.build(story)
        buffer.seek(0)

        out_name = f"MediCare_AI_Report_{(patient_name or 'Patient').replace(' ', '_')}.pdf"
        return send_file(buffer, as_attachment=True, download_name=out_name, mimetype="application/pdf")

    except Exception as e:
        print(f"PDF Export Error: {e}")
        return jsonify({"error": "Failed to generate PDF report.", "details": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("MediCare AI Backend API Starting...")
    print(f"Model: {GENERATION_MODEL}")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)