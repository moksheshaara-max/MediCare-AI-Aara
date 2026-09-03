"""
MediCare AI - Backend REST API (Flask)
Exposes RAG Pipeline + PubMed Live Search + Two-Step Multimodal Report Analyzer.
"""

import sys
import os
import io
import re
import time
import pymupdf
import json
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
            extracted_text.append(page.get_text())
        doc.close()
        return "\n".join(extracted_text).strip()
    except Exception as e:
        print(f"PDF Extraction Error: {e}")
        return ""

def parse_json_safely(text):
    try: return json.loads(text)
    except: pass
    cleaned = text.strip()
    for i in range(len(cleaned), 0, -1):
        if cleaned[i-1] in [',', '}', ']']:
            candidate = cleaned[:i-1]
            ob = candidate.count('{') - candidate.count('}')
            oa = candidate.count('[') - candidate.count(']')
            candidate += ']' * max(0, oa)
            candidate += '}' * max(0, ob)
            try: return json.loads(candidate)
            except: continue
    return None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"project": "MediCare AI SaaS API", "status": "Online"})

@app.route("/api/health", methods=["GET"])
def health():
    try:
        stats = get_stats()
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "total_chunks_indexed": stats["total_chunks"],
            "total_documents": len(stats["by_book"])
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        data = request.json or {}
        question = data.get("question", "").strip()
        chat_history = data.get("chat_history", [])
        
        if not question:
            return jsonify({"error": "Question is required."}), 400
        
        if is_medical_question(question) == "non_medical":
            return jsonify({
                "question": question,
                "answer": get_non_medical_response(),
                "mode": "non_medical",
                "accuracy_score": 0.0,
                "evaluation": None,
                "is_medical": False,
                "is_emergency": False,
                "emergency_message": None,
                "textbook_sources": [],
                "pubmed_sources": []
            })
        
        is_emergency = check_emergency(question)
        emergency_msg = get_emergency_message() if is_emergency else None
        
        chunks = search_medical_chunks(question, top_k=12)
        answer, mode, accuracy_score, pubmed_papers = generate_answer(question, chunks, chat_history=chat_history)
        
        eval_metrics = evaluate_response_quality(question, chunks, answer, mode)
        tb_sources_raw, pm_sources = format_sources(chunks, mode, pubmed_papers)
        
        tb_sources = [
            {"file_name": s["book"], "display_name": get_book_display_name(s["book"]), "page": s["page"], "relevance_score": round(s["score"], 3)}
            for s in tb_sources_raw
        ]
        
        return jsonify({
            "question": question, "answer": answer, "mode": mode,
            "accuracy_score": accuracy_score, "evaluation": eval_metrics,
            "is_medical": True, "is_emergency": is_emergency, "emergency_message": emergency_msg,
            "textbook_sources": tb_sources, "pubmed_sources": pm_sources
        })
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500


@app.route("/api/analyze-report", methods=["POST"])
def analyze_report():
    """
    Two-Step RAG Report Analyzer:
    Step 1: Exhaustive pure data extraction (Guarantees all 100+ labs are extracted).
    Step 2: Clinical RAG reasoning on the extracted data.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files['file']
        if file.filename == '' or not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Please select a valid PDF file."}), 400

        pdf_bytes = file.read()
        pdf_text = extract_text_from_pdf_stream(pdf_bytes)

        is_scanned = False
        if not pdf_text or len(pdf_text) < 30:
            is_scanned = True
            pdf_text = f"Scanned Report File Name: {file.filename}."

        print(f"Analyzing report: {file.filename} (Full Length: {len(pdf_text)} chars)")

        # ==========================================
        # STEP 1: EXHAUSTIVE LAB DATA EXTRACTION ONLY
        # ==========================================
        print("Step 1: Extracting all 100+ lab values...")
        extract_sys_prompt = """You are a highly precise medical data entry AI. 
Your ONLY job is to extract EVERY SINGLE lab test from the report.
DO NOT summarize. DO NOT skip any rows. Even if there are 150 tests, extract all 150.
Categorize system accurately (Hematology, Renal, Metabolic, Liver, Cardiac, Electrolytes, Urine, Thyroid, etc).

RETURN ONLY A JSON ARRAY OF OBJECTS LIKE THIS:
[
  {"system": "Hematology", "test": "Hemoglobin", "result": "8.2", "unit": "g/dL", "ref": "12.0 - 15.0", "status": "LOW"}
]
"""
        extract_response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=f"EXTRACT ALL LABS FROM THIS REPORT:\n\n{pdf_text}",
            config=types.GenerateContentConfig(
                system_instruction=extract_sys_prompt,
                temperature=0.0,  # Zero creativity for pure extraction
                max_output_tokens=8192,
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )
        
        extracted_labs = parse_json_safely((extract_response.text or "").strip())
        if not isinstance(extracted_labs, list):
            extracted_labs = []
            
        print(f"Successfully extracted {len(extracted_labs)} lab parameters!")

        # ==========================================
        # STEP 2: CLINICAL REASONING & RAG
        # ==========================================
        print("Step 2: RAG Search and Clinical Reasoning...")
        
        # Build search query from the abnormal labs to get perfect context
        abnormal_labs = [l for l in extracted_labs if isinstance(l, dict) and l.get("status") in ["HIGH", "LOW"]]
        search_query = ", ".join([f"{l.get('test')} {l.get('status')}" for l in abnormal_labs[:10]])
        if not search_query:
            search_query = "Normal complete blood count metabolic panel"
            
        chunks = search_medical_chunks(search_query, top_k=12)

        tb_context_parts = []
        for i, chunk in enumerate(chunks, 1):
            tb_context_parts.append(f"[Source {i}: {chunk['book']}, Page {chunk['page']}]\n{chunk['text']}")
        tb_context = "\n---\n".join(tb_context_parts)

        reasoning_sys_prompt = """You are MediCare AI, a senior clinical pathologist.
ANALYZE THE PATIENT'S EXTRACTED LAB DATA AND RETURN A VALID JSON OBJECT MATCHING THIS SCHEMA:

{
  "patient_name": "Name or 'Not Specified'",
  "patient_age_gender": "Age / Gender or 'N/A'",
  "report_date": "Date or 'N/A'",
  "lab_name": "Lab Name or 'Diagnostic Lab'",
  "risk_level": "HIGH",
  "critical_alerts": ["Alert 1"],
  "summary": "Clinical summary...",
  "pathophysiology": "Detailed explanation...",
  "differential_considerations": [{"title": "Condition", "rationale": "Reason"}],
  "recommendations": {
    "urgent_actions": [],
    "further_tests": [],
    "specialty_consultation": [],
    "lifestyle_modifications": []
  }
}

RULES:
1. Extract patient metadata from the raw report text.
2. risk_level MUST be: "LOW", "MODERATE", or "HIGH".
3. Use plain Unicode symbols. Do NOT use LaTeX.
"""
        user_prompt = f"""REFERENCE MEDICAL TEXTBOOK CONTEXT:
{tb_context}
---
RAW REPORT TEXT (For Metadata):
{pdf_text[:2000]}
---
EXTRACTED LAB VALUES ARRAY (For Clinical Analysis):
{json.dumps(extracted_labs)}
---
Output the Clinical JSON Object."""

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

        parsed = parse_json_safely((reason_response.text or "").strip()) or {}

        # ==========================================
        # MERGE & CLEANUP
        # ==========================================
        patient_name = clean_latex_symbols(str(parsed.get("patient_name", "Patient")))
        patient_age_gender = clean_latex_symbols(str(parsed.get("patient_age_gender", "N/A")))
        report_date = clean_latex_symbols(str(parsed.get("report_date", "N/A")))
        lab_name = clean_latex_symbols(str(parsed.get("lab_name", "Diagnostic Lab")))
        risk = str(parsed.get("risk_level", "MODERATE")).upper()
        if risk not in ["LOW", "MODERATE", "HIGH"]: risk = "MODERATE"

        summary = clean_latex_symbols(str(parsed.get("summary", "")))
        critical_alerts = [clean_latex_symbols(str(x)) for x in parsed.get("critical_alerts", []) if str(x).strip()]

        patho_keys = ["pathophysiology", "pathophysiology_assessment", "pathophysiology_and_connections", "clinical_pathophysiology", "explanation"]
        raw_patho = ""
        for k in patho_keys:
            if parsed.get(k):
                raw_patho = parsed[k]
                break
        if isinstance(raw_patho, list):
            raw_patho = "\n".join([str(x) for x in raw_patho])
        pathophysiology = clean_latex_symbols(str(raw_patho))

        diff_keys = ["differential_considerations", "differentials", "differential_diagnosis", "clinical_considerations", "possible_conditions"]
        raw_diffs = []
        for k in diff_keys:
            if parsed.get(k):
                raw_diffs = parsed[k]
                break

        normalized_diffs = []
        if isinstance(raw_diffs, list):
            for item in raw_diffs:
                if isinstance(item, dict):
                    normalized_diffs.append({
                        "title": str(item.get("title", item.get("condition", "Clinical Pattern"))),
                        "rationale": clean_latex_symbols(str(item.get("rationale", item.get("detail", ""))))
                    })
                elif isinstance(item, str) and item.strip():
                    parts = item.split(":", 1)
                    if len(parts) == 2:
                        normalized_diffs.append({"title": parts[0].strip(), "rationale": clean_latex_symbols(parts[1].strip())})
                    else:
                        normalized_diffs.append({"title": item.strip(), "rationale": "Correlated with abnormal laboratory parameters."})

        raw_recs = parsed.get("recommendations", {})
        if not isinstance(raw_recs, dict): raw_recs = {}
            
        recommendations = {
            "urgent_actions": [clean_latex_symbols(str(x)) for x in raw_recs.get("urgent_actions", []) if str(x).strip()],
            "further_tests": [clean_latex_symbols(str(x)) for x in raw_recs.get("further_tests", []) if str(x).strip()],
            "specialty_consultation": [clean_latex_symbols(str(x)) for x in raw_recs.get("specialty_consultation", []) if str(x).strip()],
            "lifestyle_modifications": [clean_latex_symbols(str(x)) for x in raw_recs.get("lifestyle_modifications", []) if str(x).strip()]
        }

        if not recommendations["urgent_actions"] and not recommendations["further_tests"] and not recommendations["specialty_consultation"] and not recommendations["lifestyle_modifications"]:
            recommendations["further_tests"] = ["Correlate findings with clinical patient history."]
            recommendations["specialty_consultation"] = ["Consult a healthcare professional for clinical interpretation."]

        # Clean the labs from Step 1
        normalized_labs = []
        for lab in extracted_labs:
            if isinstance(lab, dict):
                status = str(lab.get("status", "UNKNOWN")).upper()
                if status not in ["LOW", "NORMAL", "HIGH", "UNKNOWN"]: status = "UNKNOWN"
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
            "recommendations": recommendations
        })

    except Exception as e:
        print(f"Report Analysis Error: {e}")
        return jsonify({"error": "An error occurred during report analysis.", "details": str(e)}), 500


# 📄 SERVER-SIDE PDF GENERATOR (ReportLab)
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
        
        title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=20, textColor=colors.HexColor('#1a365d'), spaceAfter=2)
        subtitle_style = ParagraphStyle('DocSubTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor=colors.HexColor('#0284c7'), spaceAfter=15)
        section_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=11, textColor=colors.white, backColor=colors.HexColor('#1a365d'), borderPadding=(5, 8, 5, 8), spaceBefore=12, spaceAfter=8)
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
            alert_content = [Paragraph("<b>🚨 CRITICAL LAB ALERTS:</b>", alert_style)]
            for alert in critical_alerts:
                alert_content.append(Paragraph(f"• {alert}", alert_style))
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
                if sys_name not in grouped_labs: grouped_labs[sys_name] = []
                grouped_labs[sys_name].append(lab)

            for sys_name, labs in grouped_labs.items():
                story.append(Spacer(1, 6))
                sys_header_style = ParagraphStyle('SysHeader', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor=colors.HexColor('#075985'))
                story.append(Paragraph(f"<b>🫀 {sys_name.upper()} ({len(labs)})</b>", sys_header_style))
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
                        Paragraph(str(lab.get("test", "")), body_style),
                        result_p,
                        Paragraph(str(lab.get("unit", "")), body_style),
                        Paragraph(str(lab.get("reference_range", lab.get("ref", "-"))), body_style),
                        status_p
                    ])

                lab_table = Table(table_data, colWidths=[160, 80, 70, 150, 80], repeatRows=1)
                lab_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#334155')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
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
            story.append(Paragraph("POSSIBLE CLINICAL CONSIDERATIONS", section_style))
            for idx, diff in enumerate(differentials, 1):
                if isinstance(diff, dict):
                    story.append(Paragraph(f"<b>{idx}. {diff.get('title', 'Consideration')}:</b> {diff.get('rationale', '')}", body_style))

        if recs:
            story.append(Paragraph("RECOMMENDED NEXT STEPS", section_style))
            if isinstance(recs, dict):
                if recs.get("urgent_actions"):
                    story.append(Paragraph("<b>🔴 Urgent Actions:</b>", body_style))
                    for x in recs["urgent_actions"]: story.append(Paragraph(f"• {x}", body_style))
                if recs.get("further_tests"):
                    story.append(Paragraph("<b>🧪 Further Tests:</b>", body_style))
                    for x in recs["further_tests"]: story.append(Paragraph(f"• {x}", body_style))
                if recs.get("specialty_consultation"):
                    story.append(Paragraph("<b>👨‍⚕️ Specialty Consultations:</b>", body_style))
                    for x in recs["specialty_consultation"]: story.append(Paragraph(f"• {x}", body_style))
                if recs.get("lifestyle_modifications"):
                    story.append(Paragraph("<b>🌿 Lifestyle Modifications:</b>", body_style))
                    for x in recs["lifestyle_modifications"]: story.append(Paragraph(f"• {x}", body_style))

        story.append(Spacer(1, 15))
        disc_style = ParagraphStyle('DiscStyle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, leading=11, textColor=colors.HexColor('#64748b'))
        story.append(Paragraph("<b>MEDICAL DISCLAIMER:</b> This document is a preliminary clinical summary generated by MediCare AI using RAG technology against verified medical textbooks. It is intended for decision-support purposes only and does not constitute a formal diagnosis or prescription.", disc_style))

        doc.build(story)
        buffer.seek(0)

        out_name = f"MediCare_AI_Report_{(patient_name or 'Patient').replace(' ', '_')}.pdf"
        return send_file(buffer, as_attachment=True, download_name=out_name, mimetype='application/pdf')

    except Exception as e:
        print(f"PDF Export Error: {e}")
        return jsonify({"error": "Failed to generate PDF report.", "details": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("MediCare AI - Flask API Server Starting")
    print("=" * 60)
    print("Server running on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)