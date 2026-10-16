"""Focused offline PDF-export checks for MediCare AI v12 severity presentation.
No Flask server, Gemini request, Atlas access, network, or live RAG request.
"""
from __future__ import annotations
import ast
import io
import pathlib
import unittest
from types import SimpleNamespace

import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

APP_PATH = pathlib.Path(__file__).resolve().parents[1] / 'app.py'


def invoke_export(data):
    """Run only the export_pdf function via AST; avoid importing Flask or RAG."""
    tree = ast.parse(APP_PATH.read_text(encoding='utf-8'), filename=str(APP_PATH))
    endpoint = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'export_pdf')
    endpoint.decorator_list = []
    exported = {}

    def fake_send_file(buffer, **kwargs):
        exported['bytes'] = buffer.getvalue()
        exported['download_name'] = kwargs.get('download_name')
        return exported

    def fail_jsonify(info):
        raise AssertionError(f'Export encountered unexpected error: {info}')

    ns = {
        '__name__': 'isolated_export_test',
        'request': SimpleNamespace(json=data),
        'io': io,
        'letter': letter,
        'SimpleDocTemplate': SimpleDocTemplate,
        'Paragraph': Paragraph,
        'Spacer': Spacer,
        'Table': Table,
        'TableStyle': TableStyle,
        'ParagraphStyle': ParagraphStyle,
        'getSampleStyleSheet': getSampleStyleSheet,
        'colors': colors,
        'send_file': fake_send_file,
        'jsonify': fail_jsonify,
        'log_memory': lambda *_args, **_kwargs: None,
    }
    code = compile(ast.Module(body=[endpoint], type_ignores=[]), str(APP_PATH), 'exec')
    exec(code, ns)
    result = ns['export_pdf']()
    assert result.get('bytes', b'').startswith(b'%PDF-'), 'PDF bytes not produced'
    with pymupdf.open(stream=result['bytes'], filetype='pdf') as doc:
        pdf_text = '\n'.join(page.get_text() for page in doc)
    return pdf_text, result


class ExportSeverityPresentationTests(unittest.TestCase):
    def test_model_selected_flags_are_not_labelled_critical(self):
        pdf_text, result = invoke_export({
            'patient_name': 'Not Specified',
            'risk_level': 'HIGH',
            'critical_alerts': [
                'Fasting Blood Glucose 103 mg/dL (Prediabetes range)',
                'Vitamin D 24 ng/mL (Below illustrative interval)',
            ],
            'lab_values': [{
                'system': 'Glycemic & Metabolic', 'test': 'Fasting blood glucose',
                'result': '103', 'unit': 'mg/dL', 'reference_range': '70–99 fasting',
                'status': 'HIGH',
            }],
        })
        self.assertIn('AI-FLAGGED FINDINGS FOR CLINICIAN REVIEW', pdf_text)
        self.assertIn('Severity has not been independently assessed', pdf_text)
        self.assertIn('Fasting Blood Glucose 103 mg/dL', pdf_text)
        self.assertIn('Triage status:', pdf_text)
        self.assertIn('Not assessed', pdf_text)
        self.assertNotIn('CRITICAL LAB ALERTS:', pdf_text)
        self.assertNotIn('Triage Risk:', pdf_text)
        self.assertNotIn('MODERATE', pdf_text)
        self.assertIn('Fasting blood glucose', pdf_text)
        self.assertIn('70', pdf_text)
        self.assertEqual(result['download_name'], 'MediCare_AI_Report_Not_Specified.pdf')

    def test_no_flags_keeps_standard_lab_table_without_review_banner(self):
        pdf_text, _ = invoke_export({
            'risk_level': 'MODERATE',
            'critical_alerts': [],
            'lab_values': [{
                'system': 'Hematology', 'test': 'Hemoglobin',
                'result': '12.8', 'unit': 'g/dL',
                'reference_range': '12.0-15.0', 'status': 'NORMAL',
            }],
        })
        self.assertIn('Hemoglobin', pdf_text)
        self.assertIn('12.8', pdf_text)
        self.assertNotIn('AI-FLAGGED FINDINGS FOR CLINICIAN REVIEW', pdf_text)
        self.assertIn('Triage status:', pdf_text)
        self.assertNotIn('Triage Risk:', pdf_text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
