"""Citation reliability and fail-closed tests for v5.

No live endpoints. The evidence-level check validates references, not whether a
clinical sentence is entailed by the cited passage.
"""
import ast
from pathlib import Path
import re
import time
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'generation' / 'generator.py'


def load_functions():
    tree = ast.parse(MODULE.read_text(encoding='utf-8'))
    names = {
        '_clinical_answer_blockers', '_expand_supported_multi_page_citations',
        '_chunk_citation_token', '_only_citation_flags', '_repair_citations_only',
        '_clinical_text_unchanged_except_citations',
        '_targeted_regeneration_guidance', 'generate_answer',
    }
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {'re': re, 'time': time}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(MODULE), 'exec'), env)
    return env


class CitationRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = load_functions()

    @staticmethod
    def chunks():
        return [
            {'source_id': 'who_pph_2025', 'page': 14, 'text': 'Treat postpartum haemorrhage promptly with oxytocin.', 'score': 0.86},
            {'source_id': 'who_pph_2025', 'page': 15, 'text': 'Escalate if bleeding does not stop.', 'score': 0.85},
        ]

    def test_valid_page_merge_can_be_expanded_without_inventing_provenance(self):
        source = 'Give oxytocin [KB:who_pph_2025:p14, p15].'
        fixed = self.g['_expand_supported_multi_page_citations'](source, self.chunks())
        self.assertEqual('Give oxytocin [KB:who_pph_2025:p14] [KB:who_pph_2025:p15].', fixed)
        self.assertFalse(self.g['_clinical_answer_blockers']('PPH treatment', fixed, [], self.chunks()))

    def test_missing_page_merge_is_not_silently_laundered(self):
        source = 'Oxytocin [KB:who_pph_2025:p14,p999].'
        fixed = self.g['_expand_supported_multi_page_citations'](source, self.chunks())
        self.assertEqual(source, fixed)
        self.assertTrue(self.g['_clinical_answer_blockers']('PPH treatment', fixed, [], self.chunks()))

    def test_invalid_source_merge_is_not_silently_laundered(self):
        source = 'Oxytocin [KB:invented:p14,p15].'
        self.assertEqual(source, self.g['_expand_supported_multi_page_citations'](source, self.chunks()))
        self.assertTrue(self.g['_clinical_answer_blockers']('PPH treatment', source, [], self.chunks()))

    def test_source_only_reference_is_valid_only_for_returned_source(self):
        fn = self.g['_clinical_answer_blockers']
        self.assertFalse(fn('PPH treatment', 'Oxytocin [KB:who_pph_2025].', [], self.chunks()))
        self.assertTrue(fn('PPH treatment', 'Oxytocin [KB:invented].', [], self.chunks()))

    def test_source_only_citation_does_not_rescue_invalid_page(self):
        fn = self.g['_clinical_answer_blockers']
        self.assertTrue(fn('PPH treatment', 'Oxytocin [KB:who_pph_2025] [KB:who_pph_2025:p900].', [], self.chunks()))

    def test_pmid_citation_still_must_be_retrieved(self):
        fn = self.g['_clinical_answer_blockers']
        self.assertTrue(fn('PPH treatment', 'Evidence [KB:who_pph_2025] [PMID:999999].', [], self.chunks()))
        self.assertFalse(fn('PPH treatment', 'Evidence [KB:who_pph_2025] [PMID:123456].', [{'pmid': '123456'}], self.chunks()))

    def test_repair_classification_is_specific_to_provenance(self):
        only = self.g['_only_citation_flags']
        self.assertTrue(only(['An inline KB citation references an unprovided source/page or merged pages']))
        self.assertTrue(only(['Non-auditable numbered source citations instead of source IDs/PMIDs']))
        self.assertFalse(only(['Ergot uterotonic lacks local hypertension/pre-eclampsia contraindication']))
        self.assertFalse(only(['An inline KB citation references an unprovided source/page or merged pages',
                              'FQ-resistant DR-TB routed to a nine-month regimen']))

    def test_repair_cannot_modify_clinical_numbers_or_medications(self):
        guard = self.g['_clinical_text_unchanged_except_citations']
        self.assertTrue(guard('Oxytocin [Guideline 2].', 'Oxytocin [KB:who_pph_2025].'))
        self.assertFalse(guard('Oxytocin 10 IU [Guideline 2].', 'Oxytocin 20 IU [KB:who_pph_2025].'))
        self.assertFalse(guard('Give oxytocin [Guideline 2].', 'Give ergometrine [KB:who_pph_2025].'))

    def test_regeneration_guidance_covers_all_four_live_cases(self):
        f = self.g['_targeted_regeneration_guidance']
        self.assertIn('2HRZE/4HRE', f(['India DS-TB regimen omits ethambutol during continuation phase']))
        self.assertIn('pre-eclampsia', f(['Ergot uterotonic lacks local hypertension/pre-eclampsia contraindication']))
        self.assertIn('NOT an automatic', f(['FQ-resistant DR-TB routed to a nine-month regimen']))
        self.assertIn('ALLOWED SOURCE-LEVEL CITATIONS', f(['An inline KB citation references an unprovided source/page or merged pages']))


class ProductionGenerationWithMockGemini(unittest.TestCase):
    def gen(self, responses, max_retries=2):
        env = load_functions()

        class FakeModels:
            def __init__(self):
                self.calls = []
            def generate_content(self, model, contents, config):
                self.calls.append(contents)
                ix = min(len(self.calls) - 1, len(responses) - 1)
                return SimpleNamespace(text=responses[ix])

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake = FakeModels()
        env.update({
            'normalize_assistant_mode': lambda v: 'doctor',
            '_response_cache': {}, 'RESPONSE_CACHE_TTL_SECONDS': 600,
            '_pubmed_budget': lambda q,m: (5,'doctor'),
            'search_pubmed_live': lambda *a,**kw: [],
            'determine_mode': lambda c,p: ('books',0.86),
            'build_hybrid_prompt': lambda *a,**kw: 'Trusted source evidence [KB:who_pph_2025:p14]',
            '_presentation_instruction': lambda q: '',
            '_clinical_evidence_safety_note': lambda q: '',
            '_system_prompt_for': lambda m, fallback: 'Clinical CDS',
            'client': SimpleNamespace(models=fake), 'GENERATION_MODEL': 'mock',
            'types': SimpleNamespace(GenerateContentConfig=FakeConfig,
                                     AutomaticFunctionCallingConfig=FakeConfig),
            'clean_latex_symbols': lambda v: v,
            '_sanitize_generated_markdown': lambda v,**kw: v,
            '_has_mermaid_flowchart': lambda v: True,
            '_has_markdown_table': lambda v: True,
            '_is_medication_dose_query': lambda q: False,
            '_chunk_source_kind': lambda c: 'Guideline',
        })
        result = env['generate_answer'](
            'A patient develops postpartum hemorrhage: treatment and medications',
            CitationRecovery.chunks(), max_retries=max_retries,
            assistant_mode='doctor',
        )
        return result, fake.calls

    def test_valid_multi_page_citation_is_normalized_without_regeneration(self):
        answer = 'Oxytocin [KB:who_pph_2025:p14, p15].'
        result, calls = self.gen([answer])
        self.assertEqual('books', result[1])
        self.assertIn('[KB:who_pph_2025:p14] [KB:who_pph_2025:p15]', result[0])
        self.assertEqual(1, len(calls))

    def test_citation_only_error_gets_isolated_repair(self):
        bad = 'Oxytocin [KB:who_pph_2025:p999].'
        fixed = 'Oxytocin [KB:who_pph_2025].'
        result, calls = self.gen([bad, fixed])
        self.assertEqual('books', result[1])
        self.assertIn('[KB:who_pph_2025]', result[0])
        self.assertEqual(2, len(calls))
        self.assertIn('Fix ONLY the provenance citations', calls[1])

    def test_mixed_clinical_and_citation_error_does_not_skip_medical_rewrite(self):
        bad = 'Use ergometrine immediately [KB:fake:p3].'
        clinically_fixed = 'Avoid ergometrine in pre-eclampsia and hypertension [Guideline 2].'
        cited = 'Avoid ergometrine in pre-eclampsia and hypertension [KB:who_pph_2025].'
        result, calls = self.gen([bad, clinically_fixed, cited])
        self.assertEqual('books', result[1])
        self.assertIn('Avoid ergometrine in pre-eclampsia', result[0])
        self.assertEqual(3, len(calls))
        self.assertIn('CRITICAL CLINICAL SAFETY REGENERATION', calls[1])
        self.assertIn('Fix ONLY the provenance citations', calls[2])

    def test_persistently_invalid_attribution_still_withheld(self):
        bad = 'Oxytocin [KB:fake:p999].'
        result, calls = self.gen([bad], max_retries=2)
        self.assertEqual('error', result[1])
        self.assertIn('withheld', result[0].lower())
        self.assertNotIn('fake:p999', result[0])
        self.assertGreaterEqual(len(calls), 2)

    def test_citation_repair_cannot_silently_change_dose(self):
        original = 'Oxytocin 10 IU [KB:fake:p999].'
        unsafe_editor = 'Oxytocin 20 IU [KB:who_pph_2025].'
        safe_regen = 'Oxytocin 10 IU [KB:who_pph_2025].'
        result, calls = self.gen([original,unsafe_editor,safe_regen],max_retries=2)
        self.assertEqual('books', result[1])
        self.assertIn('Oxytocin 10 IU',result[0])
        self.assertNotIn('Oxytocin 20 IU',result[0])
        self.assertEqual(3,len(calls))

    def test_citation_repair_that_introduces_clinical_fault_is_not_accepted(self):
        original = 'Oxytocin [KB:fake:p999].'
        bad_repair = 'Use ergometrine without caution [KB:who_pph_2025].'
        safe_regen = 'Oxytocin [KB:who_pph_2025].'
        result, calls = self.gen([original,bad_repair,safe_regen],max_retries=2)
        self.assertEqual('books', result[1])
        self.assertEqual(3,len(calls))
        self.assertNotIn('Use ergometrine without caution', result[0])


if __name__ == '__main__':
    unittest.main()
