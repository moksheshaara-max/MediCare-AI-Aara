"""Regression tests for REAL v5 grouped-citation failure diagnostics.

No Gemini/Atlas calls. Evidence provenance is checked, not claim-level entailment.
These tests must run against production generation/generator.py unchanged.
"""
import ast
from pathlib import Path
import re
import unittest

PATH = Path(__file__).resolve().parents[1] / 'generation' / 'generator.py'
NAMES = {'_expand_supported_multi_page_citations', '_clinical_answer_blockers',
         '_clinical_text_unchanged_except_citations'}


def functions():
    tree = ast.parse(PATH.read_text(encoding='utf-8'))
    env = {'re': re}
    exec(compile(ast.Module(body=[n for n in tree.body
                                 if isinstance(n, ast.FunctionDef) and n.name in NAMES],
                            type_ignores=[]), str(PATH), 'exec'), env)
    return env


class DiagnosticGroupedCitations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = functions()

    def setUp(self):
        # IDs and page ranges come from the live diagnostic; text deliberately
        # omitted because verifying *claim support* is a separate clinical task.
        self.chunks = [
            {'source_id': 'ntep_differentiated_tb_care_2025', 'page': 10, 'page_start': 10, 'page_end': 10},
            {'source_id': 'ntep_tb_plhiv_art_centres', 'page': 39, 'page_start': 39, 'page_end': 40},
            {'source_id': 'ntep_tb_hiv_framework', 'page': 1, 'page_start': 1, 'page_end': 4},
            {'source_id': 'who_tb_module4_treatment_care_2026_2ed', 'page': 28, 'page_start': 28, 'page_end': 31},
            {'source_id': 'who_tb_module3_diagnosis_2025', 'page': 26, 'page_start': 26, 'page_end': 27},
            {'source_id': 'textbook_kd_tripathi_pharmacology_8e_2019', 'page': 845, 'page_start': 845, 'page_end': 846},
            {'source_id': 'textbook_harrison_internal_medicine_21e_2022', 'page': 1423},
            {'source_id': 'textbook_bailey_love_surgery_28e', 'page': 628},
        ]
        self.papers = [{'pmid': '38038600'}, {'pmid': '37061308'}]

    def norm(self, answer, chunks=None, papers=None):
        return self.g['_expand_supported_multi_page_citations'](
            answer, self.chunks if chunks is None else chunks,
            self.papers if papers is None else papers)

    def flag(self, text, chunks=None, papers=None):
        return self.g['_clinical_answer_blockers'](
            'General medical evidence question', text,
            self.papers if papers is None else papers,
            self.chunks if chunks is None else chunks)

    def test_real_gate_1_two_source_bracket(self):
        s = '[KB:ntep_differentiated_tb_care_2025, KB:ntep_tb_plhiv_art_centres]'
        n = self.norm(s)
        self.assertEqual('[KB:ntep_differentiated_tb_care_2025] [KB:ntep_tb_plhiv_art_centres]', n)
        self.assertFalse(self.flag(n))

    def test_real_gate_1_three_source_bracket(self):
        s = '[KB:ntep_differentiated_tb_care_2025, KB:ntep_tb_plhiv_art_centres, KB:who_tb_module4_treatment_care_2026_2ed]'
        n = self.norm(s)
        self.assertEqual(3, n.count('[KB:'))
        self.assertFalse(self.flag(n))

    def test_real_gate_1_three_medical_textbooks_and_who(self):
        s = '[KB:who_tb_module3_diagnosis_2025, KB:textbook_kd_tripathi_pharmacology_8e_2019, KB:textbook_harrison_internal_medicine_21e_2022]'
        n = self.norm(s)
        self.assertEqual(3, n.count('[KB:'))
        self.assertFalse(self.flag(n))

    def test_real_gate_2_page_and_other_source_bracket(self):
        s = '[KB:who_tb_module3_diagnosis_2025:p26, KB:textbook_kd_tripathi_pharmacology_8e_2019]'
        n = self.norm(s)
        self.assertEqual('[KB:who_tb_module3_diagnosis_2025:p26] [KB:textbook_kd_tripathi_pharmacology_8e_2019]', n)
        self.assertFalse(self.flag(n))

    def test_real_gate_3_source_mixed_with_no_page(self):
        s = '[KB:who_tb_module3_diagnosis_2025, KB:textbook_kd_tripathi_pharmacology_8e_2019]'
        self.assertFalse(self.flag(self.norm(s)))

    def test_real_gate_4_textbook_pair(self):
        s = '[KB:textbook_kd_tripathi_pharmacology_8e_2019, KB:textbook_harrison_internal_medicine_21e_2022]'
        self.assertFalse(self.flag(self.norm(s)))

    def test_real_gate_4_three_textbooks(self):
        s = '[KB:ntep_tb_plhiv_art_centres, KB:textbook_kd_tripathi_pharmacology_8e_2019, KB:textbook_bailey_love_surgery_28e]'
        n = self.norm(s)
        self.assertEqual(3, n.count('[KB:'))
        self.assertFalse(self.flag(n))

    def test_real_gate_4_two_guidelines(self):
        s = '[KB:ntep_differentiated_tb_care_2025, KB:who_tb_module4_treatment_care_2026_2ed]'
        self.assertFalse(self.flag(self.norm(s)))

    def test_real_spaced_pmid(self):
        n = self.norm('Result [KB:ntep_differentiated_tb_care_2025] [PMID: 38038600] and [PMID: 37061308].')
        self.assertIn('[PMID:38038600]', n)
        self.assertIn('[PMID:37061308]', n)
        self.assertFalse(self.flag(n))

    def test_unknown_spaced_pmid_remains_invalid(self):
        s = 'Result [PMID: 99999999] [KB:ntep_differentiated_tb_care_2025].'
        self.assertIn('[PMID: 99999999]', self.norm(s))
        self.assertTrue(self.flag(self.norm(s)))

    def test_unretrieved_source_in_group_never_laundered(self):
        s = '[KB:ntep_differentiated_tb_care_2025, KB:invented_source]'
        self.assertEqual(s, self.norm(s))
        self.assertTrue(self.flag(self.norm(s)))

    def test_unretrieved_page_in_group_never_laundered(self):
        s = '[KB:who_tb_module3_diagnosis_2025:p26, KB:textbook_kd_tripathi_pharmacology_8e_2019:p999]'
        self.assertEqual(s, self.norm(s))
        self.assertTrue(self.flag(self.norm(s)))

    def test_retrieved_cross_page_end_is_valid(self):
        n = self.norm('Evidence [KB:who_tb_module3_diagnosis_2025:p27].')
        self.assertFalse(self.flag(n))
        self.assertEqual('Evidence [KB:who_tb_module3_diagnosis_2025:p27].', n)

    def test_retrieved_cross_page_range_four_pages(self):
        n = self.norm('[KB:who_tb_module4_treatment_care_2026_2ed:p31]')
        self.assertFalse(self.flag(n))

    def test_outside_cross_page_range_is_rejected(self):
        s = '[KB:who_tb_module3_diagnosis_2025:p28]'
        self.assertEqual(s, self.norm(s))
        self.assertTrue(self.flag(s))

    def test_too_wide_range_is_not_trusted(self):
        c = [{'source_id': 's1', 'page': 3, 'page_start': 3, 'page_end': 300}]
        s = '[KB:s1:p150]'
        self.assertEqual(s, self.norm(s, chunks=c))
        self.assertTrue(self.flag(s, chunks=c))

    def test_supported_mixed_page_list(self):
        s = '[KB:who_tb_module3_diagnosis_2025:p26, p27]'
        n = self.norm(s)
        self.assertEqual('[KB:who_tb_module3_diagnosis_2025:p26] [KB:who_tb_module3_diagnosis_2025:p27]', n)
        self.assertFalse(self.flag(n))

    def test_invalid_mixed_page_list_kept_atomic(self):
        s = '[KB:who_tb_module3_diagnosis_2025:p26, p999]'
        self.assertEqual(s, self.norm(s))
        self.assertTrue(self.flag(s))

    def test_mixed_pmid_and_kb_group_verified(self):
        s = '[KB:who_tb_module3_diagnosis_2025:p26, PMID: 37061308]'
        n = self.norm(s)
        self.assertEqual('[KB:who_tb_module3_diagnosis_2025:p26] [PMID:37061308]', n)
        self.assertFalse(self.flag(n))

    def test_source_id_case_is_not_changed(self):
        s = '[KB:NTEP_DIFFERENTIATED_TB_CARE_2025, KB:ntep_tb_plhiv_art_centres]'
        self.assertEqual(s, self.norm(s))
        self.assertTrue(self.flag(s))

    def test_duplicate_citations_deduplicated_only_in_same_group(self):
        n = self.norm('[KB:ntep_tb_plhiv_art_centres, KB:ntep_tb_plhiv_art_centres]')
        self.assertEqual('[KB:ntep_tb_plhiv_art_centres]', n)
        self.assertFalse(self.flag(n))

    def test_clinical_text_unchanged_during_normalization(self):
        old = 'Treat with 2HRZE/4HRE [KB:ntep_differentiated_tb_care_2025, KB:who_tb_module4_treatment_care_2026_2ed]'
        new = self.norm(old)
        self.assertTrue(self.g['_clinical_text_unchanged_except_citations'](old, new))
        self.assertIn('2HRZE/4HRE', new)

    def test_citation_normalization_does_not_overrule_medical_guard(self):
        q = ('Describe drug-susceptible tuberculosis management in India under NTEP, '
             'including standard treatment regimen')
        text = self.norm('Regimen: 2HRZE / 4HR [KB:ntep_differentiated_tb_care_2025, KB:who_tb_module4_treatment_care_2026_2ed]')
        self.assertTrue(any('continuation phase' in v.lower() for v in self.g['_clinical_answer_blockers'](q, text, [], self.chunks)))

    def test_normalizer_idempotent(self):
        s = 'Evidence [KB:ntep_tb_plhiv_art_centres, KB:who_tb_module3_diagnosis_2025:p26] [PMID: 37061308].'
        self.assertEqual(self.norm(s), self.norm(self.norm(s)))

    def test_plain_clinical_prose_not_modified(self):
        s = 'Give medication according to appropriate guidance; no sources are cited.'
        self.assertEqual(s, self.norm(s))


if __name__ == '__main__':
    unittest.main()
