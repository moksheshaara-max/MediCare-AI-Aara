"""Targeted regressions for the 2026-10-11 LIVE answer failures.

No database, Gemini, or PubMed requests. These assertions detect the observed
mistakes and protect retrieval provenance; NOT a general medical validation.
"""
import ast
import importlib.util
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_generator_functions():
    path = ROOT / 'generation' / 'generator.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'_clinical_answer_blockers', '_clinical_evidence_safety_note', '_chunk_citation_token', '_format_chunk_for_prompt'}
    constants = {'NTEP_DS_TB_OFFICIAL_URL', 'WHO_MALARIA_OFFICIAL_URL', 'WHO_DRTB_2025_OFFICIAL_URL', 'WHO_PPH_2025_OFFICIAL_URL'}
    nodes = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name in names or
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
    )]
    env = {'re': re, '_chunk_source_kind': lambda chunk: 'Guideline'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def load_pubmed_module():
    path = ROOT / 'pubmed' / 'pubmed_search.py'
    spec = importlib.util.spec_from_file_location('medicare_pubmed_v4_unit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiveAnswerBlockers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = load_generator_functions()
        cls.check = staticmethod(cls.g['_clinical_answer_blockers'])

    Q_DR = 'Management of multidrug-resistant or rifampicin-resistant pulmonary tuberculosis in India'
    Q_PPH = 'Treatment of postpartum haemorrhage after vaginal delivery'
    Q_MAL = 'Explain severe falciparum malaria in an adult in India, including diagnosis and management'

    def test_real_wrong_dr_tb_mermaid_branch(self):
        ans = '''```mermaid
flowchart TD
C{"FQ Susceptible & Eligible for BPaLM?"}
C -- "No / FQ Resistant" --> E["Initiate 9-11M All-Oral or Individualized Regimen"]
```'''
        self.assertIn('FQ-resistant DR-TB routed to a nine-month regimen', self.check(self.Q_DR, ans))

    def test_dr_tb_two_line_mermaid_wrong_branch(self):
        ans = '''```mermaid
flowchart TD
C -- "FQ resistant" --> E
E["Start 9-month all-oral regimen"]
```'''
        self.assertIn('FQ-resistant DR-TB routed to a nine-month regimen', self.check(self.Q_DR, ans))

    def test_dr_tb_fq_resistant_bpal_not_wrongly_blocked(self):
        ans = '''```mermaid
flowchart TD
C -- "FQ resistant" --> E["Evaluate BPaL (omit moxifloxacin)"]
C -- "FQ susceptible" --> F["Check nine-month eligibility"]
```'''
        self.assertFalse(self.check(self.Q_DR, ans))

    def test_wrong_qualifier_bpalm_moxifloxacin(self):
        self.assertTrue(self.check(self.Q_DR,
            'BPaLM, including moxifloxacin, is appropriate for FQ resistant TB.', []))

    def test_pph_warns_near_ergot_drug_not_elsewhere(self):
        bad = ('Alternative ergometrine can be administered. '
               'Avoid fluid overload in preeclampsia and hypertension.')
        self.assertIn('Ergot uterotonic lacks local hypertension/pre-eclampsia contraindication',
                      self.check(self.Q_PPH, bad))
        good = 'Ergometrine: contraindicated in hypertension or pre-eclampsia.'
        self.assertFalse(self.check(self.Q_PPH, good))

    def test_pph_combo_uterotonic_requires_warning(self):
        self.assertTrue(self.check(self.Q_PPH, 'Use oxytocin-ergometrine as second-line therapy.'))
        self.assertFalse(self.check(self.Q_PPH,
            'Avoid oxytocin-ergometrine in hypertension and preeclampsia.'))

    def test_malaria_wrong_universal_2_percent(self):
        bad = 'Once parasitemia exceeds >2%, this is severe malaria.'
        self.assertIn('Parasitaemia above 2% presented as universal severe-malaria threshold',
                      self.check(self.Q_MAL, bad))
        bad_chart = 'A["Suspect Severe Malaria; Parasitemia >2%"] --> B["Assessment"]'
        self.assertTrue(self.check(self.Q_MAL, bad_chart))

    def test_malaria_2_percent_nonimmune_appropriately_qualified(self):
        good = 'For non-immune patients, parasitaemia >2% warrants close monitoring; it is not a universal severe threshold.'
        self.assertFalse(self.check(self.Q_MAL, good))

    def test_malaria_fake_panniculus_blocked(self):
        self.assertTrue(self.check(self.Q_MAL,
            'The key process is Panniculus and Erythrocyte Tropism.'))

    def test_dr_tb_who_note_external_and_programmatic_caution(self):
        note = self.g['_clinical_evidence_safety_note'](self.Q_DR)
        self.assertIn('fluoroquinolone-resistant', note)
        self.assertIn('9-month', note)
        self.assertIn('external WHO', note)
        self.assertIn('applicable CURRENT national protocol', note)

    def test_pph_medication_note_external(self):
        note = self.g['_clinical_evidence_safety_note'](self.Q_PPH)
        self.assertIn('hypertension', note)
        self.assertIn('2025', note)
        self.assertIn('external WHO', note)

    def test_malaria_external_note_differentiates_thresholds(self):
        note = self.g['_clinical_evidence_safety_note'](self.Q_MAL)
        self.assertIn('NOT a universal', note)
        self.assertIn('>10%', note)

    def test_citation_token_exact_indexed_page(self):
        chunk = {'source_id': 'who_pph_2025', 'page': 14, 'text': 'Uterine massage and oxytocin'}
        self.assertEqual('[KB:who_pph_2025:p14]', self.g['_chunk_citation_token'](chunk))
        self.assertIn('VERIFIED CITATION TOKEN: [KB:who_pph_2025:p14]',
                      self.g['_format_chunk_for_prompt'](chunk, 4))

    def test_citation_untrusted_page_is_not_invented(self):
        chunk = {'source_id': 'who_pph_2025', 'page': 'N/A', 'text': 'Clinical material'}
        self.assertEqual('[KB:who_pph_2025]', self.g['_chunk_citation_token'](chunk))

    def test_citation_missing_or_invented_source_is_blocked(self):
        chunks = [{'source_id': 'who_pph_2025', 'page': 14}, {'source_id': 'who_pph_2025', 'page': 15}]
        answer = 'Give oxytocin [KB:imaginary_source:p4].'
        self.assertTrue(self.check(self.Q_PPH, answer, [], chunks))
        answer = 'Give oxytocin [KB:who_pph_2025:p999].'
        self.assertTrue(self.check(self.Q_PPH, answer, [], chunks))

    def test_citation_shorthand_and_numbered_markers_are_blocked(self):
        chunks = [{'source_id': 'who_pph_2025', 'page': 14}, {'source_id': 'who_pph_2025', 'page': 15}]
        self.assertTrue(self.check(self.Q_PPH, 'Oxytocin [Guideline 2].', [], chunks))
        self.assertTrue(self.check(self.Q_PPH, 'Oxytocin [KB:who_pph_2025:p14, p15].', [], chunks))
        self.assertFalse(self.check(self.Q_PPH,
             'Oxytocin [KB:who_pph_2025:p14] and escalation [KB:who_pph_2025:p15].', [], chunks))

    def test_real_ds_tb_generic_citations_rejected(self):
        q = 'Drug-susceptible tuberculosis in India under NTEP: diagnostic workup and treatment'
        chunks = [{'source_id': 'who_tb_module4_treatment_care_2026_2ed', 'page': 10},
                  {'source_id': 'who_tb_module3_diagnosis_2025', 'page': 3}]
        answer = 'Management under NTEP includes 2HRZE/4HRE [Guideline 5].'
        self.assertTrue(self.check(q, answer, [], chunks))

    def test_unknown_pubmed_pmid_not_citable(self):
        chunks = [{'source_id': 'who_pph_2025', 'page': 14}, {'source_id': 'who_pph_2025', 'page': 15}]
        papers = [{'pmid': '37061308', 'abstract': ''}]
        answer = 'A study proves benefit [PMID:99999999]. Oxytocin [KB:who_pph_2025:p14].'
        self.assertIn('An inline PMID citation references an unprovided research paper',
                      self.check(self.Q_PPH, answer, papers, chunks))


class PubMedClinicalIntent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = load_pubmed_module()
    Q_MAL = 'Severe falciparum malaria diagnosis treatment and complications in adults in India'
    Q_PPH = 'Postpartum hemorrhage treatment after vaginal delivery: oxytocin, TXA and escalation'

    @staticmethod
    def paper(title, abstract='Severe falciparum malaria management and treatment outcomes in adults.', pmid='44', types=None):
        return {'title': title, 'abstract': abstract, 'pmid': pmid, 'year': '2024',
                'journal': 'Journal', 'authors': 'Test', 'publication_types': types or ['Review']}

    def test_malaria_house_modifications_not_severe_care(self):
        self.assertFalse(self.pm._is_relevant_paper(self.Q_MAL,
             self.paper('House modifications for preventing malaria')))

    def test_malaria_therapeutic_vaccines_not_severe_care(self):
        self.assertFalse(self.pm._is_relevant_paper(self.Q_MAL,
             self.paper('Reduced blood-stage malaria growth in humans following RH5 vaccination')))

    def test_malaria_prevention_antibody_pediatric_study_not_acute_care(self):
        self.assertFalse(self.pm._is_relevant_paper(self.Q_MAL,
             self.paper('Subcutaneous administration of a monoclonal antibody to prevent malaria')))

    def test_malaria_household_access_not_severe_care(self):
        self.assertFalse(self.pm._is_relevant_paper(self.Q_MAL,
             self.paper('Malaria medicines and miles: measuring access to treatment from a household perspective')))

    def test_malaria_diagnosis_treatment_prevention_review_allowed(self):
        self.assertTrue(self.pm._is_relevant_paper(self.Q_MAL,
             self.paper('Diagnosis Treatment and Prevention of Malaria in the US: A Review')))

    def test_severe_malaria_artemether_treatment_allowed(self):
        self.assertTrue(self.pm._is_relevant_paper(self.Q_MAL,
             self.paper('Artemether for severe malaria')))

    def test_pph_primary_prevention_study_not_active_bleeding(self):
        p = self.paper('Preventing postpartum hemorrhage with combined therapy rather than oxytocin alone',
            'Prevention of postpartum hemorrhage with two medicines.')
        self.assertFalse(self.pm._is_relevant_paper(self.Q_PPH, p))

    def test_pph_combined_prevention_treatment_guideline_kept(self):
        p = self.paper('Guideline for prevention, diagnosis and treatment of postpartum hemorrhage',
            'Management and treatment of postpartum hemorrhage in patients.')
        self.assertTrue(self.pm._is_relevant_paper(self.Q_PPH, p))

    def test_unavailable_abstract_tb_editorial_filtered(self):
        p = self.paper('Intermittent regimens for tuberculosis treatment: back to the future?',
                       'Abstract unavailable.', types=['Editorial', 'Comment'])
        q = 'Describe diagnostic workup, drug-susceptible tuberculosis treatment, monitoring and adherence'
        self.assertFalse(self.pm._is_relevant_paper(q, p))

    def test_general_diabetes_clinical_query_unaffected(self):
        p = self.paper('Management of type 2 diabetes in adults',
                       'Current treatment for type 2 diabetes in adults.')
        self.assertTrue(self.pm._is_relevant_paper('Type 2 diabetes management in adults', p))


class GeneratorCitationFailClosed(unittest.TestCase):
    """Exercise the real generation function with a fake LLM and real blockers."""

    @staticmethod
    def generate(responses):
        import time
        from types import SimpleNamespace
        path = ROOT / 'generation' / 'generator.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'generate_answer')
        env = load_generator_functions()

        class FakeModels:
            def __init__(self):
                self.calls = []
            def generate_content(self, model, contents, config):
                self.calls.append(contents)
                return SimpleNamespace(text=responses[min(len(self.calls) - 1, len(responses) - 1)])

        class FakeConfig:
            def __init__(self, **kwargs):
                self.options = kwargs

        fake = FakeModels()
        env.update({
            'normalize_assistant_mode': lambda value: 'doctor',
            '_response_cache': {}, 'RESPONSE_CACHE_TTL_SECONDS': 600, 'time': time,
            '_pubmed_budget': lambda question, mode: (5, 'doctor'),
            'search_pubmed_live': lambda *args, **kwargs: [],
            'determine_mode': lambda chunks, papers: ('books', 0.86),
            'build_hybrid_prompt': lambda *args, **kwargs: 'Evidence [KB:who_pph_2025:p14]',
            '_presentation_instruction': lambda q: 'Please output a flowchart and table',
            '_system_prompt_for': lambda m, fallback: 'Source-grounded clinical answer',
            'client': SimpleNamespace(models=fake), 'GENERATION_MODEL': 'test-model',
            'types': SimpleNamespace(GenerateContentConfig=FakeConfig,
                  AutomaticFunctionCallingConfig=FakeConfig),
            'clean_latex_symbols': lambda s: s,
            '_sanitize_generated_markdown': lambda s, **kwargs: s,
            '_has_mermaid_flowchart': lambda s: True,
            '_has_markdown_table': lambda s: True,
            '_is_medication_dose_query': lambda q: False,
            '_chunk_source_kind': lambda c: 'Guideline',
        })
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
        result = env['generate_answer'](
            'Treat postpartum hemorrhage after birth',
            [{'source_id': 'who_pph_2025', 'page': 14, 'score': 0.86},
             {'source_id': 'who_pph_2025', 'page': 15, 'score': 0.85}],
            max_retries=2, assistant_mode='doctor')
        return result, fake.calls

    def test_invalid_citation_regenerated_to_valid_citation(self):
        result, calls = self.generate([
            'IV oxytocin recommended [Guideline 2].',
            'IV oxytocin recommended [KB:who_pph_2025:p14].',
        ])
        self.assertEqual('books', result[1])
        self.assertEqual(2, len(calls))
        self.assertIn('CRITICAL CLINICAL SAFETY REGENERATION', calls[1])
        self.assertIn('[KB:who_pph_2025:p14]', result[0])

    def test_persistent_invented_citation_withheld(self):
        result, calls = self.generate(['IV oxytocin [KB:fake_guideline:p999].'])
        self.assertEqual('error', result[1])
        self.assertEqual(2, len(calls))
        self.assertIn('withheld', result[0].lower())
        self.assertNotIn('fake_guideline', result[0])


if __name__ == '__main__':
    unittest.main()
