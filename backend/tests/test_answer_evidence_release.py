"""Targeted offline evidence safety and article-intent regression checks.

No Gemini, live PubMed, or MongoDB calls. Clinical assertions here are narrow
known-error guards, not clinical validation of the overall system.
"""
import ast
import importlib.util
import pathlib
import re
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def generator_pure_helpers():
    path = ROOT / 'generation' / 'generator.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    wanted = {'_clinical_evidence_safety_note', '_clinical_answer_blockers'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    consts = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {'NTEP_DS_TB_OFFICIAL_URL', 'WHO_MALARIA_OFFICIAL_URL'}
        for t in node.targets)]
    env = {'re': re}
    exec(compile(ast.Module(body=consts + selected, type_ignores=[]), str(path), 'exec'), env)
    return env


def pubmed_module():
    path = ROOT / 'pubmed' / 'pubmed_search.py'
    spec = importlib.util.spec_from_file_location('medicare_papers_release_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnswerEvidenceSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = generator_pure_helpers()
        cls.check = staticmethod(cls.helpers['_clinical_answer_blockers'])

    def test_bad_indian_treatment_regimen_is_blocked(self):
        q = 'Describe drug-susceptible pulmonary tuberculosis treatment in India under NTEP'
        self.assertTrue(self.check(q, 'Intensive phase 2HRZE / 4HR.', []))
        self.assertFalse(self.check(q, 'NTEP regimen: 2HRZE / 4HRE daily FDCs.', []))
        self.assertFalse(self.check(q, 'Not 2HRZE / 4HR; use 2HRZE / 4HRE.', []))

    def test_wrong_regimen_not_forced_on_who_only_query(self):
        q = 'According to WHO, discuss management of drug-susceptible pulmonary tuberculosis'
        self.assertFalse(self.check(q, 'WHO regimen 2HRZE / 4HR.', []))
        self.assertEqual('', self.helpers['_clinical_evidence_safety_note'](q))
        q2 = 'According to WHO, describe drug-susceptible TB in India under NTEP'
        self.assertEqual('', self.helpers['_clinical_evidence_safety_note'](q2))
        self.assertFalse(self.check(q2, '2HRZE / 4HR may be cited by WHO.', []))

    def test_verified_indian_source_context_is_explicitly_labeled_external(self):
        q = 'Explain drug-susceptible pulmonary tuberculosis management under NTEP in India'
        note = self.helpers['_clinical_evidence_safety_note'](q)
        self.assertIn('2HRZE/4HRE', note)
        self.assertIn('external official', note)
        self.assertIn('tbcindia.mohfw.gov.in', note)

    def test_null_trial_cannot_be_reported_as_proven_success(self):
        q = 'Describe management of drug-susceptible TB in India'
        paper = {'pmid': '37061308', 'abstract': 'INTERPRETATION: intervention had no effect on unfavourable outcomes.'}
        self.assertTrue(self.check(q, 'Digital adherence technologies improve treatment success.', [paper]))
        self.assertFalse(self.check(q, 'Digital adherence monitoring had no effect on the primary outcome.', [paper]))
        self.assertFalse(self.check(q, 'Digital adherence technologies improve treatment success.', []))

    def test_pediatric_anemia_cutoff_not_applied_to_adult_malaria(self):
        q = 'Discuss severe falciparum malaria in an adult in India'
        self.assertTrue(self.check(q, 'Severe anemia (hemoglobin < 5 g/dL).', []))
        self.assertFalse(self.check(q, 'Adults with severe malarial anemia have Hb < 7 g/dL.', []))
        self.assertFalse(self.check(q, 'Severe malarial anemia in children under 12: Hb < 5 g/dL.', []))
        self.assertFalse(self.check(q, 'Do not use anemia Hb < 5 g/dL as the adult definition.', []))

    def test_malaria_error_words_blocked(self):
        q = 'Management of severe falciparum malaria in adults'
        self.assertTrue(self.check(q, 'Treat with 50% dextrose followed by 10% infusion.', []))
        self.assertTrue(self.check(q, 'Start parenteral antivenom.', []))
        self.assertFalse(self.check(q, 'Use IV artesunate according to the local protocol.', []))
        self.assertFalse(self.check(q, 'Do not give 50% dextrose undiluted.', []))

    def test_bpalm_duration_contradiction_blocked(self):
        q = 'Management of rifampicin-resistant tuberculosis'
        self.assertTrue(self.check(q, 'BPaLM lasts 6 weeks (26 weeks).', []))
        self.assertFalse(self.check(q, 'BPaLM duration is 6 months (26 weeks).', []))

    def test_pph_ergometrine_contraindication_check(self):
        q = 'Treatment of postpartum hemorrhage'
        self.assertTrue(self.check(q, 'Give ergometrine as alternative uterotonic.', []))
        self.assertFalse(self.check(q, 'Ergometrine is an option. Avoid in hypertension or preeclampsia.', []))

    def test_severe_malaria_external_note(self):
        q = 'Severe falciparum malaria in an adult in India'
        note = self.helpers['_clinical_evidence_safety_note'](q)
        self.assertIn('<7 g/dL', note)
        self.assertIn('external WHO', note)


class PubMedFocusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = pubmed_module()

    def paper(self, title, abstract, pmid='1234', types=None):
        return {'title': title, 'abstract': abstract, 'pmid': pmid,
                'year': '2024', 'journal': 'Example', 'authors': 'Demo',
                'publication_types': types or ['Review']}

    def test_adherence_null_trial_conclusion_preserved(self):
        summary = ('BACKGROUND: Drug-sensitive tuberculosis treatment requires six months. '
                   + 'METHODS: ' + 'Details of methods. '*310
                   + ' INTERPRETATION: The intervention had no effect on the primary composite outcome.')
        evidence = self.pm.format_pubmed_for_prompt([self.paper('Digital adherence technologies for pulmonary tuberculosis treatment', summary)])
        self.assertIn('no effect on the primary composite outcome', evidence)
        self.assertIn('[Middle of abstract omitted for length]', evidence)
        self.assertNotIn('Abstract: ' + summary[:1100] + '\n', evidence)

    def test_mdr_tb_prevention_paper_excluded_from_active_management(self):
        q = 'Management of rifampicin-resistant or multidrug-resistant pulmonary tuberculosis in India'
        p = self.paper('Update on multidrug-resistant tuberculosis preventive therapy',
                       'MDR TB treatment and preventive therapy for contacts.')
        self.assertFalse(self.pm._is_relevant_paper(q, p))
        self.assertTrue(self.pm._is_relevant_paper(
            'Drug resistant TB preventive therapy in contacts', p))

    def test_active_mdr_tb_treatment_paper_kept(self):
        q = 'Management of rifampicin-resistant or multidrug-resistant pulmonary tuberculosis in India'
        p = self.paper('Treatment regimens for multidrug-resistant pulmonary tuberculosis',
                       'Shorter treatment regimens with bedaquiline.')
        self.assertTrue(self.pm._is_relevant_paper(q, p))

    def test_ds_tb_paper_excludes_post_tb_and_nanomedicine(self):
        q = 'Drug-susceptible pulmonary tuberculosis diagnosis and management in India'
        self.assertFalse(self.pm._is_relevant_paper(q, self.paper(
            'Post tuberculosis treatment infectious complications', 'Pulmonary tuberculosis treatment outcomes.')))
        self.assertFalse(self.pm._is_relevant_paper(q, self.paper(
            'Advanced drug delivery and therapeutic strategies for tuberculosis treatment',
            'Nano carriers for tuberculosis treatment.')))

    def test_ds_tb_keeps_relevant_treatment_trial(self):
        q = 'Drug-susceptible pulmonary tuberculosis treatment adherence in India'
        self.assertTrue(self.pm._is_relevant_paper(q, self.paper(
            'Digital adherence technologies to improve tuberculosis treatment outcomes',
            'Drug-susceptible pulmonary tuberculosis treatment outcomes and treatment adherence.')))

    def test_severe_malaria_excludes_surveillance_and_splenomegaly(self):
        q = 'Severe falciparum malaria diagnosis and treatment in an adult in India'
        self.assertFalse(self.pm._is_relevant_paper(q, self.paper(
            'Malaria Surveillance - United States, 2017', 'Severe malaria cases and treatment surveillance.')))
        self.assertFalse(self.pm._is_relevant_paper(q, self.paper(
            'Splenomegaly: Diagnosis and Management in Adults', 'Malaria causes splenomegaly.')))
        self.assertFalse(self.pm._is_relevant_paper(q, self.paper(
            'Ivermectin treatment for reducing malaria transmission', 'Severe falciparum malaria treatment.')))

    def test_severe_malaria_keeps_specific_treatment(self):
        q = 'Severe falciparum malaria diagnosis and treatment in an adult in India'
        self.assertTrue(self.pm._is_relevant_paper(q, self.paper(
            'Artemether for severe malaria', 'Intramuscular treatment of severe malaria in adults.')))


if __name__ == '__main__':
    unittest.main()

class GenerationFailClosedTests(unittest.TestCase):
    """Execute the production generation entrypoint with deterministic fake Gemini."""

    def run_generation(self, responses, max_retries):
        import time
        from types import SimpleNamespace
        path = ROOT / 'generation' / 'generator.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'generate_answer')
        env = generator_pure_helpers()
        class FakeModels:
            def __init__(self):
                self.calls = []
            def generate_content(self, model, contents, config):
                self.calls.append(contents)
                idx = min(len(self.calls) - 1, len(responses) - 1)
                return SimpleNamespace(text=responses[idx])
        models = FakeModels()
        class FakeConfig:
            def __init__(self, **kw):
                self.options = kw
        env.update({
            'normalize_assistant_mode': lambda m: m,
            '_response_cache': {}, 'RESPONSE_CACHE_TTL_SECONDS': 600, 'time': time,
            '_pubmed_budget': lambda q, m: (5, 'doctor'),
            'search_pubmed_live': lambda *_args, **_kw: [],
            'determine_mode': lambda chunks, papers: ('books', 0.86),
            'build_hybrid_prompt': lambda *a, **kw: 'Retrieved evidence: 2HRZE/4HRE',
            '_presentation_instruction': lambda q: 'Use flowchart and table',
            '_system_prompt_for': lambda m, fallback: 'Safe clinical evidence rules',
            'client': SimpleNamespace(models=models), 'GENERATION_MODEL': 'mock-model',
            'types': SimpleNamespace(GenerateContentConfig=FakeConfig,
                                     AutomaticFunctionCallingConfig=FakeConfig),
            'clean_latex_symbols': lambda s: s,
            '_sanitize_generated_markdown': lambda s, **kw: s,
            '_has_mermaid_flowchart': lambda text: True,
            '_has_markdown_table': lambda text: True,
            '_is_medication_dose_query': lambda question: False,
            '_chunk_source_kind': lambda chunk: 'Guideline',
        })
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
        q = 'Describe treatment of drug-susceptible pulmonary tuberculosis in India under NTEP'
        chunks = [{'source_id': 'who_tb_module4_treatment_care_2026_2ed',
                   'score': 0.86, 'source_type': 'clinical_guideline'}]
        output = env['generate_answer'](q, chunks, max_retries=max_retries, assistant_mode='doctor')
        return output, models.calls

    def test_bad_claim_is_regenerated_from_original_evidence(self):
        result, calls = self.run_generation(['2HRZE / 4HR', '2HRZE / 4HRE'], 3)
        self.assertEqual('books', result[1])
        self.assertIn('4HRE', result[0])
        self.assertEqual(2, len(calls))
        self.assertIn('CRITICAL CLINICAL SAFETY REGENERATION', calls[1])
        self.assertEqual(0.86 * 100, result[2])

    def test_persistent_bad_claim_gets_withheld(self):
        result, calls = self.run_generation(['2HRZE / 4HR'], 2)
        self.assertEqual('error', result[1])
        self.assertIn('withheld', result[0].lower())
        self.assertNotIn('2HRZE / 4HR', result[0])
        self.assertEqual(2, len(calls))
