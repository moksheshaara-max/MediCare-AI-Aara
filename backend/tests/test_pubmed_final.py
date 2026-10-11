"""Strict offline acceptance checks for the single final PubMed candidate.

Tests use fake article metadata/NCBI responses; they never claim live or
clinical validation. Run after replacement with Python unittest.
"""
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    import xmltodict
except ImportError:
    sys.modules['xmltodict'] = types.ModuleType('xmltodict')
    sys.modules['xmltodict'].parse = lambda *_args, **_kw: {}

spec = importlib.util.spec_from_file_location("pubmed_final_local", ROOT / "pubmed" / "pubmed_search.py")
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)

HTN_Q = "Explain hypertension in detail for postgraduate medicine including pathophysiology, diagnosis, management and pharmacology."
CHILD_T1_Q = "Explain diagnosis, insulin management and monitoring of type 1 diabetes in a 10-year-old child."
CKD_Q = "Teach chronic kidney disease comprehensively: pathophysiology, diagnosis, staging, anemia, management, monitoring and dialysis."
HF_Q = "What is the current guideline-based first-line pharmacological treatment for HFrEF?"


def paper(title, abstract="", publication_types=(), pmid="12345", year="2025"):
    return {
        "pmid": pmid, "title": title, "abstract": abstract,
        "publication_types": list(publication_types), "year": year,
        "authors": "Demo", "journal": "Example Journal", "url": "https://pubmed.ncbi.nlm.nih.gov/12345/"
    }


class PopulationTopicTests(unittest.TestCase):
    def test_hypertension_general_guideline(self):
        self.assertTrue(pm._is_relevant_paper(HTN_Q, paper(
            "Management of Hypertension in Adults: Current Guideline", "Systemic blood pressure treatment and diagnosis in adults")))

    def test_generic_hypertension_excludes_pediatric_title(self):
        self.assertFalse(pm._is_relevant_paper(HTN_Q, paper(
            "Pediatric Hypertension", "Hypertension diagnosis and management in children")))

    def test_generic_htn_excludes_primary_aldosteronism_title(self):
        self.assertFalse(pm._is_relevant_paper(HTN_Q, paper(
            "Primary aldosteronism", "Secondary hypertension diagnosis and management")))

    def test_secondary_htn_explicit_request_allows_aldosteronism(self):
        q = "Describe primary aldosteronism as a cause of secondary hypertension and its management"
        self.assertTrue(pm._is_relevant_paper(q, paper(
            "Primary aldosteronism", "Secondary hypertension, diagnosis and management")))

    def test_htn_excludes_postpartum(self):
        self.assertFalse(pm._is_relevant_paper(HTN_Q, paper(
            "Postpartum preeclampsia or eclampsia: management in pregnancy", "Hypertension diagnosis and management")))

    def test_pregnancy_hypertension_explicitly_allowed(self):
        q = "Management of hypertension in pregnancy, including postpartum preeclampsia"
        self.assertTrue(pm._is_relevant_paper(q, paper(
            "Postpartum preeclampsia hypertension: management", "Pregnancy hypertension management")))

    def test_htn_excludes_pulmonary(self):
        self.assertFalse(pm._is_relevant_paper(HTN_Q, paper(
            "Pulmonary hypertension treatment", "Hypertension management and diagnosis")))

    def test_explicit_pediatric_hypertension_allowed(self):
        q = "Diagnosis and management of hypertension in children"
        self.assertTrue(pm._is_relevant_paper(q, paper(
            "Pediatric Hypertension Management", "Hypertension in children")))

    def test_explicit_adult_question_excludes_pediatric(self):
        q = "Diagnosis and management of hypertension in adults"
        self.assertFalse(pm._is_relevant_paper(q, paper(
            "Pediatric Hypertension Management", "Children with hypertension, adults are sometimes affected")))

    def test_pediatric_t1dm_excludes_pregnancy_study(self):
        self.assertFalse(pm._is_relevant_paper(CHILD_T1_Q, paper(
            "Continuous glucose monitoring and automated insulin delivery in pregnant women with type 1 diabetes",
            "Several pediatric societies have guidance for children. Insulin treatment and monitoring.")))

    def test_pediatric_t1dm_excludes_adult_only_title(self):
        self.assertFalse(pm._is_relevant_paper(CHILD_T1_Q, paper(
            "Insulin therapy in adults with type 1 diabetes", "Pediatric guidelines for children are cited.")))

    def test_pediatric_t1dm_includes_child_insulin_therapy(self):
        self.assertTrue(pm._is_relevant_paper(CHILD_T1_Q, paper(
            "Insulin therapy for children and adolescents with type 1 diabetes", "Clinical management")))

    def test_pediatric_t1dm_includes_cgm_youth(self):
        self.assertTrue(pm._is_relevant_paper(CHILD_T1_Q, paper(
            "Continuous glucose monitoring in youth with type 1 diabetes", "Management and HbA1c monitoring")))

    def test_pediatric_t1dm_does_not_need_child_word_when_age_given(self):
        self.assertTrue(pm._is_relevant_paper(CHILD_T1_Q, paper(
            "Insulin pump therapy in type 1 diabetes", "A 10-year-old patient receiving insulin in a school setting")))

    def test_pediatric_t1dm_excludes_unrequested_dka(self):
        self.assertFalse(pm._is_relevant_paper(CHILD_T1_Q, paper(
            "Diabetic ketoacidosis in children with type 1 diabetes", "Management with insulin and glucose monitoring")))

    def test_explicit_dka_allows_dka(self):
        q = "Management of diabetic ketoacidosis prevention for children with type 1 diabetes"
        self.assertTrue(pm._is_relevant_paper(q, paper(
            "Diabetic ketoacidosis prevention in children with type 1 diabetes", "Insulin management")))

    def test_child_query_excludes_pregnancy_even_with_pediatric_mention(self):
        self.assertTrue(pm._title_context_conflict(CHILD_T1_Q, paper(
            "Technology in pregnant women", "Pediatric guidance is cited"), "type 1 diabetes"))

    def test_hypertension_causes_may_include_aldosteronism(self):
        q = "Explain causes, differential diagnosis and management of hypertension"
        self.assertTrue(pm._is_relevant_paper(q, paper(
            "Primary aldosteronism", "A cause of hypertension, with diagnostic and treatment implications")))

    def test_explicit_children_and_adults_is_mixed(self):
        q = "Compare hypertension treatment in children and adults"
        self.assertEqual("mixed", pm._population_intent(q))
        self.assertEqual("", pm._population_clause(q))

    def test_general_htn_does_not_exclude_general_adult_htn(self):
        self.assertFalse(pm._title_context_conflict(HTN_Q, paper("Systemic hypertension in adults"), "hypertension"))

    def test_unrequested_maternal_context_across_diseases(self):
        self.assertFalse(pm._is_relevant_paper(CKD_Q, paper(
            "Chronic kidney disease in pregnancy: dialysis management", "CKD anemia in pregnant women")))


class SearchStrategyTests(unittest.TestCase):
    def test_hfref_anchored(self):
        self.assertTrue(all("ejection fraction" in x.lower() or "hfref" in x.lower() for x in pm._query_plan(HF_Q)))

    def test_unknown_condition_no_generic_search(self):
        self.assertEqual([], pm._query_plan("What is latest first-line guideline therapy?"))

    def test_pediatric_query_has_insulin_and_cgm_subqueries(self):
        plans = pm._query_plan(CHILD_T1_Q)
        self.assertTrue(any("insulin pump" in q.lower() for q in plans))
        self.assertTrue(any("continuous glucose monitoring" in q.lower() for q in plans))
        self.assertTrue(all("pediatric" in q.lower() or "children" in q.lower() for q in plans))

    def test_kid_synonym_sets_pediatric_population(self):
        self.assertEqual("child", pm._population_intent("Insulin management in kids with type 1 diabetes"))
        self.assertIn("children", pm._population_clause("Insulin management in kids with type 1 diabetes"))

    def test_hip_population_not_false_child(self):
        self.assertEqual("general", pm._population_intent("High blood pressure in postgraduate medicine"))

    def test_comprehensive_ckd_keeps_late_facet_search(self):
        plans = pm._query_plan(CKD_Q)
        self.assertTrue(any("dialysis" in p.lower() for p in plans))
        self.assertTrue(any("anemia" in p.lower() for p in plans))

    def test_search_round_robin_keeps_late_facet(self):
        plans = pm._query_plan(CKD_Q)
        dialysis_plan = next(p for p in plans if '"dialysis"' in p)
        class FakeResponse:
            def __init__(self, ids): self.ids = ids
            def json(self): return {"esearchresult": {"idlist": self.ids}}
        def fake(url, params):
            term = params['term']
            if term == dialysis_plan: return FakeResponse(['9999','9998'])
            return FakeResponse(['111','222','333','444','555'])
        with patch.object(pm, '_request', side_effect=fake):
            ids = pm.search_pubmed_ids(CKD_Q, max_results=40)
        self.assertIn('9999', ids)

    def test_search_stops_after_network_outage(self):
        attempts=[]
        def fail(url, params):
            attempts.append(params['term'])
            raise pm.requests.exceptions.ConnectionError('simulated offline')
        with patch.object(pm, '_request', side_effect=fail):
            ids=pm.search_pubmed_ids(CKD_Q, max_results=40)
        self.assertEqual([],ids)
        self.assertEqual(1,len(attempts))

    def test_dedup_exec_guideline(self):
        a='2025 Hypertension Guideline for the Prevention, Detection, Evaluation and Management of High Blood Pressure in Adults'
        b='2025 Hypertension Guideline for the Prevention, Detection, Evaluation and Management of High Blood Pressure in Adults: Executive Summary'
        self.assertEqual(pm._title_key(a), pm._title_key(b))

    def test_zero_requested_returns_zero(self):
        self.assertEqual([],pm.search_pubmed_live(HF_Q,max_results=0))

    def test_case_report_rejected_for_broad_questions(self):
        self.assertFalse(pm._is_relevant_paper(HTN_Q, paper(
            'Treatment of hypertension: A Case Report', 'Hypertension management', ('Case Reports',))))


class PaperSelectionTests(unittest.TestCase):
    def test_pediatric_pregnancy_not_selected_even_with_good_abstract(self):
        papers=[
            paper('Continuous glucose monitoring for pregnant women with type 1 diabetes',
                  'Monitoring in children and adolescents; insulin management',pmid='1'),
            paper('Insulin delivery and monitoring for children with type 1 diabetes',
                  'Pediatric glucose management',pmid='2')
        ]
        valid=[p for p in papers if pm._is_relevant_paper(CHILD_T1_Q,p)]
        self.assertEqual(['2'],[p['pmid'] for p in valid])

    def test_hypertension_pediatric_paper_excluded_end_to_end(self):
        papers=[
            paper('Pediatric Hypertension', 'Hypertension diagnosis and management in children',pmid='1'),
            paper('Management of Hypertension in Adults', 'Diagnosis pathophysiology clinical management',pmid='2')
        ]
        valid=[p for p in papers if pm._is_relevant_paper(HTN_Q,p)]
        selected=pm._select_diverse_papers(HTN_Q,valid,7)
        self.assertEqual(['2'],[p['pmid'] for p in selected])

    def test_ckd_dialysis_clinical_evidence_beats_qol(self):
        guideline=paper('Dialysis initiation and management in chronic kidney disease',
                        'Clinical indications for hemodialysis',pmid='1')
        qol=paper('Quality of life compared between peritoneal and hemodialysis in CKD',
                  'CKD patients and dialysis',pmid='2')
        self.assertGreater(pm._paper_sort_score(CKD_Q,guideline),pm._paper_sort_score(CKD_Q,qol))

    def test_ckd_dialysis_usable_aspect_prioritization(self):
        guideline=paper('Dialysis management and initiation in chronic kidney disease',
                        'Dialysis modalities and evaluation',pmid='1')
        qol=paper('Quality of life compared between peritoneal and hemodialysis in CKD',
                  'CKD dialysis management',pmid='2')
        chosen=pm._select_diverse_papers(CKD_Q,[qol,guideline],1)
        self.assertEqual('1',chosen[0]['pmid'])

    def test_title_dedup_guideline_records_not_crowd(self):
        a=paper('2025 Guideline for the Prevention, Detection and Management of High Blood Pressure in Adults',
                'Hypertension treatment and diagnosis',pmid='1')
        b=paper('2025 Guideline for the Prevention, Detection and Management of High Blood Pressure in Adults: Executive Summary',
                'Hypertension treatment and diagnosis',pmid='2')
        chosen=pm._select_diverse_papers(HTN_Q,[a,b],7)
        self.assertEqual(1,len(chosen))

    def test_budget_limits(self):
        p=[paper(f'Chronic kidney disease management review volume {n}', 'CKD dialysis anemia therapy',pmid=str(n)) for n in range(20)]
        self.assertLessEqual(len(pm._select_diverse_papers(CKD_Q,p,9)),9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
