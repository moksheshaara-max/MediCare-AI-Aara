"""Regression tests reproducing live source identity defects after master-fix rollout.
No network, MongoDB writes, or clinical validation. Uses the uploaded source registry.
"""
import csv
import unittest
from pathlib import Path
from unittest.mock import patch
from test_upgrade import load_search
import hashlib

def chunks(config):
    out=[]
    ix=1
    for sid,n,sim in config:
        for j in range(n):
            out.append({"source_id":sid,"chunk_id":f"synthetic{ix}","page_start":j+1,"page_end":j+1,"score":sim-0.0001*j,
                        "text":" ".join("".join(chr(97+(b%26)) for b in hashlib.sha256(f"{sid}:{j}:{k}".encode()).digest()) for k in range(30))})
            ix+=1
    return out

ROOT = Path(__file__).resolve().parent.parent

FALLBACK_ROWS = {
    'who_pph_2025': ('Consolidated Guidelines for the Prevention Diagnosis and Treatment of Postpartum Haemorrhage', 'postpartum haemorrhage', 'obstetrics', 'clinical_guideline', 'Global'),
    'nhm_cac_2023': ('Comprehensive Abortion Care Training and Service Delivery Guidelines', 'comprehensive abortion care', 'obstetrics / reproductive health', 'clinical_guideline', 'India'),
    'who_tb_module4_treatment_care_2026_2ed': ('WHO Consolidated Guidelines on Tuberculosis Module 4 - Treatment and Care','tuberculosis','infectious disease','clinical_guideline','Global'),
    'who_tb_module3_diagnosis_2025': ('WHO Consolidated Guidelines on Tuberculosis Module 3 - Diagnosis','tuberculosis diagnosis','infectious disease','clinical_guideline','Global'),
    'ntep_dr_tb_2025': ('National Guidelines for Management of Drug-Resistant Tuberculosis','drug-resistant tuberculosis','infectious disease / tuberculosis','clinical_guideline','India'),
    'ntep_differentiated_tb_care_2025': ('National Guidance on Differentiated TB Care','differentiated tuberculosis care','infectious disease / tuberculosis','clinical_guideline','India'),
    'ntep_family_caregiver_tb': ('Guidance Document on Engagement of Family Caregiver for TB Patients','family caregiver engagement for tuberculosis patients','infectious disease / tuberculosis / community care','clinical_programme_guideline','India'),
}
META={sid:{'source_id':sid,'official_title':title,'topic':topic,'specialty':specialty,'source_type':kind, 'country':country, 'is_current':True, 'publication_year':2025,'organization':'WHO' if country=='Global' else 'National'} for sid,(title,topic,specialty,kind,country) in FALLBACK_ROWS.items()}

class CorrectClinicalSubtypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search, _ = load_search()

    def test_pph_us_and_uk_spellings_identified_as_one_topic(self):
        for q in ('A patient develops postpartum hemorrhage after delivery.',
                  'Management of post-partum haemorrhage after vaginal delivery',
                  'Treat PPH following childbirth'):
            with self.subTest(q=q):
                self.assertIn('postpartum_hemorrhage',self.search._clinical_topic_labels(q))
                self.assertEqual('primary',self.search._topic_facet_tier(q, META['who_pph_2025']))

    def test_pph_filters_wrong_abortion_care_source_and_keeps_who(self):
        q='A patient develops postpartum hemorrhage after vaginal delivery. Explain recognition, uterotonics and tranexamic acid.'
        with patch.object(self.search,'_load_source_metadata',return_value=META):
            pool=self.search._attach_metadata_and_rerank(q,chunks([
                ('who_pph_2025',45,.897),('nhm_cac_2023',13,.87)
            ]),'doctor')
            filtered,_=self.search._apply_diversified_source_relevance_guard(q,pool)
            ids={c['source_id'] for c in filtered}
            self.assertIn('who_pph_2025',ids)
            self.assertNotIn('nhm_cac_2023',ids)
            for mode in ('doctor','pg_student'):
                selected=self.search._diversify_candidates(q,filtered,36,mode)
                selected_ids={c['source_id'] for c in selected}
                self.assertIn('who_pph_2025',selected_ids)
                self.assertNotIn('nhm_cac_2023',selected_ids)

    def test_dr_tb_rejected_for_ds_tb_not_for_mdr_request(self):
        for q in ('newly diagnosed drug-susceptible pulmonary tuberculosis under NTEP',
                  'first line regimen for drug sensitive TB',
                  'Treat DS-TB in India'):
            with self.subTest(q=q):
                self.assertTrue(self.search._has_explicit_disease_context_conflict(q,META['ntep_dr_tb_2025']))
                self.assertFalse(self.search._has_explicit_disease_context_conflict(q,META['who_tb_module4_treatment_care_2026_2ed']))
        for q in ('Treat MDR-TB under NTEP',
                  'Compare drug-susceptible and drug-resistant TB regimens'):
            with self.subTest(q=q):
                self.assertFalse(self.search._has_explicit_disease_context_conflict(q,META['ntep_dr_tb_2025']))

    def test_programmatic_tb_subtypes_supplement_not_replace_treatment(self):
        q='Describe the diagnostic workup and management of newly diagnosed drug-susceptible pulmonary tuberculosis in India under NTEP, including treatment monitoring and adherence support.'
        with patch.object(self.search,'_load_source_metadata',return_value=META):
            self.assertIsNone(self.search._resolve_explicit_source_constraint(q))
            pool=self.search._attach_metadata_and_rerank(q,chunks([
                ('ntep_dr_tb_2025',28,.888),
                ('ntep_differentiated_tb_care_2025',26,.870),
                ('ntep_family_caregiver_tb',10,.854),
                ('who_tb_module4_treatment_care_2026_2ed',55,.863),
                ('who_tb_module3_diagnosis_2025',20,.850),
            ]),'doctor')
            filtered,_=self.search._apply_diversified_source_relevance_guard(q,pool)
            tiers={c['source_id']:c['topic_tier'] for c in filtered}
            self.assertNotIn('ntep_dr_tb_2025',tiers)
            self.assertEqual('primary',tiers['who_tb_module4_treatment_care_2026_2ed'])
            self.assertEqual('primary',tiers['who_tb_module3_diagnosis_2025'])
            self.assertEqual('supplementary',tiers['ntep_differentiated_tb_care_2025'])
            self.assertEqual('supplementary',tiers['ntep_family_caregiver_tb'])
            for mode in ('doctor','pg_student'):
                chosen=self.search._diversify_candidates(q,filtered,36,mode)
                ids={c['source_id'] for c in chosen}
                self.assertIn('who_tb_module4_treatment_care_2026_2ed',ids)
                self.assertNotIn('ntep_dr_tb_2025',ids)
                self.assertGreater(sum(x['source_id']=='who_tb_module4_treatment_care_2026_2ed' for x in chosen),
                                   sum(x['source_id']=='ntep_differentiated_tb_care_2025' for x in chosen))

    def test_explicit_ntep_ds_tb_not_rescued_by_dr_source(self):
        q='According to NTEP guidelines, treat newly diagnosed drug-susceptible pulmonary tuberculosis'
        with patch.object(self.search,'_load_source_metadata',return_value=META):
            pool=self.search._attach_metadata_and_rerank(q,chunks([('ntep_dr_tb_2025',10,.92)]),'doctor')
            chosen=self.search._choose_relevant_source_ids(q,pool,allowed_ids={'ntep_dr_tb_2025'})
            self.assertEqual([],chosen)

    def test_generic_abortion_care_and_dr_tb_intents_remain_eligible(self):
        self.assertFalse(self.search._clear_topic_mismatch('Explain comprehensive abortion care',META['nhm_cac_2023']))
        self.assertEqual('primary',self.search._topic_facet_tier('Explain MDR drug-resistant tuberculosis management',META['ntep_dr_tb_2025']))

if __name__=='__main__':
    unittest.main()
