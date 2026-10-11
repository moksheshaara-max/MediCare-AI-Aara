"""Offline source identity regressions from live 2026-10-11 Atlas stage trace.

These tests use fake chunks and registry-shaped source metadata. They DO NOT
verify clinical correctness or make live Mongo/Gemini/PubMed requests.
"""
import unittest
import hashlib
from unittest.mock import patch
from test_upgrade import load_search

SOURCES = {
    'who_childhood_asthma_bronchiolitis_2026': ('WHO Consolidated Guidelines for the Management of Common Childhood Illness - Management of Asthma in Children and Adolescents and Bronchiolitis in Infants and Young Children', 'asthma and bronchiolitis', 'paediatrics','clinical_guideline'),
    'iap_stg_35_bronchial_asthma': ('Standard Treatment Guidelines 35 - Bronchial Asthma','bronchial asthma','paediatrics / pulmonology','clinical_guideline'),
    'nice_ng245_asthma': ('Asthma: diagnosis monitoring and chronic asthma management (BTS NICE SIGN)','asthma diagnosis monitoring and chronic management','pulmonology / paediatrics','clinical_guideline'),
    'mohfw_stg_critical_care': ('Guidelines for the Management of Critical Care','critical care assessment stabilization organ support monitoring and management','critical care / emergency medicine','clinical_guideline'),
    'who_tb_module4_treatment_care_2026_2ed': ('WHO Consolidated Guidelines on Tuberculosis Module 4 - Treatment and Care','tuberculosis','infectious disease','clinical_guideline'),
    'who_tb_module3_diagnosis_2025': ('WHO Consolidated Guidelines on Tuberculosis Module 3 - Diagnosis','tuberculosis','infectious disease','clinical_guideline'),
    'ntep_tb_preventive_treatment_2021': ('Guidelines for Programmatic Management of Tuberculosis Preventive Treatment in India','tuberculosis preventive treatment','infectious disease','clinical_guideline'),
    'who_hiv_clinical_management_2025': ('WHO Updated Recommendations on HIV Clinical Management - Recommendations for a Public Health Approach','HIV clinical management','infectious disease','clinical_guideline'),
    'who_advanced_hiv_disease_2025': ('WHO Guidelines on the Management of Advanced HIV Disease','advanced HIV disease','infectious disease','clinical_guideline'),
    'ntep_tb_plhiv_art_centres': ('Guidelines on Prevention and Management of Tuberculosis in PLHIV at ART Centres','tuberculosis in people living with HIV at ART centres','infectious disease / HIV / tuberculosis','clinical_guideline'),
    'who_pph_2025': ('Consolidated Guidelines for the Prevention Diagnosis and Treatment of Postpartum Haemorrhage','postpartum haemorrhage','obstetrics','clinical_guideline'),
    'nhm_iphs_shc_hwc_uhwc_2022': ('Indian Public Health Standards 2022 - Health and Wellness Centre Sub Health Centre and Urban Health and Wellness Centre','health system standards and referral','community health','health_system_standard'),
    'who_malaria_guidelines_2025_08_13': ('WHO Guidelines for Malaria','malaria','infectious disease','clinical_guideline'),
    'ncvbdc_malaria_nsp_2023_2027': ('National Strategic Plan for Malaria Elimination in India 2023-2027','malaria elimination surveillance diagnosis treatment vector control and programme strategy','infectious disease / tropical medicine','public_health_guideline'),
    'iap_stg_64_nephrotic_syndrome': ('Standard Treatment Guidelines 64 - Nephrotic Syndrome','nephrotic syndrome','pediatrics','clinical_guideline'),
    'nhm_lsas_2024': ('Life Saving Anaesthesia Skills Revised Curriculum 2024','life saving anaesthesia skills for emergency obstetric care','anaesthesiology / obstetrics','clinical_training_guideline'),
    'kdigo_ckd_2024': ('KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease','chronic kidney disease','nephrology','clinical_guideline'),
    'esc_heart_failure_2026': ('2026 ESC Guidelines for the Management of Heart Failure','heart failure','cardiology','clinical_guideline'),
    'mohfw_stg_hypertension': ('Standard Treatment Guidelines - Hypertension','hypertension diagnosis risk assessment lifestyle management and pharmacologic treatment','cardiology / internal medicine','clinical_guideline'),
}

META = {sid: {'source_id':sid,'official_title':title,'topic':topic,'specialty':specialty,'source_type':typ,'country':('India' if sid.startswith(('ntep_', 'iap_', 'mohfw_', 'nhm_', 'ncvbdc_')) else 'International'), 'is_current':True, 'publication_year':2025, 'organization':('National Tuberculosis Elimination Programme' if sid.startswith('ntep_') else title.split(' ')[0])} for sid,(title,topic,specialty,typ) in SOURCES.items()}

QUESTIONS={
    'asthma':'In a 9-year-old with recurrent wheeze, explain confirming asthma, controller treatment, inhaler technique, acute exacerbation care and an action plan.',
    'tb':'Describe the diagnostic workup and management of newly diagnosed drug-susceptible pulmonary tuberculosis in India under NTEP, including treatment monitoring and adherence support.',
    'hiv':'Teach comprehensive initial management of a newly diagnosed adult with HIV, including baseline evaluation, ART, opportunistic infection screening, preventive care, interactions and follow-up.',
    'pph':'A patient develops postpartum hemorrhage after vaginal delivery. Explain recognition, initial resuscitation, uterotonic medications, tranexamic acid and escalation of care.',
    'malaria':'Explain severe falciparum malaria in an adult in India: diagnosis, organ dysfunction, parenteral treatment, monitoring and complications.',
    'preeclampsia':'A pregnant patient at 32 weeks has severe preeclampsia. Discuss diagnostic severe features, maternal stabilization, magnesium sulfate, BP control, fetal surveillance and delivery planning.',
}

def chunks(config):
    out=[]
    ix=1
    for sid,n,sim in config:
        for j in range(n):
            out.append({'source_id':sid,'chunk_id':f'synthetic{ix}', 'page_start':j+1, 'page_end':j+1,'score':sim-0.0001*j,
                        'text':' '.join(''.join(chr(97 + (b % 26)) for b in hashlib.sha256(f'{sid}:{j}:{k}'.encode()).digest()) for k in range(30))})
            ix+=1
    return out

class GuidelineMasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r,_ = load_search()

    def test_registered_asthma_topic_tiers(self):
        q=QUESTIONS['asthma']
        with patch.object(self.r,'_load_source_metadata',return_value=META):
            pool=self.r._attach_metadata_and_rerank(q,chunks([
                ('who_childhood_asthma_bronchiolitis_2026',12,.857),
                ('iap_stg_35_bronchial_asthma',8,.854),
                ('nice_ng245_asthma',10,.853),
                ('mohfw_stg_critical_care',6,.812),
            ]),'doctor')
            filtered,_=self.r._apply_diversified_source_relevance_guard(q,pool)
            tiers={c['source_id']:c['topic_tier'] for c in filtered}
            self.assertEqual('primary',tiers['who_childhood_asthma_bronchiolitis_2026'])
            self.assertEqual('primary',tiers['iap_stg_35_bronchial_asthma'])
            self.assertEqual('primary',tiers['nice_ng245_asthma'])
            self.assertNotIn('mohfw_stg_critical_care',tiers)
            for mode in ('doctor','pg_student'):
                selected=self.r._diversify_candidates(q,filtered,36,mode)
                self.assertIn('who_childhood_asthma_bronchiolitis_2026',{c['source_id'] for c in selected})

    def test_active_tb_no_accidental_ntep_lock(self):
        q=QUESTIONS['tb']
        with patch.object(self.r,'_load_source_metadata',return_value=META):
            self.assertIsNone(self.r._resolve_explicit_source_constraint(q))
            pool=self.r._attach_metadata_and_rerank(q,chunks([
                ('who_tb_module4_treatment_care_2026_2ed',15,.863),
                ('who_tb_module3_diagnosis_2025',8,.850),
                ('ntep_tb_preventive_treatment_2021',15,.893),
            ]),'doctor')
            filtered,_=self.r._apply_diversified_source_relevance_guard(q,pool)
            names={c['source_id'] for c in filtered}
            self.assertIn('who_tb_module4_treatment_care_2026_2ed',names)
            self.assertIn('who_tb_module3_diagnosis_2025',names)
            self.assertNotIn('ntep_tb_preventive_treatment_2021',names)

    def test_strict_ntep_request_never_selects_tpt_for_active_tb(self):
        q='According to NTEP guidelines, treat newly diagnosed drug-susceptible pulmonary tuberculosis'
        with patch.object(self.r,'_load_source_metadata',return_value=META):
            self.assertIsNotNone(self.r._resolve_explicit_source_constraint(q))
            source_only=chunks([('ntep_tb_preventive_treatment_2021',5,.9)])
            ranked=self.r._attach_metadata_and_rerank(q,source_only,'doctor')
            self.assertEqual([],self.r._choose_relevant_source_ids(q,ranked,allowed_ids={'ntep_tb_preventive_treatment_2021'}))

    def test_hiv_primary_not_tb_program(self):
        q=QUESTIONS['hiv']
        with patch.object(self.r,'_load_source_metadata',return_value=META):
            pool=self.r._attach_metadata_and_rerank(q,chunks([
                ('who_hiv_clinical_management_2025',10,.778),
                ('who_advanced_hiv_disease_2025',14,.850),
                ('ntep_tb_plhiv_art_centres',10,.806),
            ]),'doctor')
            filtered,_=self.r._apply_diversified_source_relevance_guard(q,pool)
            tiers={c['source_id']:c['topic_tier'] for c in filtered}
            self.assertEqual('primary',tiers['who_hiv_clinical_management_2025'])
            self.assertEqual('supplementary',tiers['who_advanced_hiv_disease_2025'])
            self.assertEqual('supplementary',tiers['ntep_tb_plhiv_art_centres'])
            selected=self.r._diversify_candidates(q,filtered,36,'doctor')
            self.assertIn('who_hiv_clinical_management_2025',{c['source_id'] for c in selected})

    def test_pph_prevention_title_does_not_veto_treatment(self):
        q=QUESTIONS['pph']
        with patch.object(self.r,'_load_source_metadata',return_value=META):
            self.assertFalse(self.r._has_explicit_population_conflict(q,META['who_pph_2025']))
            pool=self.r._attach_metadata_and_rerank(q,chunks([
                ('who_pph_2025',20,.897),
                ('nhm_iphs_shc_hwc_uhwc_2022',3,.814),
            ]),'doctor')
            filtered,_=self.r._apply_diversified_source_relevance_guard(q,pool)
            tiers={c['source_id']:c['topic_tier'] for c in filtered}
            self.assertEqual('primary',tiers['who_pph_2025'])

    def test_malaria_who_clinical_before_elimination_strategy(self):
        q=QUESTIONS['malaria']
        with patch.object(self.r,'_load_source_metadata',return_value=META):
            pool=self.r._attach_metadata_and_rerank(q,chunks([
                ('who_malaria_guidelines_2025_08_13',25,.877),
                ('ncvbdc_malaria_nsp_2023_2027',20,.844),
            ]),'doctor')
            filtered,_=self.r._apply_diversified_source_relevance_guard(q,pool)
            tiers={c['source_id']:c['topic_tier'] for c in filtered}
            self.assertEqual('primary',tiers['who_malaria_guidelines_2025_08_13'])
            self.assertEqual('supplementary',tiers['ncvbdc_malaria_nsp_2023_2027'])
            for mode in ('doctor','pg_student'):
                chosen=self.r._diversify_candidates(q,filtered,36,mode)
                ids={c['source_id'] for c in chosen}
                self.assertIn('who_malaria_guidelines_2025_08_13',ids)

    def test_preeclampsia_broad_nhm_fallback_unchanged(self):
        q=QUESTIONS['preeclampsia']
        with patch.object(self.r,'_load_source_metadata',return_value=META):
            pool=self.r._attach_metadata_and_rerank(q,chunks([('nhm_lsas_2024',9,.89)]),'doctor')
            filtered,_=self.r._apply_diversified_source_relevance_guard(q,pool)
            self.assertTrue(any(x['source_id']=='nhm_lsas_2024' for x in filtered))

    def test_core_previous_topics_still_match(self):
        cases={
            'Explain chronic kidney disease staging, diagnosis, dialysis': 'kdigo_ckd_2024',
            'What is the guideline treatment of HFrEF heart failure': 'esc_heart_failure_2026',
            'What is first-line hypertension treatment': 'mohfw_stg_hypertension',
            'Explain pediatric nephrotic syndrome relapse management': 'iap_stg_64_nephrotic_syndrome',
        }
        for q,sid in cases.items():
            with self.subTest(q=q):
                self.assertEqual('primary',self.r._topic_facet_tier(q,META[sid]))

    def test_unrelated_topic_rejected(self):
        q=QUESTIONS['asthma']
        self.assertEqual('reject',self.r._topic_facet_tier(q,META['who_malaria_guidelines_2025_08_13']))
        self.assertEqual('reject',self.r._topic_facet_tier(q,META['mohfw_stg_critical_care']))

    def test_clinical_guidelines_with_prevention_in_title_remain_eligible(self):
        examples = [
            ('COPD diagnosis and treatment in an adult',
             {'official_title':'Global Strategy for Diagnosis, Management and Prevention of Chronic Obstructive Pulmonary Disease',
              'topic':'chronic obstructive pulmonary disease','specialty':'pulmonology'}),
            (QUESTIONS['pph'], META['who_pph_2025']),
            ('Discuss acute malaria treatment in an adult',
             META['who_malaria_guidelines_2025_08_13']),
        ]
        for q, meta in examples:
            with self.subTest(q=q):
                self.assertFalse(self.r._has_explicit_population_conflict(q,meta))

    def test_unknown_condition_uses_existing_topic_fallback(self):
        q='Explain emergency obstetric anaesthesia skills and monitoring'
        with patch.object(self.r,'_load_source_metadata',return_value=META):
            pool=self.r._attach_metadata_and_rerank(q,chunks([('nhm_lsas_2024',6,.89)]),'doctor')
            filtered,_=self.r._apply_diversified_source_relevance_guard(q,pool)
            self.assertTrue(filtered)

    def test_no_source_lock_for_national_program_context(self):
        with patch.object(self.r,'_load_source_metadata',return_value=META):
            for q in (
                'Treatment of drug-susceptible tuberculosis under NTEP',
                'Treatment of drug-susceptible tuberculosis within the National Tuberculosis Elimination Programme',
            ):
                with self.subTest(q=q):
                    self.assertIsNone(self.r._resolve_explicit_source_constraint(q))

    def test_generic_diabetes_avoids_disease_mismatch(self):
        adult_t1={'official_title':'Type 1 Diabetes Guidelines', 'topic':'type 1 diabetes', 'specialty':'endocrinology'}
        self.assertTrue(self.r._has_explicit_disease_context_conflict(
            'Treat type 2 diabetes mellitus', adult_t1))
        self.assertFalse(self.r._has_explicit_disease_context_conflict(
            'Treat type 1 diabetes mellitus', adult_t1))

    def test_a_source_with_both_populations_stays_eligible(self):
        source=dict(META['nice_ng245_asthma'], notes='current BTS NICE SIGN guideline covers adults, young people and children')
        self.assertFalse(self.r._has_explicit_population_conflict('Asthma diagnosis in an adult',source))
        self.assertFalse(self.r._has_explicit_population_conflict('Asthma in a 9-year-old child',source))

if __name__=='__main__':unittest.main()
