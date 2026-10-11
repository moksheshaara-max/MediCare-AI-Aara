"""Offline regression cases derived from 25x2 audit; never invoke real services."""
import unittest
from test_upgrade import load_search
from test_pubmed_final import pm

class AuditPubMedTests(unittest.TestCase):
    def test_new_disease_anchors_are_recognized(self):
        topics = {
            'Stable symptomatic COPD in adults, diagnosis and management': 'chronic obstructive pulmonary disease',
            'Diagnose community-acquired pneumonia in an adult': 'community acquired pneumonia',
            'Describe drug-susceptible pulmonary tuberculosis under NTEP': 'drug susceptible tuberculosis',
            'Explain rifampicin-resistant or multidrug-resistant pulmonary tuberculosis': 'drug resistant tuberculosis',
            'An adult has dengue with warning signs: monitoring and therapy': 'dengue',
            'Newly diagnosed HIV in an adult: ART management': 'hiv',
            'In a 3-year-old with a first febrile urinary tract infection: management': 'urinary tract infection',
            'A 7-year-old has generalized convulsions lasting longer than five minutes. Explain management': 'status epilepticus',
            'Explain lupus nephritis treatment': 'lupus nephritis',
            'Explain atrial fibrillation anticoagulation and stroke-risk assessment': 'atrial fibrillation',
            'Explain stroke-risk assessment for an adult with atrial fibrillation': 'atrial fibrillation',
            'Explain acute ischemic stroke treatment': 'ischemic stroke',
            'Treat postpartum hemorrhage following vaginal delivery': 'postpartum hemorrhage',
            'Severe preeclampsia at 32 weeks of pregnancy: management': 'preeclampsia',
            'Early-stage breast cancer in India: staging and management': 'breast cancer',
            'Type 2 diabetes and an infected foot ulcer: management': 'diabetic foot infection',
            'Compare Graves disease and Hashimoto thyroiditis': 'thyroid disorders',
            'Snakebite antivenom management in a district hospital': 'snakebite',
        }
        for q, topic in topics.items():
            with self.subTest(q=q):
                self.assertEqual(topic, pm._topics(q)[0])
                self.assertIn('[Title/Abstract]', pm._query_plan(q)[0])

    def test_arni_is_not_inside_warning(self):
        self.assertFalse(pm._mentions_named_drug('dengue with warning signs', 'arni'))
        self.assertTrue(pm._mentions_named_drug('ARNI in HFrEF', 'arni'))
        q = 'Adult dengue with warning signs. Explain monitoring and fluid therapy.'
        self.assertNotIn('"arni"', ' '.join(pm._query_plan(q)).lower())

    def test_known_no_topic_is_still_not_queried_as_first_line(self):
        self.assertEqual([], pm._query_plan('Current guideline based first-line treatment?'))

class AuditRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r, _ = load_search()

    def meta(self, title, topic='', source_id=''):
        return {'official_title': title, 'topic': topic, 'source_id': source_id, 'source_type': 'clinical_guideline'}

    def test_obstetric_emergency_rejects_unrelated_guidelines(self):
        q = 'Postpartum hemorrhage after vaginal delivery: treatment with uterotonics'
        self.assertTrue(self.r._has_explicit_disease_context_conflict(q, self.meta('Standard Treatment Guidelines for Burn Injuries')))
        q = 'Pregnant at 32 weeks with severe preeclampsia; BP control'
        self.assertTrue(self.r._has_explicit_disease_context_conflict(q, self.meta('Standard Treatment Guidelines for Liver Transplantation')))
        self.assertFalse(self.r._has_explicit_disease_context_conflict(q, self.meta('WHO Postpartum Haemorrhage', topic='postpartum haemorrhage')))

    def test_breast_cancer_rejects_other_organ_cancers(self):
        q = 'Teach breast cancer treatment and management in India'
        self.assertTrue(self.r._has_explicit_disease_context_conflict(q, self.meta('Urinary Bladder Cancer', 'urinary bladder cancer')))
        self.assertTrue(self.r._has_explicit_disease_context_conflict(q, self.meta('Pancreatic Cancer', 'pancreatic cancer')))
        self.assertFalse(self.r._has_explicit_disease_context_conflict(q, self.meta('Breast Cancer', 'breast cancer')))

    def test_non_match_kidney_source_is_not_kidney_match(self):
        self.assertTrue(self.r._has_explicit_disease_context_conflict(
            'Adult with acute kidney injury: diagnosis and management', self.meta('Standard Treatment Guidelines - Respiratory Diseases')))
        self.assertTrue(self.r._has_explicit_disease_context_conflict(
            'Relapse of steroid-sensitive nephrotic syndrome', self.meta('Autosomal Dominant Polycystic Kidney Disease', 'ADPKD')))

    def test_active_tb_does_not_anchor_to_preventive_tb(self):
        m=self.meta('Guidelines for Programmatic Management of Tuberculosis Preventive Treatment in India')
        self.assertTrue(self.r._has_explicit_disease_context_conflict(
            'Newly diagnosed drug-susceptible pulmonary tuberculosis management and monitoring', m))
        self.assertFalse(self.r._has_explicit_disease_context_conflict(
            'Explain TB preventive treatment for latent TB infection in child contacts', m))

    def test_age_only_pediatric_query_is_not_generic(self):
        child=self.meta('Status Epilepticus in Children','status epilepticus children')
        adults=self.meta('Treatment of Adults with Status Epilepticus','adult status epilepticus')
        all_ages=self.meta('Epilepsies in Children Young People and Adults','epilepsies')
        q='A 7-year-old has generalized convulsions lasting longer than five minutes'
        self.assertFalse(self.r._has_explicit_population_conflict(q, child))
        self.assertTrue(self.r._has_explicit_population_conflict(q, adults))
        self.assertFalse(self.r._has_explicit_population_conflict(q, all_ages))

    def test_multiple_component_pg_classification(self):
        detailed=[
          'In a 9-year-old with recurrent wheeze, explain confirming asthma, controller treatment, inhaler technique, acute exacerbation care and an action plan.',
          'Discuss diagnosis, severity assessment, empirical antimicrobial therapy, microbiology, inpatient referral and follow-up for community-acquired pneumonia in an adult.',
          'Explain evaluation and management of rifampicin-resistant or multidrug-resistant pulmonary tuberculosis in India, including susceptibility testing, regimen principles and adverse-effect monitoring.',
        ]
        for q in detailed:
            with self.subTest(q=q):
                self.assertEqual('detailed',self.r.classify_pg_query_depth(q))
        q='Teach comprehensive initial management of a newly diagnosed adult with HIV, including baseline evaluation, ART, opportunistic infection screening, preventive care, interactions and follow-up.'
        self.assertEqual('comprehensive',self.r.classify_pg_query_depth(q))
        self.assertEqual('simple',self.r.classify_pg_query_depth(
            'Explain the mechanism, adverse effects and contraindications of ACE inhibitors'))

if __name__=='__main__':
    unittest.main()
