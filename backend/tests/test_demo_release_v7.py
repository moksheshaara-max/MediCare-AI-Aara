"""Observed live v6 failure-path regression tests; no DB or API access.

These rules catch specified contradiction patterns only, not clinical validation.
"""
import ast
import importlib.util
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]


def get_gate():
    src=ROOT/'generation'/'generator.py'
    tree=ast.parse(src.read_text(encoding='utf-8'))
    chosen=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {
        '_clinical_answer_blockers','_clinical_evidence_safety_note', '_targeted_regeneration_guidance'}
        or isinstance(n, ast.Assign) and any(isinstance(x,ast.Name) and x.id in {
        'NTEP_DS_TB_OFFICIAL_URL','WHO_MALARIA_OFFICIAL_URL','WHO_DRTB_2025_OFFICIAL_URL','WHO_PPH_2025_OFFICIAL_URL'
        } for x in n.targets)]
    d={'re':re}
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(src),'exec'),d)
    return d


def get_pubmed():
    src=ROOT/'pubmed'/'pubmed_search.py'
    spec=importlib.util.spec_from_file_location('medicare_pubmed_demo_v7',src)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ObservedDRTBRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=get_gate()
    def flags(self, answer):
        return self.d['_clinical_answer_blockers'](
            'Management of multidrug-resistant or rifampicin-resistant tuberculosis in India',answer)
    def test_real_v6_direct_longer_mermaid_branch_blocked(self):
        ans='''```mermaid
flowchart TD
C{"Assess Regimen Eligibility"}
C -->|Eligible for BPaLM| D["6-Month BPaLM Regimen"]
C -->|FQ-Susceptible / Shorter Eligible| E["9–11 Month Shorter Oral Regimen"]
C -->|Complex / FQ-Resistant / Ineligible| F["Individualized Longer Oral Regimen"]
```'''
        self.assertIn('FQ-resistant DR-TB routed directly to longer therapy without six-month eligibility assessment',self.flags(ans))
    def test_real_v6_longer_table_row_blocked(self):
        ans='| **18–20 Month Longer Oral Regimen** | Complex cases, fluoroquinolone-resistant (Pre-XDR/XDR), or treatment failures | Longer |'
        self.assertIn('FQ-resistant DR-TB routed directly to longer therapy without six-month eligibility assessment',self.flags(ans))
    def test_line_with_ineligibility_after_bpal_review_is_allowed(self):
        ans='| FQ-resistant, ineligible for BPaL after specialist review | individualized longer regimen |'
        self.assertNotIn('FQ-resistant DR-TB routed directly to longer therapy without six-month eligibility assessment',self.flags(ans))
    def test_mermaid_bpal_eligibility_then_longer_allowed(self):
        ans='''```mermaid
flowchart TD
C -->|FQ resistant| F["Check BPaL eligibility; if unsuitable, longer therapy"]
```'''
        self.assertNotIn('FQ-resistant DR-TB routed directly to longer therapy without six-month eligibility assessment',self.flags(ans))
    def test_mermaid_quoted_edge_direct_longer_blocked(self):
        ans='''```mermaid
flowchart TD
C -- "FQ resistant" --> F
F["Individualized longer regimen"]
```'''
        self.assertIn('FQ-resistant DR-TB routed directly to longer therapy without six-month eligibility assessment',self.flags(ans))
    def test_new_regeneration_direction_names_bpal(self):
        s=self.d['_targeted_regeneration_guidance'](['FQ-resistant DR-TB routed directly to longer therapy without six-month eligibility assessment'])
        self.assertIn('BPaL',s)
        self.assertIn('BEFORE',s)
    def test_who_note_states_prior_short_regimen_review(self):
        note=self.d['_clinical_evidence_safety_note']('Management of MDR/RR-TB in India')
        self.assertIn('BPaL',note)
        self.assertIn('FLOWCHART',note)

class DSTBDiagnosticGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=get_gate()
    def flags(self, answer):
        return self.d['_clinical_answer_blockers']('Management of drug-susceptible pulmonary tuberculosis under NTEP India', answer)
    def test_rif_only_incorrect_confirmation_blocked(self):
        s='A positive molecular test for rifampicin susceptibility confirms eligibility for first-line standard regimens.'
        self.assertIn('Rifampicin susceptibility alone treated as full DS-TB confirmation',self.flags(s))
    def test_rif_only_cautious_statement_allowed(self):
        s='Rifampicin susceptibility does not confirm full drug susceptibility; assess isoniazid resistance.'
        self.assertNotIn('Rifampicin susceptibility alone treated as full DS-TB confirmation',self.flags(s))
    def test_india_note_reminds_isonaizid_resistance(self):
        note=self.d['_clinical_evidence_safety_note']('NTEP India drug-susceptible pulmonary tuberculosis management')
        self.assertIn('isoniazid',note.lower())

class SevereMalariaPubMed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=get_pubmed()
    QUERY='Explain severe falciparum malaria in an adult in India, including diagnosis and management'
    def scope(self, title):
        return self.p._clinical_research_scope_match(self.QUERY,{'title':title,'abstract':'malaria'},'malaria')
    def test_cambodia_epidemiology_excluded(self):
        self.assertFalse(self.scope('Malaria in Cambodia: A Retrospective Analysis of a Changing Epidemiology 2006-2019.'))
    def test_severe_care_stays(self):
        self.assertTrue(self.scope('Artemether for severe malaria.'))
    def test_query_explicitly_about_epidemiology_can_keep_it(self):
        q='Describe severe falciparum malaria epidemiology and surveillance in Cambodia'
        self.assertTrue(self.p._clinical_research_scope_match(q, {'title':'Malaria in Cambodia: A Retrospective Analysis of a Changing Epidemiology 2006-2019', 'abstract':'malaria'}, 'malaria'))

if __name__=='__main__':
    unittest.main()
