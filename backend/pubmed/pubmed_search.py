"""MediCare AI - final candidate: guarded, facet-balanced NIH PubMed retrieval.

No LLM is used for PubMed query formulation or filtering. This remains
supplementary evidence; retrieved papers never outrank applicable guidelines.
Keep public functions compatible with existing generation code.
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime
from math import ceil
from typing import Dict, List, Optional, Tuple

import requests
import xmltodict
from dotenv import load_dotenv

load_dotenv()
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "").strip()
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_RESULTS = 5
REQUEST_TIMEOUT = 12
# Bounds the full multi-facet ID-search phase on slow/unreachable PubMed servers.
SEARCH_BUDGET_SECONDS = 28
HEADERS = {
    "User-Agent": "MediCareAI/1.1 (clinical educational decision support; PubMed E-utilities)",
    "Accept": "application/json,application/xml,*/*",
}

# Use longest, clinically specific conditions first. The lexicon is only for
# anchoring known terms; general unknown conditions follow the safe fallback.
CONDITIONS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    # Clinical topic coverage is deliberately explicit. A fallback assembled
    # from arbitrary words in a long patient question is not a safe disease
    # identity (and often produced empty or irrelevant PubMed results).
    (r"\b(?:rifampicin[- ]resistant\s+or\s+multidrug[- ]resistant\s+(?:pulmonary\s+)?tuberculosis|multidrug[- ]resistant\s+(?:pulmonary\s+)?tuberculosis|rifampicin[- ]resistant\s+(?:pulmonary\s+)?tuberculosis|\bmdr[/ -]?rr[- ]?tb\b|\bmdr[- ]?tb\b|\brr[- ]?tb\b)\b", "drug resistant tuberculosis", ("multidrug resistant tuberculosis", "rifampicin resistant tuberculosis", "MDR TB")),
    (r"\b(?:drug[- ]susceptible (?:pulmonary )?tuberculosis|\bds[- ]?tb\b)\b", "drug susceptible tuberculosis", ("drug susceptible tuberculosis", "pulmonary tuberculosis", "tuberculosis treatment")),
    (r"\b(?:tuberculosis|\btb\b|\bntep\b)\b", "tuberculosis", ("tuberculosis", "pulmonary tuberculosis", "TB")),
    (r"\b(?:chronic obstructive pulmonary disease|\bcopd\b)\b", "chronic obstructive pulmonary disease", ("COPD", "chronic obstructive pulmonary disease")),
    (r"\b(?:community[- ]acquired pneumonia|\bcap\b|pneumonia)\b", "community acquired pneumonia", ("community acquired pneumonia", "pneumonia")),
    (r"\b(?:dengue(?: fever)?|dengue with warning signs)\b", "dengue", ("dengue", "dengue fever")),
    (r"\b(?:falciparum malaria|severe malaria|malaria)\b", "malaria", ("severe malaria", "falciparum malaria", "malaria")),
    (r"\b(?:hiv|human immunodeficiency virus|\baids\b)\b", "hiv", ("HIV", "human immunodeficiency virus", "antiretroviral therapy")),
    (r"\b(?:acute watery diarrh(?:ea|oea)|diarrh(?:ea|oea))\b", "acute diarrhea", ("acute diarrhea", "acute diarrhoea", "oral rehydration")),
    (r"\b(?:urinary tract infection|\buti\b|febrile uti)\b", "urinary tract infection", ("urinary tract infection", "febrile urinary tract infection", "UTI")),
    (r"\b(?:status epilepticus|prolonged convulsions|(?:generalized\s+)?convulsions\s+lasting\s+(?:longer\s+than|over|more\s+than)\s+(?:five|5)\s+minutes)\b", "status epilepticus", ("status epilepticus", "prolonged seizure")),
    (r"\b(?:focal seizures?|\bepilepsy\b|\bepilepsies\b)\b", "epilepsy", ("focal seizure", "epilepsy", "epilepsies")),
    (r"\b(?:lupus nephritis)\b", "lupus nephritis", ("lupus nephritis",)),
    (r"\b(?:atrial fibrillation|\bafib\b)\b", "atrial fibrillation", ("atrial fibrillation", "AF")),
    (r"\b(?:acute ischemic stroke|acute ischaemic stroke|ischemic stroke|ischaemic stroke|stroke(?![- ]risk))\b", "ischemic stroke", ("ischemic stroke", "ischaemic stroke", "stroke")),
    (r"\b(?:postpartum h(?:a)?emorrhage|\bpph\b)\b", "postpartum hemorrhage", ("postpartum hemorrhage", "postpartum haemorrhage", "uterine atony")),
    (r"\b(?:pre[- ]?eclampsia|\beclampsia\b)\b", "preeclampsia", ("preeclampsia", "pre-eclampsia", "eclampsia")),
    (r"\b(?:diabetic foot infection|infected diabetic foot|diabetic foot ulcer|infected foot ulcer|diabetes.{0,75}infected foot)\b", "diabetic foot infection", ("diabetic foot infection", "diabetic foot ulcer", "diabetic foot")),
    (r"\b(?:breast cancer|breast carcinoma|breast malignancy)\b", "breast cancer", ("breast cancer", "breast carcinoma")),
    (r"\b(?:graves[’']? disease|hashimoto(?:'s)? thyroiditis|subclinical hypothyroidism|hypothyroidism|hyperthyroidism|thyroid disorders?)\b", "thyroid disorders", ("Graves disease", "Hashimoto thyroiditis", "thyroid disease")),
    (r"\b(?:snake[- ]?bite|envenomation|antivenom)\b", "snakebite", ("snakebite", "snake bite", "envenomation")),
    (r"\b(?:hfr?ef|heart failure with reduced ejection fraction)\b", "heart failure with reduced ejection fraction", ("hfrEF", "heart failure with reduced ejection fraction", "systolic heart failure")),
    (r"\b(?:hfpef|heart failure with preserved ejection fraction)\b", "heart failure with preserved ejection fraction", ("hfpef", "heart failure with preserved ejection fraction")),
    (r"\b(?:heart failure|congestive heart failure|\bchf\b)\b", "heart failure", ("heart failure",)),
    (r"\b(?:t1dm|type\s*1\s+diabetes(?: mellitus)?)\b", "type 1 diabetes", ("type 1 diabetes", "t1dm")),
    (r"\b(?:t2dm|type\s*2\s+diabetes(?: mellitus)?)\b", "type 2 diabetes", ("type 2 diabetes", "t2dm")),
    (r"\b(?:gestational diabetes|\bgdm\b)\b", "gestational diabetes", ("gestational diabetes",)),
    (r"\b(?:diabetic ketoacidosis|\bdka\b)\b", "diabetic ketoacidosis", ("diabetic ketoacidosis", "dka")),
    (r"\b(?:chronic kidney disease|\bckd\b)\b", "chronic kidney disease", ("chronic kidney disease", "ckd")),
    (r"\b(?:acute kidney injury|\baki\b)\b", "acute kidney injury", ("acute kidney injury",)),
    (r"\b(?:microcytic an?emia|microcytic anaemia)\b", "microcytic anemia", ("microcytic anemia", "microcytic anaemia")),
    (r"\bnephrotic syndrome\b", "nephrotic syndrome", ("nephrotic syndrome",)),
    (r"\b(?:bronchial asthma|asthma)\b", "asthma", ("asthma",)),
    (r"\b(?:hypertension|high blood pressure|\bhtn\b)\b", "hypertension", ("hypertension", "high blood pressure")),
    (r"\b(?:sepsis|septic shock)\b", "sepsis", ("sepsis", "septic shock")),
    (r"\b(?:acute coronary syndrome|\bacs\b)\b", "acute coronary syndrome", ("acute coronary syndrome",)),
    (r"\b(?:acute myocardial infarction|myocardial infarction|heart attack)\b", "myocardial infarction", ("myocardial infarction", "heart attack")),
    (r"\b(?:nonalcoholic steatohepatitis|non-alcoholic steatohepatitis|\bnash\b|\bmash\b)\b", "steatohepatitis", ("steatohepatitis", "nash", "mash")),
    (r"\b(?:ace inhibitors?|angiotensin converting enzyme inhibitors?|\bacei\b)\b", "ACE inhibitors", ("ace inhibitor", "angiotensin converting enzyme inhibitor")),
)

# Clinical question words must never become stand-alone PubMed queries.
STOPWORDS = set("""
what which when where who how why is are was were am be being been have has had
will would could should may might can must the a an of for in on at from to with
without and or versus vs both compared compare comparison difference differences
older current latest newest recent up date updated guideline guidelines guidance
based guideline-based first-line second-line evidence-based
clinical diagnostic diagnosis diagnose management manage managing treatment treat therapy
therapeutic pharmacological pharmacology pharmacotherapy first line second line
approach initial standard recommend recommended recommendations describe explain
including include covers cover discuss detail detailed comprehensively comprehensive
teach me basic basics overview patient patients child children adult adults person
stage stages severity assessment investigation investigations criteria drug drugs
agents medications medication medical postgraduate student important major common
according evidence review papers research studies clinical trial trial trials
pathophysiology etiology aetiology mechanism mechanisms history prevention monitor
monitoring follow followup follow-up and major primary complete initial
""".split())

DRUG_TERMS = (
    "tirzepatide", "semaglutide", "dapagliflozin", "empagliflozin", "metformin",
    "sacubitril", "valsartan", "spironolactone", "finerenone", "beta blocker",
    "sglt2", "glp-1", "arni", "ace inhibitor", "insulin", "antibiotic",
)


ASPECT_SPECS = (
    ("management", r"\b(?:management|treatment|therapy|therapeutic|pharmacology|pharmacological|pharmacotherapy|drug|drugs|medication|medications|first[- ]line|second[- ]line|insulin therapy|insulin delivery|insulin pump|automated insulin delivery|vasopressor|antimicrobial)\b",
     ("management", "treatment", "therapy", "therapeutics", "pharmacotherapy", "insulin therapy", "insulin delivery", "insulin pump", "automated insulin delivery")),
    ("diagnosis", r"\b(?:diagnosis|diagnostic|criteria|evaluation|assessment|investigation|investigations|staging|stage|severity)\b",
     ("diagnosis", "diagnostic", "evaluation", "staging", "assessment")),
    ("pathophysiology", r"\b(?:pathophysiology|pathogenesis|physiology|mechanism|mechanisms|remodeling|inflammatory|circulatory)\b",
     ("pathophysiology", "pathogenesis", "mechanism", "physiology")),
    ("monitoring", r"\b(?:monitoring|monitor|follow[- ]?up|surveillance|reassessment|glucose monitoring|continuous glucose monitoring|cgm|hba1c)\b",
     ("monitoring", "follow-up", "follow up", "surveillance", "glucose monitoring", "continuous glucose monitoring", "cgm", "hba1c")),
    ("complications", r"\b(?:complication|complications|adverse outcomes?|organ support|red flags?)\b",
     ("complications", "outcomes", "organ dysfunction")),
    ("anemia", r"\b(?:anemia|anaemia)\b", ("anemia", "anaemia")),
    ("dialysis", r"\b(?:dialysis|renal replacement therapy|hemodialysis|haemodialysis|peritoneal dialysis)\b",
     ("dialysis", "renal replacement therapy", "hemodialysis", "peritoneal dialysis")),
)

# Wrong-disease subtypes that must not be allowed to masquerade as the generic
# condition. We only apply these when the user did not explicitly request them.
TOPIC_EXCLUSIONS = {
    "hypertension": (
        "pulmonary hypertension", "pulmonary arterial hypertension",
        "chronic thromboembolic pulmonary hypertension", "portal hypertension",
        "intracranial hypertension", "ocular hypertension",
    ),
}

# Contexts that are valid clinical topics but should not replace a generic
# condition question unless the user explicitly asks for that context.
TITLE_CONTEXT_EXCLUSIONS = {
    "hypertension": (
        "pregnancy", "pregnant", "gestational hypertension",
        "preeclampsia", "pre-eclampsia", "eclampsia", "postpartum",
    ),
}

LOW_VALUE_PUBLICATION_TYPES = ("case reports", "case report")

# A paper can mention "children" in its abstract while describing research
# exclusively on pregnancy or adults. Title-level population conflicts must
# be excluded *before* calculating relevance/quality scores.
PEDIATRIC_MARKERS = r"\b(?:child(?:ren|hood)?|schoolchildren|kid(?:s)?|pediatric|paediatric|adolescent(?:s)?|teen(?:s|ager(?:s)?)?|juvenile|youth|young people|school[- ]aged|\d{1,2}[- ]year[- ]old)\b"
ADULT_TITLE_MARKERS = r"\b(?:adult(?:s)?|elderly|older people)\b"
MATERNAL_TITLE_MARKERS = r"\b(?:pregnan(?:t|cy)|gestational|maternal|postpartum|prenatal|antenatal|obstetric|perinatal|preeclampsia|pre-eclampsia|eclampsia)\b"
HTN_SECONDARY_TITLE_MARKERS = r"\b(?:primary aldosteronism|secondary hypertension|endocrine hypertension|renovascular hypertension|renal artery stenosis|pheochromocytoma|phaeochromocytoma)\b"
TITLE_INCIDENTAL_MARKERS = r"\b(?:quality of life|patient satisfaction|psychosocial|social determinants)\b"


def _population_intent(question: str) -> str:
    q = str(question or "").lower()
    has_child = bool(re.search(PEDIATRIC_MARKERS, q))
    has_adult = bool(re.search(r"\badults?\b", q))
    if has_child and has_adult:
        return "mixed"
    if has_child:
        return "child"
    if has_adult:
        return "adult"
    return "general"


def _title_context_conflict(question: str, paper: Dict, canonical: str) -> bool:
    """Reject clear *title-level* population/clinical-context mismatches.

    An abstract may mention many populations as background; that must not
    rescue a paper whose primary study is about a different population.
    We do not ban comorbidity papers just for mentioning an unrelated group
    in an abstract. Title-specific exclusion is deliberately conservative.
    """
    title = str(paper.get("title", "")).lower()
    q = str(question or "").lower()
    pop = _population_intent(question)

    maternal_in_q = bool(re.search(MATERNAL_TITLE_MARKERS, q))
    if re.search(MATERNAL_TITLE_MARKERS, title) and not maternal_in_q:
        return True

    if pop == "child":
        if re.search(ADULT_TITLE_MARKERS, title) and not re.search(PEDIATRIC_MARKERS, title):
            return True
    elif pop == "adult":
        if re.search(PEDIATRIC_MARKERS, title) and not re.search(ADULT_TITLE_MARKERS, title):
            return True
    elif canonical == "hypertension":
        # Unspecified systemic hypertension: don't substitute pediatric,
        # neonatal, or niche secondary-etiology papers for adult/general HTN.
        if re.search(PEDIATRIC_MARKERS + r"|\b(?:infant|neonat(?:e|al))\b", title):
            return True
        secondary_explicit = bool(re.search(HTN_SECONDARY_TITLE_MARKERS, q))
        secondary_learning = bool(re.search(r"\b(?:cause|causes|etiology|aetiology|differential|secondary|resistant|endocrine)\b", q))
        if re.search(HTN_SECONDARY_TITLE_MARKERS, title) and not (secondary_explicit or secondary_learning):
            return True
    return False


def _publication_types_text(paper: Dict) -> str:
    return " ".join(str(x).lower() for x in paper.get("publication_types", []) if x)


def _is_comparison_question(question: str) -> bool:
    q = str(question or "").lower()
    return bool(re.search(r"\b(?:compare|comparison|versus|\bvs\b|older and current|old and new|changes over time)\b", q))


def _title_key(title: str, preserve_year: bool = False) -> str:
    """Normalize duplicate guideline/review versions into one title family.

    Executive summaries and full guideline records frequently coexist in PubMed.
    For ordinary questions we also collapse different-year versions of the same
    guideline family so the newest/highest-ranked record does not crowd the result
    set with historical duplicates. Comparison questions preserve the year.
    """
    raw = str(title or "").lower()
    year_match = re.search(r"\b(?:19|20)\d{2}\b", raw)
    text = raw
    if not preserve_year:
        text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = re.sub(r"\bexecutive summary\b\s*:?\s*", " ", text)
    text = re.sub(r"\s*:?\s*(?:a|an) report of the\b.*$", " ", text)

    # Guideline records often prepend long society acronym lists that differ
    # between the full paper and executive summary. Canonicalize from the
    # clinically meaningful 'guideline for ...' phrase when present.
    m = re.search(r"\bguideline for\b.*", text)
    if m:
        text = m.group(0)

    text = re.sub(r"[^a-z0-9]+", " ", text)
    core = " ".join(text.split())[:220]
    if preserve_year and year_match:
        return f"{year_match.group(0)} {core}"[:220]
    return core


def _title_aspects(paper: Dict) -> List[str]:
    return _paper_aspects({"title": str(paper.get("title", "")), "abstract": ""})


def _narrow_title_group(canonical: str, title: str) -> str:
    """Identify narrow CKD subtopics so one subtopic cannot dominate broad PG sets."""
    if canonical != "chronic kidney disease":
        return ""
    t = str(title or "").lower()
    groups = (
        ("anemia", ("anemia", "anaemia", "iron deficiency", "erythropoietin")),
        ("dialysis", ("dialysis", "hemodialysis", "haemodialysis", "peritoneal dialysis", "renal replacement therapy")),
        ("hypertension", ("hypertension", "blood pressure")),
        ("diabetes", ("diabetes", "diabetic")),
        ("fatigue", ("fatigue",)),
        ("microbiome", ("microbiome",)),
    )
    for name, markers in groups:
        if any(marker in t for marker in markers):
            return name
    return ""


def _is_broad_clinical_question(question: str) -> bool:
    q = str(question or "").lower()
    return (
        len(_requested_aspects(question)) >= 2
        or bool(re.search(r"\b(?:comprehensive|comprehensively|in detail|postgraduate|teach)\b", q))
    )


def _field_or(terms: Tuple[str, ...]) -> str:
    return "(" + " OR ".join(f'"{t}"[Title/Abstract]' for t in terms) + ")"


def _population_clause(question: str) -> str:
    population = _population_intent(question)
    if population == "child":
        return _field_or(("child", "children", "pediatric", "paediatric", "adolescent", "youth"))
    if population == "adult":
        return _field_or(("adult", "adults"))
    return ""


def _requested_aspects(question: str) -> List[str]:
    q = str(question or "").lower()
    found = []
    for name, pattern, _terms in ASPECT_SPECS:
        if re.search(pattern, q):
            found.append(name)
    return found


def _aspect_clause(name: str) -> str:
    for aspect_name, _pattern, terms in ASPECT_SPECS:
        if aspect_name == name:
            return _field_or(terms)
    return ""


def _mentions_named_drug(text: str, drug: str) -> bool:
    """Avoid matching drug acronyms inside ordinary words (ARNI in 'warning')."""
    return bool(re.search(
        r"(?<![a-z0-9])" + re.escape(drug.lower()) + r"s?(?![a-z0-9])",
        str(text or "").lower(),
    ))


def clean_query_terms(query: str) -> List[str]:
    """Backward-compatible helper: return medical content tokens, not task words."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", str(query).lower())
    return [w for w in tokens if len(w) > 2 and w not in STOPWORDS]


def _topics(query: str) -> Tuple[str, Tuple[str, ...], List[str]]:
    text = str(query or "").lower()
    matches = []
    for pattern, canonical, aliases in CONDITIONS:
        hit = re.search(pattern, text, flags=re.IGNORECASE)
        if hit:
            matches.append((hit.start(), -len(hit.group()), canonical, aliases))
    # Some questions start with a comorbidity/risk (e.g. "type 2 diabetes")
    # before naming the actual target ("infected foot ulcer"). Prefer the
    # specific named clinical problem over its background condition. This is
    # deliberately limited to unambiguous subtopic identities.
    focal_priorities = {
        "diabetic foot infection": 2,
        "drug resistant tuberculosis": 2,
        "drug susceptible tuberculosis": 2,
        "status epilepticus": 2,
        "postpartum hemorrhage": 2,
        "preeclampsia": 2,
        "atrial fibrillation": 2,
        "lupus nephritis": 2,
    }
    matches.sort(key=lambda item: (-focal_priorities.get(item[2], 0), item[0], item[1]))
    seen = set()
    unique = []
    for _, _, canonical, aliases in matches:
        if canonical in seen:
            continue
        if canonical == "heart failure" and any(t[2].startswith("heart failure with") for t in matches):
            continue
        seen.add(canonical)
        unique.append((canonical, aliases))
    if unique:
        canonical, aliases = unique[0]
        return canonical, tuple(aliases), [m[0] for m in unique[:3]]
    terms = clean_query_terms(query)
    if not terms:
        return "", tuple(), []
    # Unknown-condition fallback remains anchored to several clinical words.
    terms = list(dict.fromkeys(terms))[:5]
    return " ".join(terms), (" ".join(terms[:3]),), []


def _query_plan(question: str) -> List[str]:
    """Build several disease-anchored searches that mirror requested facets.

    Broad PG questions should not collapse into one generic disease search.
    Population terms are pushed into PubMed itself, then checked again after fetch.
    """
    canonical, aliases, all_conditions = _topics(question)
    if not canonical:
        return []

    anchor = _field_or(tuple(aliases[:3]))
    population = _population_clause(question)
    suffix = f" AND {population}" if population else ""
    qlow = str(question or "").lower()

    plans: List[str] = []

    # Named co-diseases / interventions are highest precision. For explicit
    # head-to-head questions, preserve both interventions in the first search.
    secondary = [f'"{other}"[Title/Abstract]' for other in all_conditions[1:]]
    named_drugs = [
        f'"{drug}"[Title/Abstract]'
        for drug in DRUG_TERMS
        if _mentions_named_drug(qlow, drug) and not any(drug in a.lower() for a in aliases)
    ]
    secondary.extend(named_drugs)
    secondary = list(dict.fromkeys(secondary))[:3]
    if secondary:
        if len(secondary) > 1:
            plans.append(anchor + " AND " + " AND ".join(secondary) + suffix)
        for term in secondary:
            plans.append(anchor + " AND " + term + suffix)

    # Search each requested clinical facet separately. This creates a candidate
    # pool with management, diagnostic and teaching depth rather than whichever
    # generic papers happen to rank for the disease name alone.
    requested_aspects = _requested_aspects(question)
    for aspect in requested_aspects:
        clause = _aspect_clause(aspect)
        if clause:
            plans.append(anchor + " AND " + clause + suffix)

    # Focused pediatric T1DM questions benefit from clinically specific insulin
    # and glucose-monitoring searches. These are still deterministic PubMed
    # queries, not LLM-generated terms.
    if canonical == "type 1 diabetes" and population:
        if "insulin" in qlow and "management" in requested_aspects:
            plans.append(
                anchor
                + " AND "
                + _field_or(("insulin therapy", "insulin delivery", "insulin pump", "automated insulin delivery", "continuous subcutaneous insulin infusion"))
                + suffix
            )
        if "monitoring" in requested_aspects:
            plans.append(
                anchor
                + " AND "
                + _field_or(("continuous glucose monitoring", "glucose monitoring", "cgm", "hba1c"))
                + suffix
            )

    if re.search(r"\b(?:trial|research|evidence|studies|randomized|randomised)\b", qlow):
        plans.append(
            anchor
            + ' AND ("randomized controlled trial"[Publication Type] OR "clinical trial"[Publication Type])'
            + suffix
        )

    # Always keep one broader, still population-anchored fallback.
    plans.append(anchor + suffix)
    return list(dict.fromkeys(plans))



def _request(url: str, params: dict):
    last_error = None
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as exc:
            last_error = exc
            code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)) or code in (429, 500, 502, 503, 504)
            if not retryable or attempt == 1:
                break
            time.sleep(0.4 * (attempt + 1))
    raise last_error


def search_pubmed_ids(query: str, max_results: int = MAX_RESULTS) -> List[str]:
    """Collect a facet-balanced candidate pool from PubMed.

    Each query-plan facet receives a small quota before the
    candidate pool is truncated. This prevents broad PG questions from spending
    the whole ID budget on management/diagnosis before later facets such as
    anemia or dialysis are searched.
    """
    if not str(query or "").strip():
        return []

    max_results = max(1, min(80, int(max_results)))
    plans = _query_plan(query)
    if not plans:
        return []

    current_year = datetime.now().year
    recent_start = max(2000, current_year - 8)
    per_plan = max(4, min(9, ceil(max_results / max(1, len(plans)))))

    buckets: List[List[str]] = []

    def fetch_ids(term: str, recent_only: bool, retmax: int) -> List[str]:
        nonlocal network_unavailable
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
            "tool": "medicare_ai",
        }
        if recent_only:
            params.update({
                "datetype": "pdat",
                "mindate": str(recent_start),
                "maxdate": str(current_year),
            })
        if PUBMED_API_KEY:
            params["api_key"] = PUBMED_API_KEY

        label = "recent" if recent_only else "fallback"
        print(f"    PubMed Searching ({label}): {term[:110]}...")
        try:
            data = _request(f"{BASE_URL}/esearch.fcgi", params).json()
            return [str(x) for x in data.get("esearchresult", {}).get("idlist", []) if str(x).isdigit()]
        except Exception as exc:
            network_unavailable = True
            print(f"    PubMed search unavailable ({type(exc).__name__}); stopping remaining searches")
            return []

    # Stop promptly when PubMed is offline; don't time out once per facet.
    start = time.monotonic()
    network_unavailable = False
    for term in plans:
        if network_unavailable or time.monotonic() - start > SEARCH_BUDGET_SECONDS:
            print("    PubMed ID-search budget reached or network unavailable; using collected IDs")
            break
        ids = fetch_ids(term, recent_only=True, retmax=per_plan)
        # Only use older literature for a facet that is itself sparse.
        if len(ids) < min(3, per_plan) and not network_unavailable and time.monotonic() - start < SEARCH_BUDGET_SECONDS:
            older = fetch_ids(term, recent_only=False, retmax=per_plan)
            ids = list(dict.fromkeys(ids + older))
        buckets.append(ids[:per_plan])

    # Round-robin merge means no early facet can crowd out later facets.
    collected: List[str] = []
    seen = set()
    depth = max((len(b) for b in buckets), default=0)
    for i in range(depth):
        for bucket in buckets:
            if i >= len(bucket):
                continue
            pmid = bucket[i]
            if pmid in seen:
                continue
            seen.add(pmid)
            collected.append(pmid)
            if len(collected) >= max_results:
                print(f"    PubMed unique candidate IDs after facet balancing: {len(collected)}")
                return collected

    print(f"    PubMed unique candidate IDs after facet balancing: {len(collected)}")
    return collected


def _flatten_xml_text(obj) -> str:
    if isinstance(obj, list):
        return " ".join(filter(None, (_flatten_xml_text(x) for x in obj))).strip()
    if isinstance(obj, dict):
        sections = []
        if "#text" in obj:
            sections.append(_flatten_xml_text(obj["#text"]))
        for key, val in obj.items():
            if key.startswith("@") or key == "#text":
                continue
            sections.append(_flatten_xml_text(val))
        return " ".join(filter(None, sections)).strip()
    return str(obj or "").strip()


def parse_article(article: Dict) -> Optional[Dict]:
    """Extract PubMed metadata robustly; never invent an unavailable abstract."""
    try:
        medline = article.get("MedlineCitation") or {}
        data = medline.get("Article") or {}
        pmid = _flatten_xml_text(medline.get("PMID"))
        if not pmid.isdigit():
            return None
        title = _flatten_xml_text(data.get("ArticleTitle")) or "Untitled article"
        abstract = data.get("Abstract") or {}
        parts = abstract.get("AbstractText", []) if isinstance(abstract, dict) else []
        if not isinstance(parts, list):
            parts = [parts]
        abstract_text = " ".join(
            (f"{part.get('@Label')}: " if isinstance(part, dict) and part.get('@Label') else "")
            + _flatten_xml_text(part)
            for part in parts
        ).strip()
        authors = data.get("AuthorList") or {}
        author_items = authors.get("Author", []) if isinstance(authors, dict) else []
        if isinstance(author_items, dict):
            author_items = [author_items]
        author_names = []
        for author in author_items[:3]:
            if isinstance(author, dict) and author.get("LastName"):
                author_names.append(str(author.get("LastName")) + " " + str(author.get("Initials", "")))
        author_str = ", ".join(author_names)
        if len(author_items) > 3:
            author_str += ", et al."
        journal = data.get("Journal") or {}
        date = (journal.get("JournalIssue") or {}).get("PubDate") or {}
        year = str(date.get("Year", ""))
        if not year:
            year_match = re.search(r"(?:19|20)\d{2}", str(date.get("MedlineDate", "")))
            year = year_match.group(0) if year_match else "Unknown"

        pub_types_raw = data.get("PublicationTypeList") or {}
        pub_types = pub_types_raw.get("PublicationType", []) if isinstance(pub_types_raw, dict) else []
        if not isinstance(pub_types, list):
            pub_types = [pub_types]
        publication_types = [
            _flatten_xml_text(x) for x in pub_types if _flatten_xml_text(x)
        ]

        return {
            "pmid": pmid, "title": title, "abstract": abstract_text or "Abstract unavailable.",
            "authors": author_str or "Unknown authors",
            "journal": str(journal.get("Title") or "Medical Journal"),
            "year": year, "publication_types": publication_types,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
    except (AttributeError, TypeError, ValueError) as exc:
        print(f"    PubMed article parse skipped ({type(exc).__name__})")
        return None


def fetch_pubmed_papers(ids: List[str]) -> List[Dict]:
    ids = list(dict.fromkeys(str(x) for x in ids if str(x).isdigit()))
    if not ids:
        return []
    params = {
        "db": "pubmed", "id": ",".join(ids), "rettype": "abstract",
        "retmode": "xml", "tool": "medicare_ai",
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY
    try:
        data = xmltodict.parse(_request(f"{BASE_URL}/efetch.fcgi", params).content)
        articles = (data.get("PubmedArticleSet") or {}).get("PubmedArticle", [])
        if isinstance(articles, dict):
            articles = [articles]
        found = {}
        for article in articles:
            paper = parse_article(article)
            if paper:
                found[paper["pmid"]] = paper
        return [found[i] for i in ids if i in found]
    except Exception as exc:
        print(f"    PubMed fetch unavailable ({type(exc).__name__})")
        return []


def _paper_aspects(paper: Dict) -> List[str]:
    text = (str(paper.get("title", "")) + " " + str(paper.get("abstract", ""))).lower()
    matched = []
    for name, _pattern, terms in ASPECT_SPECS:
        if any(term.lower() in text for term in terms):
            matched.append(name)
    if "management" not in matched and re.search(r"\b(?:glucose lowering|blood pressure lowering|disease-modifying|guideline-directed medical therapy)\b", text):
        matched.append("management")
    return matched


def _clinical_research_scope_match(question: str, paper: Dict, canonical: str) -> bool:
    """Conservative title-level vetoes for specific, high-stakes research intents.

    Broad abstract background is not enough: e.g. an active MDR-TB treatment
    question must not return prophylaxis in exposed *contacts*. This function
    is not a comprehensive clinical quality validator.
    """
    q = str(question or "").lower()
    title = str(paper.get("title", "")).lower()
    if not title:
        return False

    def wanted(*terms):
        return any(t in q for t in terms)

    if canonical in {"drug susceptible tuberculosis", "drug resistant tuberculosis"}:
        # A new pulmonary TB question must not be answered by an
        # extrapulmonary/subtype-only or post-treatment complications paper.
        if wanted("pulmonary", "drug-susceptible", "drug susceptible", "drug-resistant", "drug resistant"):
            other_sites = ("tuberculous meningitis", "abdominal tuberculosis",
                           "cutaneous tuberculosis", "intestinal tuberculosis")
            if any(t in title for t in other_sites) and not any(t in q for t in other_sites):
                return False
        if any(t in title for t in ("post-tuberculosis treatment", "post tuberculosis treatment",
                                  "post-tb treatment", "post tb treatment", "after tuberculosis treatment")):
            if not wanted("after treatment", "post-tb", "post tuberculosis", "post-tuberculosis", "sequelae"):
                return False
        # Prevention of *infection* in contacts is distinct from treating an
        # already diagnosed active case. Respect specific prevention questions.
        if any(t in title for t in ("preventive therapy", "preventive treatment", "prevention of tuberculosis",
                                  "preventing tuberculosis", "tuberculosis prevention", "tuberculosis contacts")):
            if not wanted("preventive therapy", "preventive treatment", "contacts", "latent tb", "tpt"):
                return False
        if canonical == "drug susceptible tuberculosis":
            if re.search(r"\b(?:multidrug[- ]resistan\w*|rifampicin[- ]resistan\w*|mdr[-/ ]?tb|rr[-/ ]?tb)\b", title):
                if not wanted("compare", "comparison", "versus", "resistance development"):
                    return False
            # Nanocarriers are investigational drug-delivery research, not
            # an established regimen or a direct monitoring protocol.
            if re.search(r"\b(?:nanomedicine|nanoparticles?|nanocarriers?|advanced drug delivery)\b", title):
                if not wanted("nanomedicine", "nanoparticle", "drug delivery", "novel therapies"):
                    return False
            if "stool specimen" in title and "stool" not in q:
                return False
            # Research about detecting a second malignancy in someone with a
            # TB history does not answer routine DS-TB regimen/monitoring.
            if ("lung cancer" in title and not re.search(
                r"\b(?:lung cancer|malignancy|cancer|nodules?|differential)\b", q
            )):
                return False

    if canonical == "postpartum hemorrhage" and re.search(
        r"\b(?:manage|management|treatment|treat|bleeding|resuscitation|"
        r"emergency|tranexamic|uterotonic)\b", q
    ) and not re.search(r"\b(?:prevention|prevent|prophylaxis)\b", q):
        # Established PPH treatment is a different clinical task from
        # prophylaxis in an unbleeding mother. Keep comprehensive guidelines
        # covering both, but drop trials exclusively about prevention.
        preventive_title = bool(re.search(
            r"\b(?:preventing|prevention of|prophylaxis for|preventive)\b", title
        ))
        treatment_title = bool(re.search(
            r"\b(?:management|treatment|treating|hemorrhagic shock|"
            r"first response|first-response)\b", title
        ))
        if preventive_title and not treatment_title:
            return False

    if canonical == "drug resistant tuberculosis" and not re.search(
        r"\b(?:child|children|pediatric|paediatric|adolescent)\b", q
    ):
        abstract = str(paper.get("abstract", "")).lower()
        # "adult tablets" in a paediatric PK paper does not make its trial
        # population adult. Require an explicit adult *study population* cue.
        adult_study_population = bool(re.search(
            r"\b(?:adult patients?|adults? with|adult participants?|"
            r"adults? aged|adult cohorts?|both adults? and children)\b", abstract
        ))
        if (re.search(r"\b(?:children|pediatric|paediatric)\b", abstract)
                and not adult_study_population
                and re.search(r"\b(?:pharmacokinetics|pediatric|paediatric|children)\b", title)):
            return False

    if canonical == "malaria" and wanted("severe", "falciparum", "cerebral"):
        # Severe-disease care must not be represented by population prevention,
        # vector control, or surveillance statistics in a different country.
        excluded = ("malaria surveillance", "malaria transmission", "reducing malaria transmission",
                    "malaria elimination", "chemoprophylaxis", "mass drug administration",
                    "mosquito control", "malaria prevention")
        if any(term in title for term in excluded) and not any(term in q for term in excluded):
            return False
        # Clinical management must not be padded to k with prevention-only
        # vaccines, prophylaxis, household access, housing modifications, or
        # public-health interventions. Do NOT apply this veto to questions
        # explicitly asking about vaccine/prevention research instead.
        asks_prevention = bool(re.search(
            r"\b(?:vaccin\w*|prevent\w*|prophylaxis|mosquito|transmission|"
            r"household|vector control|healthcare access)\b", q
        ))
        clinical_management = bool(re.search(
            r"\b(?:treat\w*|manage\w*|care|diagnos\w*|therap\w*|"
            r"organ dysfunction|complications?)\b", q
        ))
        if clinical_management and not asks_prevention:
            public_health_only = bool(re.search(
                r"\b(?:house modifications?|house screening|vector control|"
                r"healthcare access|access to (?:malaria )?treatment|"
                r"treatment from a household perspective|mosquito net\w*|"
                r"measuring access to treatment|malaria transmission)\b", title
            ))
            prevention_focus = bool(re.search(
                r"\b(?:vaccin\w*|immuni[sz]ation|prevent\w*|prophylaxis|"
                r"monoclonal antibod\w*)\b", title
            ))
            also_acute_care = bool(re.search(
                r"\b(?:diagnos\w*|treat\w*|clinical management|"
                r"artesunate|artemether|quinine|severe malaria)\b", title
            ))
            if public_health_only or (prevention_focus and not also_acute_care):
                return False
            # A geographical surveillance/epidemiology report about incidence
            # and spatial clusters does not answer an acute severe-malaria
            # treatment request, even if the abstract says "malaria" often.
            epidemiology_only = bool(re.search(
                r"\b(?:epidemiolog\w*|spatial clusters?|spatiotemporal|"
                r"geographic distribution|surveillance data|incidence trends?)\b", title
            )) and not bool(re.search(
                r"\b(?:severe|critical care|treat\w*|clinical management|"
                r"case management|intensive care|artesunate|artemether|"
                r"mortality outcomes?)\b", title
            ))
            if epidemiology_only and not re.search(
                r"\b(?:epidemiolog\w*|surveillance|incidence|"
                r"geographic distribution|spatial)\b", q
            ):
                return False
        if "splenomegaly" in title and "splenomegaly" not in q:
            return False
        # The disease must be a title-level focus, not a sentence buried in
        # the abstract of a paper on an unrelated clinical syndrome.
        if not re.search(r"\b(?:malaria|malarial|plasmodium|falciparum|artesunate|artemether)\b", title):
            return False

    return True


def _is_relevant_paper(question: str, paper: Dict) -> bool:
    canonical, aliases, explicit_topics = _topics(question)
    if not canonical:
        return False

    title = str(paper.get("title", "")).lower()
    abstract = str(paper.get("abstract", "")).lower()
    text = title + " " + abstract
    qlow = str(question or "").lower()

    # Respect requested population twice: once in the PubMed search and again
    # after fetch. Fewer papers are safer than silently crossing populations.
    population = _population_intent(question)
    query_child = population == "child"
    query_adult = population == "adult"
    child_markers = bool(re.search(PEDIATRIC_MARKERS, text))
    adult_markers = bool(re.search(r"\badults?\b", text))
    if query_child and not child_markers:
        return False
    if query_adult and child_markers and not adult_markers:
        return False
    if _title_context_conflict(question, paper, canonical):
        return False
    if not _clinical_research_scope_match(question, paper, canonical):
        return False

    # Generic-condition questions must not silently cross into a different
    # disease entity that happens to contain the same word (e.g. pulmonary
    # hypertension for systemic hypertension).
    for excluded in TOPIC_EXCLUSIONS.get(canonical, ()):
        if excluded in text and excluded not in qlow:
            return False

    # Likewise, a generic hypertension teaching question should not be replaced
    # by pregnancy/postpartum hypertensive disorders unless that context was asked.
    context_terms = TITLE_CONTEXT_EXCLUSIONS.get(canonical, ())
    if context_terms and not any(term in qlow for term in context_terms):
        if any(term in title for term in context_terms):
            return False

    # Disease/subtype identity is mandatory.
    if explicit_topics:
        # Hyphenated clinical labels ("multidrug-resistant") and whitespace
        # variants represent the same indexed disease, not different topics.
        norm_text = re.sub(r"[\s/–—_-]+", " ", text)
        topic_ok = any(re.sub(r"[\s/–—_-]+", " ", a.lower()) in norm_text for a in aliases)
        if not topic_ok and canonical == "drug resistant tuberculosis":
            topic_ok = bool(re.search(
                r"\b(?:multidrug|multi drug|rifampicin|drug) resistant\b.{0,45}\b(?:tuberculosis|tb)\b",
                norm_text,
            ))
        if not topic_ok and canonical == "drug susceptible tuberculosis":
            topic_ok = bool(re.search(
                r"\b(?:drug sensitive|drug susceptible|rifampicin susceptible)\b.{0,45}\b(?:tuberculosis|tb)\b",
                norm_text,
            ))
    else:
        topic_words = [w for w in clean_query_terms(canonical) if len(w) > 3]
        topic_ok = bool(topic_words) and sum(w in text for w in topic_words) >= min(2, len(topic_words))
    if not topic_ok:
        return False

    requested_drugs = [d for d in DRUG_TERMS if _mentions_named_drug(qlow, d)]
    requested_aspects = _requested_aspects(question)
    paper_aspects = set(_paper_aspects(paper)) if requested_aspects else set()

    # A multi-facet question such as "insulin management and glucose monitoring"
    # should allow a high-quality monitoring paper even if that specific paper
    # does not discuss insulin. Drug presence is mandatory only when the paper is
    # being used for the management/therapy facet rather than another explicitly
    # requested facet.
    if requested_drugs and not any(d in text for d in requested_drugs):
        non_management_aspects = set(requested_aspects) - {"management"}
        if not paper_aspects.intersection(non_management_aspects):
            return False

    if requested_aspects and not paper_aspects.intersection(requested_aspects):
        return False

    if (_is_broad_clinical_question(question)
            and "abstract unavailable" in abstract
            and re.search(r"\b(?:editorial|comment|letter)\b", _publication_types_text(paper))
            and not re.search(r"\b(?:guideline|recommendation)\b", title)):
        return False

    # Case reports are useful for unusual diagnostic puzzles, but they are poor
    # default evidence for broad guideline/teaching/management questions.
    if _is_broad_clinical_question(question):
        pub_types = _publication_types_text(paper)
        if any(x in pub_types for x in LOW_VALUE_PUBLICATION_TYPES) or "case report" in title:
            return False

    # Very narrow intervention papers should not occupy broad teaching slots
    # unless the user asked for that intervention.
    if _is_broad_clinical_question(question):
        if "implant-mediated" in title and not re.search(r"\b(?:implant|device)\b", qlow):
            return False

    # Focused pediatric T1DM management/monitoring should not drift toward
    # unrequested complications, diets, psychosocial topics, or rescue drugs.
    if canonical == "type 1 diabetes" and query_child:
        unrequested_title_topics = (
            (("diabetic ketoacidosis", "ketoacidosis"), ("dka", "ketoacidosis")),
            (("low-carb", "low carb", "ketogenic"), ("diet", "nutrition", "low-carb", "low carb", "ketogenic")),
            (("psychosocial",), ("psychosocial", "mental health", "family")),
            (("glucagon", "dasiglucagon"), ("glucagon", "hypoglycemia")),
        )
        for title_terms, question_terms in unrequested_title_topics:
            if any(term in title for term in title_terms) and not any(term in qlow for term in question_terms):
                return False

        # For a focused question with a named therapy such as insulin, require
        # title-level alignment with either that therapy or a requested clinical
        # facet. This blocks papers that only mention insulin incidentally.
        if requested_drugs and not _is_broad_clinical_question(question):
            title_aspects = set(_title_aspects(paper))
            title_has_requested_drug = any(drug in title for drug in requested_drugs)
            if not title_has_requested_drug and not title_aspects.intersection(requested_aspects):
                return False

    return True

def _paper_sort_score(question: str, paper: Dict) -> float:
    canonical, aliases, _topics_found = _topics(question)
    title = str(paper.get("title", "")).lower()
    abstract = str(paper.get("abstract", "")).lower()
    qlow = str(question or "").lower()
    score = 0.0

    for alias in aliases:
        a = alias.lower()
        if a in title:
            score += 6.0
        elif a in abstract:
            score += 2.0

    requested_aspects = _requested_aspects(question)
    paper_aspects = set(_paper_aspects(paper))
    title_only_aspects = set(_paper_aspects({"title": title, "abstract": ""}))
    for aspect in requested_aspects:
        clause_terms = next((terms for name, _p, terms in ASPECT_SPECS if name == aspect), ())
        if any(t.lower() in title for t in clause_terms):
            score += 3.2
        elif aspect in paper_aspects:
            score += 0.7

    for drug in DRUG_TERMS:
        if _mentions_named_drug(qlow, drug):
            if drug in title:
                score += 4.0
            elif drug in abstract:
                score += 0.8

    # Broad disease reviews/guidelines are strong anchors. A paper that only
    # reaches a requested facet in its abstract is still eligible but ranks below
    # a paper whose title is explicitly aligned with the requested clinical task.
    if canonical.lower() in title and re.search(r"\b(?:management|treatment|therapy|guideline|consensus|recommendations?|overview|update|diagnosis|staging|pathophysiology)\b", title):
        score += 3.0
    if requested_aspects and not title_only_aspects.intersection(requested_aspects):
        score -= 1.4

    pub_types = _publication_types_text(paper)
    if "systematic review" in pub_types or "meta-analysis" in pub_types:
        score += 2.0
    elif "practice guideline" in pub_types or "guideline" in pub_types:
        score += 1.8
    elif "randomized controlled trial" in pub_types or "clinical trial" in pub_types:
        score += 1.4
    elif "review" in pub_types:
        score += 1.0
    if any(x in pub_types for x in LOW_VALUE_PUBLICATION_TYPES):
        score -= 6.0

    # De-emphasize very narrow interventions in broad teaching questions unless
    # the user actually asked for them. This is a ranking penalty, not a ban.
    if _is_broad_clinical_question(question):
        narrow_terms = ("implant-mediated", "device-based", "psychosocial", "ketogenic diet", "low-carb", "microbiome")
        if any(term in title for term in narrow_terms) and not any(term in qlow for term in narrow_terms):
            score -= 2.8
        if canonical == "chronic kidney disease" and "dialysis" in _requested_aspects(question):
            if re.search(TITLE_INCIDENTAL_MARKERS, title) and not re.search(TITLE_INCIDENTAL_MARKERS, qlow):
                score -= 7.0

    try:
        year = int(paper.get("year", ""))
        current = datetime.now().year
        age = max(0, current - year)
        score += max(0.0, 1.4 - 0.14 * age)
    except (TypeError, ValueError):
        pass

    return score


def _select_diverse_papers(question: str, papers: List[Dict], k: int) -> List[Dict]:
    """Cover requested facets with title-level evidence first, then fill safely.

    Guarded selection prevents a narrow CKD subtopic (especially anemia) from representing
    every broad facet merely because its abstract contains many generic terms.
    """
    if not papers or k <= 0:
        return []

    ranked = sorted(papers, key=lambda p: _paper_sort_score(question, p), reverse=True)
    requested = _requested_aspects(question)
    canonical, _aliases, _topics_found = _topics(question)
    selected: List[Dict] = []
    used_pmids = set()
    used_titles = set()
    narrow_counts: Dict[str, int] = {}
    preserve_year = _is_comparison_question(question)

    def add(paper: Dict, force: bool = False) -> bool:
        pmid = paper.get("pmid")
        tkey = _title_key(paper.get("title", ""), preserve_year=preserve_year)
        if not pmid or pmid in used_pmids or (tkey and tkey in used_titles):
            return False

        group = _narrow_title_group(canonical, paper.get("title", ""))
        if group and not force:
            # Requested narrow facets may contribute two papers; incidental
            # subtopics get one slot at most.
            limit = 2 if group in requested else 1
            if narrow_counts.get(group, 0) >= limit:
                return False

        selected.append(paper)
        used_pmids.add(pmid)
        if tkey:
            used_titles.add(tkey)
        if group:
            narrow_counts[group] = narrow_counts.get(group, 0) + 1
        return True

    broad_facets = {"management", "diagnosis", "pathophysiology", "monitoring", "complications"}

    # One representative per requested facet. Prefer title-level matches. For
    # broad CKD facets, prefer a general CKD paper instead of an anemia/diabetes/
    # hypertension subtopic paper; dedicated narrow facets are handled separately.
    for aspect in requested:
        title_matches = [p for p in ranked if aspect in _title_aspects(p)]
        all_matches = [p for p in ranked if aspect in _paper_aspects(p)]

        if canonical == "chronic kidney disease" and aspect == "dialysis":
            # A paper comparing quality-of-life in dialysis modalities is not
            # primary evidence for dialysis indications or clinical management.
            usable = lambda p: not re.search(TITLE_INCIDENTAL_MARKERS, str(p.get("title", "")).lower()) or re.search(TITLE_INCIDENTAL_MARKERS, str(question).lower())
            title_matches = [p for p in title_matches if usable(p)]
            all_matches = [p for p in all_matches if usable(p)]

        if canonical == "chronic kidney disease" and aspect in broad_facets:
            general_title = [p for p in title_matches if not _narrow_title_group(canonical, p.get("title", ""))]
            general_all = [p for p in all_matches if not _narrow_title_group(canonical, p.get("title", ""))]
            ordered = general_title + [p for p in title_matches if p not in general_title] + general_all + [p for p in all_matches if p not in general_all]
        else:
            ordered = title_matches + [p for p in all_matches if p not in title_matches]

        for paper in ordered:
            if add(paper, force=True):
                break
        if len(selected) >= k:
            return selected[:k]

    for paper in ranked:
        add(paper, force=False)
        if len(selected) >= k:
            break
    return selected[:k]

def search_pubmed_live(query: str, max_results: int = MAX_RESULTS, assistant_mode: str = "doctor", question_depth: str = "simple") -> List[Dict]:
    """Return topically and clinically checked PubMed papers.

    The mode/depth arguments remain API-compatible; the caller controls the
    paper budget while this module controls quality, population and facet match.
    """
    if int(max_results) <= 0:
        return []
    k = min(10, int(max_results))
    candidate_budget = min(64, max(24, k * 6))
    ids = search_pubmed_ids(query, max_results=candidate_budget)
    if not ids:
        return []

    papers = fetch_pubmed_papers(ids)
    unique = {p["pmid"]: p for p in papers if _is_relevant_paper(query, p)}
    relevant = list(unique.values())
    if not relevant:
        print("    PubMed: found articles, but none passed topic/population/facet verification")
        return []

    selected = _select_diverse_papers(query, relevant, k)
    print(f"    PubMed relevant papers after verification: {len(relevant)}; selected: {len(selected)}")
    return selected



def _abstract_for_prompt(abstract: str, char_limit: int = 4800) -> str:
    """Preserve results/conclusions, which are often AFTER long method sections.

    The previous [:1100] slice removed the negative result of PMID 37061308,
    allowing the synthesizer to make the opposite efficacy claim. Very long
    abstracts retain their introduction *and their final conclusions*.
    """
    content = str(abstract or "Abstract unavailable.").strip()
    if len(content) <= char_limit:
        return content
    head = 1950
    tail = char_limit - head - 64
    return (content[:head] + "\n[Middle of abstract omitted for length]\n" +
            content[-tail:])


def format_pubmed_for_prompt(papers: List[Dict]) -> str:
    if not papers:
        return ""
    formatted = []
    for i, p in enumerate(papers, 1):
        formatted.append(
            f"[Research Paper {i} | PMID: {p['pmid']}]\n"
            f"Title: {p['title']}\n"
            f"Authors: {p.get('authors', 'N/A')}\n"
            f"Journal: {p['journal']} ({p['year']})\n"
            f"Abstract: {_abstract_for_prompt(p.get('abstract', ''))}\n"
            "Evidence role: supplementary research. Interpret the actual results, "
            "effect sizes, certainty and study conclusion; a negative/null trial "
            "must NEVER be summarized as a demonstrated benefit. "
            "Do not override applicable clinical guidelines."
        )
    return "\n---\n".join(formatted)


if __name__ == "__main__":
    sample = "What is the current guideline-based first-line pharmacological treatment for HFrEF?"
    print("QUERY PLAN:", _query_plan(sample))
    papers = search_pubmed_live(sample)
    print("TOPIC-MATCHED PUBMED PAPERS:", len(papers))
    for paper in papers:
        print(paper["pmid"], paper["title"][:100])
