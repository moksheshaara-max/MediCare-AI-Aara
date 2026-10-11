"""
MediCare AI - Retrieval v3.1 (Adaptive PG Depth + Context Guards + Comparisons)

Architecture-preserving retrieval pipeline:
1. Expand medical acronyms and embed the query with all-mpnet-base-v2.
2. Fetch a broad vector candidate pool from MongoDB Atlas Vector Search.
3. Join normalized source metadata from medical_sources in memory.
4. Detect strict retrieval overrides before normal diversification:
   - Explicit named source/organization -> retrieve from that source family only.
   - Explicit latest/current/newest/recent guidance request -> use newest current
     applicable guideline source(s), without padding from older/superseded sources.
5. Otherwise rerank according to Clinical Assistant mode:
   - Doctor Mode: guideline-heavy, current guidance first, India-first when appropriate.
   - PG Student Mode: current-guideline management + stronger textbook depth.
6. Deduplicate overlapping/near-identical chunks and diversify by source.
7. Apply a source-level topical relevance guard in normal diversified retrieval so
   authority/recency bonuses cannot rescue clearly off-topic guideline sources.
8. Classify guideline sources into primary-topic / supplementary / reject tiers.
   Primary-topic guidance may contribute more chunks; narrower/comorbid guidance is
   capped tightly so same-specialty sources cannot crowd out the actual condition.
9. Apply explicit disease/subtype conflict guards (e.g. T1DM vs T2DM, generic
   anemia vs CKD-specific anemia, heart failure vs hypertension) before source choice.
10. For focused topics, use more distinct chunks from the few strongly relevant
   primary sources instead of re-admitting off-topic sources merely to fill quota.
11. If strict topic filtering leaves fewer than 30 normal-retrieval chunks, refill only
    from already-proven PRIMARY guideline sources using additional non-duplicate chunks.
    Never reopen rejected sources merely to hit quota.
12. Return ~30-40 useful chunks for normal retrieval when enough relevant evidence exists.
    Source-constrained retrieval may return fewer chunks when the requested source
    contains less relevant material.

Important compatibility rule:
- chunk["score"] remains the raw MongoDB vectorSearchScore because generator.py
  uses it for confidence/mode decisions.
- chunk["retrieval_score"] is the metadata-aware score used only for ranking.

No ingestion/chunking changes are required.
"""

import os
import re
import sys
import time
from collections import Counter

# Ensure backend root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, ConnectionFailure, NetworkTimeout, ServerSelectionTimeoutError


# -----------------------------------------------------------------------------
# Environment / collections
# -----------------------------------------------------------------------------
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=env_path)
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "medicare_rag"
COLLECTION_NAME = "medical_chunks"
SOURCES_COLLECTION_NAME = "medical_sources"
VECTOR_INDEX_NAME = "vector_index"

mongo_client = MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]
sources_collection = mongo_client[DB_NAME][SOURCES_COLLECTION_NAME]


# -----------------------------------------------------------------------------
# Retrieval v2 configuration
# -----------------------------------------------------------------------------
VECTOR_CANDIDATE_LIMIT = 360
VECTOR_NUM_CANDIDATES = 5000

# Explicit source/latest requests use a wider recall pool before in-memory
# metadata filtering. This preserves the normalized medical_sources architecture
# and does not require a second vector index or duplicated metadata in chunks.
CONSTRAINED_VECTOR_CANDIDATE_LIMIT = 1200
CONSTRAINED_VECTOR_NUM_CANDIDATES = 10000
CONSTRAINED_PAGE_RANGE_CAP = 3
MAX_CONSTRAINED_SOURCES = 3
SOURCE_RELEVANCE_MARGIN = 0.035

# Diversified retrieval guardrail: a guideline with weak source-level semantic
# relevance must have registry metadata support for the query topic. This keeps
# authority/recency/India bonuses from promoting incidentally related guidelines.
DIVERSIFIED_STRONG_SOURCE_MARGIN = 0.020
DIVERSIFIED_METADATA_SOURCE_MARGIN = 0.120
DIVERSIFIED_BROAD_SOURCE_MARGIN = 0.060

DEFAULT_FINAL_K = 36
MIN_FINAL_K = 30
MAX_FINAL_K = 40
PG_MAX_FINAL_K = 50  # PG-only ceiling; Doctor stays capped at 40.

# Per-source diversity caps. These prevent one broad/large reference from
# dominating final context even when it has many nearby vector matches.
DOCTOR_GUIDELINE_PER_SOURCE_CAP = 7
DOCTOR_TEXTBOOK_PER_SOURCE_CAP = 2
DOCTOR_OTHER_PER_SOURCE_CAP = 2

PG_GUIDELINE_PER_SOURCE_CAP = 6
PG_TEXTBOOK_PER_SOURCE_CAP = 3
PG_OTHER_PER_SOURCE_CAP = 2

# PG domain-aware candidate augmentation. Only normal diversified PG retrieval uses
# these extra searches; Doctor Mode and strict source/latest policies are untouched.
PG_DOMAIN_VECTOR_CANDIDATE_LIMIT = 140
PG_DOMAIN_VECTOR_NUM_CANDIDATES = 2500

# Overall supplementary-guideline ceilings prevent subtype/comorbidity guidance from
# crowding out the educational textbook domains requested by a PG student.
PG_SUPPLEMENTARY_TOTAL_CAP = {
    "simple": 7,
    "detailed": 8,
    "comprehensive": 9,
}

# Broad PG questions may go a little deeper into a strongly matched PRIMARY
# guideline source while retaining the existing near-duplicate/page safeguards.
PG_PRIMARY_SOURCE_CAP_BY_DEPTH = {
    "simple": 9,
    "detailed": 11,
    "comprehensive": 13,
}

# Detailed/comprehensive PG queries may need more depth from sources that have
# ALREADY proven relevant. This refill never reopens rejected guideline sources;
# it only deepens selected primary guidelines and selected role-matched textbooks.
PG_HIGH_DEPTH_PRIMARY_REFILL_CAP = {
    "detailed": 18,
    "comprehensive": 22,
}
PG_HIGH_DEPTH_TEXTBOOK_PER_SOURCE_CAP = {
    "detailed": 5,
    "comprehensive": 6,
}
PG_HIGH_DEPTH_TEXTBOOK_CEILING = {
    "detailed": 24,
    "comprehensive": 28,
}
PG_HIGH_DEPTH_PAGE_RANGE_CAP = 3

# Topic-tier caps for normal diversified guideline retrieval. Exact primary-topic
# guidance may dominate more strongly than broad comorbidity/subtype guidance.
DOCTOR_PRIMARY_GUIDELINE_CAP = 12
DOCTOR_SUPPLEMENTARY_GUIDELINE_CAP = 3
PG_PRIMARY_GUIDELINE_CAP = 9
PG_SUPPLEMENTARY_GUIDELINE_CAP = 3

# When a focused query has only one/few truly relevant guideline sources, normal
# diversification can finish below the 30-chunk target after rejecting off-topic
# neighbors. In that case only, allow deeper retrieval from PRIMARY guideline
# sources. This is safer than re-admitting mismatched supplementary sources.
DOCTOR_PRIMARY_REFILL_PER_SOURCE_CAP = 24
PG_PRIMARY_REFILL_PER_SOURCE_CAP = 18
PRIMARY_REFILL_PAGE_RANGE_CAP = 3

# Allow more than one distinct chunk from the same page range. The ingestion
# pipeline can legitimately create multiple non-duplicate 360-token chunks from
# one dense page (for example a one-page STW). Near-duplicate filtering remains
# the primary overlap guard.
DIVERSIFIED_PAGE_RANGE_CAP = 2

# Final-context textbook ceilings/targets are calculated dynamically from K.
# Doctor mode uses a ceiling; PG mode reserves a target when sufficiently
# relevant textbook candidates exist.

SOURCE_METADATA_CACHE_SECONDS = 600
_source_metadata_cache = {}
_source_metadata_cache_loaded_at = 0.0

GUIDELINE_SOURCE_TYPES = {
    "clinical_guideline",
    "public_health_guideline",
    "clinical_programme_guideline",
    "clinical_operational_guideline",
    "clinical_programme_manual",
    "diagnostic_guideline",
    "clinical_training_guideline",
    "clinical_guideline_web_archive",
    "laboratory_guideline",
    "health_system_standard",
    "clinical_ethics_guideline",
}

MANAGEMENT_TERMS = {
    "manage", "management", "treat", "treatment", "therapy", "therapeutic",
    "first line", "first-line", "second line", "second-line", "drug", "dose",
    "dosing", "protocol", "guideline", "guidelines", "recommendation",
    "recommendations", "diagnose", "diagnosis", "diagnostic", "criteria",
    "screen", "screening", "threshold", "target", "monitor", "monitoring",
    "prophylaxis", "prevention", "vaccine", "vaccination", "algorithm",
    "workflow", "standard treatment", "antimicrobial", "antibiotic",
    "contraindication", "contraindications", "escalation", "follow-up",
    "follow up", "red flag", "red flags", "admission", "discharge",
}

EDUCATIONAL_TERMS = {
    "pathophysiology", "pathogenesis", "mechanism", "mechanisms", "physiology",
    "explain", "explanation", "why", "rationale", "concept", "concepts",
    "differential", "differentials", "differential diagnosis", "classification",
    "etiology", "aetiology", "causes", "clinical features", "manifestations",
    "exam", "viva", "theory", "short note", "long answer", "approach to",
}

INDIA_TERMS = {
    "india", "indian", "icmr", "mohfw", "nhm", "ntep", "naco", "ncdc",
    "ncvbdc", "dghs", "national programme", "national program",
    "standard treatment workflow", "stw",
}

INDIAN_NATIONAL_AUTHORITY_TERMS = {
    "indian council of medical research",
    "ministry of health and family welfare",
    "national health mission",
    "national tuberculosis elimination programme",
    "national aids control programme",
    "national centre for disease control",
    "national center for disease control",
    "national center for vector borne diseases control",
    "national centre for vector borne diseases control",
    "directorate general of health services",
    "central tb division",
    "department of health research",
}

INDIAN_NATIONAL_AUTHORITY_ABBREVIATIONS = {
    "icmr", "mohfw", "nhm", "ntep", "naco", "ncdc", "dghs", "dhr",
}

PREFERRED_SOURCE_NOTE_TERMS = {
    "should take precedence",
    "newer topic-specific",
    "latest master",
}

# Explicit source/organization aliases. Matching is done against normalized
# medical_sources metadata, not filenames hard-coded into retrieval results.
SOURCE_ALIAS_METADATA_TERMS = {
    "kdigo": ("kdigo", "kidney disease: improving global outcomes", "kidney disease improving global outcomes"),
    "icmr": ("icmr", "indian council of medical research"),
    "ntep": ("ntep", "national tuberculosis elimination programme", "central tb division"),
    "mohfw": ("mohfw", "ministry of health and family welfare"),
    "nhm": ("nhm", "national health mission"),
    "naco": ("naco", "national aids control programme"),
    "ncdc": ("ncdc", "national centre for disease control", "national center for disease control"),
    "dghs": ("dghs", "directorate general of health services"),
    "who": ("world health organization", "who_"),
    "nice": ("national institute for health and care excellence", "nice_"),
    "esc": ("european society of cardiology", "esc_"),
    "aha_acc": ("american heart association", "american college of cardiology", "aha_acc"),
    "ada": ("american diabetes association", "ada_"),
    "gold": ("global initiative for chronic obstructive lung disease", "gold_"),
    "gina": ("global initiative for asthma", "gina_"),
    "iap": ("indian academy of pediatrics", "iap_"),
    "idsa": ("infectious diseases society of america", "idsa_"),
    "harrison": ("harrison",),
    "davidson": ("davidson",),
    "robbins": ("robbins",),
    "guyton": ("guyton",),
    "kd_tripathi": ("tripathi", "kdt-essentials", "kd_tripathi"),
    "ghai": ("ghai",),
    "bailey_love": ("bailey", "bailey & love", "bailey and love"),
    "hutchison": ("hutchison",),
    "alagappan": ("alagappan", "manual of practical medicine"),
}

# Most aliases are unambiguous as standalone medical acronyms/names.
SOURCE_ALIAS_QUERY_PATTERNS = {
    "kdigo": (r"\bkdigo\b",),
    "icmr": (r"\bicmr\b", r"\bindian council of medical research\b"),
    "ntep": (r"\bntep\b", r"\bnational tuberculosis elimination programme\b"),
    "mohfw": (r"\bmohfw\b", r"\bministry of health and family welfare\b"),
    "nhm": (r"\bnhm\b", r"\bnational health mission\b"),
    "naco": (r"\bnaco\b", r"\bnational aids control programme\b"),
    "ncdc": (r"\bncdc\b", r"\bnational (?:centre|center) for disease control\b"),
    "dghs": (r"\bdghs\b", r"\bdirectorate general of health services\b"),
    "esc": (r"\besc\b", r"\beuropean society of cardiology\b"),
    "aha_acc": (r"\baha\s*/?\s*acc\b", r"\bamerican heart association\b", r"\bamerican college of cardiology\b"),
    "ada": (r"\bamerican diabetes association\b", r"\bada\s+(?:guideline|guidelines|standards|recommendation|recommendations)\b"),
    "gina": (r"\bgina\b", r"\bglobal initiative for asthma\b"),
    "iap": (r"\bindian academy of pediatrics\b", r"\biap\s+(?:guideline|guidelines|recommendation|recommendations)\b"),
    "idsa": (r"\bidsa\b", r"\binfectious diseases society of america\b"),
    "harrison": (r"\bharrison(?:'s|s)?\b",),
    "davidson": (r"\bdavidson(?:'s|s)?\b",),
    "robbins": (r"\brobbins\b",),
    "guyton": (r"\bguyton\b",),
    "kd_tripathi": (r"\b(?:kd\s+tripathi|k\.?\s*d\.?\s+tripathi|tripathi)\b",),
    "ghai": (r"\bghai\b",),
    "bailey_love": (r"\bbailey\s*(?:&|and)?\s*love\b",),
    "hutchison": (r"\bhutchison(?:'s|s)?\b",),
    "alagappan": (r"\balagappan\b", r"\bmanual of practical medicine\b"),
}

FRESHNESS_PATTERNS = (
    r"\blatest\b",
    r"\bnewest\b",
    r"\bmost recent\b",
    r"\bcurrent\b",
    r"\brecent\b",
    r"\bup[- ]to[- ]date\b",
    r"\bupdated\b",
)

FRESH_GUIDANCE_CONTEXT_TERMS = {
    "guideline", "guidelines", "guidance", "recommendation", "recommendations",
    "management", "manage", "treatment", "treat", "therapy", "first line",
    "first-line", "second line", "second-line", "criteria", "diagnostic",
    "diagnosis", "threshold", "target", "screening", "protocol", "algorithm",
    "workflow", "dose", "dosing", "monitoring", "prevention", "prophylaxis",
}

COMPARISON_TERMS = {
    "compare", "comparison", "versus", " vs ", "older", "previous", "prior",
    "historical", "superseded", "changed", "change from", "difference",
    "differences", "evolution",
}

QUERY_METADATA_STOPWORDS = {
    "what", "which", "when", "where", "who", "how", "why", "the", "and", "or",
    "for", "from", "with", "without", "into", "about", "according", "current",
    "latest", "first", "second", "line", "management", "manage", "treatment", "treat",
    "therapy", "diagnosis", "diagnostic", "criteria", "guideline", "guidelines",
    "recommendation", "recommendations", "protocol", "algorithm", "workflow", "monitoring",
    "monitor", "screening", "screen", "threshold", "target", "dose", "dosing", "drug",
    "explain", "explanation", "pathophysiology", "pathogenesis", "mechanism", "mechanisms",
    "rationale", "concept", "concepts", "adult", "adults", "patient", "patients",
    "clinical", "clinical", "disease", "condition", "approach", "overview",
}


# Clinically meaningful condition identity is checked against curated source
# title/topic, not whichever other guideline happened to rank first.  The
# vocabulary is intentionally modest: an unrecognised topic falls back to the
# existing conservative matching rather than guessing a diagnosis.
CLINICAL_TOPIC_PATTERNS = {
    "asthma": r"\b(?:asthma|asthmatic|bronchial\s+asthma)\b",
    "tuberculosis": r"\b(?:tuberculosis|\btb\b|mycobacterium\s+tuberculosis)\b",
    "hiv": r"\b(?:hiv(?:[- ]?[12])?|human\s+immunodeficiency|\baids\b|\bplhiv\b)\b",
    "postpartum_hemorrhage": r"\b(?:post[- ]?partum\s+h(?:ae|e)morrhag\w*|pph)\b",
    "preeclampsia": r"\b(?:pre[- ]?eclampsia|eclampsia)\b",
    "malaria": r"\b(?:malaria|malarial|plasmodium|falciparum)\b",
    "dengue": r"\b(?:dengue)\b",
    "nephrotic_syndrome": r"\b(?:nephrotic\s+syndrome)\b",
    "ckd": r"\b(?:chronic\s+kidney\s+disease|\bckd\b|chronic\s+renal\s+disease)\b",
    "aki": r"\b(?:acute\s+kidney\s+injury|\baki\b|acute\s+renal\s+(?:injury|failure))\b",
    "heart_failure": r"\b(?:heart\s+failure|\bhfref\b|\bhfpef\b|\bhfmref\b|\bchf\b)\b",
    "hypertension": r"\b(?:hypertension|\bhtn\b|high\s+blood\s+pressure)\b",
    "copd": r"\b(?:\bcopd\b|chronic\s+obstructive\s+pulmonary\s+disease)\b",
    "pneumonia": r"\b(?:pneumonia|pneumonitis)\b",
    "type1_diabetes": r"\b(?:type\s*[ -]?1\s+diabetes|\bt1dm\b)\b",
    "type2_diabetes": r"\b(?:type\s*[ -]?2\s+diabetes|\bt2dm\b)\b",
    "diabetes": r"\b(?:diabetes|diabetic)\b",
    "breast_cancer": r"\b(?:breast\s+(?:cancer|carcinoma|neoplasm))\b",
    "sepsis": r"\b(?:sepsis|septic\s+shock)\b",
    "atrial_fibrillation": r"\b(?:atrial\s+fibrillation|\bafib\b)\b",
    "stroke": r"\b(?:stroke|cerebrovascular\s+accident|ischemic\s+cerebral\s+infarct)\b",
    "urinary_tract_infection": r"\b(?:urinary\s+tract\s+infection|\buti\b|pyelonephritis)\b",
    "epilepsy": r"\b(?:epilepsy|epilepsies|status\s+epilepticus|seizure)\b",
}


def _clinical_topic_labels(text: str) -> set:
    value = str(text or "").lower()
    labels = {
        label for label, pattern in CLINICAL_TOPIC_PATTERNS.items()
        if re.search(pattern, value)
    }
    # Diabetes in general is a parent condition, not another disease to
    # compete with its own specifically requested subtypes.
    if labels & {"type1_diabetes", "type2_diabetes"}:
        labels.add("diabetes")
    return labels


def _source_clinical_topic_labels(meta: dict) -> set:
    # Title/topic/source id, not specialty or notes: those often list adjacent
    # conditions (e.g. HIV/TB collaborations or pediatrics/respiratory medicine).
    return _clinical_topic_labels(" ".join([
        str(meta.get("official_title", "")),
        str(meta.get("topic", "")),
        str(meta.get("source_id", "")),
    ]))


def _topic_facet_tier(question: str, meta: dict):
    """Return primary, supplementary, reject, or None (unknown condition)."""
    query_topics = _clinical_topic_labels(question)
    if not query_topics:
        return None
    source_topics = _source_clinical_topic_labels(meta)
    shared = query_topics & source_topics

    # Topic identity must come before organization/country and similarity.
    if not shared:
        # Non-disease operational guidance can be relevant, but must not become
        # the primary disease-management source when disease-specific material
        # is available. Do not force a broad-source fallback here.
        return "reject"

    title_topic = " ".join([
        str(meta.get("official_title", "")),
        str(meta.get("topic", "")),
    ]).lower()

    # Disease-adjacent programme, screening, and implementation documents are
    # supplementary. They must not outrank full treatment guidelines.
    if ("hiv" in query_topics and "tuberculosis" in source_topics
            and "tuberculosis" not in query_topics):
        return "supplementary"
    if ("tuberculosis" in query_topics and "hiv" in source_topics
            and "hiv" not in query_topics):
        return "supplementary"
    # Differentiated-care and family-support manuals are useful operational
    # references but not primary antimicrobial treatment recommendations.
    if "tuberculosis" in query_topics and re.search(
        r"\b(?:differentiated\s+(?:tb|tuberculosis)\s+care|"
        r"family\s+caregiver|caregiver\s+engagement|"
        r"community\s+engagement|tb\s+elimination\s+strategy)\b",
        title_topic,
    ):
        return "supplementary"
    if ("hiv" in query_topics and re.search(r"\badvanced\s+hiv\b", title_topic)
            and not re.search(r"\badvanced\s+hiv\b", str(question).lower())):
        return "supplementary"
    if "malaria" in query_topics and re.search(
            r"\b(?:national\s+strategic\s+plan|malaria\s+elimination|surveillance\s+programme)\b",
            title_topic):
        return "supplementary"
    return "primary"


def _named_organization_is_context_only(question: str, alias: str) -> bool:
    """Distinguish jurisdiction/programme context from a strict citation order."""
    if alias != "ntep":
        return False
    q = str(question).lower()
    contextual = bool(re.search(
        r"\b(?:under|within|in|through)\s+(?:the\s+)?(?:ntep|national\s+tuberculosis\s+elimination\s+programme)\b",
        q,
    ))
    strict = bool(re.search(
        r"\b(?:according\s+to|cite|use\s+only|exclusively\s+(?:from|using)|"
        r"specified\s+by|issued\s+by)\s+(?:the\s+)?(?:ntep|national\s+tuberculosis\s+elimination\s+programme)\b"
        r"|\bntep\s+(?:guideline|guidelines|manual|document|source|sources|recommendations?)\b",
        q,
    ))
    return contextual and not strict


VALID_ASSISTANT_MODES = {"doctor", "pg_student"}


# -----------------------------------------------------------------------------
# Medical acronym expansion
# -----------------------------------------------------------------------------
MEDICAL_ACRONYMS = {
    r"\bt2dm\b": "Type 2 Diabetes Mellitus",
    r"\bt1dm\b": "Type 1 Diabetes Mellitus",
    r"\bdm\b": "Diabetes Mellitus",
    r"\bckd\b": "Chronic Kidney Disease",
    r"\bakd\b": "Acute Kidney Injury",
    r"\baki\b": "Acute Kidney Injury",
    r"\bhtn\b": "Hypertension High Blood Pressure",
    r"\bmi\b": "Myocardial Infarction Heart Attack",
    r"\bcad\b": "Coronary Artery Disease",
    r"\bhfref\b": "Heart Failure with Reduced Ejection Fraction HFrEF",
    r"\bhfpef\b": "Heart Failure with Preserved Ejection Fraction HFpEF",
    r"\bhfmref\b": "Heart Failure with Mildly Reduced Ejection Fraction HFmrEF",
    r"\bchf\b": "Congestive Heart Failure",
    r"\bhf\b": "Heart Failure",
    r"\bdka\b": "Diabetic Ketoacidosis",
    r"\bcopd\b": "Chronic Obstructive Pulmonary Disease",
    r"\bsob\b": "Shortness of Breath Dyspnea",
    r"\bcva\b": "Cerebrovascular Accident Stroke",
    r"\btia\b": "Transient Ischemic Attack",
    r"\bdvt\b": "Deep Vein Thrombosis",
    r"\bpe\b": "Pulmonary Embolism",
    r"\bgurd\b": "Gastroesophageal Reflux Disease",
    r"\bgerd\b": "Gastroesophageal Reflux Disease",
    r"\bibd\b": "Inflammatory Bowel Disease",
    r"\bibs\b": "Irritable Bowel Syndrome",
    r"\bpcos\b": "Polycystic Ovary Syndrome",
    r"\buti\b": "Urinary Tract Infection",
    r"\burti\b": "Upper Respiratory Tract Infection",
    r"\blrti\b": "Lower Respiratory Tract Infection",
    r"\bafib\b": "Atrial Fibrillation",
    r"\bsle\b": "Systemic Lupus Erythematosus",
    r"\bra\b": "Rheumatoid Arthritis",
    r"\boa\b": "Osteoarthritis",
    r"\bptsd\b": "Post Traumatic Stress Disorder",
    r"\bcbc\b": "Complete Blood Count",
    r"\blft\b": "Liver Function Test",
    r"\bkft\b": "Kidney Function Test",
    r"\brft\b": "Renal Function Test",
    r"\babg\b": "Arterial Blood Gas",
    r"\bhba1c\b": "Glycated Hemoglobin HbA1c",
    r"\bldl\b": "Low Density Lipoprotein Cholesterol",
    r"\bhdl\b": "High Density Lipoprotein Cholesterol",
    r"\btg\b": "Triglycerides",
    r"\bcrp\b": "C-Reactive Protein",
    r"\besr\b": "Erythrocyte Sedimentation Rate",
    r"\begfr\b": "Estimated Glomerular Filtration Rate",
}


def expand_medical_acronyms(query: str) -> str:
    """Expand common medical abbreviations before query embedding."""
    expanded = query
    for pattern, replacement in MEDICAL_ACRONYMS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


def embed_query(question: str):
    """Convert the query into a 768-D all-mpnet-base-v2 vector."""
    try:
        from embeddings.local_embedder import embed_single_text

        enriched_query = expand_medical_acronyms(question)
        return embed_single_text(enriched_query)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None


# -----------------------------------------------------------------------------
# Assistant mode / query intent
# -----------------------------------------------------------------------------
def normalize_assistant_mode(assistant_mode: str) -> str:
    """Normalize product/UI mode labels without breaking older callers."""
    value = str(assistant_mode or "doctor").strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "doctor": "doctor",
        "doctor_mode": "doctor",
        "clinician": "doctor",
        "clinical": "doctor",
        "pg": "pg_student",
        "pg_mode": "pg_student",
        "pg_student": "pg_student",
        "pg_student_mode": "pg_student",
        "student": "pg_student",
    }
    return aliases.get(value, "doctor")


def _query_intent(question: str) -> str:
    q = question.lower()
    management = any(term in q for term in MANAGEMENT_TERMS)
    educational = any(term in q for term in EDUCATIONAL_TERMS)

    if management and educational:
        return "mixed"
    if management:
        return "management"
    if educational:
        return "educational"
    return "general"


# PG depth classification controls context BUDGET, not diagnosis or source priority.
# Heuristic version 1: deterministic and deliberately conservative.
PG_DEPTH_BUDGETS = {"simple": 36, "detailed": 45, "comprehensive": 50}

PG_DEPTH_DOMAIN_PATTERNS = (
    r"\b(?:pathophysiology|pathogenesis|physiology|mechanisms?|etiology|aetiology)\b",
    r"\b(?:diagnos(?:is|e|tic)|criteria|investigations?|work\s*up|evaluation|screening)\b",
    r"\b(?:management|treatment|therapy|therapeutic|protocol|guidelines?|algorithms?)\b",
    r"\b(?:pharmacology|drug|drugs|medications?|dosages?|doses?|adverse effects?|contraindications?)\b",
    r"\b(?:complications?|prognosis|long.term outcomes?|sequelae)\b",
    r"\b(?:clinical features?|presentation|signs?|symptoms?|physical examination)\b",
    r"\b(?:differential(?: diagnosis|s)?|classification|staging|grades?)\b",
    r"\b(?:recent evidence|clinical trials?|research|emerging therapies?)\b",
)


def classify_pg_query_depth(question: str) -> str:
    """Classify PG learning scope from what was actually requested.

    Uses explicit breadth/cue phrases and distinct knowledge domains, not query
    length alone. It never modifies the question or selects medical sources.
    """
    q = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    if not q:
        return "simple"

    domains = sum(bool(re.search(pattern, q)) for pattern in PG_DEPTH_DOMAIN_PATTERNS)
    comprehensive_cue = bool(re.search(
        r"\b(?:comprehensive(?:ly)?|everything about|complete overview|full overview|"
        r"complete review|full review|from basics to advanced|cover all aspects|"
        r"teach me .* comprehensively)\b", q
    ))
    detailed_cue = bool(re.search(
        r"\b(?:in detail|explain in detail|detailed explanation|discuss in detail|"
        r"long answer|elaborate|step.by.step|in.depth)\b", q
    ))

    # Explicit basic/short teaching questions should not automatically consume
    # 45 chunks just because they mention two connected learning domains.
    basic_cue = bool(re.search(r"\b(?:basic|basics|brief|simple|short\s+answer|short\s+note)\b", q))
    word_count = len(q.split())

    # Specific guideline comparisons need enough context for both dated sides.
    # This does not turn a focused "current guideline" lookup into a broad one.
    guidance_comparison = bool(re.search(
        r"\b(?:compare|comparison|differences?|changes?|older|previous)\b", q
    )) and bool(re.search(
        r"\b(?:guidelines?|approaches?|diagnos(?:is|tic)|treatment|management|current)\b", q
    ))

    # Detect genuinely multi-part practical care questions, even when their
    # wording doesn't explicitly say "in detail" or "pharmacology".
    # Counting DISTINCT elements prevents a repeated keyword from increasing depth.
    care_elements = (
        r"\b(?:diagnos(?:is|e|tic)|confirm(?:ing|ation)?|assess(?:ment)?|investigations?|evaluation)\b",
        r"\b(?:management|treatment|therapy|insulin|medications?)\b",
        r"\b(?:monitor(?:ing)?|glucose\s+monitoring|follow\s*up\s+testing)\b",
        r"\b(?:hypoglyc(?:a?emia)|complications?|adverse\s+effects?|red\s+flags?)\b",
        r"\b(?:prevent(?:ion|ive)?|prophylaxis|vaccination)\b",
        r"\b(?:follow[ -]?up|long[ -]?term\s+care|reassess(?:ment)?)\b",
        r"\b(?:inhaler\s+technique|action\s+plan|self[- ]management|treatment\s+adherence)\b",
        r"\b(?:acute\s+exacerbation|emergency\s+care|exacerbation\s+care|inpatient\s+referral)\b",
    )
    care_breadth = sum(bool(re.search(pattern, q)) for pattern in care_elements)

    # Reserve 50 chunks for genuinely broad multi-domain teaching requests.
    if domains >= 5 or (comprehensive_cue and (domains >= 2 or care_breadth >= 3)):
        return "comprehensive"
    if guidance_comparison or care_breadth >= 4:
        return "detailed"
    if detailed_cue or comprehensive_cue or domains >= 3:
        return "detailed"
    # Multi-component clinical questions are not necessarily labelled "in
    # detail". Avoid treating a long assessment+therapy+monitoring request
    # as simple merely because it omits those exact cue words. Preserve
    # explicit short/basic queries and short focused pharmacology questions.
    if not basic_cue and word_count >= 17 and (care_breadth >= 3 or domains >= 2):
        return "detailed"
    # A deliberate list of three or more independent clinical tasks is a
    # detailed teaching request even if no exact classifier keywords appear
    # (e.g. "ECG, troponin interpretation, antithrombotics, invasive care").
    if not basic_cue and word_count >= 16 and q.count(",") >= 2 and re.search(
        r"\b(?:explain|discuss|describe|teach|cover|include|including)\b", q
    ):
        return "detailed"
    if basic_cue or (domains <= 2 and word_count <= 27):
        return "simple"
    if domains >= 2:
        return "detailed"
    return "simple"


def pg_retrieval_budget(question: str) -> int:
    return PG_DEPTH_BUDGETS[classify_pg_query_depth(question)]


def _is_broad_pg_learning_request(question: str) -> bool:
    """Return True when PG intent is broad education rather than a freshness-only lookup."""
    depth = classify_pg_query_depth(question)
    roles = _pg_requested_textbook_roles(question)
    return depth in {"detailed", "comprehensive"} and len(roles) >= 2


PG_TEXTBOOK_ROLE_PATTERNS = {
    "physiology": (
        r"\bphysiology\b", r"\bphysiologic", r"\bguyton\b",
    ),
    "pathology": (
        r"\bpathology\b", r"\bpathologic", r"\brobbins\b", r"\bmorpholog",
    ),
    "pharmacology": (
        r"\bpharmacology\b", r"\bpharmacodynamic", r"\bpharmacokinetic",
        r"\btherapeutics?\b", r"\btripathi\b",
    ),
    "clinical_medicine": (
        r"\binternal medicine\b", r"\bclinical medicine\b",
        r"\bpractical medicine\b", r"\bharrison", r"\bdavidson",
        r"\bapi textbook of medicine\b", r"\balagappan\b",
    ),
}


def _textbook_learning_roles(chunk_or_meta: dict) -> set:
    """Infer educational role(s) from normalized source metadata, not hard-coded IDs."""
    blob = " ".join([
        str(chunk_or_meta.get("official_title", "")),
        str(chunk_or_meta.get("specialty", "")),
        str(chunk_or_meta.get("topic", "")),
        str(chunk_or_meta.get("source_id", "")),
        str(chunk_or_meta.get("book", "")),
    ]).lower()

    roles = set()
    for role, patterns in PG_TEXTBOOK_ROLE_PATTERNS.items():
        if any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in patterns):
            roles.add(role)
    return roles


def _pg_requested_textbook_roles(question: str) -> set:
    """Map the student's requested learning domains to textbook evidence roles."""
    q = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    roles = set()

    mechanism = bool(re.search(
        r"\b(?:pathophysiology|pathogenesis|physiology|mechanisms?|homeostasis|hemodynamics?)\b", q
    ))
    pathology = bool(re.search(
        r"\b(?:pathology|pathogenesis|etiology|aetiology|morphology|histology|lesions?)\b", q
    ))
    pharmacology = bool(re.search(
        r"\b(?:pharmacology|pharmacotherapy|drug|drugs|medications?|medicine|medicines|"
        r"dose|doses|dosage|dosing|mechanism of action|adverse effects?|side effects?|"
        r"contraindications?|interactions?|titration|antimicrobials?|antibiotics?|"
        r"vasopressors?|inotropes?|insulin|anticoagulants?|diuretics?|"
        r"beta[ -]?blockers?|sglt2(?:[ -]?inhibitors?)?)\b", q
    ))
    clinical = bool(re.search(
        r"\b(?:clinical features?|presentation|diagnos(?:is|tic)|differential|investigations?|"
        r"work\s*up|management|treatment|complications?|prognosis|approach|staging|classification)\b", q
    ))

    if mechanism:
        roles.add("physiology")
    # Pathology is reserved for disease pathogenesis/pathology signals, not every
    # generic "mechanism" request (e.g. mechanism of action of a drug).
    disease_pathology = pathology or bool(re.search(
        r"\b(?:pathophysiology|pathogenesis|etiology|aetiology)\b", q
    ))
    if disease_pathology:
        roles.add("pathology")
    if pharmacology:
        roles.add("pharmacology")
    if clinical or not roles:
        roles.add("clinical_medicine")

    return roles


def _pg_role_target(depth: str, role: str) -> int:
    """Small role reservations; these are relevance-gated targets, never forced quotas."""
    if depth == "comprehensive":
        return 4 if role == "clinical_medicine" else 3
    if depth == "detailed":
        return 3 if role == "clinical_medicine" else 2
    return 2


def _pg_textbook_budget(final_k: int, intent: str, depth: str) -> tuple:
    """Return (target, ceiling) for PG textbook context at the detected depth."""
    if depth == "comprehensive":
        target = 21 if intent in {"mixed", "educational", "general"} else 17
        ceiling = 22 if intent in {"mixed", "educational", "general"} else 19
    elif depth == "detailed":
        target = 17 if intent in {"mixed", "educational", "general"} else 14
        ceiling = 18 if intent in {"mixed", "educational", "general"} else 16
    else:
        if intent == "educational":
            target, ceiling = round(final_k * 0.39), min(15, round(final_k * 0.42))
        elif intent == "management":
            target, ceiling = round(final_k * 0.28), min(12, round(final_k * 0.34))
        else:
            target, ceiling = round(final_k * 0.33), min(14, round(final_k * 0.39))

    return min(target, final_k), min(ceiling, final_k)


def _query_prefers_indian_guidance(question: str) -> bool:
    q = question.lower()
    explicit_india = any(term in q for term in INDIA_TERMS)
    management_or_threshold = any(term in q for term in MANAGEMENT_TERMS)

    # MediCare AI is India-oriented. For management/threshold/algorithm questions,
    # prefer relevant Indian guidance even when the user does not explicitly say
    # "India". For explanatory questions, do not force an India preference.
    return explicit_india or management_or_threshold


# -----------------------------------------------------------------------------
# Source metadata helpers
# -----------------------------------------------------------------------------
def _load_source_metadata():
    """Cache the small normalized medical_sources collection for reranking."""
    global _source_metadata_cache, _source_metadata_cache_loaded_at

    now = time.time()
    if (
        _source_metadata_cache
        and now - _source_metadata_cache_loaded_at < SOURCE_METADATA_CACHE_SECONDS
    ):
        return _source_metadata_cache

    projection = {
        "_id": 0,
        "source_id": 1,
        "official_title": 1,
        "organization": 1,
        "publication_year": 1,
        "source_type": 1,
        "specialty": 1,
        "topic": 1,
        "country": 1,
        "is_current": 1,
        "local_filename": 1,
        "status": 1,
        "notes": 1,
    }

    retryable = (AutoReconnect, ConnectionFailure, NetworkTimeout, ServerSelectionTimeoutError)
    for attempt in range(3):
        try:
            metadata = {}
            for doc in sources_collection.find({}, projection):
                source_id = doc.get("source_id")
                if source_id:
                    metadata[source_id] = doc
            break
        except retryable as exc:
            if attempt == 2:
                raise
            pause = 0.6 * (2 ** attempt)
            print(f"MongoDB metadata read retry {attempt + 2}/3 ({type(exc).__name__})")
            time.sleep(pause)

    _source_metadata_cache = metadata
    _source_metadata_cache_loaded_at = now
    return metadata


def _source_type(meta: dict) -> str:
    return str(meta.get("source_type", "")).strip().lower()


def _is_guideline(meta: dict) -> bool:
    return _source_type(meta) in GUIDELINE_SOURCE_TYPES


def _is_textbook(meta: dict) -> bool:
    return _source_type(meta) == "textbook"


def _is_current(meta: dict) -> bool:
    value = meta.get("is_current", "")
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() == "TRUE"


def _publication_year(meta: dict):
    try:
        raw = str(meta.get("publication_year", "")).strip()
        if not raw:
            return None
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _is_indian_national_guidance(meta: dict) -> bool:
    country = str(meta.get("country", "")).strip().lower()
    if country != "india":
        return False

    organization = str(meta.get("organization", "")).strip().lower()
    if any(term in organization for term in INDIAN_NATIONAL_AUTHORITY_TERMS):
        return True

    org_tokens = set(re.findall(r"[a-z0-9]+", organization))
    return bool(org_tokens & INDIAN_NATIONAL_AUTHORITY_ABBREVIATIONS)


def _has_explicit_precedence_note(meta: dict) -> bool:
    notes = str(meta.get("notes", "")).strip().lower()
    return any(term in notes for term in PREFERRED_SOURCE_NOTE_TERMS)


def _metadata_terms(text: str) -> set:
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return {
        token for token in tokens
        if len(token) >= 3 and token not in QUERY_METADATA_STOPWORDS
    }


def _topic_identity_terms(meta: dict) -> set:
    """Return compact disease/topic identity terms from curated registry metadata."""
    topic = str(meta.get("topic", ""))
    title = str(meta.get("official_title", ""))

    generic = {
        "guideline", "guidelines", "guidance", "practice", "clinical", "national",
        "standard", "standards", "treatment", "management", "evaluation", "diagnosis",
        "diagnostic", "prevention", "care", "programme", "program", "protocol",
        "framework", "workflow", "workflows", "recommendation", "recommendations",
        "adult", "adults", "patient", "patients", "india", "indian", "global",
        "edition", "update", "updated", "revised", "document", "module", "part",
        "disease", "diseases", "infection", "infections", "syndrome", "syndromes",
        "fever", "disorder", "disorders",
    }

    topic_terms = _metadata_terms(topic) - generic
    title_terms = _metadata_terms(title) - generic

    # Topic is curator-authored and usually the cleanest identity field. Fall back
    # to title only when topic metadata is absent/too sparse.
    return topic_terms if topic_terms else title_terms


def _clear_topic_mismatch(question: str, meta: dict) -> bool:
    """Reject manifestly wrong-disease clinical-guideline metadata.

    This is a conservative veto, not an attempt to diagnose/validate a guideline
    from a title. General textbooks/critical-care containers are still eligible;
    no source is *forced in* to fill the context. Registry titles/topics are the
    evidence for this check, not a similarity-score cutoff.
    """
    q = str(question or "").lower()
    meta_identity = " ".join([
        str(meta.get("official_title", "")),
        str(meta.get("topic", "")),
        str(meta.get("source_id", "")),
    ]).lower()
    # The exact disease requested should not be answered from clinical
    # guidance about an unrelated malignant organ, obstetric condition,
    # kidney disease, burns or liver transplantation.
    conflicts = (
        (r"\bbreast\s+(?:cancer|carcinoma)\b", r"\b(?:bladder|pancreatic|prostate|cervical|ovarian|lung)\s+cancer\b"),
        (r"\b(?:post[- ]?partum\s+h(?:ae|e)morrhag\w*|pph)\b", r"\b(?:burn\s+injur|burns|liver\s+transplant|solid\s+organ\s+transplant)"),
        (r"\b(?:pre[- ]?eclampsia|eclampsia)\b", r"\b(?:liver\s+transplant|burn\s+injur|burns)"),
        (r"\b(?:nephrotic\s+syndrome|steroid[- ]sensitive\s+nephrotic)\b", r"\b(?:adpkd|autosomal\s+dominant\s+polycystic\s+kidney)\b"),
        (r"\b(?:acute\s+kidney\s+injury|\baki\b)\b", r"\b(?:respiratory\s+(?:diseases|medicine)|liver\s+transplant|burns?)\b"),
    )
    if any(re.search(qpat, q) and re.search(spat, meta_identity)
           for qpat, spat in conflicts):
        return True

    # Drug-susceptible TB and drug-resistant TB are different treatment
    # pathways. Similarity on the shared words "TB", "management", and
    # "monitoring" must not make an MDR/RR-TB protocol a primary source for
    # explicitly drug-susceptible disease. Comparison requests asking about
    # both subtypes retain both source families.
    ds_tb = bool(re.search(
        r"\b(?:drug[-\s]susceptible|drug[-\s]sensitive|ds[-\s]?tb|"
        r"rifampicin[-\s]susceptible)\b", q
    ))
    dr_tb = bool(re.search(
        r"\b(?:drug[-\s]resistan\w*|multi[-\s]?drug[-\s]resistan\w*|"
        r"mdr[-/\s]?(?:rr[-/\s]?)?tb|rr[-\s]?tb|"
        r"rifampicin[-\s]resistan\w*)\b", q
    ))
    source_dr_tb = bool(re.search(
        r"\b(?:drug[-\s]resistan\w*|multi[-\s]?drug[-\s]resistan\w*|"
        r"mdr[-/\s]?(?:rr[-/\s]?)?tb|rr[-\s]?tb|"
        r"rifampicin[-\s]resistan\w*)\b", meta_identity
    )) or str(meta.get("source_id", "")).lower().startswith("ntep_dr_tb_")
    if ds_tb and not dr_tb and source_dr_tb:
        return True

    # A *prevention* programme document is not a treatment guideline for
    # newly diagnosed active pulmonary TB. Preventive therapy should still be
    # retrievable for explicit latent infection/contact preventive questions.
    active_tb = bool(re.search(
        r"\b(?:newly\s+diagnosed|drug[- ]susceptible|rifampicin[- ]resistant|multidrug[- ]resistant|active\s+tb|active\s+tuberculosis)\b", q
    ))
    wants_tpt = bool(re.search(
        r"\b(?:latent\s+tb|tb\s+preventive\s+treatment|tpt|preventive\s+therapy|contact\s+preventive)\b", q
    ))
    if active_tb and not wants_tpt and re.search(
        r"\b(?:tb\s+preventive\s+treatment|tuberculosis\s+preventive\s+treatment)\b", meta_identity
    ):
        return True

    return False


def _has_explicit_disease_context_conflict(question: str, meta: dict) -> bool:
    """Reject clearly incompatible disease/subtype-specific sources.

    This guard is intentionally conservative. It only acts when the query names a
    specific disease/subtype (or a generic condition that should not be narrowed
    to a comorbidity-specific guideline). Broad sources that cover both requested
    subtypes remain eligible.
    """
    if _clear_topic_mismatch(question, meta):
        return True

    q_raw = str(question).lower()
    q = expand_medical_acronyms(q_raw).lower()
    blob = " ".join([
        str(meta.get("official_title", "")),
        str(meta.get("topic", "")),
        str(meta.get("specialty", "")),
    ]).lower()

    # Diabetes subtype identity. A source explicitly covering BOTH type 1 and
    # type 2 diabetes remains eligible; a type-1-only source cannot become primary
    # evidence for an explicit type-2 query and vice versa.
    query_t1 = bool(re.search(r"\b(?:t1dm|type\s*1\s+diabetes(?:\s+mellitus)?)\b", q_raw)) or "type 1 diabetes" in q
    query_t2 = bool(re.search(r"\b(?:t2dm|type\s*2\s+diabetes(?:\s+mellitus)?)\b", q_raw)) or "type 2 diabetes" in q
    source_t1 = bool(re.search(r"\b(?:t1dm|type\s*1\s+diabetes(?:\s+mellitus)?)\b", blob))
    source_t2 = bool(re.search(r"\b(?:t2dm|type\s*2\s+diabetes(?:\s+mellitus)?)\b", blob))

    if query_t1 and not query_t2 and source_t2 and not source_t1:
        return True
    if query_t2 and not query_t1 and source_t1 and not source_t2:
        return True

    # A diabetes-in-CKD guideline is comorbidity-specific, not core diabetes
    # guidance. Do not use it for diabetes without renal involvement. Allow it
    # whenever the user actually asks about CKD/renal disease or renal markers.
    query_diabetes = bool(re.search(r"\b(?:diabetes|diabetic|t[12]dm)\b", q))
    query_renal_context = bool(re.search(
        r"\b(?:ckd|chronic\s+kidney\s+disease|diabetic\s+kidney\s+disease|"
        r"kidneys?|renal|nephro\w*|e-?gfr|dialysis|albuminuria|proteinuria|"
        r"creatinine|uacr)\b", q
    ))
    source_diabetes = bool(re.search(r"\b(?:diabetes|diabetic)\b", blob))
    source_ckd_specific = bool(re.search(
        r"\b(?:ckd|chronic\s+kidney\s+disease|diabetic\s+kidney\s+disease)\b",
        blob,
    ))
    if query_diabetes and not query_renal_context and source_diabetes and source_ckd_specific:
        return True

    # Heart failure and hypertension share many drug terms. Do not allow a
    # hypertension-only guideline to become the topic anchor for an explicit
    # heart-failure query (or vice versa) unless the query actually asks about both.
    query_hf = bool(re.search(
        r"\b(?:heart\s+failure|hfr?ef|hfpef|hfmref|congestive\s+heart\s+failure|chf)\b",
        q,
    ))
    query_htn = bool(re.search(r"\b(?:hypertension|high\s+blood\s+pressure|htn)\b", q))
    source_hf = bool(re.search(r"\b(?:heart\s+failure|hfr?ef|hfpef|hfmref|congestive\s+heart\s+failure)\b", blob))
    source_htn = bool(re.search(r"\b(?:hypertension|high\s+blood\s+pressure)\b", blob))

    if query_hf and not query_htn and source_htn and not source_hf:
        return True
    if query_htn and not query_hf and source_hf and not source_htn:
        return True

    # Generic anemia questions should not be anchored to CKD-specific anemia
    # guidance unless the user supplied renal/CKD context.
    query_anemia = bool(re.search(r"\b(?:anemia|anaemia|microcytic|macrocytic)\b", q))
    query_renal = bool(re.search(
        r"\b(?:ckd|chronic\s+kidney\s+disease|kidney|renal|nephro\w*|egfr|dialysis)\b",
        q,
    ))
    source_anemia = bool(re.search(r"\b(?:anemia|anaemia)\b", blob))
    source_renal = bool(re.search(
        r"\b(?:ckd|chronic\s+kidney\s+disease|kidney|renal|nephro\w*)\b",
        blob,
    ))

    if query_anemia and not query_renal and source_anemia and source_renal:
        return True

    return False


def _build_guideline_topic_tiers(question: str, reranked: list) -> dict:
    """Classify guideline sources as primary / supplementary / reject.

    The best topic-matched guideline establishes the query's clinical anchor. Other
    guideline sources must meaningfully share that anchor. Sources that merely sit
    in the same specialty/programme but describe another disease are rejected.

    This is used only for normal diversified retrieval. Explicit source/latest
    retrieval remains governed by the strict resolver path.
    """
    metadata_by_id = _load_source_metadata()
    stats = _source_stats(reranked)
    if not stats:
        return {}

    # First try an anchor based on the QUESTION, not the top-ranked source.
    # Previously an incidental national/critical-care source could become the
    # anchor and reject all 100+ chunks from the exact WHO clinical guideline.
    # Retain the original algorithm for unrecognised/implicit conditions.
    qlabels = _clinical_topic_labels(question)
    if qlabels:
        query_tiers = {}
        found_primary = False
        for sid in stats:
            meta = metadata_by_id.get(sid, {})
            if not _is_guideline(meta):
                continue
            if (_has_explicit_population_conflict(question, meta)
                    or _has_explicit_disease_context_conflict(question, meta)):
                continue
            tier = _topic_facet_tier(question, meta)
            if tier == "primary":
                found_primary = True
            query_tiers[sid] = tier
        if found_primary:
            return query_tiers

    guideline_items = []
    for source_id, item in stats.items():
        meta = metadata_by_id.get(source_id, {})
        if not _is_guideline(meta):
            continue
        if _has_explicit_population_conflict(question, meta):
            continue
        if _has_explicit_disease_context_conflict(question, meta):
            continue
        topic_match = _source_topic_match_score(question, meta)
        guideline_items.append((source_id, item, topic_match))

    if not guideline_items:
        return {}

    guideline_items.sort(
        key=lambda row: (row[2], row[1]["max_vector_score"], row[1]["max_retrieval_score"]),
        reverse=True,
    )
    primary_source_id, primary_item, _ = guideline_items[0]
    primary_meta = metadata_by_id.get(primary_source_id, {})
    anchor_terms = _topic_identity_terms(primary_meta)

    # If metadata is too sparse to establish a safe anchor, leave surviving
    # guideline sources as primary rather than guessing.
    if not anchor_terms:
        return {source_id: "primary" for source_id, _, _ in guideline_items}

    tiers = {}
    primary_best_vector = float(primary_item.get("max_vector_score", 0.0) or 0.0)

    for source_id, item, topic_match in guideline_items:
        meta = metadata_by_id.get(source_id, {})
        if source_id == primary_source_id:
            tiers[source_id] = "primary"
            continue

        source_terms = _topic_identity_terms(meta)
        if not source_terms:
            tiers[source_id] = "reject"
            continue

        shared = anchor_terms & source_terms
        anchor_coverage = len(shared) / max(1, len(anchor_terms))
        source_precision = len(shared) / max(1, len(source_terms))
        vector_score = float(item.get("max_vector_score", 0.0) or 0.0)

        # A source whose curated topic is essentially the same condition can be
        # treated as another primary guideline (e.g. ICMR/AHA/ESC hypertension).
        if anchor_coverage >= 0.80 and source_precision >= 0.70:
            tiers[source_id] = "primary"
            continue

        # Narrow subtypes/comorbid guidance can supplement the core topic, but it
        # gets a small source cap. Require meaningful anchor overlap and reasonable
        # semantic proximity to the best source.
        if (
            anchor_coverage >= 0.50
            and source_precision >= 0.30
            and vector_score >= (primary_best_vector - 0.10)
        ):
            tiers[source_id] = "supplementary"
            continue

        tiers[source_id] = "reject"

    return tiers


def _metadata_topic_bonus(question: str, meta: dict) -> float:
    """
    Modest lexical alignment bonus using curated source metadata only.

    This helps primary-topic guidance outrank incidentally relevant comorbidity
    documents without hard-filtering useful supplementary guidelines.
    """
    query_terms = _metadata_terms(expand_medical_acronyms(question))
    if not query_terms:
        return 0.0

    metadata_text = " ".join([
        str(meta.get("official_title", "")),
        str(meta.get("topic", "")),
        str(meta.get("specialty", "")),
    ])
    source_terms = _metadata_terms(metadata_text)
    overlap = query_terms & source_terms

    if len(overlap) >= 3:
        return 0.040
    if len(overlap) == 2:
        return 0.030
    if len(overlap) == 1:
        return 0.020
    return 0.0


def _is_broad_registry_source(meta: dict) -> bool:
    """Return True for deliberately broad/multidisciplinary guideline containers."""
    topic = str(meta.get("topic", "")).strip().lower()
    specialty = str(meta.get("specialty", "")).strip().lower()
    title = str(meta.get("official_title", "")).strip().lower()

    broad_markers = (
        "multidisciplinary",
        "standard treatment workflows",
        "standard treatment workflow volume",
        "general clinical",
    )
    blob = " ".join([topic, specialty, title])
    return any(marker in blob for marker in broad_markers)


def _has_explicit_population_conflict(question: str, meta: dict) -> bool:
    """Reject obvious age-population mismatches in normal diversified retrieval.

    This is deliberately conservative: it only acts when the user explicitly
    states an adult vs paediatric population. It does not infer age from a
    disease name, so paediatric diseases can still retrieve paediatric sources.
    """
    q = str(question).lower()
    blob = " ".join([
        str(meta.get("official_title", "")),
        str(meta.get("topic", "")),
        str(meta.get("specialty", "")),
    ]).lower()
    notes = str(meta.get("notes", "")).lower()
    # Source notes may explicitly document mixed-population coverage. They can
    # broaden an age restriction but must not accidentally tag general guidance
    # as pregnancy-only merely because pregnancy is mentioned in notes.
    age_blob = blob + " " + notes

    query_adult = bool(re.search(r"\badults?\b", q))
    query_paediatric = bool(re.search(
        r"\b(?:paediatric|pediatric|child|children|infant|infants|newborn|newborns|neonate|neonates|\d{1,2}[- ]year[- ]old)\b",
        q,
    ))

    source_paediatric = bool(re.search(
        r"\b(?:paediatric|pediatric|paediatrics|pediatrics|child|children|neonatal|neonatology)\b",
        age_blob,
    ))
    source_adult = bool(re.search(r"\badults?\b", age_blob))

    query_pregnancy = bool(re.search(
        r"\b(?:pregnan(?:t|cy)|gestational|antenatal|maternal|obstetric|obstetrics|post[- ]?partum|postnatal|peri[- ]?partum|puerperal|delivery|childbirth)\b",
        q,
    ))
    source_pregnancy = bool(re.search(
        r"\b(?:pregnan(?:t|cy)|gestational|antenatal|maternal|obstetric|obstetrics)\b",
        blob,
    ))

    # A guideline covering BOTH children and adults is appropriate for either
    # subgroup; reject only exclusive population mismatches.
    if query_adult and source_paediatric and not source_adult and not query_paediatric:
        return True
    if query_paediatric and source_adult and not source_paediatric and not query_adult:
        return True
    if query_paediatric and source_pregnancy and not query_pregnancy:
        return True
    if query_pregnancy and source_paediatric and not source_pregnancy and not query_paediatric:
        return True

    # Population-specific sources should not fill generic questions unless the
    # user actually asked for that population.
    if source_paediatric and not query_paediatric and not source_adult:
        return True
    if source_pregnancy and not query_pregnancy:
        return True

    # Do not reject a guideline simply because its title contains "prevention".
    # WHO PPH guidance, GOLD COPD, and other current clinical guidelines cover
    # prevention AND treatment. True active-TB/TPT conflicts are handled by
    # _clear_topic_mismatch instead of blanket substring matching.

    return False


def _apply_diversified_source_relevance_guard(question: str, reranked: list) -> tuple:
    """Remove off-topic guideline sources and annotate surviving guideline topic tiers."""
    if not reranked:
        return reranked, 0

    metadata_by_id = _load_source_metadata()
    best_vector = max(float(c.get("vector_score", 0.0) or 0.0) for c in reranked)

    source_best = {}
    for chunk in reranked:
        source_id = chunk.get("source_id")
        if not source_id:
            continue
        score = float(chunk.get("vector_score", 0.0) or 0.0)
        source_best[source_id] = max(source_best.get(source_id, 0.0), score)

    allowed_guideline_sources = set()
    rejected_guideline_sources = set()
    population_conflict_sources = set()
    disease_context_conflict_sources = set()

    for source_id, source_score in source_best.items():
        meta = metadata_by_id.get(source_id, {})

        if _has_explicit_population_conflict(question, meta):
            population_conflict_sources.add(source_id)
            if _is_guideline(meta):
                rejected_guideline_sources.add(source_id)
            continue

        if _has_explicit_disease_context_conflict(question, meta):
            disease_context_conflict_sources.add(source_id)
            if _is_guideline(meta):
                rejected_guideline_sources.add(source_id)
            continue

        if not _is_guideline(meta):
            continue

        topic_bonus = _metadata_topic_bonus(question, meta)
        strong_semantic = source_score >= (best_vector - DIVERSIFIED_STRONG_SOURCE_MARGIN)
        clinical_topic_supported = (
            _topic_facet_tier(question, meta) in {"primary", "supplementary"}
            and source_score >= max(0.65, best_vector - 0.23)
        )
        metadata_supported = (
            topic_bonus > 0.0
            and source_score >= (best_vector - DIVERSIFIED_METADATA_SOURCE_MARGIN)
        )
        broad_source_supported = (
            _is_broad_registry_source(meta)
            and source_score >= (best_vector - DIVERSIFIED_BROAD_SOURCE_MARGIN)
        )

        if clinical_topic_supported or strong_semantic or metadata_supported or broad_source_supported:
            allowed_guideline_sources.add(source_id)
        else:
            rejected_guideline_sources.add(source_id)

    prelim = []
    for chunk in reranked:
        source_id = chunk.get("source_id")
        meta = metadata_by_id.get(source_id, {})
        if source_id in population_conflict_sources or source_id in disease_context_conflict_sources:
            continue
        if _is_guideline(meta) and source_id not in allowed_guideline_sources:
            continue
        prelim.append(chunk)

    topic_tiers = _build_guideline_topic_tiers(question, prelim)
    filtered = []
    for chunk in prelim:
        source_id = chunk.get("source_id")
        meta = metadata_by_id.get(source_id, {})
        item = dict(chunk)

        if _is_guideline(meta):
            tier = topic_tiers.get(source_id, "reject")
            if tier == "reject":
                rejected_guideline_sources.add(source_id)
                continue
            item["topic_tier"] = tier
        else:
            item["topic_tier"] = "textbook" if _is_textbook(meta) else "other"

        filtered.append(item)

    return filtered, len(rejected_guideline_sources)


# -----------------------------------------------------------------------------
# Explicit source / freshness policy detection
# -----------------------------------------------------------------------------
def _normalize_lookup_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def _requested_years(question: str) -> set:
    return {
        int(match)
        for match in re.findall(r"\b(?:19|20)\d{2}\b", str(question))
    }


def _is_comparison_request(question: str) -> bool:
    q = f" {str(question).lower()} "
    return any(term in q for term in COMPARISON_TERMS)


def _wants_fresh_guidance(question: str) -> bool:
    """
    Trigger strict freshness only when freshness language is tied to a clinical
    guidance/management request. "Recent pathophysiology research" should not
    accidentally become a guideline-only query.
    """
    q = str(question).lower()
    has_freshness = any(re.search(pattern, q) for pattern in FRESHNESS_PATTERNS)
    has_guidance_context = any(term in q for term in FRESH_GUIDANCE_CONTEXT_TERMS)
    return has_freshness and has_guidance_context


def _metadata_search_blob(meta: dict) -> str:
    return " ".join([
        str(meta.get("source_id", "")),
        str(meta.get("official_title", "")),
        str(meta.get("organization", "")),
        str(meta.get("local_filename", "")),
    ]).lower()


def _detect_source_aliases(question: str) -> list:
    q_lower = str(question).lower()
    found = []

    for alias, patterns in SOURCE_ALIAS_QUERY_PATTERNS.items():
        if any(re.search(pattern, q_lower, flags=re.IGNORECASE) for pattern in patterns):
            found.append(alias)

    # WHO and NICE are common English words/acronyms that can be ambiguous.
    # Accept uppercase mentions, full names, or explicit source/guideline phrasing.
    original = str(question)
    if (
        re.search(r"\bWHO\b", original)
        or re.search(r"\bworld health organization\b", q_lower)
        or re.search(r"\baccording to who\b", q_lower)
        or re.search(r"\bwho\s+(?:guideline|guidelines|guidance|recommendation|recommendations)\b", q_lower)
    ):
        if "who" not in found:
            found.append("who")

    if (
        re.search(r"\bNICE\b", original)
        or re.search(r"\bnational institute for health and care excellence\b", q_lower)
        or re.search(r"\baccording to nice\b", q_lower)
        or re.search(r"\bnice\s+(?:guideline|guidelines|guidance|recommendation|recommendations)\b", q_lower)
    ):
        if "nice" not in found:
            found.append("nice")

    # "gold standard" is ordinary clinical language, so GOLD source lock only
    # activates for an uppercase acronym/full organization or clear COPD context.
    if (
        re.search(r"\bGOLD\b", original)
        or re.search(r"\bglobal initiative for chronic obstructive lung disease\b", q_lower)
        or re.search(r"\bgold\s+(?:copd|guideline|guidelines|report|strategy)\b", q_lower)
        or re.search(r"\baccording to gold\b", q_lower)
    ):
        if "gold" not in found:
            found.append("gold")

    return found


def _direct_named_source_ids(question: str, metadata_by_id: dict) -> set:
    """
    Catch explicit source IDs / full official titles / recognizable filenames.

    This is intentionally conservative. Broad organization matching is handled by
    aliases; direct matching should only fire when the user clearly named a source.
    """
    q_lower = str(question).lower()
    q_norm = _normalize_lookup_text(question)
    matches = set()

    for source_id, meta in metadata_by_id.items():
        source_id_lower = str(source_id).lower()
        if source_id_lower and source_id_lower in q_lower:
            matches.add(source_id)
            continue

        title_norm = _normalize_lookup_text(meta.get("official_title", ""))
        if title_norm and len(title_norm.split()) >= 4 and title_norm in q_norm:
            matches.add(source_id)
            continue

        filename = str(meta.get("local_filename", "")).rsplit(".", 1)[0]
        filename_norm = _normalize_lookup_text(filename)
        if filename_norm and len(filename_norm.split()) >= 4 and filename_norm in q_norm:
            matches.add(source_id)

    return matches


def _metadata_matches_alias(meta: dict, alias: str) -> bool:
    blob = _metadata_search_blob(meta)
    return any(term in blob for term in SOURCE_ALIAS_METADATA_TERMS.get(alias, ()))


def _resolve_explicit_source_constraint(question: str):
    """
    Resolve an explicit organization/source request to allowed source IDs.

    Examples:
      KDIGO 2024 -> KDIGO source IDs with publication_year=2024
      ICMR guideline -> ICMR-family source IDs only
      NTEP -> NTEP-family source IDs only
      Harrison -> Harrison textbook source only

    Returns None when the user did not explicitly name a source/organization.
    """
    metadata_by_id = _load_source_metadata()
    direct_ids = _direct_named_source_ids(question, metadata_by_id)
    aliases = [
        alias for alias in _detect_source_aliases(question)
        if not _named_organization_is_context_only(question, alias)
    ]

    if not direct_ids and not aliases:
        return None

    if direct_ids:
        allowed_ids = set(direct_ids)
        label = "direct:" + ",".join(sorted(direct_ids))
    else:
        allowed_ids = {
            source_id
            for source_id, meta in metadata_by_id.items()
            if any(_metadata_matches_alias(meta, alias) for alias in aliases)
        }
        label = "+".join(aliases)

    years = _requested_years(question)
    if years:
        year_filtered = {
            source_id
            for source_id in allowed_ids
            if _publication_year(metadata_by_id.get(source_id, {})) in years
        }
        # Explicit year is a strict constraint. Do not silently substitute a
        # different edition/year when the requested one is not present.
        allowed_ids = year_filtered

    fresh = _wants_fresh_guidance(question)
    comparison = _is_comparison_request(question)

    if fresh and not comparison and not years and allowed_ids:
        current_ids = {
            source_id
            for source_id in allowed_ids
            if _is_current(metadata_by_id.get(source_id, {}))
        }
        if current_ids:
            allowed_ids = current_ids

    return {
        "allowed_ids": allowed_ids,
        "aliases": aliases,
        "direct_ids": direct_ids,
        "requested_years": years,
        "fresh": fresh,
        "comparison": comparison,
        "label": label,
    }


def _source_stats(candidates: list) -> dict:
    stats = {}
    metadata_by_id = _load_source_metadata()

    for chunk in candidates:
        source_id = chunk.get("source_id")
        if not source_id:
            continue

        entry = stats.setdefault(
            source_id,
            {
                "source_id": source_id,
                "max_vector_score": 0.0,
                "max_retrieval_score": 0.0,
                "meta": metadata_by_id.get(source_id, {}),
            },
        )
        entry["max_vector_score"] = max(
            entry["max_vector_score"],
            float(chunk.get("vector_score", chunk.get("score", 0.0)) or 0.0),
        )
        entry["max_retrieval_score"] = max(
            entry["max_retrieval_score"],
            float(chunk.get("retrieval_score", chunk.get("score", 0.0)) or 0.0),
        )

    return stats


SOURCE_RESOLUTION_STOPWORDS = QUERY_METADATA_STOPWORDS | {
    "according", "say", "says", "does", "should", "managed", "managing",
    "recommend", "recommends", "recommended", "practice", "clinical",
    "guideline", "guidelines", "guidance", "organization", "source",
}


def _source_resolution_query_terms(question: str) -> set:
    """Return clinical topic terms for resolving a document within a named source family."""
    cleaned = expand_medical_acronyms(str(question)).lower()
    cleaned = re.sub(r"\b(?:19|20)\d{2}\b", " ", cleaned)

    # Remove explicit organization/source aliases from the topic signal. The
    # organization/year filter has already been applied before this resolver.
    for patterns in SOURCE_ALIAS_QUERY_PATTERNS.values():
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    tokens = re.findall(r"[a-z0-9]+", cleaned)
    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in SOURCE_RESOLUTION_STOPWORDS
    }


def _source_topic_match_score(question: str, meta: dict) -> tuple:
    """
    Score how specifically the user's clinical topic matches one registry source.

    This score is deliberately independent of chunk vector score. In an explicit
    source-family request such as "KDIGO 2024 chronic kidney disease", document
    identity/topic must be resolved before chunk-level semantic similarity.
    """
    query_terms = _source_resolution_query_terms(question)
    if not query_terms:
        return (0, 0.0, 0, 0)

    topic = str(meta.get("topic", ""))
    title = str(meta.get("official_title", ""))
    specialty = str(meta.get("specialty", ""))

    topic_terms = _metadata_terms(topic)
    title_terms = _metadata_terms(title)
    specialty_terms = _metadata_terms(specialty)

    primary_terms = topic_terms | title_terms
    primary_overlap = query_terms & primary_terms
    specialty_overlap = query_terms & specialty_terms

    # Exact curated topic phrase is the strongest possible signal.
    q_norm = _normalize_lookup_text(expand_medical_acronyms(question))
    topic_norm = _normalize_lookup_text(topic)
    exact_topic_phrase = int(
        bool(topic_norm)
        and len(topic_norm.split()) >= 2
        and topic_norm in q_norm
    )

    # Coverage is useful when two sources share one broad word (e.g. "kidney")
    # but only one source matches the full disease phrase.
    coverage = (
        len(primary_overlap) / max(1, len(query_terms))
    )

    return (
        exact_topic_phrase,
        round(coverage, 6),
        len(primary_overlap),
        len(specialty_overlap),
    )


def _choose_relevant_source_ids(
    question: str,
    candidates: list,
    allowed_ids: set = None,
    max_sources: int = MAX_CONSTRAINED_SOURCES,
) -> list:
    """
    Resolve the relevant document(s) inside an explicit source/organization family.

    Priority is intentionally different from normal retrieval:
      1. registry topic/title match to the user's requested clinical topic
      2. chunk vector relevance
      3. metadata-aware retrieval score

    This prevents a nearby document from the same organization/year (for example
    KDIGO 2024 lupus nephritis) from leaking into a request explicitly about the
    KDIGO 2024 chronic kidney disease guideline.
    """
    metadata_by_id = _load_source_metadata()
    filtered = [
        c for c in candidates
        if (allowed_ids is None or c.get("source_id") in allowed_ids)
        and not _has_explicit_disease_context_conflict(
            question, metadata_by_id.get(c.get("source_id"), {})
        )
        and not _has_explicit_population_conflict(
            question, metadata_by_id.get(c.get("source_id"), {})
        )
    ]
    stats = _source_stats(filtered)
    if not stats:
        return []

    for item in stats.values():
        item["topic_match"] = _source_topic_match_score(question, item["meta"])

    ranked = sorted(
        stats.values(),
        key=lambda item: (
            item["topic_match"],
            item["max_vector_score"],
            item["max_retrieval_score"],
        ),
        reverse=True,
    )

    best = ranked[0]
    best_topic_match = best["topic_match"]
    best_vector = best["max_vector_score"]

    # When registry metadata clearly identifies the clinical topic, source lock
    # becomes strict to that best topic match (or exact ties). This is the normal
    # case for curated guideline families such as KDIGO/ICMR/NTEP.
    if best_topic_match > (0, 0.0, 0, 0):
        chosen = [
            item["source_id"]
            for item in ranked
            if item["topic_match"] == best_topic_match
        ][:max_sources]
        return chosen or [best["source_id"]]

    # If the query names only an organization/year but no resolvable clinical
    # topic, retain the previous semantic fallback and allow several closely
    # relevant documents from that requested family.
    chosen = []
    for item in ranked:
        if item["max_vector_score"] < (best_vector - SOURCE_RELEVANCE_MARGIN):
            continue
        chosen.append(item["source_id"])
        if len(chosen) >= max_sources:
            break

    return chosen or [best["source_id"]]

def _choose_freshest_applicable_source_ids(
    question: str,
    candidates: list,
    allowed_ids: set = None,
) -> tuple:
    """
    Select newest/current applicable guideline source(s).

    Order of operations is deliberate:
      1. current guideline only
      2. explicit population/disease-subtype compatibility for auto-freshness
      3. registry topic identity (before vector similarity)
      4. vector applicability within the resolved topic
      5. India-national preference for India-oriented management
      6. newest known publication year
      7. strongest source relevance within that newest year

    Topic identity is resolved before recency so a newer but clinically adjacent
    guideline (for example hypertension) cannot displace the correct disease
    guideline (for example HFrEF/heart failure).
    """
    filtered = []
    metadata_by_id = _load_source_metadata()

    for chunk in candidates:
        source_id = chunk.get("source_id")
        if not source_id:
            continue
        if allowed_ids is not None and source_id not in allowed_ids:
            continue

        meta = metadata_by_id.get(source_id, {})
        if not _is_guideline(meta) or not _is_current(meta):
            continue
        if _has_explicit_population_conflict(question, meta):
            continue
        if _has_explicit_disease_context_conflict(question, meta):
            continue
        filtered.append(chunk)

    stats = _source_stats(filtered)
    if not stats:
        return [], None

    for item in stats.values():
        item["topic_match"] = _source_topic_match_score(question, item["meta"])

    ranked = sorted(
        stats.values(),
        key=lambda item: (
            item["topic_match"],
            item["max_vector_score"],
            item["max_retrieval_score"],
        ),
        reverse=True,
    )

    best_topic_match = ranked[0]["topic_match"]

    # When curated metadata gives us a real disease/topic signal, resolve that
    # identity first. Exact ties are allowed so multiple genuinely equivalent
    # current sources can still compete on authority/year/vector relevance.
    if best_topic_match > (0, 0.0, 0, 0):
        applicable = [
            item for item in ranked
            if item["topic_match"] == best_topic_match
        ]
    else:
        best_vector = ranked[0]["max_vector_score"]
        applicable = [
            item
            for item in ranked
            if item["max_vector_score"] >= (best_vector - SOURCE_RELEVANCE_MARGIN)
        ]
        if not applicable:
            applicable = [ranked[0]]

    # For management/threshold questions in this India-oriented CDS, choose the
    # latest relevant Indian national guidance only AFTER the disease topic has
    # been resolved. This prevents unrelated India guidance from winning merely
    # because it is national/current.
    if _query_prefers_indian_guidance(question):
        indian_national = [
            item for item in applicable
            if _is_indian_national_guidance(item["meta"])
        ]
        if indian_national:
            applicable = indian_national

    dated = [
        item for item in applicable
        if _publication_year(item["meta"]) is not None
    ]

    newest_year = None
    if dated:
        newest_year = max(_publication_year(item["meta"]) for item in dated)
        applicable = [
            item for item in dated
            if _publication_year(item["meta"]) == newest_year
        ]

    applicable.sort(
        key=lambda item: (
            item["topic_match"],
            item["max_retrieval_score"],
            item["max_vector_score"],
        ),
        reverse=True,
    )

    chosen = [item["source_id"] for item in applicable[:MAX_CONSTRAINED_SOURCES]]
    return chosen, newest_year


def _select_constrained_chunks(
    reranked: list,
    allowed_source_ids: set,
    final_k: int,
) -> list:
    """
    Select only chunks from the requested/chosen source IDs.

    No cross-source padding is permitted. Per-source caps from normal diversified
    retrieval are intentionally disabled because source domination is expected
    when the user explicitly asked for that source.
    """
    if not allowed_source_ids:
        return []

    selected = []
    selected_ids = set()
    selected_signatures = []
    source_page_counts = Counter()

    for chunk in reranked:
        if len(selected) >= final_k:
            break

        if chunk.get("source_id") not in allowed_source_ids:
            continue

        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id in selected_ids:
            continue

        source_id = chunk.get("source_id") or chunk.get("book") or "unknown"
        page_start = chunk.get("page_start", chunk.get("page"))
        page_end = chunk.get("page_end", page_start)
        source_page_key = (source_id, page_start, page_end)

        if source_page_counts[source_page_key] >= CONSTRAINED_PAGE_RANGE_CAP:
            continue

        signature = _token_signature(chunk.get("text", ""))
        if _is_near_duplicate(signature, selected_signatures):
            continue

        selected.append(chunk)
        if chunk_id:
            selected_ids.add(chunk_id)
        selected_signatures.append(signature)
        source_page_counts[source_page_key] += 1

    return selected


def _vector_search_candidates(query_embedding, candidate_limit: int, num_candidates: int) -> list:
    """Execute Atlas Vector Search, retrying *only* transient connection failures.

    Do not turn a failed network operation into an apparently valid empty result.
    Non-transient issues (invalid index/pipeline/authentication) still propagate.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": candidate_limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "source_id": 1,
                "book": 1,
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "chunk_index": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    retryable = (AutoReconnect, ConnectionFailure, NetworkTimeout, ServerSelectionTimeoutError)
    for attempt in range(3):
        try:
            return list(collection.aggregate(pipeline))
        except retryable as exc:
            if attempt == 2:
                raise
            wait_seconds = 0.6 * (2 ** attempt)
            print(
                f"MongoDB transient connection failure; "
                f"retrying search {attempt + 2}/3 in {wait_seconds:.1f}s "
                f"({type(exc).__name__})"
            )
            time.sleep(wait_seconds)
    return []


def _comparison_guideline_pairs(question: str, reranked: list) -> tuple:
    """Find an older-dated and newer-dated guideline on the *same query topic*.

    An older publication is not automatically superseded: return source-specific
    publication years so the synthesis can describe exactly what was compared.
    Never fabricate a pair when the vector candidate pool lacks two dated sources.
    """
    if not _is_comparison_request(question):
        return [], []

    query_terms = _metadata_terms(expand_medical_acronyms(question))
    by_source = {}
    for chunk in reranked:
        if _candidate_kind(chunk) != "guideline":
            continue
        if _has_explicit_population_conflict(question, chunk):
            continue
        if _has_explicit_disease_context_conflict(question, chunk):
            continue
        year = _publication_year(chunk)
        if year is None:
            continue
        # Curated source topic/title must contain at least one substantive
        # disease term from the user's question. A nearby specialty alone is not enough.
        identity = _topic_identity_terms(chunk)
        if not (identity & query_terms):
            continue
        source_id = chunk.get("source_id")
        if not source_id:
            continue
        by_source.setdefault(source_id, []).append(chunk)

    if len(by_source) < 2:
        return [], []

    best = []
    for source_id, chunks in by_source.items():
        chunks.sort(key=lambda c: (
            c.get("vector_score", c.get("score", 0.0)),
            c.get("retrieval_score", 0.0)
        ), reverse=True)
        meta = chunks[0]
        year = _publication_year(meta)
        topic_overlap = len(_topic_identity_terms(meta) & query_terms)
        best.append((year, topic_overlap, float(chunks[0].get("vector_score", 0.0)), source_id, chunks))

    # Keep only sources with the best *disease identity*, then select dates.
    # Different publication years can represent different authorities, not
    # necessarily revisions of the same guideline; metadata is preserved.
    max_overlap = max(item[1] for item in best)
    eligible = [item for item in best if item[1] == max_overlap]
    years = sorted({item[0] for item in eligible})
    if len(years) < 2:
        return [], []

    older_year, newer_year = years[0], years[-1]
    def best_for_year(year):
        return max(
            (item for item in eligible if item[0] == year),
            key=lambda item: item[2]
        )

    old = best_for_year(older_year)
    new = best_for_year(newer_year)
    result = []
    for row, temporal_role in ((old, "older_dated"), (new, "newer_dated")):
        # Search beyond the first three candidate chunks, because nearby vector
        # hits can be near duplicates from the same page. Reserve up to three
        # DISTINCT passages without manufacturing evidence when fewer exist.
        signatures = []
        for c in row[4]:
            signature = _token_signature(c.get("text", ""))
            if _is_near_duplicate(signature, signatures):
                continue
            item = dict(c)
            item["comparison_role"] = temporal_role
            result.append(item)
            signatures.append(signature)
            if len(signatures) >= 3:
                break
    return result, [(old[3], older_year), (new[3], newer_year)]


def _ensure_comparison_pair(selected: list, extra: list, final_k: int) -> list:
    """Reserve up to three distinct evidence passages per dated comparison side."""
    if not extra:
        return selected
    wanted = {}
    for c in extra:
        sid = c.get("source_id")
        wanted.setdefault(sid, []).append(c)
    if len(wanted) < 2:
        return selected
    output = list(selected)
    keys = {c.get("chunk_id") for c in output if c.get("chunk_id")}
    for source_id, items in wanted.items():
        current = sum(1 for c in output if c.get("source_id") == source_id)
        for item in items:
            if current >= 3:
                break
            if item.get("chunk_id") and item.get("chunk_id") in keys:
                continue
            signature = _token_signature(item.get("text", ""))
            # Distinct guideline sources are not duplicates of each other for
            # historical/current comparison, even when wording overlaps.
            same_source_signatures = [
                _token_signature(c.get("text", "")) for c in output
                if c.get("source_id") == source_id
            ]
            if _is_near_duplicate(signature, same_source_signatures):
                continue
            item = dict(item)
            if len(output) >= final_k:
                # Drop the weakest non-comparison context, never the counterpart.
                evict = next((i for i in range(len(output)-1, -1, -1)
                              if output[i].get("source_id") not in wanted), None)
                if evict is None:
                    break
                output.pop(evict)
            output.append(item)
            if item.get("chunk_id"):
                keys.add(item["chunk_id"])
            current += 1
    output.sort(key=lambda c: (c.get("retrieval_score", 0.0), c.get("vector_score", 0.0)), reverse=True)
    return output[:final_k]


def _pg_domain_search_queries(question: str) -> list:
    """Build targeted textbook subqueries for requested PG learning domains."""
    roles = _pg_requested_textbook_roles(question)
    depth = classify_pg_query_depth(question)
    queries = []

    if "physiology" in roles:
        queries.append((
            "physiology",
            f"{question} physiology homeostasis organ function hemodynamics physiological mechanisms",
        ))

    if "pathology" in roles:
        queries.append((
            "pathology",
            f"{question} pathology pathogenesis morphology disease mechanisms histopathology",
        ))

    if "pharmacology" in roles:
        queries.append((
            "pharmacology",
            f"{question} pharmacology therapeutics mechanism of action dosing adverse effects contraindications drug interactions",
        ))

    # Clinical-medicine sources are normally strong in the broad base query, so
    # they do not need a separate search. Keep extra Atlas calls bounded.
    max_queries = 3 if depth == "comprehensive" else 2
    return queries[:max_queries]


def _merge_pg_reranked_textbooks(base_reranked: list, domain_textbooks: list) -> list:
    """Merge targeted PG textbook results after base guideline filtering is complete."""
    merged = {}
    order = []

    def key_for(chunk):
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            return ("id", chunk_id)
        return (
            "fallback",
            chunk.get("source_id") or chunk.get("book"),
            chunk.get("page_start", chunk.get("page")),
            chunk.get("page_end", chunk.get("page")),
            str(chunk.get("text", ""))[:120],
        )

    for chunk in base_reranked:
        key = key_for(chunk)
        merged[key] = dict(chunk)
        order.append(key)

    for domain_label, chunks in domain_textbooks:
        for chunk in chunks:
            key = key_for(chunk)
            domain_score = float(chunk.get("vector_score", chunk.get("score", 0.0)) or 0.0)
            if key in merged:
                item = merged[key]
                origins = list(item.get("pg_domain_origins", []))
                if domain_label not in origins:
                    origins.append(domain_label)
                item["pg_domain_origins"] = origins
                item["pg_domain_score"] = max(
                    float(item.get("pg_domain_score", 0.0) or 0.0), domain_score
                )
            else:
                item = dict(chunk)
                item["pg_domain_origins"] = [domain_label]
                item["pg_domain_score"] = domain_score
                item["pg_domain_only"] = True
                merged[key] = item
                order.append(key)

    output = [merged[key] for key in order]
    output.sort(
        key=lambda c: (c.get("retrieval_score", 0.0), c.get("vector_score", 0.0)),
        reverse=True,
    )
    return output


def _augment_pg_reranked_textbooks(question: str, base_reranked: list) -> tuple:
    """Recover requested textbook domains without altering guideline source filtering."""
    domain_textbooks = []

    for label, domain_query in _pg_domain_search_queries(question):
        query_embedding = embed_query(domain_query)
        if not query_embedding:
            continue

        raw_candidates = _vector_search_candidates(
            query_embedding,
            PG_DOMAIN_VECTOR_CANDIDATE_LIMIT,
            PG_DOMAIN_VECTOR_NUM_CANDIDATES,
        )
        reranked = _attach_metadata_and_rerank(question, raw_candidates, "pg_student")

        # Domain augmentation is textbook-only. This is deliberate: the normal
        # base query remains solely responsible for guideline authority/topic tiers.
        # It prevents a targeted physiology/pharmacology query from weakening or
        # reshaping the established guideline guard.
        textbooks = [
            chunk for chunk in reranked
            if _candidate_kind(chunk) == "textbook"
            and not _has_explicit_population_conflict(question, chunk)
        ]
        domain_textbooks.append((label, textbooks))

    merged = _merge_pg_reranked_textbooks(base_reranked, domain_textbooks)
    added = max(0, len(merged) - len(base_reranked))
    tagged = sum(1 for chunk in merged if chunk.get("pg_domain_origins"))
    return merged, domain_textbooks, added, tagged


# -----------------------------------------------------------------------------
# Candidate reranking
# -----------------------------------------------------------------------------
def _priority_adjustment(question: str, meta: dict, assistant_mode: str) -> float:
    """
    Metadata adjustment layered over vector similarity.

    Vector relevance remains the base signal. Mode-aware authority bonuses decide
    which otherwise-relevant chunks should reach the final 30-40 chunk context.
    """
    mode = normalize_assistant_mode(assistant_mode)
    intent = _query_intent(question)

    guideline = _is_guideline(meta)
    textbook = _is_textbook(meta)
    current = _is_current(meta)
    country = str(meta.get("country", "")).strip().lower()
    year = _publication_year(meta)

    adjustment = 0.0

    if mode == "doctor":
        if guideline:
            adjustment += 0.090
            if current:
                adjustment += 0.025

            if year is not None:
                if year >= 2024:
                    adjustment += 0.025
                elif year >= 2020:
                    adjustment += 0.012

            if _query_prefers_indian_guidance(question) and country == "india":
                adjustment += 0.035

            if _query_prefers_indian_guidance(question) and _is_indian_national_guidance(meta):
                adjustment += 0.035

        elif textbook:
            adjustment -= 0.020

            # Doctor mode should use textbooks as supporting background, not the
            # primary source for current thresholds/treatment algorithms.
            if intent in {"management", "mixed"}:
                adjustment -= 0.015

    else:  # PG Student Mode
        if guideline:
            adjustment += 0.060
            if current:
                adjustment += 0.020

            if year is not None:
                if year >= 2024:
                    adjustment += 0.018
                elif year >= 2020:
                    adjustment += 0.010

            # Guidelines still lead treatment/threshold decisions in PG mode.
            if intent in {"management", "mixed"}:
                adjustment += 0.020

            if _query_prefers_indian_guidance(question) and country == "india":
                adjustment += 0.025

            if _query_prefers_indian_guidance(question) and _is_indian_national_guidance(meta):
                adjustment += 0.020

        elif textbook:
            # Textbooks get a positive role in PG mode for mechanisms, reasoning,
            # conceptual background and exam-relevant explanation.
            adjustment += 0.015
            if intent in {"educational", "mixed", "general"}:
                adjustment += 0.025

            # Purposeful evidence mixture: reward a textbook when its curated
            # metadata matches a learning domain explicitly requested by the PG.
            requested_roles = _pg_requested_textbook_roles(question)
            source_roles = _textbook_learning_roles(meta)
            matched_roles = requested_roles & source_roles
            if matched_roles:
                adjustment += min(0.045, 0.022 + 0.010 * len(matched_roles))

    # Curated metadata relevance is intentionally modest: it breaks ties in favor
    # of sources whose title/topic/specialty matches the actual query, but vector
    # similarity remains the dominant signal.
    adjustment += _metadata_topic_bonus(question, meta)

    # Registry notes may explicitly mark a newer topic-specific document as the
    # preferred/superseding source. Honor that curated signal without hard filters.
    if guideline and _has_explicit_precedence_note(meta):
        adjustment += 0.030 if mode == "doctor" else 0.020

    return adjustment


def _attach_metadata_and_rerank(question: str, candidates: list, assistant_mode: str) -> list:
    metadata_by_id = _load_source_metadata()
    reranked = []

    for candidate in candidates:
        item = dict(candidate)
        meta = metadata_by_id.get(item.get("source_id"), {})

        vector_score = float(item.get("score", 0.0) or 0.0)
        priority_adjustment = _priority_adjustment(question, meta, assistant_mode)
        retrieval_score = vector_score + priority_adjustment

        item.update(
            {
                # IMPORTANT: preserve score as raw vector similarity for
                # generator.py confidence / accuracy compatibility.
                "score": vector_score,
                "vector_score": vector_score,
                "priority_adjustment": round(priority_adjustment, 6),
                "retrieval_score": round(retrieval_score, 6),
                "official_title": meta.get("official_title", ""),
                "organization": meta.get("organization", ""),
                "publication_year": meta.get("publication_year", ""),
                "source_type": meta.get("source_type", ""),
                "specialty": meta.get("specialty", ""),
                "topic": meta.get("topic", ""),
                "country": meta.get("country", ""),
                "is_current": meta.get("is_current", ""),
                "notes": meta.get("notes", ""),
            }
        )
        reranked.append(item)

    reranked.sort(
        key=lambda c: (c.get("retrieval_score", 0.0), c.get("vector_score", 0.0)),
        reverse=True,
    )
    return reranked


# -----------------------------------------------------------------------------
# Deduplication / diversification
# -----------------------------------------------------------------------------
def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _token_signature(text: str) -> set:
    # 180 terms is sufficient to identify near-duplicate chunks while remaining
    # cheap for a candidate pool of only a few hundred chunks.
    return set(_normalized_text(text).split()[:180])


def _is_near_duplicate(signature: set, selected_signatures: list) -> bool:
    if not signature:
        return True

    for previous in selected_signatures:
        if not previous:
            continue
        intersection = len(signature & previous)
        smaller = min(len(signature), len(previous))
        if smaller and (intersection / smaller) >= 0.84:
            return True
    return False


def _source_cap(chunk: dict, assistant_mode: str) -> int:
    mode = normalize_assistant_mode(assistant_mode)
    source_type = str(chunk.get("source_type", "")).strip().lower()
    topic_tier = str(chunk.get("topic_tier", "primary")).strip().lower()

    if mode == "doctor":
        if source_type in GUIDELINE_SOURCE_TYPES:
            if topic_tier == "supplementary":
                return DOCTOR_SUPPLEMENTARY_GUIDELINE_CAP
            return DOCTOR_PRIMARY_GUIDELINE_CAP
        if source_type == "textbook":
            return DOCTOR_TEXTBOOK_PER_SOURCE_CAP
        return DOCTOR_OTHER_PER_SOURCE_CAP

    if source_type in GUIDELINE_SOURCE_TYPES:
        if topic_tier == "supplementary":
            return PG_SUPPLEMENTARY_GUIDELINE_CAP
        return PG_PRIMARY_GUIDELINE_CAP
    if source_type == "textbook":
        return PG_TEXTBOOK_PER_SOURCE_CAP
    return PG_OTHER_PER_SOURCE_CAP


def _candidate_kind(chunk: dict) -> str:
    source_type = str(chunk.get("source_type", "")).strip().lower()
    if source_type in GUIDELINE_SOURCE_TYPES:
        return "guideline"
    if source_type == "textbook":
        return "textbook"
    return "other"


def _is_relevant_enough_for_reserved_slot(chunk: dict, best_vector_score: float) -> bool:
    """
    Do not force weak category matches merely to satisfy a mode quota.

    Reserved guideline/textbook slots are filled only when their raw vector
    similarity is reasonably close to the best vector match for this query.
    """
    score = float(chunk.get("vector_score", 0.0) or 0.0)
    floor = max(0.48, best_vector_score - 0.18)
    return score >= floor


def _pg_textbook_effective_score(chunk: dict) -> float:
    """Use domain-query similarity for PG textbook selection without overwriting raw score."""
    return max(
        float(chunk.get("vector_score", 0.0) or 0.0),
        float(chunk.get("pg_domain_score", 0.0) or 0.0),
    )


def _is_relevant_enough_pg_textbook(chunk: dict, best_vector_score: float) -> bool:
    score = _pg_textbook_effective_score(chunk)
    floor = max(0.48, best_vector_score - 0.20)
    return score >= floor


def _try_add_chunk(
    chunk: dict,
    selected: list,
    selected_ids: set,
    selected_signatures: list,
    source_counts: Counter,
    source_page_counts: Counter,
    assistant_mode: str,
    textbook_ceiling: int,
    textbook_count: int,
    source_cap_override: int = None,
    page_range_cap_override: int = None,
) -> tuple:
    chunk_id = chunk.get("chunk_id")
    if chunk_id and chunk_id in selected_ids:
        return False, textbook_count

    source_id = chunk.get("source_id") or chunk.get("book") or "unknown"
    page_start = chunk.get("page_start", chunk.get("page"))
    page_end = chunk.get("page_end", page_start)
    source_page_key = (source_id, page_start, page_end)
    kind = _candidate_kind(chunk)

    source_cap = (
        int(source_cap_override)
        if source_cap_override is not None
        else _source_cap(chunk, assistant_mode)
    )
    page_range_cap = (
        int(page_range_cap_override)
        if page_range_cap_override is not None
        else DIVERSIFIED_PAGE_RANGE_CAP
    )

    if source_counts[source_id] >= source_cap:
        return False, textbook_count
    if source_page_counts[source_page_key] >= page_range_cap:
        return False, textbook_count
    if kind == "textbook" and textbook_count >= textbook_ceiling:
        return False, textbook_count

    signature = _token_signature(chunk.get("text", ""))
    if _is_near_duplicate(signature, selected_signatures):
        return False, textbook_count

    selected.append(chunk)
    if chunk_id:
        selected_ids.add(chunk_id)
    selected_signatures.append(signature)
    source_counts[source_id] += 1
    source_page_counts[source_page_key] += 1

    if kind == "textbook":
        textbook_count += 1

    return True, textbook_count


def _diversify_candidates(question: str, reranked: list, final_k: int, assistant_mode: str) -> list:
    mode = normalize_assistant_mode(assistant_mode)
    intent = _query_intent(question)

    selected = []
    selected_ids = set()
    selected_signatures = []
    source_counts = Counter()
    source_page_counts = Counter()
    textbook_count = 0

    if not reranked:
        return selected

    best_vector_score = max(float(c.get("vector_score", 0.0) or 0.0) for c in reranked)

    # Mode-aware context composition.
    if mode == "doctor":
        # Doctor Mode: guideline-heavy. Textbooks can support but cannot dominate.
        textbook_ceiling = max(5, round(final_k * 0.20))  # ~7 of 36
        guideline_target = round(final_k * (0.78 if intent in {"management", "mixed"} else 0.70))

        # Pass 1: reserve a strong guideline core when sufficiently relevant.
        for chunk in reranked:
            if len(selected) >= min(guideline_target, final_k):
                break
            if _candidate_kind(chunk) != "guideline":
                continue
            if not _is_relevant_enough_for_reserved_slot(chunk, best_vector_score):
                continue
            _, textbook_count = _try_add_chunk(
                chunk, selected, selected_ids, selected_signatures,
                source_counts, source_page_counts, mode,
                textbook_ceiling, textbook_count,
            )

        # Pass 2: fill by overall reranked relevance with strict source/textbook caps.
        for chunk in reranked:
            if len(selected) >= final_k:
                break
            _, textbook_count = _try_add_chunk(
                chunk, selected, selected_ids, selected_signatures,
                source_counts, source_page_counts, mode,
                textbook_ceiling, textbook_count,
            )

    else:
        # PG Student Mode: adaptive, domain-aware evidence composition. Current
        # guidelines still control management, while requested textbook domains
        # receive small relevance-gated reservations.
        depth = classify_pg_query_depth(question)
        requested_roles = _pg_requested_textbook_roles(question)
        textbook_target, textbook_ceiling = _pg_textbook_budget(final_k, intent, depth)
        supplementary_total_cap = PG_SUPPLEMENTARY_TOTAL_CAP.get(depth, 7)
        primary_source_cap = PG_PRIMARY_SOURCE_CAP_BY_DEPTH.get(depth, PG_PRIMARY_GUIDELINE_CAP)

        # Pass 1: reserve the explicitly requested textbook learning roles.
        # Example: a comprehensive CKD question can deliberately recover Guyton
        # (physiology), Robbins (pathology), KDT (pharmacology), and clinical
        # medicine texts without forcing any weak match.
        for role in ("clinical_medicine", "pathology", "physiology", "pharmacology"):
            if role not in requested_roles:
                continue
            role_target = _pg_role_target(depth, role)
            role_count = 0
            for chunk in reranked:
                if role_count >= role_target or len(selected) >= final_k:
                    break
                if _candidate_kind(chunk) != "textbook":
                    continue
                if role not in _textbook_learning_roles(chunk):
                    continue
                if not _is_relevant_enough_pg_textbook(chunk, best_vector_score):
                    continue
                added, textbook_count = _try_add_chunk(
                    chunk, selected, selected_ids, selected_signatures,
                    source_counts, source_page_counts, mode,
                    textbook_ceiling, textbook_count,
                )
                if added:
                    role_count += 1

        # Pass 2: fill the remaining textbook target by overall relevance. Per-source
        # caps keep this spread across API/Harrison/Davidson/Robbins/Guyton/KDT/etc.
        for chunk in reranked:
            if textbook_count >= textbook_target or len(selected) >= final_k:
                break
            if _candidate_kind(chunk) != "textbook":
                continue
            if not _is_relevant_enough_pg_textbook(chunk, best_vector_score):
                continue
            _, textbook_count = _try_add_chunk(
                chunk, selected, selected_ids, selected_signatures,
                source_counts, source_page_counts, mode,
                textbook_ceiling, textbook_count,
            )

        # Pass 3: reserve the guideline core. Primary-topic guidance is considered
        # first and can go slightly deeper for detailed/comprehensive PG questions.
        guideline_target = round(final_k * (0.50 if intent in {"management", "mixed"} else 0.42))
        guideline_count = 0
        supplementary_count = 0

        for chunk in selected:
            if _candidate_kind(chunk) == "guideline":
                guideline_count += 1
                if str(chunk.get("topic_tier", "")).lower() == "supplementary":
                    supplementary_count += 1

        for chunk in reranked:
            if guideline_count >= guideline_target or len(selected) >= final_k:
                break
            if _candidate_kind(chunk) != "guideline":
                continue
            if str(chunk.get("topic_tier", "")).lower() != "primary":
                continue
            if not _is_relevant_enough_for_reserved_slot(chunk, best_vector_score):
                continue
            added, textbook_count = _try_add_chunk(
                chunk, selected, selected_ids, selected_signatures,
                source_counts, source_page_counts, mode,
                textbook_ceiling, textbook_count,
                source_cap_override=primary_source_cap,
            )
            if added:
                guideline_count += 1

        # Pass 4: add relevant supplementary guideline evidence, but cap the total
        # so CKD-anemia/BP/diabetes/etc. cannot consume all educational slots.
        for chunk in reranked:
            if guideline_count >= guideline_target or len(selected) >= final_k:
                break
            if supplementary_count >= supplementary_total_cap:
                break
            if _candidate_kind(chunk) != "guideline":
                continue
            if str(chunk.get("topic_tier", "")).lower() != "supplementary":
                continue
            if not _is_relevant_enough_for_reserved_slot(chunk, best_vector_score):
                continue
            added, textbook_count = _try_add_chunk(
                chunk, selected, selected_ids, selected_signatures,
                source_counts, source_page_counts, mode,
                textbook_ceiling, textbook_count,
            )
            if added:
                guideline_count += 1
                supplementary_count += 1

        # Pass 5: fill remaining slots by overall relevance while preserving the
        # textbook ceiling and supplementary-guideline ceiling.
        for chunk in reranked:
            if len(selected) >= final_k:
                break
            kind = _candidate_kind(chunk)
            tier = str(chunk.get("topic_tier", "")).lower()
            if kind == "guideline" and tier == "supplementary":
                if supplementary_count >= supplementary_total_cap:
                    continue
            added, textbook_count = _try_add_chunk(
                chunk, selected, selected_ids, selected_signatures,
                source_counts, source_page_counts, mode,
                textbook_ceiling, textbook_count,
                source_cap_override=(primary_source_cap if kind == "guideline" and tier == "primary" else None),
            )
            if added and kind == "guideline" and tier == "supplementary":
                supplementary_count += 1

        # Pass 6: high-depth safe refill. Detailed/comprehensive PG questions can
        # deepen sources that have ALREADY proved relevant, so the 45/50 budgets
        # are achievable without reopening rejected guideline neighbors.
        if depth in {"detailed", "comprehensive"} and len(selected) < final_k:
            proven_source_ids = {
                c.get("source_id") or c.get("book")
                for c in selected
                if (c.get("source_id") or c.get("book"))
            }
            refill_primary_cap = PG_HIGH_DEPTH_PRIMARY_REFILL_CAP[depth]
            refill_textbook_source_cap = PG_HIGH_DEPTH_TEXTBOOK_PER_SOURCE_CAP[depth]
            refill_textbook_ceiling = PG_HIGH_DEPTH_TEXTBOOK_CEILING[depth]

            for chunk in reranked:
                if len(selected) >= final_k:
                    break

                source_id = chunk.get("source_id") or chunk.get("book")
                if not source_id or source_id not in proven_source_ids:
                    continue

                kind = _candidate_kind(chunk)
                tier = str(chunk.get("topic_tier", "")).lower()

                if kind == "guideline":
                    if tier != "primary":
                        continue
                    if not _is_relevant_enough_for_reserved_slot(chunk, best_vector_score):
                        continue
                    source_override = refill_primary_cap
                    ceiling_for_add = refill_textbook_ceiling

                elif kind == "textbook":
                    source_roles = _textbook_learning_roles(chunk)
                    if requested_roles and not (source_roles & requested_roles):
                        continue
                    if not _is_relevant_enough_pg_textbook(chunk, best_vector_score):
                        continue
                    source_override = refill_textbook_source_cap
                    ceiling_for_add = refill_textbook_ceiling

                else:
                    continue

                _, textbook_count = _try_add_chunk(
                    chunk, selected, selected_ids, selected_signatures,
                    source_counts, source_page_counts, mode,
                    ceiling_for_add, textbook_count,
                    source_cap_override=source_override,
                    page_range_cap_override=PG_HIGH_DEPTH_PAGE_RANGE_CAP,
                )

    # Safe focused-topic refill: if strict filtering leaves fewer than the normal
    # 30-chunk target, deepen only already-proven PRIMARY guideline sources. This
    # never reopens rejected/supplementary disease neighbors just to satisfy quota.
    if len(selected) < min(MIN_FINAL_K, final_k):
        primary_refill_cap = (
            DOCTOR_PRIMARY_REFILL_PER_SOURCE_CAP
            if mode == "doctor"
            else PG_PRIMARY_REFILL_PER_SOURCE_CAP
        )
        for chunk in reranked:
            if len(selected) >= min(MIN_FINAL_K, final_k):
                break
            if _candidate_kind(chunk) != "guideline":
                continue
            if str(chunk.get("topic_tier", "")).lower() != "primary":
                continue
            if not _is_relevant_enough_for_reserved_slot(chunk, best_vector_score):
                continue
            _, textbook_count = _try_add_chunk(
                chunk, selected, selected_ids, selected_signatures,
                source_counts, source_page_counts, mode,
                textbook_ceiling, textbook_count,
                source_cap_override=primary_refill_cap,
                page_range_cap_override=PRIMARY_REFILL_PAGE_RANGE_CAP,
            )

    # Keep normal source caps strict; only the explicit primary-only refill above
    # may deepen a strongly matched guideline when the context would otherwise be
    # underfilled. Re-sort final context by retrieval quality.
    selected.sort(
        key=lambda c: (c.get("retrieval_score", 0.0), c.get("vector_score", 0.0)),
        reverse=True,
    )
    return selected[:final_k]


# -----------------------------------------------------------------------------
# Public retrieval function
# -----------------------------------------------------------------------------
def search_medical_chunks(
    question: str,
    top_k: int = DEFAULT_FINAL_K,
    assistant_mode: str = "doctor",
):
    """
    Retrieval v2.7 policies:

    A. Explicit source/organization request
       -> strict source-constrained retrieval only.

    B. Explicit latest/current/newest/recent guidance request
       -> newest current applicable guideline source(s) only.

    C. Otherwise
       -> normal mode-aware diversified retrieval.

    assistant_mode:
      "doctor"     = guideline-heavy, India-first for management when appropriate
      "pg_student" = balanced guideline + textbook context

    Backward compatibility:
    - Existing callers passing only (question, top_k) continue to work.
    - Until app.py is wired to send the UI mode, the safe default is Doctor Mode.
    - Calls requesting 12/15 chunks are upgraded to the v2 default context of 36.

    Important:
    - Doctor normal target remains 30-40; PG can request up to 50.
    - Strict source/latest retrieval never pads with unrelated sources merely to
      reach the target; it may therefore return fewer chunks.
    """
    mode = normalize_assistant_mode(assistant_mode)

    query_embedding = embed_query(question)
    if not query_embedding:
        print("Failed to embed query")
        return []

    try:
        requested_k = int(top_k)
    except (TypeError, ValueError):
        requested_k = DEFAULT_FINAL_K

    mode_max_k = PG_MAX_FINAL_K if mode == "pg_student" else MAX_FINAL_K
    final_k = max(MIN_FINAL_K, min(requested_k, mode_max_k))
    if requested_k < MIN_FINAL_K:
        final_k = DEFAULT_FINAL_K

    source_constraint = _resolve_explicit_source_constraint(question)
    wants_fresh = _wants_fresh_guidance(question)
    comparison = _is_comparison_request(question)

    # A broad PG teaching request may mention "current guidelines" as one domain
    # among many. In that case, keep diversified retrieval so current guidance can
    # coexist with physiology/pathology/pharmacology textbooks. Explicit named
    # sources still remain strict, and focused freshness-only questions remain strict.
    broad_pg_learning = mode == "pg_student" and _is_broad_pg_learning_request(question)
    auto_strict_freshness = wants_fresh and not comparison and not broad_pg_learning

    # ------------------------------------------------------------------
    # Policy A/B: explicit source lock and/or strict freshness
    # ------------------------------------------------------------------
    if source_constraint is not None or auto_strict_freshness:
        try:
            candidates = _vector_search_candidates(
                query_embedding,
                CONSTRAINED_VECTOR_CANDIDATE_LIMIT,
                CONSTRAINED_VECTOR_NUM_CANDIDATES,
            )
            reranked = _attach_metadata_and_rerank(question, candidates, mode)

            allowed_ids = None
            if source_constraint is not None:
                allowed_ids = set(source_constraint.get("allowed_ids", set()))

                # Explicit source + explicit year is fully strict. If the exact
                # requested edition/year is absent, return no substitute source.
                if not allowed_ids:
                    years = sorted(source_constraint.get("requested_years", set()))
                    year_text = ",".join(str(y) for y in years) if years else "any"
                    print(
                        "Retrieval v2 | policy=source_locked | "
                        f"source={source_constraint.get('label', 'requested')} | "
                        f"year={year_text} | no matching source in medical_sources"
                    )
                    return []

            requested_years = (
                source_constraint.get("requested_years", set())
                if source_constraint is not None
                else set()
            )

            # Explicit year wins over "latest/current" language. Example:
            # "According to KDIGO 2024..." must stay on 2024.
            if source_constraint is not None and requested_years:
                chosen_source_ids = _choose_relevant_source_ids(
                    question,
                    reranked,
                    allowed_ids=allowed_ids,
                    max_sources=MAX_CONSTRAINED_SOURCES,
                )
                policy = "source_locked"

            elif wants_fresh and not comparison:
                chosen_source_ids, newest_year = _choose_freshest_applicable_source_ids(
                    question,
                    reranked,
                    allowed_ids=allowed_ids,
                )
                policy = "source_locked_latest" if source_constraint is not None else "freshest_guidance"

                if not chosen_source_ids:
                    # If a named source has no current guideline metadata, keep
                    # the source lock rather than leaking to unrelated sources.
                    if source_constraint is not None:
                        chosen_source_ids = _choose_relevant_source_ids(
                            question,
                            reranked,
                            allowed_ids=allowed_ids,
                            max_sources=MAX_CONSTRAINED_SOURCES,
                        )
                        newest_year = None
                    else:
                        print(
                            "Retrieval v2 | policy=freshest_guidance | "
                            "no current applicable guideline candidates found"
                        )
                        return []

            else:
                # Explicit source request without freshness: lock to the named
                # source family, then choose only the relevant source(s) within it.
                chosen_source_ids = _choose_relevant_source_ids(
                    question,
                    reranked,
                    allowed_ids=allowed_ids,
                    max_sources=MAX_CONSTRAINED_SOURCES,
                )
                newest_year = None
                policy = "source_locked"

            chosen_set = set(chosen_source_ids)
            selected = _select_constrained_chunks(reranked, chosen_set, final_k)

            metadata_by_id = _load_source_metadata()
            selected_labels = [
                metadata_by_id.get(source_id, {}).get("official_title", source_id)
                for source_id in chosen_source_ids
            ]

            year_label = ""
            if requested_years:
                year_label = ",".join(str(y) for y in sorted(requested_years))
            elif 'newest_year' in locals() and newest_year is not None:
                year_label = str(newest_year)

            print(
                "Retrieval v2 | "
                f"policy={policy} | mode={mode} | "
                f"source={source_constraint.get('label') if source_constraint else 'auto'} | "
                f"year={year_label or 'auto'} | "
                f"{len(candidates)} candidates -> {len(selected)} final | "
                f"selected_sources={len(chosen_source_ids)}"
            )
            if selected_labels:
                print("  Locked sources: " + " | ".join(selected_labels))

            return selected

        except Exception as e:
            print(f"Search error: {type(e).__name__}: {e}")
            raise

    # ------------------------------------------------------------------
    # Policy C: normal mode-aware diversified retrieval
    # ------------------------------------------------------------------
    candidate_limit = (
        CONSTRAINED_VECTOR_CANDIDATE_LIMIT if comparison
        else max(VECTOR_CANDIDATE_LIMIT, final_k * 5)
    )

    try:
        base_candidates = _vector_search_candidates(
            query_embedding,
            candidate_limit,
            VECTOR_NUM_CANDIDATES,
        )
        reranked = _attach_metadata_and_rerank(question, base_candidates, mode)
        comparison_candidates, comparison_sources = _comparison_guideline_pairs(question, reranked)
        reranked, filtered_guideline_sources = _apply_diversified_source_relevance_guard(
            question, reranked
        )

        pg_domain_pools = []
        pg_domain_added = 0
        pg_domain_tagged = 0
        if mode == "pg_student":
            reranked, pg_domain_pools, pg_domain_added, pg_domain_tagged = _augment_pg_reranked_textbooks(
                question, reranked
            )

        selected = _diversify_candidates(question, reranked, final_k, mode)
        if comparison_candidates:
            selected = _ensure_comparison_pair(selected, comparison_candidates, final_k)
            print(f"  Comparison guideline pair: {comparison_sources}")
        elif comparison:
            print("  Comparison note: no valid dated older/newer guideline pair in candidate pool")

        if selected:
            guideline_count = sum(1 for c in selected if _candidate_kind(c) == "guideline")
            textbook_count = sum(1 for c in selected if _candidate_kind(c) == "textbook")
            other_count = len(selected) - guideline_count - textbook_count
            indian_count = sum(
                1 for c in selected if str(c.get("country", "")).strip().lower() == "india"
            )
            national_india_count = sum(
                1
                for c in selected
                if _is_indian_national_guidance(
                    {
                        "country": c.get("country", ""),
                        "organization": c.get("organization", ""),
                    }
                )
            )

            primary_guideline_count = sum(
                1 for c in selected
                if _candidate_kind(c) == "guideline" and c.get("topic_tier") == "primary"
            )
            supplementary_guideline_count = sum(
                1 for c in selected
                if _candidate_kind(c) == "guideline" and c.get("topic_tier") == "supplementary"
            )

            print(
                "Retrieval v2 | "
                + f"policy=diversified | mode={mode} | intent={_query_intent(question)} | "
                + (f"depth={classify_pg_query_depth(question)} | target={final_k} | " if mode == "pg_student" else "")
                + ("freshness=blended | " if mode == "pg_student" and wants_fresh and broad_pg_learning else "")
                + (
                    f"{len(base_candidates)} base | PG-domain tagged={pg_domain_tagged}, new={pg_domain_added} "
                    if mode == "pg_student" and pg_domain_pools
                    else f"{len(base_candidates)} candidates "
                )
                + f"-> {len(selected)} final | "
                + f"guidelines={guideline_count} "
                f"(primary={primary_guideline_count}, supplementary={supplementary_guideline_count}), "
                f"textbooks={textbook_count}, other={other_count}, "
                f"India={indian_count}, India-national={national_india_count}, "
                f"filtered_guideline_sources={filtered_guideline_sources}"
            )

        return selected

    except Exception as e:
        print(f"Search error: {type(e).__name__}: {e}")
        raise


# -----------------------------------------------------------------------------
# Existing medical/non-medical/emergency helpers (preserved)
# -----------------------------------------------------------------------------
MEDICAL_KEYWORDS = [
    "symptom", "symptoms", "disease", "diseases", "condition", "syndrome", "disorder",
    "infection", "illness", "sick", "pain", "ache", "fever", "cough", "headache", "nausea",
    "heart", "lung", "kidney", "liver", "brain", "blood", "bone", "muscle", "skin", "eye",
    "ear", "throat", "stomach", "intestine", "chest", "back", "joint", "diagnose", "diagnosis",
    "treatment", "treat", "cure", "therapy", "medication", "medicine", "drug", "prescribe",
    "surgery", "operation", "test", "screening", "examination", "diabetes", "cancer",
    "hypertension", "asthma", "arthritis", "pneumonia", "tuberculosis", "hepatitis", "hiv",
    "aids", "stroke", "seizure", "epilepsy", "migraine", "blood pressure", "heart rate",
    "glucose", "cholesterol", "hemoglobin", "hba1c", "doctor", "hospital", "clinic", "health",
    "t2dm", "t1dm", "ckd", "aki", "copd", "gerd", "dvt", "uti", "cad", "chf",
]

NON_MEDICAL_KEYWORDS = [
    "recipe", "cook", "bake", "cooking", "baking", "meal", "dinner", "breakfast",
    "python", "javascript", "code", "programming", "software", "computer", "laptop",
    "movie", "film", "song", "music", "game", "sport", "sports", "celebrity",
    "weather", "vacation", "tourism", "flight", "hotel", "stock", "bitcoin", "crypto",
]


def is_medical_question(question: str) -> str:
    question_lower = question.lower()
    medical_matches = sum(1 for kw in MEDICAL_KEYWORDS if kw in question_lower)
    non_medical_matches = sum(1 for kw in NON_MEDICAL_KEYWORDS if kw in question_lower)

    if non_medical_matches > 0 and medical_matches == 0:
        return "non_medical"
    if medical_matches > 0 and non_medical_matches == 0:
        return "medical"
    return "ambiguous"


def get_non_medical_response() -> str:
    return """I am MediCare AI, specialized in medical knowledge and clinical decision support.

Your question does not appear to be medical in nature.

I can assist you with:
• Symptoms, Differential Diagnoses, and Diseases
• Clinical Guidelines and First-Line Treatment Protocols
• Laboratory Test Interpretation & Reference Ranges
• Pharmacology, Mechanisms of Action, and Drug Interactions
• Live NIH PubMed Research & Clinical Trial Evidence

Please ask a healthcare-related question."""


EMERGENCY_KEYWORDS = [
    "chest pain", "chest tightness", "heart attack", "difficulty breathing",
    "cant breathe", "can't breathe", "shortness of breath", "severe bleeding",
    "unconscious", "passed out", "fainting", "stroke", "face drooping",
    "severe headache", "seizure", "convulsion", "poisoning", "overdose",
    "anaphylaxis", "suicide", "self harm", "vomiting blood", "coughing blood",
]


def check_emergency(question: str) -> bool:
    q_low = question.lower()
    return any(kw in q_low for kw in EMERGENCY_KEYWORDS)


def get_emergency_message() -> str:
    return """
🚨 MEDICAL EMERGENCY ALERT 🚨

You may be describing symptoms of a critical medical emergency.

IMMEDIATE ACTIONS:
1. Call Emergency Services immediately:
   • India: 102 / 108
   • USA: 911 | UK: 999 | EU: 112
2. Alert someone nearby immediately.
3. Do not drive yourself; proceed to the nearest Emergency Department.

MediCare AI provides educational decision-support only and cannot replace immediate emergency care.
"""


if __name__ == "__main__":
    print("Testing Acronym Expansion:")
    sample = "How to manage T2DM with CKD and HTN?"
    print(f"Original: {sample}")
    print(f"Expanded: {expand_medical_acronyms(sample)}")
    print(f"Doctor mode normalized: {normalize_assistant_mode('Doctor Mode')}")
    print(f"PG mode normalized: {normalize_assistant_mode('PG Student Mode')}")
