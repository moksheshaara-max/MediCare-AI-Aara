"""
MediCare AI - Retrieval v2.7 (Mode-Aware + Strict Source Resolution + Topic Tiers + Safe Primary Refill)

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
9. For focused topics, use more distinct chunks from the few strongly relevant
   primary sources instead of re-admitting off-topic sources merely to fill quota.
10. If strict topic filtering leaves fewer than 30 normal-retrieval chunks, refill only
    from already-proven PRIMARY guideline sources using additional non-duplicate chunks.
    Never reopen rejected sources merely to hit quota.
11. Return ~30-40 useful chunks for normal retrieval when enough relevant evidence exists.
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

# Per-source diversity caps. These prevent one broad/large reference from
# dominating final context even when it has many nearby vector matches.
DOCTOR_GUIDELINE_PER_SOURCE_CAP = 7
DOCTOR_TEXTBOOK_PER_SOURCE_CAP = 2
DOCTOR_OTHER_PER_SOURCE_CAP = 2

PG_GUIDELINE_PER_SOURCE_CAP = 6
PG_TEXTBOOK_PER_SOURCE_CAP = 3
PG_OTHER_PER_SOURCE_CAP = 2

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
    r"\bchf\b": "Congestive Heart Failure",
    r"\bhf\b": "Heart Failure",
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

    metadata = {}
    for doc in sources_collection.find({}, projection):
        source_id = doc.get("source_id")
        if source_id:
            metadata[source_id] = doc

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

    guideline_items = []
    for source_id, item in stats.items():
        meta = metadata_by_id.get(source_id, {})
        if not _is_guideline(meta):
            continue
        if _has_explicit_population_conflict(question, meta):
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

    query_adult = bool(re.search(r"\badults?\b", q))
    query_paediatric = bool(re.search(
        r"\b(?:paediatric|pediatric|child|children|infant|infants|newborn|newborns|neonate|neonates)\b",
        q,
    ))

    source_paediatric = bool(re.search(
        r"\b(?:paediatric|pediatric|paediatrics|pediatrics|child|children|neonatal|neonatology)\b",
        blob,
    ))
    source_adult = bool(re.search(r"\badults?\b", blob))

    query_pregnancy = bool(re.search(
        r"\b(?:pregnan(?:t|cy)|gestational|antenatal|maternal|obstetric|obstetrics)\b",
        q,
    ))
    source_pregnancy = bool(re.search(
        r"\b(?:pregnan(?:t|cy)|gestational|antenatal|maternal|obstetric|obstetrics)\b",
        blob,
    ))

    if query_adult and source_paediatric and not query_paediatric:
        return True
    if query_paediatric and source_adult and not query_adult:
        return True
    if query_paediatric and source_pregnancy and not query_pregnancy:
        return True
    if query_pregnancy and source_paediatric and not query_paediatric:
        return True

    # Population-specific sources should not fill generic questions unless the
    # user actually asked for that population.
    if source_paediatric and not query_paediatric:
        return True
    if source_pregnancy and not query_pregnancy:
        return True

    # Active-disease treatment questions should not be padded with prevention/
    # prophylaxis programme guidance unless prevention was actually requested.
    query_prevention = bool(re.search(
        r"\b(?:prevent|prevention|preventive|prophylaxis|prophylactic|screen|screening|vaccin(?:e|ation))\b",
        q,
    ))
    source_prevention = bool(re.search(
        r"\b(?:prevent|prevention|preventive|prophylaxis|prophylactic|screen|screening|vaccin(?:e|ation))\b",
        blob,
    ))
    query_active_management = bool(re.search(
        r"\b(?:manage|managed|management|treat|treated|treatment|therapy|therapeutic)\b",
        q,
    ))
    if query_active_management and source_prevention and not query_prevention:
        return True

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

    for source_id, source_score in source_best.items():
        meta = metadata_by_id.get(source_id, {})

        if _has_explicit_population_conflict(question, meta):
            population_conflict_sources.add(source_id)
            if _is_guideline(meta):
                rejected_guideline_sources.add(source_id)
            continue

        if not _is_guideline(meta):
            continue

        topic_bonus = _metadata_topic_bonus(question, meta)
        strong_semantic = source_score >= (best_vector - DIVERSIFIED_STRONG_SOURCE_MARGIN)
        metadata_supported = (
            topic_bonus > 0.0
            and source_score >= (best_vector - DIVERSIFIED_METADATA_SOURCE_MARGIN)
        )
        broad_source_supported = (
            _is_broad_registry_source(meta)
            and source_score >= (best_vector - DIVERSIFIED_BROAD_SOURCE_MARGIN)
        )

        if strong_semantic or metadata_supported or broad_source_supported:
            allowed_guideline_sources.add(source_id)
        else:
            rejected_guideline_sources.add(source_id)

    prelim = []
    for chunk in reranked:
        source_id = chunk.get("source_id")
        meta = metadata_by_id.get(source_id, {})
        if source_id in population_conflict_sources:
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
    aliases = _detect_source_aliases(question)

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
    filtered = [
        c for c in candidates
        if not allowed_ids or c.get("source_id") in allowed_ids
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
      2. topical/vector applicability
      3. India-national preference for India-oriented management
      4. newest known publication year
      5. strongest source relevance within that newest year

    Undated sources are not allowed to displace a dated current guideline when a
    clearly applicable dated source exists.
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
        filtered.append(chunk)

    stats = _source_stats(filtered)
    if not stats:
        return [], None

    ranked = sorted(
        stats.values(),
        key=lambda item: (
            item["max_vector_score"],
            _metadata_topic_bonus(question, item["meta"]),
            item["max_retrieval_score"],
        ),
        reverse=True,
    )

    best_vector = ranked[0]["max_vector_score"]
    applicable = [
        item
        for item in ranked
        if item["max_vector_score"] >= (best_vector - SOURCE_RELEVANCE_MARGIN)
    ]
    if not applicable:
        applicable = [ranked[0]]

    # For management/threshold questions in this India-oriented CDS, choose the
    # latest relevant Indian national guidance when such guidance exists.
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
    return list(collection.aggregate(pipeline))


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
        # PG Student Mode: management remains guideline-led, but textbook depth is
        # intentionally reserved for mechanism/background/differential reasoning.
        if intent == "educational":
            textbook_target = round(final_k * 0.39)  # ~14 of 36
            textbook_ceiling = min(15, round(final_k * 0.42))
        elif intent == "management":
            textbook_target = round(final_k * 0.28)  # ~10 of 36
            textbook_ceiling = min(12, round(final_k * 0.34))
        else:  # mixed/general
            textbook_target = round(final_k * 0.33)  # ~12 of 36
            textbook_ceiling = min(14, round(final_k * 0.39))

        # Pass 1: reserve relevant textbook depth. Per-source caps ensure this is
        # spread across multiple textbooks rather than Harrison dominating.
        for chunk in reranked:
            if textbook_count >= textbook_target or len(selected) >= final_k:
                break
            if _candidate_kind(chunk) != "textbook":
                continue
            if not _is_relevant_enough_for_reserved_slot(chunk, best_vector_score):
                continue
            _, textbook_count = _try_add_chunk(
                chunk, selected, selected_ids, selected_signatures,
                source_counts, source_page_counts, mode,
                textbook_ceiling, textbook_count,
            )

        # Pass 2: reserve current guideline evidence for treatment/thresholds.
        guideline_target = round(final_k * (0.58 if intent in {"management", "mixed"} else 0.50))
        guideline_count = 0

        for chunk in selected:
            if _candidate_kind(chunk) == "guideline":
                guideline_count += 1

        for chunk in reranked:
            if guideline_count >= guideline_target or len(selected) >= final_k:
                break
            if _candidate_kind(chunk) != "guideline":
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

        # Pass 3: fill remaining slots by overall reranked relevance.
        for chunk in reranked:
            if len(selected) >= final_k:
                break
            _, textbook_count = _try_add_chunk(
                chunk, selected, selected_ids, selected_signatures,
                source_counts, source_page_counts, mode,
                textbook_ceiling, textbook_count,
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
    - The 30-40 target applies to normal diversified retrieval.
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

    final_k = max(MIN_FINAL_K, min(requested_k, MAX_FINAL_K))
    if requested_k < MIN_FINAL_K:
        final_k = DEFAULT_FINAL_K

    source_constraint = _resolve_explicit_source_constraint(question)
    wants_fresh = _wants_fresh_guidance(question)
    comparison = _is_comparison_request(question)

    # ------------------------------------------------------------------
    # Policy A/B: explicit source lock and/or strict freshness
    # ------------------------------------------------------------------
    if source_constraint is not None or (wants_fresh and not comparison):
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
            print(f"Search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Policy C: normal mode-aware diversified retrieval
    # ------------------------------------------------------------------
    candidate_limit = max(VECTOR_CANDIDATE_LIMIT, final_k * 5)

    try:
        candidates = _vector_search_candidates(
            query_embedding,
            candidate_limit,
            VECTOR_NUM_CANDIDATES,
        )
        reranked = _attach_metadata_and_rerank(question, candidates, mode)
        reranked, filtered_guideline_sources = _apply_diversified_source_relevance_guard(
            question, reranked
        )
        selected = _diversify_candidates(question, reranked, final_k, mode)

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
                f"policy=diversified | mode={mode} | intent={_query_intent(question)} | "
                f"{len(candidates)} candidates -> {len(selected)} final | "
                f"guidelines={guideline_count} "
                f"(primary={primary_guideline_count}, supplementary={supplementary_guideline_count}), "
                f"textbooks={textbook_count}, other={other_count}, "
                f"India={indian_count}, India-national={national_india_count}, "
                f"filtered_guideline_sources={filtered_guideline_sources}"
            )

        return selected

    except Exception as e:
        print(f"Search error: {e}")
        return []


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
