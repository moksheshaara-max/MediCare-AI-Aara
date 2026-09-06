"""
MediCare AI - Retrieval Module
Includes Medical Acronym Expansion, Query Enrichment, and MongoDB Vector Search.
"""

import sys
import os
import re

# Ensure backend root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=env_path)
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "medicare_rag"
COLLECTION_NAME = "medical_chunks"
VECTOR_INDEX_NAME = "vector_index"
EMBEDDING_DIMENSIONS = 768

# Initialize Gemini client & MongoDB
client = genai.Client(api_key=GOOGLE_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

# Comprehensive Medical Acronym & Abbreviation Dictionary
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
    r"\begfr\b": "Estimated Glomerular Filtration Rate"
}


def expand_medical_acronyms(query: str) -> str:
    """
    Expand standard medical abbreviations to full clinical terms.
    Improves vector cosine similarity matching against medical textbooks.
    """
    expanded = query
    for pattern, replacement in MEDICAL_ACRONYMS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


def embed_query(question: str):
    """
    Convert query into 768-D embedding vector using lightweight embedder.
    """
    try:
        from embeddings.local_embedder import embed_single_text
        enriched_query = expand_medical_acronyms(question)
        return embed_single_text(enriched_query)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None


# Medical topic indicators
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
    "t2dm", "t1dm", "ckd", "aki", "copd", "gerd", "dvt", "uti", "cad", "chf"
]

NON_MEDICAL_KEYWORDS = [
    "recipe", "cook", "bake", "cooking", "baking", "meal", "dinner", "breakfast",
    "python", "javascript", "code", "programming", "software", "computer", "laptop",
    "movie", "film", "song", "music", "game", "sport", "sports", "celebrity",
    "weather", "vacation", "tourism", "flight", "hotel", "stock", "bitcoin", "crypto"
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


def search_medical_chunks(question: str, top_k: int = 15):
    """
    Search top chunks in MongoDB Vector Search with 200 candidates.
    """
    query_embedding = embed_query(question)
    
    if not query_embedding:
        print("Failed to embed query")
        return []
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 200,
                "limit": top_k
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "book": 1,
                "page": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    try:
        return list(collection.aggregate(pipeline))
    except Exception as e:
        print(f"Search error: {e}")
        return []


EMERGENCY_KEYWORDS = [
    "chest pain", "chest tightness", "heart attack", "difficulty breathing",
    "cant breathe", "can't breathe", "shortness of breath", "severe bleeding",
    "unconscious", "passed out", "fainting", "stroke", "face drooping",
    "severe headache", "seizure", "convulsion", "poisoning", "overdose",
    "anaphylaxis", "suicide", "self harm", "vomiting blood", "coughing blood"
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