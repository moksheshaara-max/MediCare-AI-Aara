"""
MediCare AI - Retrieval Module

Takes a user question and finds the most relevant
medical chunks from MongoDB using Vector Search.
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "medicare_rag"
COLLECTION_NAME = "medical_chunks"
VECTOR_INDEX_NAME = "vector_index"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

# Initialize Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Initialize MongoDB
mongo_client = MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]


def embed_query(question):
    """
    Convert user question into embedding vector using local model.
    """
    try:
        from embeddings.local_embedder import embed_single_text
        return embed_single_text(question)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None
# Medical topic indicators (any of these = likely medical)
MEDICAL_KEYWORDS = [
    # Symptoms/conditions
    "symptom", "symptoms", "disease", "diseases", "condition",
    "syndrome", "disorder", "infection", "illness", "sick",
    "pain", "ache", "fever", "cough", "headache", "nausea",
    
    # Body parts
    "heart", "lung", "kidney", "liver", "brain", "blood",
    "bone", "muscle", "skin", "eye", "ear", "throat",
    "stomach", "intestine", "chest", "back", "joint",
    
    # Medical actions
    "diagnose", "diagnosis", "treatment", "treat", "cure",
    "therapy", "medication", "medicine", "drug", "prescribe",
    "surgery", "operation", "test", "screening", "examination",
    
    # Medical specialties/topics
    "diabetes", "cancer", "hypertension", "asthma", "arthritis",
    "pneumonia", "tuberculosis", "hepatitis", "hiv", "aids",
    "stroke", "seizure", "epilepsy", "migraine",
    
    # Medical measures
    "blood pressure", "heart rate", "temperature", "pulse",
    "glucose", "cholesterol", "hemoglobin", "hba1c",
    
    # Anatomy
    "artery", "vein", "nerve", "cell", "tissue", "organ",
    "hormone", "enzyme", "antibody", "vaccine",
    
    # Medical roles
    "doctor", "physician", "nurse", "patient", "hospital",
    "clinic", "emergency", "medical", "health", "healthcare",
    
    # Common medical terms
    "chronic", "acute", "benign", "malignant", "inflammation",
    "swelling", "bleeding", "fracture", "wound",
]


# Non-medical topic indicators (any of these = NOT medical)
NON_MEDICAL_KEYWORDS = [
    # Food/cooking
    "recipe", "cook", "bake", "cooking", "baking", "meal",
    "dinner", "breakfast", "lunch", "cuisine", "restaurant",
    
    # Technology
    "python", "javascript", "code", "programming", "software",
    "computer", "laptop", "phone", "app", "website", "coding",
    
    # Entertainment
    "movie", "film", "song", "music", "game", "sport", "sports",
    "celebrity", "actor", "singer", "book review", "novel",
    
    # Weather/travel
    "weather", "temperature outside", "climate", "travel",
    "vacation", "tourism", "flight", "hotel",
    
    # Money/business
    "stock", "money", "investment", "bitcoin", "crypto",
    "salary", "job", "career", "resume",
    
    # Random topics
    "cookies", "cake", "pizza", "burger",
    "programming language", "web development",
]


def is_medical_question(question):
    """
    Simple keyword-based check if question is medical.
    
    Returns:
        "medical"     - Definitely medical
        "non_medical" - Definitely NOT medical
        "ambiguous"   - Unclear, needs further checking
    """
    question_lower = question.lower()
    
    # Count matches
    medical_matches = sum(1 for kw in MEDICAL_KEYWORDS if kw in question_lower)
    non_medical_matches = sum(1 for kw in NON_MEDICAL_KEYWORDS if kw in question_lower)
    
    # Clear non-medical
    if non_medical_matches > 0 and medical_matches == 0:
        return "non_medical"
    
    # Clear medical
    if medical_matches > 0 and non_medical_matches == 0:
        return "medical"
    
    # Both or neither - let vector search decide
    return "ambiguous"


def get_non_medical_response():
    """Return polite rejection for non-medical questions."""
    return """I'm MediCare AI, designed to answer medical and healthcare questions.

Your question doesn't appear to be medical in nature.

I can help you with:
• Symptoms and diseases
• Diagnoses and treatments  
• Medications and therapies
• Anatomy and physiology
• Medical procedures
• Health conditions

Please ask me a medical question, and I'll search my medical textbooks for you.

Examples:
• "What are the symptoms of diabetes?"
• "How is pneumonia treated?"
• "What causes chest pain?"
"""

def search_medical_chunks(question, top_k=12):
    """
    Search for top 12 medical chunks using MongoDB Vector Search.
    Evaluates 200 candidate vectors for maximum retrieval precision.
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
                "numCandidates": 200,   # Increased from 100 to evaluate more candidates
                "limit": top_k           # Fetches top 12
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
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []


def display_results(question, results):
    """Pretty print the search results."""
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} relevant chunks:\n")
    
    for i, chunk in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Source     : {chunk['book']}, Page {chunk['page']}")
        print(f"Chunk ID   : {chunk['chunk_id']}")
        print(f"Similarity : {chunk['score']:.4f}")
        print(f"Text preview:")
        print(f"  {chunk['text'][:300]}...")
        print()
# Emergency symptoms - trigger urgent warning
EMERGENCY_KEYWORDS = [
    "chest pain", "chest tightness", "heart attack",
    "difficulty breathing", "cant breathe", "can't breathe",
    "shortness of breath", "trouble breathing",
    "severe bleeding", "heavy bleeding", "bleeding won't stop",
    "unconscious", "passed out", "fainting",
    "stroke", "face drooping", "arm weakness",
    "severe headache", "worst headache",
    "seizure", "convulsion", "fit",
    "poisoning", "overdose",
    "severe allergic reaction", "anaphylaxis",
    "suicide", "self harm", "kill myself",
    "severe abdominal pain", "severe stomach pain",
    "vomiting blood", "coughing blood",
]


def check_emergency(question):
    """
    Check if question mentions emergency symptoms.
    
    Returns:
        True if emergency detected, False otherwise
    """
    question_lower = question.lower()
    
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in question_lower:
            return True
    
    return False


def get_emergency_message():
    """Return the emergency warning message."""
    return """
🚨 MEDICAL EMERGENCY ALERT 🚨

You may be describing a medical emergency.

IMMEDIATE ACTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Call emergency services NOW:
   • India: 102 (Ambulance) or 108 (Emergency)
   • USA: 911
   • UK: 999
   • EU: 112

2. If someone is with you, tell them immediately.

3. Do NOT wait for online information.

4. Go to the nearest emergency room if possible.

This app is for information only.
Emergency situations require immediate professional help.
"""

# Test this module directly
if __name__ == "__main__":
    print("=" * 70)
    print("MediCare AI - Retrieval System Test")
    print("=" * 70)
    
    # Test questions
    test_questions = [
        "What are the symptoms of dengue fever?",
        "How to diagnose hypertension?",
        "What is the treatment for pneumonia?"
    ]
    
    for question in test_questions:
        results = search_medical_chunks(question, top_k=3)
        display_results(question, results)
        print("\n")