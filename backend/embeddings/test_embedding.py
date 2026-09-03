import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Create client using the NEW SDK
client = genai.Client(api_key=api_key)

print("=" * 50)
print("MediCare AI — Embedding API Test")
print("=" * 50)

# Test texts
test_texts = [
    "What are the symptoms of dengue fever?",
    "Common signs of dengue include high fever and headache.",
    "How to treat heart disease?"
]

print("\nTesting embedding generation...\n")

embeddings = []

for i, text in enumerate(test_texts):
    print(f"Text {i+1}: {text[:60]}")
    
    try:
        # Call the embedding API using new SDK
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        
        # Extract the embedding vector
        embedding = result.embeddings[0].values
        embeddings.append(embedding)
        
        print(f"  Embedding size: {len(embedding)}")
        print(f"  First 5 values: {[round(v, 4) for v in embedding[:5]]}")
        print()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        print()

# Similarity test
if len(embeddings) == 3:
    print("=" * 50)
    print("SIMILARITY TEST")
    print("=" * 50)
    
    def cosine_similarity(v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = sum(a * a for a in v1) ** 0.5
        magnitude2 = sum(b * b for b in v2) ** 0.5
        return dot_product / (magnitude1 * magnitude2)
    
    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
    
    print(f"\n'dengue symptoms?' vs 'dengue signs include fever'")
    print(f"  Similarity: {sim_1_2:.4f}  (should be HIGH)")
    
    print(f"\n'dengue symptoms?' vs 'heart disease treatment'")
    print(f"  Similarity: {sim_1_3:.4f}  (should be LOWER)")
    
    print("\n" + "=" * 50)
    if sim_1_2 > sim_1_3:
        print("Embeddings work correctly")
        print("Similar meanings produce higher similarity")
    else:
        print("Something is off with embeddings")
    print("=" * 50)