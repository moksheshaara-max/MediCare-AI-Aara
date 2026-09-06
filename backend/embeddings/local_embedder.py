"""
MediCare AI - Lightweight Cloud Embedder
Uses official Hugging Face InferenceClient for sentence-transformers/all-mpnet-base-v2.
Output: Exactly 768 dimensions (100% compatible with MongoDB Vector Search).
Uses zero PyTorch memory on Render.
"""

import os
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Ensure .env is loaded
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=env_path)
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSIONS = 768

# Initialize official Hugging Face client
client = InferenceClient(token=HF_TOKEN if HF_TOKEN else None)


def embed_single_text(text: str, max_retries: int = 5):
    """
    Generate 768-D embedding vector for a single text.
    """
    if not text:
        return [0.0] * EMBEDDING_DIMENSIONS

    for attempt in range(max_retries):
        try:
            # Official feature extraction call
            emb = client.feature_extraction(text, model=MODEL_ID)
            
            # Handle numpy arrays or nested lists
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            
            if isinstance(emb, list) and len(emb) > 0:
                # If nested [[0.1, 0.2, ...]], flatten to 1D
                if isinstance(emb[0], list):
                    # Mean pooling if multi-token shape
                    if len(emb) > 1 and len(emb[0]) == EMBEDDING_DIMENSIONS:
                        pooled = [sum(col) / len(col) for col in zip(*emb)]
                        return pooled
                    return emb[0]
                return emb
            
            return emb

        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "loading" in err_str.lower():
                time.sleep(3)
                continue
            time.sleep(1.5)

    print("⚠️ Fallback: Unable to generate embedding.")
    return None


def embed_batch(texts: list, max_retries: int = 5):
    """
    Generate embeddings for multiple texts.
    """
    if not texts:
        return []
    
    embeddings = []
    for text in texts:
        vec = embed_single_text(text, max_retries=max_retries)
        if vec:
            embeddings.append(vec)
    return embeddings


def embed_all_chunks(chunks):
    for chunk in chunks:
        chunk["embedding"] = embed_single_text(chunk.get("text", ""))
    return chunks


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Official Hugging Face Embedder")
    print(f"Target Model: {MODEL_ID}")
    print(f"Token Configured: {'YES' if HF_TOKEN else 'NO'}")
    print("=" * 60)

    test_sentence = "What are the symptoms of Type 2 Diabetes?"
    vector = embed_single_text(test_sentence)

    if vector and isinstance(vector, list) and len(vector) == 768:
        print(f"\n✓ SUCCESS: Generated {len(vector)}-dimensional vector!")
        print(f"First 5 dimensions: {vector[:5]}")
    else:
        print("\n❌ Error generating embedding.")