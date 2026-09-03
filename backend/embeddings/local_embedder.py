"""
MediCare AI - Local Embedder
Uses sentence-transformers for FREE unlimited embeddings.
Same 768 dimensions as Google - MongoDB compatible.
"""

from sentence_transformers import SentenceTransformer
import time

# Load model - downloads ~420MB on first run only
print("Loading local embedding model (first time downloads ~420MB)...")
_model = SentenceTransformer('all-mpnet-base-v2')
print(f"Model loaded! Dimensions: {_model.get_sentence_embedding_dimension()}")

# Configuration
EMBEDDING_DIMENSIONS = 768   # Same as Google!
BATCH_SIZE = 100             # Process 100 at once
DELAY_BETWEEN_BATCHES = 0    # No delay for local


def embed_single_text(text, task_type=None):
    """Generate embedding for a single text."""
    embedding = _model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding.tolist()


def embed_batch(texts, task_type=None):
    """Generate embeddings for multiple texts at once."""
    if not texts:
        return []
    
    embeddings = _model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return [emb.tolist() for emb in embeddings]


def embed_all_chunks(chunks):
    """Generate embeddings for ALL chunks - no limits."""
    total = len(chunks)
    print(f"Generating {total} embeddings locally...")
    print(f"Dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"NO API LIMITS - running on your laptop")
    print()
    
    start = time.time()
    all_texts = [c["text"] for c in chunks]
    
    all_embeddings = []
    for i in range(0, total, BATCH_SIZE):
        batch = all_texts[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"Batch {batch_num}/{total_batches} (chunks {i+1}-{min(i+BATCH_SIZE, total)})...")
        embeddings = embed_batch(batch)
        all_embeddings.extend(embeddings)
    
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding
    
    elapsed = time.time() - start
    print()
    print(f"Complete in {elapsed:.1f} seconds")
    print(f"Speed: {total/elapsed:.1f} chunks/second")
    
    return chunks


class QuotaExhaustedError(Exception):
    """Placeholder - not used in local embedder."""
    pass


if __name__ == "__main__":
    print("=" * 50)
    print("Local Embedder Test")
    print("=" * 50)
    
    sample_texts = [
        "Diabetes is a metabolic disease.",
        "Common symptoms include high fever.",
        "Heart attack requires urgent care."
    ]
    
    print(f"\nEmbedding {len(sample_texts)} test texts...")
    embeddings = embed_batch(sample_texts)
    
    print(f"\nResults:")
    for i, (text, emb) in enumerate(zip(sample_texts, embeddings)):
        print(f"  Text {i+1}: {text[:40]}...")
        print(f"    Size: {len(emb)}, First 3: {emb[:3]}")
    
    print("\n" + "=" * 50)
    print("Local embedder working!")
    print("=" * 50)