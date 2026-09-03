import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Create client (reused across calls)
client = genai.Client(api_key=api_key)

# Configuration
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768   # for MongoDB Vector Search compatibility
BATCH_SIZE = 20
DELAY_BETWEEN_BATCHES = 30
MAX_RETRIES = 3          # reduced from 5
def embed_single_text(text, task_type="RETRIEVAL_DOCUMENT"):
    """
    Generate embedding for a single text.
    Used for user queries later.
    """
    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS
            )
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None


def embed_batch(texts, task_type="RETRIEVAL_DOCUMENT"):
    """
    Generate embeddings for multiple texts in one API call.
    Raises QuotaExhaustedError if daily quota hit.
    """
    if not texts:
        return []
    
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIMENSIONS
                )
            )
            
            embeddings = [emb.values for emb in result.embeddings]
            return embeddings
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Detect daily quota exhausted - stop immediately
            if "quota" in error_msg and ("free_tier" in error_msg or "per day" in error_msg or "1000" in error_msg):
                print(f"    DAILY QUOTA EXHAUSTED")
                raise QuotaExhaustedError("Daily quota reached")
            
            # Per-minute rate limit - retry
            if "429" in error_msg or "rate" in error_msg:
                wait_time = 60 * (attempt + 1)
                print(f"    Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            # Other errors
            if attempt < MAX_RETRIES - 1:
                wait_time = 5 * (attempt + 1)
                print(f"    Error: {e}")
                print(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    Failed after {MAX_RETRIES} attempts")
                return None
    
    return None


class QuotaExhaustedError(Exception):
    """Raised when daily API quota is exhausted."""
    pass


def embed_all_chunks(chunks):
    """
    Generate embeddings for ALL chunks.
    """
    total = len(chunks)
    print(f"Generating embeddings for {total} chunks...")
    print(f"Dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total batches: {(total + BATCH_SIZE - 1) // BATCH_SIZE}")
    print()
    
    success_count = 0
    failed_count = 0
    
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = chunks[batch_start:batch_end]
        batch_texts = [c["text"] for c in batch]
        
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"Batch {batch_num}/{total_batches} "
              f"(chunks {batch_start+1}-{batch_end})...")
        
        embeddings = embed_batch(batch_texts)
        
        if embeddings and len(embeddings) == len(batch):
            for chunk, embedding in zip(batch, embeddings):
                chunk["embedding"] = embedding
            success_count += len(batch)
            print(f"  Success: {len(batch)} embeddings created")
        else:
            for chunk in batch:
                chunk["embedding"] = None
            failed_count += len(batch)
            print(f"  Failed: batch could not be embedded")
        
        if batch_end < total:
            time.sleep(DELAY_BETWEEN_BATCHES)
    
    print()
    print(f"Embedding complete:")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {failed_count}")
    
    return chunks


if __name__ == "__main__":
    print("=" * 50)
    print("MediCare AI — Embedder Module Test")
    print("=" * 50)
    
    sample_chunks = [
        {"chunk_id": "test_001", "text": "Dengue fever is caused by a virus."},
        {"chunk_id": "test_002", "text": "Common symptoms include high fever."},
        {"chunk_id": "test_003", "text": "Treatment involves fluid replacement."},
        {"chunk_id": "test_004", "text": "Heart attack requires urgent care."},
        {"chunk_id": "test_005", "text": "Diabetes affects blood sugar levels."}
    ]
    
    print(f"\nTesting with {len(sample_chunks)} sample chunks...\n")
    
    embedded = embed_all_chunks(sample_chunks)
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    for chunk in embedded:
        if chunk["embedding"]:
            print(f"{chunk['chunk_id']}: "
                  f"vector size {len(chunk['embedding'])}, "
                  f"first 3 values {[round(v, 4) for v in chunk['embedding'][:3]]}")
        else:
            print(f"{chunk['chunk_id']}: FAILED")
    
    print("\n" + "=" * 50)
    print("Embedder module ready for full ingestion")
    print("=" * 50)