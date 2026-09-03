"""
MediCare AI - Full Ingestion Pipeline

This script runs the complete pipeline:
1. Extract text from all PDFs in data/documents/
2. Clean extracted text
3. Chunk into smaller pieces
4. Check MongoDB for existing chunks (skip duplicates)
5. Generate embeddings for new chunks
6. Store in MongoDB

Safe to run multiple times.
Only processes NEW chunks.
"""

import sys
import time
from ingestion.pdf_extractor import extract_all_pdfs
from ingestion.text_cleaner import clean_all_pages
from ingestion.chunker import chunk_pages
from embeddings.local_embedder import embed_all_chunks
from database.mongo_store import (
    get_collection,
    create_chunk_id_index,
    filter_new_chunks,
    insert_chunks,
    get_stats
)


PDF_FOLDER = "data/documents"


def print_header(text):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(number, text):
    """Print a step header."""
    print(f"\n[STEP {number}] {text}")
    print("-" * 60)


def main():
    """Main ingestion pipeline with incremental saving."""
    
    start_time = time.time()
    
    print_header("MediCare AI - Full Ingestion Pipeline")
    print(f"PDF folder: {PDF_FOLDER}")
    
    # STEP 1: Setup MongoDB
    print_step(1, "Setting up MongoDB")
    try:
        collection = get_collection()
        create_chunk_id_index()
        stats = get_stats()
        print(f"MongoDB ready. Current chunks in DB: {stats['total_chunks']}")
    except Exception as e:
        print(f"MongoDB setup failed: {e}")
        sys.exit(1)
    
    # STEP 2: Extract PDFs
    print_step(2, "Extracting text from PDFs")
    try:
        raw_pages = extract_all_pdfs(PDF_FOLDER)
        if not raw_pages:
            print("No PDFs found.")
            sys.exit(1)
        print(f"Extracted {len(raw_pages)} pages total")
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        sys.exit(1)
    
    # STEP 3: Clean text
    print_step(3, "Cleaning extracted text")
    cleaned_pages = clean_all_pages(raw_pages)
    print(f"Cleaned pages: {len(cleaned_pages)}")
    
    # STEP 4: Chunk text
    print_step(4, "Chunking text into pieces")
    all_chunks = chunk_pages(cleaned_pages)
    print(f"Total chunks created: {len(all_chunks)}")
    
    # STEP 5: Filter out already-ingested chunks
    print_step(5, "Checking for existing chunks in MongoDB")
    new_chunks, skipped = filter_new_chunks(all_chunks)
    print(f"Already in DB: {skipped}")
    print(f"New to ingest: {len(new_chunks)}")
    
    if not new_chunks:
        print("\nAll chunks already ingested. Nothing to do.")
        return
    
    # STEP 6: Confirmation
    print_step(6, "Confirmation required")
    print(f"About to embed and store {len(new_chunks)} new chunks")
    print(f"Estimated batches: ~{(len(new_chunks) + 19) // 20}")
    print(f"Estimated time: ~{max(1, len(new_chunks) // 20 * 30 // 60)} minutes")
    
    confirm = input("\nProceed? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled by user.")
        return
    
    # STEP 7: EMBED AND STORE INCREMENTALLY
    print_step(7, "Embedding and storing incrementally")
    
    from embeddings.local_embedder import embed_batch, BATCH_SIZE, DELAY_BETWEEN_BATCHES, QuotaExhaustedError
    total = len(new_chunks)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    
    total_success = 0
    total_failed = 0
    total_stored = 0
    quota_hit = False
    
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = new_chunks[batch_start:batch_end]
        batch_texts = [c["text"] for c in batch]
        
        batch_num = (batch_start // BATCH_SIZE) + 1
        
        print(f"\nBatch {batch_num}/{total_batches} (chunks {batch_start+1}-{batch_end})...")
        
        try:
            embeddings = embed_batch(batch_texts)
        except QuotaExhaustedError:
            print(f"\n{'!' * 60}")
            print("  DAILY QUOTA EXHAUSTED - Stopping cleanly")
            print("  Re-run tomorrow to continue from where we stopped")
            print(f"{'!' * 60}")
            quota_hit = True
            break
        
        if embeddings and len(embeddings) == len(batch):
            for chunk, embedding in zip(batch, embeddings):
                chunk["embedding"] = embedding
            
            total_success += len(batch)
            print(f"  Embeddings: {len(batch)} created")
            
            # SAVE IMMEDIATELY to MongoDB
            try:
                inserted = insert_chunks(batch)
                total_stored += inserted
                print(f"  Stored to MongoDB: {inserted} chunks")
                print(f"  === TOTAL STORED SO FAR: {total_stored}/{total} ===")
            except Exception as e:
                print(f"  MongoDB storage error: {e}")
        else:
            total_failed += len(batch)
            print(f"  Failed batch: could not embed")
        
        if batch_end < total:
            time.sleep(DELAY_BETWEEN_BATCHES)
    
    # STEP 8: Final summary
    print_step(8, "Final Summary")
    
    if quota_hit:
        print("Ingestion stopped due to daily quota.")
        print("Re-run tomorrow to continue.")
    
    print(f"\nEmbeddings created this run: {total_success}")
    print(f"Chunks stored in MongoDB    : {total_stored}")
    print(f"Failed embeddings           : {total_failed}")
    
    try:
        stats = get_stats()
        print(f"\nTotal chunks in database now: {stats['total_chunks']}")
        for book_info in stats["by_book"]:
            print(f"  {book_info['_id']}: {book_info['count']} chunks")
    except Exception as e:
        print(f"Could not fetch final stats: {e}")
    
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print_header(f"Session ended in {minutes}m {seconds}s")

    

if __name__ == "__main__":
    main()