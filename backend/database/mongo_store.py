import os
from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, ConnectionFailure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "medicare_rag"
COLLECTION_NAME = "medical_chunks"

# Batch size for MongoDB operations
INSERT_BATCH_SIZE = 100


def get_mongo_client():
    """Create and return a MongoDB client."""
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI not set in .env file")
    
    client = MongoClient(MONGODB_URI)
    
    # Test the connection
    try:
        client.admin.command("ping")
    except ConnectionFailure:
        raise ConnectionFailure("Failed to connect to MongoDB")
    
    return client


def get_collection():
    """Get the medical_chunks collection."""
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection


def get_existing_chunk_ids():
    """
    Get set of chunk_ids already in database.
    Used to skip already-ingested chunks.
    
    Returns:
        set of chunk_id strings
    """
    collection = get_collection()
    
    # Only fetch chunk_id field (not full documents)
    existing = collection.find({}, {"chunk_id": 1, "_id": 0})
    chunk_ids = set(doc["chunk_id"] for doc in existing)
    
    return chunk_ids


def filter_new_chunks(chunks):
    """
    Filter out chunks that are already in database.
    
    Args:
        chunks: List of chunk dictionaries
    
    Returns:
        (new_chunks, skipped_count)
    """
    existing_ids = get_existing_chunk_ids()
    
    new_chunks = []
    skipped = 0
    
    for chunk in chunks:
        if chunk["chunk_id"] in existing_ids:
            skipped += 1
        else:
            new_chunks.append(chunk)
    
    return new_chunks, skipped


def insert_chunks(chunks):
    """
    Insert chunks into MongoDB.
    Uses bulk insert for performance.
    
    Args:
        chunks: List of chunk dictionaries with 'embedding' field
    
    Returns:
        Number of chunks successfully inserted
    """
    if not chunks:
        return 0
    
    collection = get_collection()
    total = len(chunks)
    inserted_count = 0
    
    print(f"Inserting {total} chunks into MongoDB...")
    
    # Insert in batches
    for batch_start in range(0, total, INSERT_BATCH_SIZE):
        batch_end = min(batch_start + INSERT_BATCH_SIZE, total)
        batch = chunks[batch_start:batch_end]
        
        # Prepare documents for insertion
        documents = []
        for chunk in batch:
            # Skip chunks without embeddings
            if not chunk.get("embedding"):
                continue
            
            doc = {
                "chunk_id"    : chunk["chunk_id"],
                "book"        : chunk["book"],
                "page"        : chunk["page"],
                "chunk_index" : chunk["chunk_index"],
                "text"        : chunk["text"],
                "word_count"  : chunk.get("word_count", 0),
                "embedding"   : chunk["embedding"],
                "ingested_at" : datetime.now(timezone.utc)
            }
            documents.append(doc)
        
        if not documents:
            continue
        
        try:
            # Bulk insert
            result = collection.insert_many(documents, ordered=False)
            inserted_count += len(result.inserted_ids)
            
            batch_num = (batch_start // INSERT_BATCH_SIZE) + 1
            total_batches = (total + INSERT_BATCH_SIZE - 1) // INSERT_BATCH_SIZE
            print(f"  Batch {batch_num}/{total_batches}: "
                  f"inserted {len(result.inserted_ids)} chunks")
            
        except BulkWriteError as e:
            print(f"  Batch error: {e.details}")
    
    return inserted_count


def create_chunk_id_index():
    """
    Create unique index on chunk_id for fast lookups.
    Only creates if it doesn't already exist.
    """
    collection = get_collection()
    
    # Get list of existing indexes
    existing_indexes = collection.index_information()
    
    if "chunk_id_1" not in existing_indexes:
        collection.create_index("chunk_id", unique=True)
        print("Created unique index on chunk_id")
    else:
        print("Index on chunk_id already exists")


def get_stats():
    """Get statistics about stored chunks."""
    collection = get_collection()
    
    total = collection.count_documents({})
    
    # Count by book
    pipeline = [
        {"$group": {
            "_id": "$book",
            "count": {"$sum": 1}
        }}
    ]
    by_book = list(collection.aggregate(pipeline))
    
    return {
        "total_chunks": total,
        "by_book": by_book
    }


def clear_all_chunks():
    """
    DANGER: Delete ALL chunks from database.
    Only use for fresh start.
    """
    collection = get_collection()
    result = collection.delete_many({})
    return result.deleted_count


# Test this module
if __name__ == "__main__":
    print("=" * 50)
    print("MediCare AI — MongoDB Storage Test")
    print("=" * 50)
    
    # Test 1: Connection
    print("\n1. Testing connection...")
    try:
        collection = get_collection()
        print("   Connection: SUCCESS")
    except Exception as e:
        print(f"   Connection FAILED: {e}")
        exit()
    
    # Test 2: Create index
    print("\n2. Setting up index...")
    create_chunk_id_index()
    
    # Test 3: Current stats
    print("\n3. Current database stats:")
    stats = get_stats()
    print(f"   Total chunks: {stats['total_chunks']}")
    for book_info in stats["by_book"]:
        print(f"   {book_info['_id']}: {book_info['count']} chunks")
    
    # Test 4: Insert sample chunks with fake embeddings
    print("\n4. Testing insertion with sample chunks...")
    
    sample_chunks = [
        {
            "chunk_id"    : "test_sample_001",
            "book"        : "test.pdf",
            "page"        : 1,
            "chunk_index" : 0,
            "text"        : "This is a test chunk for MongoDB storage.",
            "word_count"  : 9,
            "embedding"   : [0.1] * 768  # fake 768-dim vector
        },
        {
            "chunk_id"    : "test_sample_002",
            "book"        : "test.pdf",
            "page"        : 1,
            "chunk_index" : 1,
            "text"        : "Another test chunk to verify storage works.",
            "word_count"  : 8,
            "embedding"   : [0.2] * 768
        }
    ]
    
    # Check for duplicates first
    new_chunks, skipped = filter_new_chunks(sample_chunks)
    print(f"   New chunks: {len(new_chunks)}, Skipped (already exist): {skipped}")
    
    if new_chunks:
        inserted = insert_chunks(new_chunks)
        print(f"   Inserted: {inserted} chunks")
    
    # Test 5: Verify
    print("\n5. Verifying storage...")
    stats = get_stats()
    print(f"   Total chunks now: {stats['total_chunks']}")
    
    # Test 6: Run again to test duplicate prevention
    print("\n6. Running insert AGAIN (should skip duplicates)...")
    new_chunks, skipped = filter_new_chunks(sample_chunks)
    print(f"   New chunks: {len(new_chunks)}, Skipped: {skipped}")
    
    print("\n" + "=" * 50)
    print("MongoDB storage module ready")
    print("=" * 50)