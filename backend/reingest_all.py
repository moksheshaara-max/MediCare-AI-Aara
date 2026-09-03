"""
Delete all chunks from MongoDB before re-ingesting.
Run once to clear database.
"""

from database.mongo_store import get_collection

print("=" * 50)
print("Clearing MongoDB for Fresh Ingestion")
print("=" * 50)

collection = get_collection()

count_before = collection.count_documents({})
print(f"\nChunks currently in database: {count_before}")

if count_before == 0:
    print("Database already empty. Nothing to delete.")
else:
    print(f"\nDeleting all {count_before} chunks...")
    result = collection.delete_many({})
    print(f"Deleted: {result.deleted_count} chunks")
    
    count_after = collection.count_documents({})
    print(f"Chunks remaining: {count_after}")

print("\n" + "=" * 50)
print("Database ready for fresh ingestion")
print("=" * 50)