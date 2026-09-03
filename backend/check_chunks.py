from database.mongo_store import get_collection

collection = get_collection()

# Total chunks
total = collection.count_documents({})
print(f"Total chunks in DB: {total}")

# Chunks per book
from collections import Counter
books = collection.distinct("book")
print(f"\nBooks: {books}")

for book in books:
    count = collection.count_documents({"book": book})
    print(f"  {book}: {count} chunks")

# Check for duplicates
pipeline = [
    {"$group": {
        "_id": "$chunk_id",
        "count": {"$sum": 1}
    }},
    {"$match": {"count": {"$gt": 1}}},
    {"$limit": 5}
]

duplicates = list(collection.aggregate(pipeline))
if duplicates:
    print(f"\nDuplicate chunk_ids found: {len(duplicates)}")
    for d in duplicates:
        print(f"  {d['_id']}: {d['count']} times")
else:
    print("\nNo duplicates found")

# Sample chunks
print("\nSample chunk_ids:")
samples = collection.find({}, {"chunk_id": 1, "book": 1, "page": 1}).limit(5)
for s in samples:
    print(f"  {s['chunk_id']} - {s['book']} p{s['page']}")