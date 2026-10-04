import os
from collections import Counter
from datetime import datetime, timezone

from bson.binary import Binary, BinaryVectorDtype
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, ConnectionFailure


load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "medicare_rag"
CHUNKS_COLLECTION_NAME = "medical_chunks"
SOURCES_COLLECTION_NAME = "medical_sources"
INSERT_BATCH_SIZE = 100
EMBEDDING_DIMENSIONS = 768

SOURCE_METADATA_FIELDS = [
    "source_id",
    "official_title",
    "organization",
    "publication_year",
    "edition_version",
    "source_type",
    "specialty",
    "topic",
    "country",
    "official_url",
    "download_date",
    "is_current",
    "local_filename",
    "folder",
    "status",
    "notes",
]


def get_mongo_client():
    """Create and verify a MongoDB client."""
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI not set in .env file")

    client = MongoClient(MONGODB_URI)
    try:
        client.admin.command("ping")
    except ConnectionFailure:
        raise ConnectionFailure("Failed to connect to MongoDB")
    return client


def get_collection():
    """Backward-compatible accessor for the medical_chunks collection."""
    return get_mongo_client()[DB_NAME][CHUNKS_COLLECTION_NAME]


def get_sources_collection():
    """Get the normalized medical_sources collection."""
    return get_mongo_client()[DB_NAME][SOURCES_COLLECTION_NAME]


def embedding_to_bindata(embedding):
    """Convert a 768-D Python vector into compact FLOAT32 BSON BinData."""
    if embedding is None:
        return None
    if isinstance(embedding, Binary):
        return embedding
    if not isinstance(embedding, (list, tuple)):
        raise TypeError(
            f"Embedding must be list/tuple/Binary, got {type(embedding).__name__}"
        )
    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS}-D embedding, got {len(embedding)}"
        )
    return Binary.from_vector(
        [float(v) for v in embedding],
        BinaryVectorDtype.FLOAT32,
    )


def create_indexes():
    """Create small deterministic indexes used by ingestion and retrieval."""
    chunks = get_collection()
    sources = get_sources_collection()

    chunks.create_index("chunk_id", unique=True)
    chunks.create_index("source_id")
    sources.create_index("source_id", unique=True)


def create_chunk_id_index():
    """Backward-compatible wrapper used by existing pipeline code."""
    create_indexes()
    print("MongoDB indexes ready")


def upsert_sources(source_rows):
    """
    Store registry metadata once per source instead of repeating it in every
    chunk document.
    """
    if not source_rows:
        return 0

    collection = get_sources_collection()
    now = datetime.now(timezone.utc)
    operations = []

    for source in source_rows:
        source_id = source.get("source_id", "").strip()
        if not source_id:
            raise ValueError("Source row missing source_id")

        doc = {field: source.get(field, "") for field in SOURCE_METADATA_FIELDS}
        doc["metadata_updated_at"] = now

        operations.append(
            UpdateOne(
                {"source_id": source_id},
                {"$set": doc},
                upsert=True,
            )
        )

    result = collection.bulk_write(operations, ordered=False)
    return result.upserted_count + result.modified_count + result.matched_count


def get_existing_chunk_ids():
    """Return chunk IDs already stored, for safe resumable ingestion."""
    collection = get_collection()
    return {
        doc["chunk_id"]
        for doc in collection.find({}, {"chunk_id": 1, "_id": 0})
        if doc.get("chunk_id")
    }


def filter_new_chunks(chunks):
    existing_ids = get_existing_chunk_ids()
    new_chunks = []
    skipped = 0

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            raise ValueError("Chunk is missing required field: chunk_id")
        if chunk_id in existing_ids:
            skipped += 1
        else:
            new_chunks.append(chunk)

    return new_chunks, skipped


def _build_document(chunk):
    """
    Build the compact medical_chunks document.

    Full source metadata is normalized into medical_sources. Chunks retain only
    fields required for vector retrieval, source linking, provenance, and UI
    compatibility.
    """
    required_fields = [
        "chunk_id",
        "source_id",
        "book",
        "page_start",
        "page_end",
        "chunk_index",
        "text",
        "embedding",
    ]

    missing = [field for field in required_fields if field not in chunk]
    if missing:
        raise ValueError(
            f"Chunk {chunk.get('chunk_id', '<unknown>')} missing fields: "
            + ", ".join(missing)
        )

    if not chunk.get("embedding"):
        return None

    return {
        "chunk_id": chunk["chunk_id"],
        "source_id": chunk["source_id"],
        "book": chunk["book"],
        # Keep 'page' for current retrieval/UI compatibility.
        "page": chunk["page_start"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "chunk_index": chunk["chunk_index"],
        "chunking_version": chunk.get("chunking_version", ""),
        "text": chunk["text"],
        "word_count": chunk.get("word_count", 0),
        "token_count": chunk.get("token_count", 0),
        "embedding": embedding_to_bindata(chunk["embedding"]),
        "ingested_at": datetime.now(timezone.utc),
    }


def insert_chunks(chunks):
    """Insert compact chunk documents using FLOAT32 BinData embeddings."""
    if not chunks:
        return 0

    collection = get_collection()
    total = len(chunks)
    inserted_count = 0

    print(f"Inserting {total} chunks into MongoDB...")

    for batch_start in range(0, total, INSERT_BATCH_SIZE):
        batch_end = min(batch_start + INSERT_BATCH_SIZE, total)
        batch = chunks[batch_start:batch_end]
        documents = []

        for chunk in batch:
            try:
                doc = _build_document(chunk)
                if doc is not None:
                    documents.append(doc)
            except Exception as e:
                print(
                    f"  Skipping chunk {chunk.get('chunk_id', '<unknown>')}: {e}"
                )

        if not documents:
            continue

        try:
            result = collection.insert_many(documents, ordered=False)
            inserted_count += len(result.inserted_ids)
            batch_num = (batch_start // INSERT_BATCH_SIZE) + 1
            total_batches = (total + INSERT_BATCH_SIZE - 1) // INSERT_BATCH_SIZE
            print(
                f"  Batch {batch_num}/{total_batches}: "
                f"inserted {len(result.inserted_ids)} chunks"
            )

        except BulkWriteError as e:
            details = e.details or {}
            inserted_in_batch = details.get("nInserted", 0)
            inserted_count += inserted_in_batch
            print(f"  Batch partially completed: {inserted_in_batch} inserted")
            for write_error in details.get("writeErrors", []):
                print(
                    f"    MongoDB write error (code {write_error.get('code')}): "
                    f"{write_error.get('errmsg')}"
                )

    return inserted_count


def get_stats():
    """Get chunk counts without duplicating source metadata in every chunk."""
    chunks = get_collection()
    sources = get_sources_collection()

    total = chunks.count_documents({})

    by_book = list(
        chunks.aggregate([
            {"$group": {"_id": "$book", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ])
    )

    source_counts = list(
        chunks.aggregate([
            {"$group": {"_id": "$source_id", "count": {"$sum": 1}}},
        ])
    )

    source_type_by_id = {
        doc.get("source_id"): doc.get("source_type", "")
        for doc in sources.find({}, {"source_id": 1, "source_type": 1, "_id": 0})
    }

    type_counts = Counter()
    for item in source_counts:
        source_type = source_type_by_id.get(item.get("_id"), "")
        type_counts[source_type] += item.get("count", 0)

    by_source_type = [
        {"_id": source_type, "count": count}
        for source_type, count in type_counts.most_common()
    ]

    return {
        "total_chunks": total,
        "total_sources": sources.count_documents({}),
        "by_book": by_book,
        "by_source_type": by_source_type,
    }


def clear_all_chunks():
    """DANGER: delete all vector chunks, leaving normalized source metadata."""
    return get_collection().delete_many({}).deleted_count


if __name__ == "__main__":
    print("=" * 60)
    print("MediCare AI - MongoDB Storage Test")
    print("=" * 60)

    try:
        get_collection()
        get_sources_collection()
        print("Connection: SUCCESS")
    except Exception as e:
        print(f"Connection FAILED: {e}")
        raise SystemExit(1)

    create_indexes()
    stats = get_stats()
    print(f"Chunks : {stats['total_chunks']}")
    print(f"Sources: {stats['total_sources']}")
