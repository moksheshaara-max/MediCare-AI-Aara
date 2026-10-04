"""
MediCare AI - Registry-Aware Token-Safe Ingestion Pipeline

Pipeline:
1. Read source_registry.csv
2. Select status=registered and is_current=TRUE
3. Resolve each registered PDF inside data/documents/
4. Store source metadata once in medical_sources
5. Extract + clean text
6. Build packed token-safe 360/30 chunks across pages
7. Skip chunk_ids already present in MongoDB
8. Generate 768-D all-mpnet-base-v2 embeddings locally in CPU batches
9. Store compact chunks incrementally in medical_chunks
"""

import csv
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from ingestion.pdf_extractor import extract_pdf
from ingestion.text_cleaner import clean_all_pages
from ingestion.chunker import (
    chunk_pages,
    CHUNKING_VERSION,
    MAX_FINAL_TOKENS,
    OVERLAP_TOKENS,
)
from embeddings.local_embedder import (
    MODEL_ID,
    EMBEDDING_DIMENSIONS,
)
from database.mongo_store import (
    get_collection,
    create_chunk_id_index,
    get_existing_chunk_ids,
    insert_chunks,
    get_stats,
    upsert_sources,
)


PDF_ROOT = Path("data/documents")
REGISTRY_PATH = Path("source_registry.csv")
LOCAL_EMBED_BATCH_SIZE = 64
TORCH_CPU_THREADS = 16

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


def print_header(text):
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def print_step(number, text):
    print(f"\n[STEP {number}] {text}")
    print("-" * 72)


def load_active_registry():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    with REGISTRY_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    active = []
    seen_source_ids = set()
    seen_filenames = set()

    for row in rows:
        row = {
            str(k).strip(): (v.strip() if isinstance(v, str) else v)
            for k, v in row.items()
        }

        if row.get("status", "").lower() != "registered":
            continue
        if row.get("is_current", "").upper() != "TRUE":
            continue

        source_id = row.get("source_id", "")
        filename = row.get("local_filename", "")

        if not source_id:
            raise ValueError("Active registry row is missing source_id")
        if not filename:
            raise ValueError(
                f"Active registry row {source_id!r} is missing local_filename"
            )
        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate active source_id: {source_id}")
        if filename in seen_filenames:
            raise ValueError(f"Duplicate active local_filename: {filename}")

        seen_source_ids.add(source_id)
        seen_filenames.add(filename)
        active.append(row)

    return active


def resolve_pdf_path(source):
    filename = source["local_filename"]
    folder = source.get("folder", "").strip()

    expected = (
        PDF_ROOT / Path(folder.replace("\\", "/")) / filename
        if folder
        else PDF_ROOT / filename
    )

    if expected.is_file():
        return expected

    matches = list(PDF_ROOT.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Registered PDF not found: {filename} (expected under {PDF_ROOT})"
        )
    raise RuntimeError(
        f"Multiple PDFs found for registered filename {filename}: "
        + "; ".join(str(p) for p in matches)
    )


def validate_active_sources(active_sources):
    resolved = []
    problems = []

    for source in active_sources:
        try:
            resolved.append((source, resolve_pdf_path(source)))
        except Exception as e:
            problems.append((source.get("source_id", "<unknown>"), str(e)))

    if problems:
        print("\nRegistry/PDF validation problems:")
        for source_id, message in problems:
            print(f"  - {source_id}: {message}")
        raise RuntimeError(
            f"{len(problems)} active registry source(s) could not be resolved"
        )

    return resolved


def attach_source_link(chunks, source):
    """
    Attach only the source link required by compact chunk documents.
    Full registry metadata is stored once in medical_sources.
    """
    source_id = source["source_id"]
    filename = source["local_filename"]

    for chunk in chunks:
        chunk["source_id"] = source_id
        chunk["book"] = filename

    return chunks


def build_active_chunks(resolved_sources):
    all_chunks = []
    total_sources = len(resolved_sources)
    total_raw_pages = 0
    total_cleaned_pages = 0

    for index, (source, pdf_path) in enumerate(resolved_sources, 1):
        filename = source["local_filename"]

        print()
        print(f"[{index:3}/{total_sources}] {source['source_id']} | {filename}")

        raw_pages = extract_pdf(str(pdf_path))
        total_raw_pages += len(raw_pages)

        cleaned_pages = clean_all_pages(raw_pages)
        total_cleaned_pages += len(cleaned_pages)

        chunks = chunk_pages(
            cleaned_pages,
            source_id=source["source_id"],
            book_name=filename,
        )
        attach_source_link(chunks, source)
        all_chunks.extend(chunks)

        print(
            f"  Source result: {len(raw_pages)} pages -> "
            f"{len(cleaned_pages)} kept -> {len(chunks)} chunks"
        )

    return all_chunks, total_raw_pages, total_cleaned_pages


def embed_and_store(new_chunks):
    """
    Generate embeddings locally with SentenceTransformer in CPU batches, then
    store each completed batch immediately in MongoDB. Existing chunk IDs are
    filtered before this function is called, so the process remains resumable.
    """
    total = len(new_chunks)
    total_batches = (total + LOCAL_EMBED_BATCH_SIZE - 1) // LOCAL_EMBED_BATCH_SIZE
    total_success = 0
    total_failed = 0
    total_stored = 0

    torch.set_num_threads(TORCH_CPU_THREADS)

    print(f"Loading local SentenceTransformer on CPU: {MODEL_ID}")
    print(f"Local embedding batch size : {LOCAL_EMBED_BATCH_SIZE}")
    print(f"Torch CPU threads          : {torch.get_num_threads()}")

    model = SentenceTransformer(MODEL_ID, device="cpu")

    if model.max_seq_length < MAX_FINAL_TOKENS:
        raise RuntimeError(
            f"Model max_seq_length={model.max_seq_length} is smaller than "
            f"chunk maximum {MAX_FINAL_TOKENS}"
        )

    print(f"Model max sequence length  : {model.max_seq_length}")

    for batch_start in range(0, total, LOCAL_EMBED_BATCH_SIZE):
        batch_end = min(batch_start + LOCAL_EMBED_BATCH_SIZE, total)
        batch = new_chunks[batch_start:batch_end]
        batch_num = (batch_start // LOCAL_EMBED_BATCH_SIZE) + 1

        print()
        print(
            f"Batch {batch_num}/{total_batches} "
            f"(chunks {batch_start + 1}-{batch_end})"
        )

        valid_chunks = []
        texts = []

        for offset, chunk in enumerate(batch, start=batch_start + 1):
            token_count = chunk.get("token_count", 0)
            if token_count > MAX_FINAL_TOKENS:
                total_failed += 1
                print(
                    f"  [{offset}/{total}] skipped: token_count={token_count} "
                    f"exceeds {MAX_FINAL_TOKENS}: {chunk['chunk_id']}"
                )
                continue

            valid_chunks.append(chunk)
            texts.append(chunk["text"])

        if not valid_chunks:
            continue

        try:
            vectors = model.encode(
                texts,
                batch_size=LOCAL_EMBED_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as e:
            total_failed += len(valid_chunks)
            print(f"  Local embedding batch failed: {e}")
            print(
                "  Stopping so this entire unstored batch can be retried safely."
            )
            break

        successful_chunks = []

        for chunk, vector in zip(valid_chunks, vectors):
            embedding = vector.tolist()

            if len(embedding) != EMBEDDING_DIMENSIONS:
                total_failed += 1
                print(
                    f"  Invalid embedding dimensions for {chunk['chunk_id']}: "
                    f"{len(embedding)}"
                )
                continue

            chunk["embedding"] = embedding
            successful_chunks.append(chunk)
            total_success += 1

        if successful_chunks:
            try:
                inserted = insert_chunks(successful_chunks)
                total_stored += inserted
                print(f"  Embedded locally       : {len(successful_chunks)}")
                print(f"  Stored this batch      : {inserted}")
                print(f"  TOTAL STORED THIS RUN  : {total_stored}/{total}")
            except Exception as e:
                print(f"  MongoDB storage error: {e}")
                print(
                    "  Stopping so this successful-but-unstored batch can be retried."
                )
                break

    return total_success, total_failed, total_stored


def main():
    start_time = time.time()

    print_header("MediCare AI - Registry-Aware Token-Safe Ingestion")
    print(f"Registry          : {REGISTRY_PATH}")
    print(f"PDF root          : {PDF_ROOT}")
    print(f"Embedding         : {MODEL_ID}")
    print(f"Dimensions        : {EMBEDDING_DIMENSIONS}")
    print(f"Chunking version  : {CHUNKING_VERSION}")
    print(f"Max final tokens  : {MAX_FINAL_TOKENS}")
    print(f"Overlap tokens    : {OVERLAP_TOKENS}")

    print_step(1, "Loading active sources from registry")
    try:
        active_sources = load_active_registry()
        print(f"Active sources selected: {len(active_sources)}")
        print("Rule: status=registered AND is_current=TRUE")
    except Exception as e:
        print(f"Registry load failed: {e}")
        sys.exit(1)

    print_step(2, "Validating registered PDF paths")
    try:
        resolved_sources = validate_active_sources(active_sources)
        print(f"Resolved PDFs: {len(resolved_sources)}/{len(active_sources)}")
    except Exception as e:
        print(f"PDF validation failed: {e}")
        sys.exit(1)

    print_step(3, "Setting up MongoDB and normalized source metadata")
    try:
        get_collection()
        create_chunk_id_index()
        upsert_sources(active_sources)
        stats = get_stats()
        print(f"MongoDB ready. Current chunks : {stats['total_chunks']}")
        print(f"Source metadata documents     : {stats['total_sources']}")
    except Exception as e:
        print(f"MongoDB setup failed: {e}")
        sys.exit(1)

    print_step(4, "Extracting, cleaning and token-safe chunking")
    try:
        all_chunks, raw_pages, cleaned_pages = build_active_chunks(resolved_sources)
    except Exception as e:
        print(f"Chunk preparation failed: {e}")
        sys.exit(1)

    print()
    print(f"Active sources processed : {len(resolved_sources)}")
    print(f"Raw PDF pages            : {raw_pages:,}")
    print(f"Cleaned/kept pages       : {cleaned_pages:,}")
    print(f"Chunks prepared          : {len(all_chunks):,}")

    if all_chunks:
        max_tokens = max(c.get("token_count", 0) for c in all_chunks)
        avg_tokens = sum(c.get("token_count", 0) for c in all_chunks) / len(all_chunks)
        print(f"Average tokens/chunk     : {avg_tokens:.1f}")
        print(f"Maximum tokens/chunk     : {max_tokens}")

    print_step(5, "Filtering chunks already stored in MongoDB")
    try:
        existing_ids = get_existing_chunk_ids()
        new_chunks = [
            chunk for chunk in all_chunks
            if chunk["chunk_id"] not in existing_ids
        ]
        skipped = len(all_chunks) - len(new_chunks)
        print(f"Already in MongoDB : {skipped:,}")
        print(f"New chunks         : {len(new_chunks):,}")
    except Exception as e:
        print(f"Duplicate check failed: {e}")
        sys.exit(1)

    if not new_chunks:
        print("\nEverything is already ingested. Nothing to do.")
        return

    print_step(6, "Confirmation required")
    print(f"Sources             : {len(resolved_sources)}")
    print(f"New chunks          : {len(new_chunks):,}")
    print(f"Local embed batch   : {LOCAL_EMBED_BATCH_SIZE}")
    print(f"Embedding model     : {MODEL_ID}")
    print("Mongo vector format : FLOAT32 BinData")
    print("Source metadata     : normalized in medical_sources")
    print()
    print(
        "The next step WILL generate embeddings locally on this computer and "
        "WILL write successful chunks to MongoDB."
    )

    confirm = input("\nProceed with embedding + storage? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled before embedding. No chunk embeddings were written.")
        return

    print_step(7, "Local batch embedding and incremental storage")
    total_success, total_failed, total_stored = embed_and_store(new_chunks)

    print_step(8, "Final summary")
    print(f"Embeddings created this run : {total_success:,}")
    print(f"Failed embeddings           : {total_failed:,}")
    print(f"Chunks stored this run      : {total_stored:,}")

    try:
        stats = get_stats()
        print(f"Total chunks in database    : {stats['total_chunks']:,}")
        print(f"Source metadata documents   : {stats['total_sources']:,}")

        if stats["by_source_type"]:
            print("\nDatabase chunks by source type:")
            for item in stats["by_source_type"]:
                label = item["_id"] or "<missing>"
                print(f"  {label}: {item['count']:,}")
    except Exception as e:
        print(f"Could not fetch final stats: {e}")

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print_header(f"Session ended in {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
