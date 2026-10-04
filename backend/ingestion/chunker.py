import re
from functools import lru_cache

from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# Token-safe chunking configuration for sentence-transformers/all-mpnet-base-v2
# -----------------------------------------------------------------------------
TOKENIZER_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNKING_VERSION = "mpnet360o30_v1"
MAX_FINAL_TOKENS = 360
OVERLAP_TOKENS = 30
# Hard-split long sentence/table blocks with a little headroom because decode ->
# re-tokenize can occasionally change token counts slightly.
HARD_SPLIT_CONTENT_TOKENS = 350


@lru_cache(maxsize=1)
def get_tokenizer():
    """Load the MPNet tokenizer once, only when chunking is actually used."""
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def split_into_sentences(text):
    """Split text after . ! ? while preserving punctuation."""
    pattern = r"(?<=[.!?])\s+"
    return [s.strip() for s in re.split(pattern, text) if s.strip()]


def split_into_paragraphs(text):
    """Split text into paragraphs separated by blank lines."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def count_words(text):
    """Count whitespace-delimited words."""
    return len(text.split())


def _token_ids(text, add_special_tokens=False):
    tokenizer = get_tokenizer()
    return tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )["input_ids"]


def count_tokens(text, add_special_tokens=True):
    """Count MPNet tokens for text."""
    return len(_token_ids(text, add_special_tokens=add_special_tokens))


def _decode_ids(token_ids):
    return get_tokenizer().decode(token_ids, skip_special_tokens=True).strip()


def _fit_decoded_piece(token_ids):
    """
    Decode token ids and, if decode/re-tokenize expansion occurs, trim until the
    final text is guaranteed to fit MAX_FINAL_TOKENS including special tokens.
    """
    ids = list(token_ids)
    while ids:
        text = _decode_ids(ids)
        if text and count_tokens(text, add_special_tokens=True) <= MAX_FINAL_TOKENS:
            return text
        ids = ids[:-1]
    return ""


def _hard_split(text, page_num):
    """
    Force-split a sentence/table-like unit that is too large for MPNet.
    Every returned piece is guaranteed to fit MAX_FINAL_TOKENS.
    """
    ids = _token_ids(text, add_special_tokens=False)

    if count_tokens(text, add_special_tokens=True) <= MAX_FINAL_TOKENS:
        return [{"text": text, "page_start": page_num, "page_end": page_num}]

    step = HARD_SPLIT_CONTENT_TOKENS - OVERLAP_TOKENS
    pieces = []

    for start in range(0, len(ids), step):
        raw_piece_ids = ids[start:start + HARD_SPLIT_CONTENT_TOKENS]
        if not raw_piece_ids:
            break

        piece = _fit_decoded_piece(raw_piece_ids)
        if piece:
            pieces.append({
                "text": piece,
                "page_start": page_num,
                "page_end": page_num,
            })

        if start + HARD_SPLIT_CONTENT_TOKENS >= len(ids):
            break

    return pieces


def _build_units(cleaned_pages):
    """
    Convert cleaned pages into paragraph/sentence units while preserving source
    page numbers. Oversized units are hard-split safely.
    """
    units = []

    for page in sorted(cleaned_pages, key=lambda p: p["page"]):
        page_num = page["page"]
        text = page["text"]

        for paragraph in split_into_paragraphs(text):
            if count_tokens(paragraph, add_special_tokens=True) <= MAX_FINAL_TOKENS:
                units.append({
                    "text": paragraph,
                    "page_start": page_num,
                    "page_end": page_num,
                })
                continue

            sentences = split_into_sentences(paragraph)

            # If sentence splitting failed to produce useful smaller units, the
            # hard splitter below still guarantees safety.
            for sentence in sentences or [paragraph]:
                if count_tokens(sentence, add_special_tokens=True) <= MAX_FINAL_TOKENS:
                    units.append({
                        "text": sentence,
                        "page_start": page_num,
                        "page_end": page_num,
                    })
                else:
                    units.extend(_hard_split(sentence, page_num))

    return units


def _overlap_text(previous_text):
    """Return approximately OVERLAP_TOKENS from the end of a completed chunk."""
    ids = _token_ids(previous_text, add_special_tokens=False)
    if not ids:
        return ""
    return _decode_ids(ids[-OVERLAP_TOKENS:])


def create_chunks_from_pages(cleaned_pages):
    """
    Pack cleaned pages from one source into token-safe chunks.

    Strategy:
      - max final chunk size: 360 MPNet tokens, including special tokens
      - overlap: ~30 MPNet tokens
      - pack across page boundaries within the same source
      - hard-split oversized sentence/table blocks
      - preserve page_start/page_end provenance

    Returns a list of dictionaries with text + page range + counts.
    """
    units = _build_units(cleaned_pages)
    chunks = []

    current_text = ""
    current_page_start = None
    current_page_end = None

    for unit in units:
        unit_text = unit["text"]

        if not current_text:
            current_text = unit_text
            current_page_start = unit["page_start"]
            current_page_end = unit["page_end"]
            continue

        candidate = f"{current_text} {unit_text}".strip()

        if count_tokens(candidate, add_special_tokens=True) <= MAX_FINAL_TOKENS:
            current_text = candidate
            current_page_end = unit["page_end"]
            continue

        # Finalize the current packed chunk.
        chunks.append({
            "text": current_text.strip(),
            "page_start": current_page_start,
            "page_end": current_page_end,
        })

        overlap = _overlap_text(current_text)
        candidate = f"{overlap} {unit_text}".strip() if overlap else unit_text

        # Prefer overlap, but never exceed the hard token ceiling.
        if count_tokens(candidate, add_special_tokens=True) <= MAX_FINAL_TOKENS:
            current_text = candidate
            # The overlap originated at the end of the previous chunk.
            current_page_start = current_page_end
            current_page_end = unit["page_end"]
        else:
            current_text = unit_text
            current_page_start = unit["page_start"]
            current_page_end = unit["page_end"]

    if current_text:
        chunks.append({
            "text": current_text.strip(),
            "page_start": current_page_start,
            "page_end": current_page_end,
        })

    # Final defensive check. If this ever fails, stop ingestion rather than
    # silently embedding truncated text.
    for chunk in chunks:
        final_tokens = count_tokens(chunk["text"], add_special_tokens=True)
        if final_tokens > MAX_FINAL_TOKENS:
            raise ValueError(
                f"Chunk exceeded token ceiling: {final_tokens} > {MAX_FINAL_TOKENS}"
            )
        chunk["token_count"] = final_tokens
        chunk["word_count"] = count_words(chunk["text"])

    return chunks


def chunk_pages(cleaned_pages, source_id=None, book_name=None):
    """
    Convert cleaned pages into packed token-safe chunks with stable metadata.

    The ingestion pipeline calls this once per source. chunk_id includes a
    chunking-version marker so a future chunking change cannot silently collide
    with IDs produced by this strategy.
    """
    if not cleaned_pages:
        return []

    inferred_book = book_name or cleaned_pages[0].get("book", "unknown.pdf")
    source_key = source_id or inferred_book.rsplit(".pdf", 1)[0]
    safe_source_key = re.sub(r"[^A-Za-z0-9_-]+", "_", str(source_key)).strip("_")

    packed = create_chunks_from_pages(cleaned_pages)
    all_chunks = []

    for chunk_index, chunk in enumerate(packed):
        page_start = chunk["page_start"]
        page_end = chunk["page_end"]
        chunk_id = (
            f"{safe_source_key}_{CHUNKING_VERSION}_"
            f"c{chunk_index:06d}"
        )

        all_chunks.append({
            "chunk_id": chunk_id,
            "book": inferred_book,
            # Legacy field retained for current retrieval/UI compatibility.
            "page": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_index": chunk_index,
            "text": chunk["text"],
            "word_count": chunk["word_count"],
            "token_count": chunk["token_count"],
            "chunking_version": CHUNKING_VERSION,
        })

    print(
        f"Chunking {inferred_book} ({len(cleaned_pages)} kept pages)... "
        f"created {len(all_chunks)} chunks"
    )
    return all_chunks


# Backward-compatible helper for small tests that still call the old function.
def create_chunks_from_text(text, chunk_size=None, overlap=None):
    """
    Token-safe replacement for the old word-based helper.

    chunk_size/overlap are accepted only for call compatibility; production
    settings are fixed by MAX_FINAL_TOKENS and OVERLAP_TOKENS.
    """
    pseudo_page = [{"book": "text", "page": 1, "text": text}]
    return [item["text"] for item in create_chunks_from_pages(pseudo_page)]


if __name__ == "__main__":
    print("=" * 60)
    print("MediCare AI - Token-Safe Chunker")
    print("=" * 60)
    print(f"Tokenizer       : {TOKENIZER_MODEL}")
    print(f"Chunking version: {CHUNKING_VERSION}")
    print(f"Max final tokens: {MAX_FINAL_TOKENS}")
    print(f"Overlap tokens  : {OVERLAP_TOKENS}")
