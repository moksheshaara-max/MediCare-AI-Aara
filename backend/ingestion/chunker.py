import re


# Chunking configuration
CHUNK_SIZE = 600       # target words per chunk
CHUNK_OVERLAP = 100    # words to overlap between chunks
MIN_CHUNK_SIZE = 100   # skip chunks smaller than this


def split_into_sentences(text):
    """
    Split text into sentences.
    
    Handles common sentence endings: . ! ?
    Preserves the punctuation with the sentence.
    """
    # Regex: split after . ! ? followed by space or newline
    # But preserve the punctuation
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)
    
    # Remove empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def split_into_paragraphs(text):
    """
    Split text into paragraphs (separated by blank lines).
    """
    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def count_words(text):
    """Count words in text."""
    return len(text.split())


def create_chunks_from_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Recursive chunking:
    1. First try to split by paragraphs
    2. If paragraph is too big, split by sentences
    3. Combine into chunks of ~chunk_size words
    4. Add overlap between chunks
    
    Returns list of chunk texts (just strings, no metadata yet).
    """
    
    # Step 1: Split into paragraphs
    paragraphs = split_into_paragraphs(text)
    
    # Step 2: Build a list of "units" to chunk
    # Units are paragraphs, or sentences if paragraph is too big
    units = []
    
    for para in paragraphs:
        para_words = count_words(para)
        
        if para_words <= chunk_size:
            # Paragraph fits, use as single unit
            units.append(para)
        else:
            # Paragraph too big, split into sentences
            sentences = split_into_sentences(para)
            units.extend(sentences)
    
    # Step 3: Combine units into chunks
    chunks = []
    current_chunk_units = []
    current_word_count = 0
    
    for unit in units:
        unit_words = count_words(unit)
        
        # If adding this unit exceeds chunk size, finalize current chunk
        if current_word_count + unit_words > chunk_size and current_chunk_units:
            chunk_text = " ".join(current_chunk_units)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            # Take last few units for overlap
            overlap_units = []
            overlap_word_count = 0
            
            # Add units from end of previous chunk until overlap size reached
            for prev_unit in reversed(current_chunk_units):
                prev_words = count_words(prev_unit)
                if overlap_word_count + prev_words <= overlap:
                    overlap_units.insert(0, prev_unit)
                    overlap_word_count += prev_words
                else:
                    break
            
            current_chunk_units = overlap_units
            current_word_count = overlap_word_count
        
        # Add current unit to chunk
        current_chunk_units.append(unit)
        current_word_count += unit_words
    
    # Don't forget the last chunk
    if current_chunk_units:
        chunk_text = " ".join(current_chunk_units)
        chunks.append(chunk_text)
    
    # Filter out chunks that are too small
    chunks = [c for c in chunks if count_words(c) >= MIN_CHUNK_SIZE // 4]
    
    return chunks


def chunk_pages(cleaned_pages):
    """
    Convert cleaned pages into chunks with metadata.
    
    Strategy: 
    - Combine consecutive pages of same book
    - Then split into chunks
    - Track which pages each chunk spans
    
    For simplicity in this version, we chunk each page separately.
    Later we can improve to combine pages.
    
    Args:
        cleaned_pages: List of {"book", "page", "text"}
    
    Returns:
        List of chunk dictionaries with metadata:
        {
          "chunk_id": "book1_p294_c00",
          "book": "book1.pdf",
          "page": 294,
          "chunk_index": 0,
          "text": "...",
          "word_count": 587
        }
    """
    all_chunks = []
    
    # Group pages by book
    from collections import defaultdict
    pages_by_book = defaultdict(list)
    
    for page in cleaned_pages:
        pages_by_book[page["book"]].append(page)
    
    # Process each book
    for book_name, pages in pages_by_book.items():
        print(f"Chunking {book_name} ({len(pages)} pages)...")
        book_chunk_count = 0
        
        # Sort pages by page number
        pages.sort(key=lambda p: p["page"])
        
        # Chunk each page
        for page in pages:
            page_chunks = create_chunks_from_text(page["text"])
            
            for chunk_index, chunk_text in enumerate(page_chunks):
                # Create unique chunk ID
                book_id = book_name.replace(".pdf", "")
                chunk_id = f"{book_id}_p{page['page']:04d}_c{chunk_index:02d}"
                
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "book": book_name,
                    "page": page["page"],
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "word_count": count_words(chunk_text)
                })
                book_chunk_count += 1
        
        print(f"  Created {book_chunk_count} chunks")
    
    return all_chunks


# Test this module directly
if __name__ == "__main__":
    print("=" * 50)
    print("MediCare AI — Chunker Test")
    print("=" * 50)
    
    # Import previous modules
    from pdf_extractor import extract_all_pdfs
    from text_cleaner import clean_all_pages
    
    PDF_FOLDER = "data/documents"
    
    print("\n1. Extracting PDFs...")
    raw_pages = extract_all_pdfs(PDF_FOLDER)
    
    print(f"\n2. Cleaning {len(raw_pages)} pages...")
    cleaned = clean_all_pages(raw_pages)
    
    print(f"\n3. Chunking {len(cleaned)} cleaned pages...")
    chunks = chunk_pages(cleaned)
    
    print("\n" + "=" * 50)
    print("CHUNKING RESULTS")
    print("=" * 50)
    print(f"Total chunks created: {len(chunks)}")
    
    # Statistics
    word_counts = [c["word_count"] for c in chunks]
    if word_counts:
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        print(f"Average words/chunk : {avg_words:.0f}")
        print(f"Min words/chunk     : {min_words}")
        print(f"Max words/chunk     : {max_words}")
    
    # Chunks per book
    from collections import Counter
    book_counts = Counter(c["book"] for c in chunks)
    print("\nChunks per book:")
    for book, count in book_counts.items():
        print(f"  {book}: {count} chunks")
    
    # Show sample chunks
    print("\n" + "=" * 50)
    print("SAMPLE CHUNKS")
    print("=" * 50)
    
    # Sample from middle of collection
    sample_indices = [len(chunks) // 4, len(chunks) // 2, len(chunks) * 3 // 4]
    
    for idx in sample_indices:
        if idx < len(chunks):
            chunk = chunks[idx]
            print(f"\nChunk ID    : {chunk['chunk_id']}")
            print(f"Book        : {chunk['book']}")
            print(f"Page        : {chunk['page']}")
            print(f"Word count  : {chunk['word_count']}")
            print(f"Text preview: {chunk['text'][:300]}...")
            print("-" * 50)