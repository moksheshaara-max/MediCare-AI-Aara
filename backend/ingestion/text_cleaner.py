import re


def fix_hyphenation(text):
    """
    Fix words broken by hyphenation across lines.
    
    Example:
    "symp-\ntoms" → "symptoms"
    """
    # Pattern: word ending with hyphen, then newline, then word
    pattern = r"(\w+)-\s*\n\s*(\w+)"
    fixed = re.sub(pattern, r"\1\2", text)
    return fixed


def collapse_whitespace(text):
    """
    Replace multiple spaces with single space.
    Replace multiple newlines with single newline.
    """
    # Multiple spaces → single space
    text = re.sub(r" +", " ", text)
    
    # Multiple newlines → double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def merge_broken_lines(text):
    """
    Merge lines that are broken mid-sentence.
    
    PDFs often break lines at word boundaries
    even when it's the same paragraph.
    
    We only join lines if:
    - Previous line does NOT end with punctuation
    - Next line does NOT start with capital letter
      after a paragraph break
    """
    lines = text.split("\n")
    merged = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Empty line = paragraph break, keep as is
        if not stripped:
            merged.append("")
            continue
        
        # First line, just add
        if not merged:
            merged.append(stripped)
            continue
        
        # Check if previous line looks like end of sentence
        prev = merged[-1]
        if prev and prev[-1] in ".!?:;":
            # Sentence ended, new line is new sentence
            merged.append(stripped)
        elif prev and prev == "":
            # Previous was paragraph break
            merged.append(stripped)
        else:
            # Continuation of previous line
            merged[-1] = prev + " " + stripped
    
    return "\n".join(merged)


def is_meaningful_page(text, min_length=100):
    """
    Check if page has meaningful content.
    
    Skip pages with very little text
    (blank pages, chapter dividers, etc.)
    """
    # Remove whitespace to count real characters
    cleaned = re.sub(r"\s+", "", text)
    return len(cleaned) >= min_length


def clean_page_text(text):
    """
    Apply all cleaning steps to a page's text.
    """
    # Step 1: Fix hyphenated words across line breaks
    text = fix_hyphenation(text)
    
    # Step 2: Merge broken lines
    text = merge_broken_lines(text)
    
    # Step 3: Collapse whitespace
    text = collapse_whitespace(text)
    
    return text


def clean_all_pages(pages_data, min_length=100):
    """
    Clean all extracted pages.
    
    Args:
        pages_data: List of {"book", "page", "text"}
        min_length: Minimum text length to keep page
    
    Returns:
        Cleaned pages, empty pages removed
    """
    cleaned_pages = []
    skipped_count = 0
    
    for page in pages_data:
        cleaned_text = clean_page_text(page["text"])
        
        # Skip pages with little meaningful content
        if not is_meaningful_page(cleaned_text, min_length):
            skipped_count += 1
            continue
        
        cleaned_pages.append({
            "book": page["book"],
            "page": page["page"],
            "text": cleaned_text
        })
    
    print(f"  Kept    : {len(cleaned_pages)} pages")
    print(f"  Skipped : {skipped_count} pages (empty/short)")
    
    return cleaned_pages


# Test this module directly
if __name__ == "__main__":
    print("=" * 50)
    print("MediCare AI — Text Cleaner Test")
    print("=" * 50)
    
    # Import extractor to get real data
    from pdf_extractor import extract_all_pdfs
    
    PDF_FOLDER = "data/documents"
    
    print("\n1. Extracting PDFs...")
    all_pages = extract_all_pdfs(PDF_FOLDER)
    
    print(f"\n2. Cleaning {len(all_pages)} pages...")
    cleaned = clean_all_pages(all_pages)
    
    print("\n" + "=" * 50)
    print("BEFORE and AFTER comparison")
    print("=" * 50)
    
    # Find a page with real content (middle of book 2)
    sample_index = 300  # arbitrary middle page
    
    if sample_index < len(all_pages):
        original = all_pages[sample_index]
        
        # Find same page in cleaned data
        cleaned_match = None
        for c in cleaned:
            if c["book"] == original["book"] and c["page"] == original["page"]:
                cleaned_match = c
                break
        
        print(f"\nBook: {original['book']}, Page: {original['page']}")
        print("-" * 50)
        print("BEFORE cleaning (first 400 chars):")
        print(original["text"][:400])
        print("-" * 50)
        if cleaned_match:
            print("AFTER cleaning (first 400 chars):")
            print(cleaned_match["text"][:400])
        else:
            print("(This page was skipped as not meaningful)")
    
    print("\n" + "=" * 50)
    print(f"Original pages : {len(all_pages)}")
    print(f"Cleaned pages  : {len(cleaned)}")
    print("=" * 50)