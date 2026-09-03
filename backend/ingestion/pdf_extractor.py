import pymupdf  # PyMuPDF (new import name)
import os


def extract_pdf(pdf_path):
    """
    Extract text from a PDF file page by page.
    
    Args:
        pdf_path: Full path to the PDF file
    
    Returns:
        List of dictionaries with page number and text
        Example:
        [
          {"page": 1, "text": "..."},
          {"page": 2, "text": "..."},
          ...
        ]
    """
    
    # Verify file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Open the PDF
    doc = pymupdf.open(pdf_path)
    
    # Get book filename (without path)
    book_name = os.path.basename(pdf_path)
    
    print(f"Extracting: {book_name}")
    print(f"Total pages: {doc.page_count}")
    
    # Store all pages here
    pages_data = []
    
    # Loop through every page
    for page_num in range(doc.page_count):
        # Load one page at a time (memory efficient)
        page = doc.load_page(page_num)
        
        # Extract text from this page
        text = page.get_text()
        
        # Store page data
        pages_data.append({
            "book": book_name,
            "page": page_num + 1,  # human-readable page number
            "text": text
        })
        
        # Show progress every 100 pages
        if (page_num + 1) % 100 == 0:
            print(f"  Processed {page_num + 1}/{doc.page_count} pages")
    
    # Always close the file
    doc.close()
    
    print(f"  Extraction complete: {len(pages_data)} pages")
    print()
    
    return pages_data


def extract_all_pdfs(pdf_folder):
    """
    Extract text from ALL PDFs in a folder.
    
    Args:
        pdf_folder: Path to folder containing PDFs
    
    Returns:
        List of all pages from all PDFs
    """
    
    # Find all PDFs in folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDFs found in {pdf_folder}")
        return []
    
    print(f"Found {len(pdf_files)} PDF(s) to process:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    print()
    
    # Extract text from each PDF
    all_pages = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        pages = extract_pdf(pdf_path)
        all_pages.extend(pages)
    
    return all_pages


# Run test only if this file is executed directly
if __name__ == "__main__":
    print("=" * 50)
    print("MediCare AI — PDF Extractor Test")
    print("=" * 50)
    
    PDF_FOLDER = "data/documents"
    all_pages = extract_all_pdfs(PDF_FOLDER)
    
    print("=" * 50)
    print(f"TOTAL pages extracted: {len(all_pages)}")
    print("=" * 50)
    
    # Show sample: first page and a middle page
    if all_pages:
        print("\nSample from first page:")
        print("-" * 50)
        first = all_pages[0]
        print(f"Book: {first['book']}")
        print(f"Page: {first['page']}")
        print(f"Text (first 200 chars): {first['text'][:200]}")
        
        # Middle page for meaningful content
        mid_index = len(all_pages) // 2
        print("\nSample from middle page:")
        print("-" * 50)
        mid = all_pages[mid_index]
        print(f"Book: {mid['book']}")
        print(f"Page: {mid['page']}")
        print(f"Text (first 200 chars): {mid['text'][:200]}")