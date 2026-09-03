import fitz  # PyMuPDF is imported as "fitz"
import os

# Path to your PDFs folder
PDF_FOLDER = "data/documents"

print("=" * 50)
print("MediCare AI — PDF Reader Test")
print("=" * 50)

# List all PDFs in the folder
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

if not pdf_files:
    print(f"No PDFs found in {PDF_FOLDER}")
    print("Please add your medical PDF books there")
    exit()

print(f"Found {len(pdf_files)} PDF file(s):")
for pdf in pdf_files:
    print(f"  - {pdf}")
print()

# Take the first PDF and analyze it
first_pdf_path = os.path.join(PDF_FOLDER, pdf_files[0])
print(f"Analyzing: {pdf_files[0]}")
print("-" * 50)

# Open the PDF
doc = fitz.open(first_pdf_path)

# Basic information
print(f"Total pages    : {doc.page_count}")
print(f"Metadata title : {doc.metadata.get('title', 'N/A')}")
print(f"Metadata author: {doc.metadata.get('author', 'N/A')}")
print("-" * 50)

# Extract text from first 3 pages as a test
print("Text preview from first 3 pages:")
print()

for page_number in range(min(3, doc.page_count)):
    page = doc.load_page(page_number)
    text = page.get_text()
    
    # Show first 300 characters only
    preview = text[:300].strip()
    
    print(f"--- Page {page_number + 1} ---")
    print(preview)
    print()

# Close the PDF
doc.close()

print("=" * 50)
print("PDF reader is working correctly")
print("=" * 50)