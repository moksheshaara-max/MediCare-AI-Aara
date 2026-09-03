"""
Analyze all PDFs before ingestion.
Detects potential issues.
"""

import os
import pymupdf
from collections import defaultdict

PDF_FOLDER = "data/documents"

print("=" * 60)
print("PDF Analysis Report")
print("=" * 60)

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

print(f"\nTotal PDFs found: {len(pdf_files)}")
print()

total_pages = 0
total_size_mb = 0
issues = []
file_info = []

size_to_files = defaultdict(list)

for pdf_file in pdf_files:
    file_path = os.path.join(PDF_FOLDER, pdf_file)
    
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        total_size_mb += size_mb
        
        size_to_files[size_bytes].append(pdf_file)
        
        doc = pymupdf.open(file_path)
        pages = doc.page_count
        total_pages += pages
        
        first_page = doc.load_page(0)
        first_text = first_page.get_text().strip()
        has_text = len(first_text) > 50
        
        doc.close()
        
        file_info.append({
            "name": pdf_file,
            "pages": pages,
            "size_mb": size_mb,
            "has_text": has_text
        })
        
        if not has_text:
            issues.append(f"WARNING: {pdf_file} - may be scanned (no text extractable)")
        if pages > 500:
            issues.append(f"NOTE: {pdf_file} - large file ({pages} pages)")
            
    except Exception as e:
        issues.append(f"ERROR: {pdf_file} - Cannot open: {e}")

duplicates = {size: files for size, files in size_to_files.items() if len(files) > 1}

print(f"SUMMARY:")
print(f"  Total pages: {total_pages}")
print(f"  Total size: {total_size_mb:.1f} MB")
print(f"  Estimated chunks: ~{int(total_pages * 1.2)}")
print(f"  Estimated ingestion time: ~{max(15, int(total_pages * 1.2 / 100))} minutes")
print()

if duplicates:
    print("=" * 60)
    print("POTENTIAL DUPLICATES (same file size)")
    print("=" * 60)
    for size, files in duplicates.items():
        size_mb = size / (1024 * 1024)
        print(f"\nSize: {size_mb:.2f} MB - {len(files)} files:")
        for f in files:
            print(f"  - {f}")
else:
    print("No exact size duplicates detected")

if issues:
    print("\n" + "=" * 60)
    print("ISSUES FOUND")
    print("=" * 60)
    for issue in issues:
        print(f"  {issue}")

print("\n" + "=" * 60)
print("TOP 10 LARGEST PDFs (by pages)")
print("=" * 60)
file_info.sort(key=lambda x: x["pages"], reverse=True)
for i, info in enumerate(file_info[:10], 1):
    print(f"{i}. {info['name']:<50} {info['pages']:>4} pages ({info['size_mb']:.1f} MB)")

print("\n" + "=" * 60)
print("SMALLEST PDFs (may be low value)")
print("=" * 60)
file_info.sort(key=lambda x: x["pages"])
for info in file_info[:5]:
    print(f"  {info['name']:<50} {info['pages']:>4} pages ({info['size_mb']:.1f} MB)")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)