"""
Remove ONLY duplicate files (same content).
Keep all originals including "scanned" ones.
"""

import os
import shutil
from collections import defaultdict

PDF_FOLDER = "data/documents"
BACKUP_FOLDER = "data/removed_pdfs"

os.makedirs(BACKUP_FOLDER, exist_ok=True)

print("=" * 60)
print("Removing Only Duplicate PDFs")
print("=" * 60)

# Group by file size
size_to_files = defaultdict(list)

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f"\nTotal PDFs: {len(pdf_files)}")

for pdf_file in pdf_files:
    file_path = os.path.join(PDF_FOLDER, pdf_file)
    size = os.path.getsize(file_path)
    size_to_files[size].append(pdf_file)

# Find duplicates
duplicates_to_remove = []

for size, files in size_to_files.items():
    if len(files) > 1:
        size_mb = size / (1024 * 1024)
        print(f"\nSame size ({size_mb:.2f} MB):")
        print(f"  KEEP: {files[0]}")
        for f in files[1:]:
            print(f"  REMOVE: {f}")
            duplicates_to_remove.append(f)

if not duplicates_to_remove:
    print("\nNo duplicates found!")
    exit()

print(f"\n{'=' * 60}")
print(f"Total duplicates to remove: {len(duplicates_to_remove)}")
print(f"PDFs after cleanup: {len(pdf_files) - len(duplicates_to_remove)}")
print(f"{'=' * 60}")

confirm = input("\nMove duplicates to backup? (yes/no): ").strip().lower()

if confirm != "yes":
    print("Cancelled.")
    exit()

moved = 0
for f in duplicates_to_remove:
    try:
        src = os.path.join(PDF_FOLDER, f)
        dst = os.path.join(BACKUP_FOLDER, f)
        shutil.move(src, dst)
        moved += 1
        print(f"  Moved: {f}")
    except Exception as e:
        print(f"  Error with {f}: {e}")

print(f"\n{'=' * 60}")
print(f"Done! Moved {moved} duplicate files")

remaining = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f"PDFs remaining: {len(remaining)}")
print(f"{'=' * 60}")