"""
MediCare AI - Hybrid Text Extraction Module
- Normal PDFs -> PyMuPDF (Fast text layer extraction)
- Images (PNG/JPG) -> Tesseract OCR
- Scanned PDFs -> PyMuPDF Page Render + Tesseract OCR
"""

import os
import io
import shutil
import pymupdf
from PIL import Image

try:
    import pytesseract
    # Auto-detect Tesseract installation path on Windows
    if os.name == "nt":
        default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(default_win_path):
            pytesseract.pytesseract.tesseract_cmd = default_win_path
    
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False


def ocr_image_bytes(image_bytes: bytes) -> str:
    """Extract text from raw image bytes using Tesseract OCR."""
    if not pytesseract:
        raise RuntimeError("pytesseract library is not installed.")
    
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
        
    text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
    return (text or "").strip()


def extract_document_text(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Extract text from file.
    Returns tuple: (extracted_text, method_used)
    """
    fname = (filename or "").lower()

    # 1. IMAGES -> Tesseract OCR
    if fname.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
        text = ocr_image_bytes(file_bytes)
        return text, "tesseract_image"

    # 2. PDFs -> PyMuPDF First, Fallback to Tesseract if Scanned
    if fname.endswith(".pdf"):
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text_parts.append(page.get_text() or "")
        doc.close()

        extracted_text = "\n".join(text_parts).strip()

        # If text PDF was readable
        if len(extracted_text) >= 50:
            return extracted_text, "pymupdf"

        # Scanned PDF Fallback -> Render pages to images & run Tesseract OCR
        print("  PDF has little text layer (scanned). Running Tesseract OCR on pages...")
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        ocr_parts = []
        mat = pymupdf.Matrix(2.0, 2.0)  # 2x zoom for high accuracy OCR
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_bytes = pix.tobytes("png")
            page_text = ocr_image_bytes(png_bytes)
            if page_text:
                ocr_parts.append(page_text)
        doc.close()

        return "\n\n".join(ocr_parts).strip(), "tesseract_scanned_pdf"

    raise ValueError("Unsupported file format. Please upload PDF or PNG/JPG image.")