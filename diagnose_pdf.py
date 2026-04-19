"""
diagnose_pdf.py
---------------
Quick diagnostic — checks what text PyMuPDF actually extracted
from the PDF and whether pages are empty (scanned images).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/input/pdfs/celir_case.pdf"

doc = fitz.open(pdf_path)
print(f"Total pages: {len(doc)}")
print(f"\nFirst 5 pages text preview:")
print("="*60)

empty_pages = 0
for i in range(min(5, len(doc))):
    page = doc[i]
    text = page.get_text("text").strip()
    word_count = len(text.split())
    print(f"\nPage {i+1}: {word_count} words")
    if word_count < 10:
        empty_pages += 1
        print(" EMPTY — likely scanned image")
    else:
        print(f"  Preview: {text[:200]}")

# Check overall
total_empty = 0
for i in range(len(doc)):
    text = doc[i].get_text("text").strip()
    if len(text.split()) < 10:
        total_empty += 1

print(f"\n{'='*60}")
print(f"Summary: {total_empty}/{len(doc)} pages are empty/scanned")
if total_empty > len(doc) * 0.5:
    print("PDF is mostly scanned — Tesseract OCR is required")
else:
    print("PDF has extractable text — OCR not needed")