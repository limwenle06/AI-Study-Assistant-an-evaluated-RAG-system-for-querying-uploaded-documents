"""Utilities for extracting page-aware text from PDF documents."""

from pathlib import Path
from pypdf import PdfReader

def extract_pdf_pages(pdf_path: str | Path) -> list[dict[str, str | int]]:

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    reader = PdfReader(path)
    extracted_pages: list[dict[str, str | int]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text() or ""

        page_record = {
            "document": path.name,
            "page": page_number,
            "text": extracted_text.strip(),
        }
        extracted_pages.append(page_record)

    return extracted_pages
