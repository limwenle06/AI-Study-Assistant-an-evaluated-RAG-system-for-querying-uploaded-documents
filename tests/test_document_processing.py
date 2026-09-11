from pathlib import Path
import pytest
import src.document_processing as document_processing

class FakePage:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


def create_fake_reader(page_texts: list[str | None]):
    class FakeReader:
        def __init__(self, _pdf_path: Path) -> None:
            self.pages = [FakePage(text) for text in page_texts]

    return FakeReader


def test_extract_pdf_pages_preserves_text_and_metadata(tmp_path, monkeypatch):
    pdf_path = tmp_path / "lecture_notes.pdf"
    pdf_path.write_bytes(b"fake PDF contents")
    fake_reader = create_fake_reader([" First page text. ", "Second page text."])
    monkeypatch.setattr(document_processing, "PdfReader", fake_reader)

    pages = document_processing.extract_pdf_pages(pdf_path)

    assert pages == [
        {
            "document": "lecture_notes.pdf",
            "page": 1,
            "text": "First page text.",
        },
        {
            "document": "lecture_notes.pdf",
            "page": 2,
            "text": "Second page text.",
        },
    ]


def test_extract_pdf_pages_keeps_pages_without_extractable_text(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned_notes.pdf"
    pdf_path.write_bytes(b"fake PDF contents")
    fake_reader = create_fake_reader([None])
    monkeypatch.setattr(document_processing, "PdfReader", fake_reader)

    pages = document_processing.extract_pdf_pages(pdf_path)

    assert pages == [
        {
            "document": "scanned_notes.pdf",
            "page": 1,
            "text": "",
        }
    ]


def test_extract_pdf_pages_rejects_a_missing_file(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        document_processing.extract_pdf_pages(missing_pdf)

def test_extract_pdf_pages_rejects_a_non_pdf_file(tmp_path):
    text_path = tmp_path / "notes.txt"
    text_path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a PDF file"):
        document_processing.extract_pdf_pages(text_path)
