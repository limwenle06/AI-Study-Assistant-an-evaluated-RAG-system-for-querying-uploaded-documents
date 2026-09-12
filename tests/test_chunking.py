import pytest

from src.chunking import chunk_pages


def test_short_page_produces_one_chunk():
    extracted_pages = [
        {
            "document": "lecture1.pdf",
            "page": 1,
            "text": "Artificial intelligence learns patterns from data.",
        }
    ]

    chunks = chunk_pages(extracted_pages, chunk_size_words=10, overlap_words=2)

    assert chunks == [
        {
            "document": "lecture1.pdf",
            "page": 1,
            "chunk": 1,
            "text": "Artificial intelligence learns patterns from data.",
        }
    ]


def test_long_page_produces_overlapping_chunks():
    extracted_pages = [
        {
            "document": "lecture1.pdf",
            "page": 3,
            "text": "one two three four five six seven eight",
        }
    ]

    chunks = chunk_pages(extracted_pages, chunk_size_words=4, overlap_words=2)

    assert chunks == [
        {
            "document": "lecture1.pdf",
            "page": 3,
            "chunk": 1,
            "text": "one two three four",
        },
        {
            "document": "lecture1.pdf",
            "page": 3,
            "chunk": 2,
            "text": "three four five six",
        },
        {
            "document": "lecture1.pdf",
            "page": 3,
            "chunk": 3,
            "text": "five six seven eight",
        },
    ]


def test_empty_page_produces_no_chunks():
    extracted_pages = [
        {
            "document": "scanned_notes.pdf",
            "page": 2,
            "text": "",
        }
    ]

    chunks = chunk_pages(extracted_pages)

    assert chunks == []


def test_invalid_chunk_settings_raise_errors():
    with pytest.raises(ValueError, match="greater than zero"):
        chunk_pages([], chunk_size_words=0)

    with pytest.raises(ValueError, match="cannot be negative"):
        chunk_pages([], overlap_words=-1)

    with pytest.raises(ValueError, match="must be smaller"):
        chunk_pages([], chunk_size_words=40, overlap_words=40)
