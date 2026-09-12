"""Utilities for splitting extracted PDF pages into smaller text chunks."""


def chunk_pages(extracted_pages: list[dict], chunk_size_words: int = 200, overlap_words: int = 40) -> list[dict]:
    """Split page text into overlapping word-based chunks.

    Each chunk keeps the source document and page number from its original
    page record.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    chunks = []

    for page_record in extracted_pages:
        words = page_record["text"].split()

        if not words:
            continue

        start_index = 0
        chunk_number = 1

        while start_index < len(words):
            end_index = start_index + chunk_size_words
            chunk_words = words[start_index:end_index]

            chunk_record = {
                "document": page_record["document"],
                "page": page_record["page"],
                "chunk": chunk_number,
                "text": " ".join(chunk_words)
            }
            chunks.append(chunk_record)

            if end_index >= len(words):
                break

            start_index = end_index - overlap_words
            chunk_number += 1

    return chunks
