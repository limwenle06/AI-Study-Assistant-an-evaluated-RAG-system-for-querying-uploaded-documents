import sqlite3

import pytest

from src.database import (
    calculate_file_hash,
    find_document_by_hash,
    initialize_database,
    load_document_chunks,
    save_document,
)
from src.embeddings import EMBEDDING_MODEL


def test_initialize_database_creates_documents_and_chunks_tables(tmp_path):
    db_path = tmp_path / "test.db"

    initialize_database(db_path)

    with sqlite3.connect(db_path) as connection:
        table_rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name IN ('documents', 'chunks')
            ORDER BY name
            """
        ).fetchall()

    assert table_rows == [("chunks",), ("documents",)]


def test_calculate_file_hash_uses_file_contents(tmp_path):
    first_file = tmp_path / "lecture.pdf"
    renamed_copy = tmp_path / "lecture-copy.pdf"
    changed_file = tmp_path / "lecture-edited.pdf"
    first_file.write_bytes(b"same PDF contents")
    renamed_copy.write_bytes(b"same PDF contents")
    changed_file.write_bytes(b"different PDF contents")

    first_hash = calculate_file_hash(first_file)
    renamed_copy_hash = calculate_file_hash(renamed_copy)
    changed_hash = calculate_file_hash(changed_file)

    assert first_hash == renamed_copy_hash
    assert first_hash != changed_hash


def test_save_find_and_load_document(tmp_path):
    db_path = tmp_path / "test.db"
    embedded_chunks = [
        {
            "document": "lecture1.pdf",
            "page": 1,
            "chunk": 1,
            "text": "Artificial intelligence learns from data.",
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "document": "lecture1.pdf",
            "page": 2,
            "chunk": 1,
            "text": "Training-data quality affects model quality.",
            "embedding": [0.4, 0.5, 0.6],
        },
    ]

    document_id = save_document(
        "lecture1.pdf",
        "sample-hash",
        embedded_chunks,
        db_path=db_path,
    )

    document = find_document_by_hash("sample-hash", db_path=db_path)
    loaded_chunks = load_document_chunks(document_id, db_path=db_path)

    assert document == {
        "id": document_id,
        "filename": "lecture1.pdf",
        "file_hash": "sample-hash",
        "embedding_model": EMBEDDING_MODEL,
    }
    assert loaded_chunks == embedded_chunks


def test_find_document_by_hash_returns_none_when_missing(tmp_path):
    db_path = tmp_path / "test.db"

    document = find_document_by_hash("missing-hash", db_path=db_path)

    assert document is None


def test_save_document_rejects_empty_chunks(tmp_path):
    db_path = tmp_path / "test.db"

    with pytest.raises(ValueError, match="embedded_chunks cannot be empty"):
        save_document(
            "lecture1.pdf",
            "sample-hash",
            [],
            db_path=db_path,
        )
