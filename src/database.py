"""Store processed documents and chunk embeddings in SQLite."""
import hashlib
import json
import sqlite3
from pathlib import Path

from src.embeddings import EMBEDDING_MODEL

DATABASE_PATH = Path("data/study_assistant.db")
def calculate_file_hash(file_path: str | Path) -> str:
    """Calculate a SHA-256 fingerprint from a file's contents."""
    path = Path(file_path)
    file_hash = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            file_bytes = file.read(8192) 

            if not file_bytes:
                break

            file_hash.update(file_bytes)

    return file_hash.hexdigest()


def initialize_database(db_path: str | Path = DATABASE_PATH) -> None:
    """Create the SQLite database and its tables if they do not exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute("PRAGMA foreign_keys = ON") #Default sqlite setting
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    embedding_model TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE,
                    UNIQUE (document_id, page_number, chunk_number)
                )
                """
            )
    finally:
        connection.close()


def save_document(
    filename: str,
    file_hash: str,
    embedded_chunks: list[dict],
    db_path: str | Path = DATABASE_PATH,
    embedding_model: str = EMBEDDING_MODEL,
) -> int:
    """Save one document and all its embedded chunks."""
    if not embedded_chunks:
        raise ValueError("embedded_chunks cannot be empty")

    initialize_database(db_path)

    connection = sqlite3.connect(db_path)
    try:
        with connection:
            connection.execute("PRAGMA foreign_keys = ON")
            cursor = connection.execute(
                """
                INSERT INTO documents (filename, file_hash, embedding_model)
                VALUES (?, ?, ?)
                """,
                (filename, file_hash, embedding_model),
            )
            document_id = cursor.lastrowid

            for chunk in embedded_chunks:
                connection.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        page_number,
                        chunk_number,
                        text,
                        embedding
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        chunk["page"],
                        chunk["chunk"],
                        chunk["text"],
                        json.dumps(chunk["embedding"]),
                    ),
                )
    finally:
        connection.close()

    return int(document_id)


def find_document_by_hash(file_hash: str, db_path: str | Path = DATABASE_PATH) -> dict | None:
    """Find a saved document using its file fingerprint."""
    initialize_database(db_path)

    connection = sqlite3.connect(db_path)
    try:
        with connection:
            row = connection.execute(
                """
                SELECT id, filename, file_hash, embedding_model
                FROM documents
                WHERE file_hash = ?
                """,
                (file_hash,),
            ).fetchone()
    finally:
        connection.close()

    if row is None:
        return None

    return {
        "id": row[0],
        "filename": row[1],
        "file_hash": row[2],
        "embedding_model": row[3],
    }


def load_document_chunks(document_id: int, db_path: str | Path = DATABASE_PATH) -> list[dict]:
    """Load a saved document's chunks and convert embeddings back to lists."""
    initialize_database(db_path)

    connection = sqlite3.connect(db_path)
    try:
        with connection:
            rows = connection.execute(
                """
                SELECT
                    documents.filename,
                    chunks.page_number,
                    chunks.chunk_number,
                    chunks.text,
                    chunks.embedding
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE documents.id = ?
                ORDER BY chunks.page_number, chunks.chunk_number
                """,
                (document_id,),
            ).fetchall()
    finally:
        connection.close()

    loaded_chunks = []
    for row in rows:
        chunk = {
            "document": row[0],
            "page": row[1],
            "chunk": row[2],
            "text": row[3],
            "embedding": json.loads(row[4]),
        }
        loaded_chunks.append(chunk)

    return loaded_chunks
