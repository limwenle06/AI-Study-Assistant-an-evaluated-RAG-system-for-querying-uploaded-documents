from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.pipeline as pipeline
from src.database import calculate_file_hash, find_document_by_hash, load_document_chunks, save_document
from src.embeddings import EMBEDDING_MODEL


def test_process_document_saves_new_pdf_and_reuses_it_later(monkeypatch, tmp_path):
    pdf_path = tmp_path / "lecture1.pdf"
    pdf_path.write_bytes(b"sample PDF contents")
    db_path = tmp_path / "test.db"
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.7, 0.3])]
    )
    extracted_pages = [
        {
            "document": "lecture1.pdf",
            "page": 1,
            "text": "Artificial intelligence learns from data.",
        }
    ]
    chunks = [
        {
            "document": "lecture1.pdf",
            "page": 1,
            "chunk": 1,
            "text": "Artificial intelligence learns from data.",
        }
    ]
    embedded_chunks = [
        {
            **chunks[0],
            "embedding": [0.7, 0.3],
        }
    ]

    extract_pdf_pages = Mock(return_value=extracted_pages)
    chunk_pages = Mock(return_value=chunks)
    monkeypatch.setattr(pipeline, "extract_pdf_pages", extract_pdf_pages)
    monkeypatch.setattr(pipeline, "chunk_pages", chunk_pages)

    first_result = pipeline.process_document(pdf_path, client, db_path=db_path)

    extract_pdf_pages.assert_called_once_with(pdf_path)
    chunk_pages.assert_called_once_with(extracted_pages)
    client.embeddings.create.assert_called_once_with(
        model=EMBEDDING_MODEL,
        input=["Artificial intelligence learns from data."],
    )
    assert first_result == embedded_chunks

    saved_document = find_document_by_hash(calculate_file_hash(pdf_path), db_path=db_path)
    assert saved_document["filename"] == "lecture1.pdf"
    assert saved_document["embedding_model"] == EMBEDDING_MODEL
    assert load_document_chunks(saved_document["id"], db_path=db_path) == embedded_chunks

    new_client = Mock()
    second_result = pipeline.process_document(pdf_path, new_client, db_path=db_path)

    assert second_result == embedded_chunks
    extract_pdf_pages.assert_called_once_with(pdf_path)
    chunk_pages.assert_called_once_with(extracted_pages)
    new_client.embeddings.create.assert_not_called()


def test_process_document_reprocesses_changed_pdf(monkeypatch, tmp_path):
    pdf_path = tmp_path / "lecture1.pdf"
    pdf_path.write_bytes(b"original PDF contents")
    db_path = tmp_path / "test.db"
    old_hash = calculate_file_hash(pdf_path)
    save_document(
        "lecture1.pdf",
        old_hash,
        [{
            "page": 1,
            "chunk": 1,
            "text": "Original text.",
            "embedding": [0.1, 0.2],
        }],
        db_path=db_path,
    )
    pdf_path.write_bytes(b"changed PDF contents")

    extracted_pages = [{"document": "lecture1.pdf", "page": 1, "text": "Updated text."}]
    chunks = [{"document": "lecture1.pdf", "page": 1, "chunk": 1, "text": "Updated text."}]
    monkeypatch.setattr(pipeline, "extract_pdf_pages", Mock(return_value=extracted_pages))
    monkeypatch.setattr(pipeline, "chunk_pages", Mock(return_value=chunks))
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.3, 0.4])]
    )

    result = pipeline.process_document(pdf_path, client, db_path=db_path)

    assert result[0]["text"] == "Updated text."
    client.embeddings.create.assert_called_once()
    assert find_document_by_hash(old_hash, db_path=db_path) is not None
    assert find_document_by_hash(calculate_file_hash(pdf_path), db_path=db_path) is not None


def test_process_document_rejects_different_saved_embedding_model(monkeypatch, tmp_path):
    pdf_path = tmp_path / "lecture1.pdf"
    pdf_path.write_bytes(b"sample PDF contents")
    db_path = tmp_path / "test.db"
    save_document(
        "lecture1.pdf",
        calculate_file_hash(pdf_path),
        [{
            "page": 1,
            "chunk": 1,
            "text": "Artificial intelligence learns from data.",
            "embedding": [0.7, 0.3],
        }],
        db_path=db_path,
        embedding_model="older-embedding-model",
    )
    extract_pdf_pages = Mock()
    monkeypatch.setattr(pipeline, "extract_pdf_pages", extract_pdf_pages)
    client = Mock()

    with pytest.raises(ValueError, match="different embedding model"):
        pipeline.process_document(pdf_path, client, db_path=db_path)

    extract_pdf_pages.assert_not_called()
    client.embeddings.create.assert_not_called()


def test_answer_question_connects_retrieval_and_generation(monkeypatch):
    client = Mock()
    embedded_chunks = [
        {
            "document": "lecture1.pdf",
            "page": 4,
            "chunk": 1,
            "text": "Poor-quality training data can produce a poor model.",
            "embedding": [0.7, 0.3],
        }
    ]
    retrieved_chunks = [
        {
            **embedded_chunks[0],
            "score": 0.91,
        }
    ]

    embed_text = Mock(return_value=[0.8, 0.2])
    retrieve_top_chunks = Mock(return_value=retrieved_chunks)
    generate_answer = Mock(
        return_value="Poor-quality data can produce a poor model. [Source 1]"
    )
    monkeypatch.setattr(pipeline, "embed_text", embed_text)
    monkeypatch.setattr(pipeline, "retrieve_top_chunks", retrieve_top_chunks)
    monkeypatch.setattr(pipeline, "generate_answer", generate_answer)

    result = pipeline.answer_question(
        "Why is training-data quality important?",
        embedded_chunks,
        client,
        top_k=1,
    )

    embed_text.assert_called_once_with(
        "Why is training-data quality important?",
        client,
    )
    retrieve_top_chunks.assert_called_once_with(
        [0.8, 0.2],
        embedded_chunks,
        top_k=1,
    )
    generate_answer.assert_called_once_with(
        "Why is training-data quality important?",
        retrieved_chunks,
        client,
    )
    assert result == {
        "answer": "Poor-quality data can produce a poor model. [Source 1]",
        "sources": [
            {
                "document": "lecture1.pdf",
                "page": 4,
                "chunk": 1,
                "text": "Poor-quality training data can produce a poor model.",
                "score": 0.91,
            }
        ],
    }


def test_answer_question_rejects_empty_embedded_chunks_without_api_calls():
    client = Mock()

    with pytest.raises(ValueError, match="embedded_chunks cannot be empty"):
        pipeline.answer_question(
            "What is artificial intelligence?",
            [],
            client,
        )

    client.embeddings.create.assert_not_called()
    client.responses.create.assert_not_called()
