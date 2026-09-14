from unittest.mock import Mock

import pytest

import src.pipeline as pipeline


def test_process_document_connects_extraction_chunking_and_embeddings(monkeypatch):
    client = Mock()
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
    embed_chunks = Mock(return_value=embedded_chunks)
    monkeypatch.setattr(pipeline, "extract_pdf_pages", extract_pdf_pages)
    monkeypatch.setattr(pipeline, "chunk_pages", chunk_pages)
    monkeypatch.setattr(pipeline, "embed_chunks", embed_chunks)

    result = pipeline.process_document("lecture1.pdf", client)

    extract_pdf_pages.assert_called_once_with("lecture1.pdf")
    chunk_pages.assert_called_once_with(extracted_pages)
    embed_chunks.assert_called_once_with(chunks, client)
    assert result == embedded_chunks


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
