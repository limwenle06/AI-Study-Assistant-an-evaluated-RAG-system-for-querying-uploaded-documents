from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from src.embeddings import EMBEDDING_MODEL, embed_chunks, embed_text


def test_embed_chunks_adds_embeddings_and_preserves_metadata():
    chunks = [
        {
            "document": "lecture1.pdf",
            "page": 1,
            "chunk": 1,
            "text": "Artificial intelligence learns from data.",
        },
        {
            "document": "lecture1.pdf",
            "page": 1,
            "chunk": 2,
            "text": "Machine learning is one area of AI.",
        },
    ]

    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
            SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
        ]
    )

    embedded_chunks = embed_chunks(chunks, client)

    client.embeddings.create.assert_called_once_with(
        model=EMBEDDING_MODEL,
        input=[
            "Artificial intelligence learns from data.",
            "Machine learning is one area of AI.",
        ],
    )
    assert embedded_chunks == [
        {
            "document": "lecture1.pdf",
            "page": 1,
            "chunk": 1,
            "text": "Artificial intelligence learns from data.",
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "document": "lecture1.pdf",
            "page": 1,
            "chunk": 2,
            "text": "Machine learning is one area of AI.",
            "embedding": [0.4, 0.5, 0.6],
        },
    ]
    assert "embedding" not in chunks[0]
    assert "embedding" not in chunks[1]


def test_embed_chunks_returns_empty_list_without_calling_api():
    client = Mock()

    embedded_chunks = embed_chunks([], client)

    assert embedded_chunks == []
    client.embeddings.create.assert_not_called()


def test_embed_text_returns_one_embedding():
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.7, 0.8, 0.9])]
    )

    embedding = embed_text("What is artificial intelligence?", client)

    client.embeddings.create.assert_called_once_with(
        model=EMBEDDING_MODEL,
        input="What is artificial intelligence?",
    )
    assert embedding == [0.7, 0.8, 0.9]


def test_embed_text_rejects_empty_text_without_calling_api():
    client = Mock()

    with pytest.raises(ValueError, match="text cannot be empty"):
        embed_text("   ", client)

    client.embeddings.create.assert_not_called()
