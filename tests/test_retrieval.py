import pytest

from src.retrieval import cosine_similarity, retrieve_top_chunks


def test_cosine_similarity_compares_vector_directions():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [1, 1]) == pytest.approx(0.7071, abs=0.0001)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_retrieve_top_chunks_returns_highest_scores_first():
    embedded_chunks = [
        {
            "document": "lecture1.pdf",
            "page": 4,
            "chunk": 1,
            "text": "Unrelated topic.",
            "embedding": [0, 1],
        },
        {
            "document": "lecture1.pdf",
            "page": 10,
            "chunk": 1,
            "text": "Document chunking divides text into smaller sections.",
            "embedding": [1, 0],
        },
        {
            "document": "lecture1.pdf",
            "page": 7,
            "chunk": 1,
            "text": "Partially related topic.",
            "embedding": [1, 1],
        },
    ]

    results = retrieve_top_chunks([1, 0], embedded_chunks, top_k=2)

    assert len(results) == 2
    assert results[0]["page"] == 10
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["page"] == 7
    assert results[1]["score"] == pytest.approx(0.7071, abs=0.0001)
    assert "score" not in embedded_chunks[0]
    assert "score" not in embedded_chunks[1]
    assert "score" not in embedded_chunks[2]


def test_retrieval_rejects_invalid_values():
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1, 0], [1, 0, 1])

    with pytest.raises(ValueError, match="zero length"):
        cosine_similarity([0, 0], [1, 0])

    with pytest.raises(ValueError, match="greater than zero"):
        retrieve_top_chunks([1, 0], [], top_k=0)
