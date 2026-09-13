"""Utilities for retrieving document chunks by vector similarity."""
import numpy as np

def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calculate the cosine similarity between two vectors."""
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same length")

    array_a = np.array(vector_a, dtype=float)
    array_b = np.array(vector_b, dtype=float)

    length_a = np.linalg.norm(array_a)
    length_b = np.linalg.norm(array_b)

    if length_a == 0 or length_b == 0:
        raise ValueError("vectors cannot have zero length")

    dot_product = np.dot(array_a, array_b)

    return float(dot_product / (length_a * length_b))


def get_chunk_score(chunk: dict) -> float:
    """Return a chunk's score for sorting."""
    return chunk["score"]


def retrieve_top_chunks(question_embedding: list[float], embedded_chunks: list[dict], top_k: int = 3) -> list[dict]:
    """Score chunks against a question embedding and return the top results."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    scored_chunks = []

    for chunk in embedded_chunks:
        score = cosine_similarity(question_embedding, chunk["embedding"])

        scored_chunk = chunk.copy()
        scored_chunk["score"] = score
        scored_chunks.append(scored_chunk)

    scored_chunks.sort(key=get_chunk_score, reverse=True)

    return scored_chunks[:top_k]
