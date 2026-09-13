"""Utilities for generating embeddings for document chunks."""

from openai import OpenAI
EMBEDDING_MODEL = "text-embedding-3-small"

def embed_chunks(chunks: list[dict], client: OpenAI, model: str = EMBEDDING_MODEL) -> list[dict]:
    """Generate an embedding for each chunk while preserving its metadata."""
    if not chunks:
        return []

    chunk_texts = []

    for chunk in chunks:
        chunk_texts.append(chunk["text"])

    response = client.embeddings.create(
        model=model,
        input=chunk_texts,
    )

    embedded_chunks = []

    for index, chunk in enumerate(chunks):
        embedded_chunk = chunk.copy()
        embedded_chunk["embedding"] = response.data[index].embedding
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks
