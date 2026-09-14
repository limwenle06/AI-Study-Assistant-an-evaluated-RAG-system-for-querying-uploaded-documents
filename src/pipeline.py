"""Coordinate document processing, retrieval, and answer generation."""
from pathlib import Path
from openai import OpenAI

from src.answer_generation import generate_answer
from src.chunking import chunk_pages
from src.document_processing import extract_pdf_pages
from src.embeddings import embed_chunks, embed_text
from src.retrieval import retrieve_top_chunks


def process_document(pdf_path: str | Path, client: OpenAI) -> list[dict]:
    """Extract, chunk, and embed a PDF for reuse across questions."""
    extracted_pages = extract_pdf_pages(pdf_path)
    chunks = chunk_pages(extracted_pages)

    return embed_chunks(chunks, client)


def answer_question(question: str, embedded_chunks: list[dict], client: OpenAI, top_k: int = 3) -> dict:
    """Retrieve relevant chunks and generate a grounded answer."""
    if not embedded_chunks:
        raise ValueError("embedded_chunks cannot be empty")

    question_embedding = embed_text(question, client)
    retrieved_chunks = retrieve_top_chunks(question_embedding,
                                           embedded_chunks,
                                           top_k=top_k,)
    
    answer = generate_answer(question, retrieved_chunks, client)

    sources = []

    for chunk in retrieved_chunks:
        source = {
            "document": chunk["document"],
            "page": chunk["page"],
            "chunk": chunk["chunk"],
            "text": chunk["text"],
            "score": chunk["score"],
        }
        sources.append(source)

    return {
        "answer": answer,
        "sources": sources,
    }
