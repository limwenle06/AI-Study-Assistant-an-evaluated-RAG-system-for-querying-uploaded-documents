"""Coordinate document processing, retrieval, and answer generation."""
from pathlib import Path
from openai import OpenAI

from src.answer_generation import generate_answer
from src.chunking import chunk_pages
from src.database import DATABASE_PATH, calculate_file_hash, find_document_by_hash, load_document_chunks, save_document
from src.document_processing import extract_pdf_pages
from src.embeddings import EMBEDDING_MODEL, embed_chunks, embed_text
from src.retrieval import retrieve_top_chunks


def process_document(pdf_path: str | Path, client: OpenAI, db_path: str | Path = DATABASE_PATH) -> list[dict]:
    """Load a saved PDF or process and save it for future questions."""
    file_hash = calculate_file_hash(pdf_path)
    saved_document = find_document_by_hash(file_hash, db_path=db_path)

    if saved_document is not None:
        if saved_document["embedding_model"] != EMBEDDING_MODEL:
            raise ValueError("saved document uses a different embedding model")

        return load_document_chunks(saved_document["id"], db_path=db_path)

    extracted_pages = extract_pdf_pages(pdf_path)
    chunks = chunk_pages(extracted_pages)
    embedded_chunks = embed_chunks(chunks, client)

    save_document(
        filename=Path(pdf_path).name,
        file_hash=file_hash,
        embedded_chunks=embedded_chunks,
        db_path=db_path,
        embedding_model=EMBEDDING_MODEL,
    )

    return embedded_chunks


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
