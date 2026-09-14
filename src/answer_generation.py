"""Utilities for generating answers grounded in retrieved document chunks."""
from openai import OpenAI

GENERATION_MODEL = "gpt-5.6-luna"
ANSWER_INSTRUCTIONS = (
    "You are an AI study assistant. Answer the question using only the provided sources. "
    "Cite supporting sources using labels such as [Source 1]. "
    "If the sources do not contain enough information, say: "
    '"I could not find enough information in the provided document."'
)

def format_sources(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks as labelled sources for the model."""
    formatted_sources = []

    for source_number, chunk in enumerate(retrieved_chunks, start=1):
        formatted_source = (
            f"[Source {source_number}]\n"
            f"Document: {chunk['document']}\n"
            f"Page: {chunk['page']}\n"
            f"Chunk: {chunk['chunk']}\n"
            f"Text: {chunk['text']}"
        )
        formatted_sources.append(formatted_source)

    return "\n\n".join(formatted_sources)

def generate_answer(question: str, retrieved_chunks: list[dict], client: OpenAI, model: str = GENERATION_MODEL) -> str:
    """Generate an answer using only the supplied document chunks."""
    if not question.strip():
        raise ValueError("question cannot be empty")

    if not retrieved_chunks:
        raise ValueError("retrieved_chunks cannot be empty")

    sources = format_sources(retrieved_chunks)
    prompt = f"Question:\n{question}\n\nSources:\n{sources}"

    response = client.responses.create(
        model=model,
        instructions=ANSWER_INSTRUCTIONS,
        input=prompt,
        store=False,
    )

    return response.output_text
