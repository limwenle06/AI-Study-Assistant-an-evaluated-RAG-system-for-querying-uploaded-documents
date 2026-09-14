from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.answer_generation import ANSWER_INSTRUCTIONS, GENERATION_MODEL, format_sources, generate_answer


def make_retrieved_chunks():
    return [
        {
            "document": "lecture1.pdf",
            "page": 4,
            "chunk": 1,
            "text": "Poor-quality training data can produce a poor model.",
            "embedding": [0.7, 0.3],
            "score": 0.91,
        },
        {
            "document": "lecture1.pdf",
            "page": 5,
            "chunk": 1,
            "text": "Machine-learning models learn patterns from examples.",
            "embedding": [0.6, 0.4],
            "score": 0.82,
        },
    ]


def test_format_sources_labels_chunks_and_includes_metadata():
    retrieved_chunks = make_retrieved_chunks()

    sources = format_sources(retrieved_chunks)

    assert "[Source 1]" in sources
    assert "Document: lecture1.pdf" in sources
    assert "Page: 4" in sources
    assert "Chunk: 1" in sources
    assert "Poor-quality training data can produce a poor model." in sources
    assert "[Source 2]" in sources
    assert "Page: 5" in sources


def test_generate_answer_sends_grounded_prompt_and_returns_text():
    retrieved_chunks = make_retrieved_chunks()
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="Training-data quality affects model quality. [Source 1]"
    )

    answer = generate_answer(
        "Why is training-data quality important?",
        retrieved_chunks,
        client,
    )

    expected_sources = format_sources(retrieved_chunks)
    expected_prompt = (
        "Question:\nWhy is training-data quality important?\n\n"
        f"Sources:\n{expected_sources}"
    )
    client.responses.create.assert_called_once_with(
        model=GENERATION_MODEL,
        instructions=ANSWER_INSTRUCTIONS,
        input=expected_prompt,
        store=False,
    )
    assert answer == "Training-data quality affects model quality. [Source 1]"


def test_generate_answer_rejects_empty_question_without_calling_api():
    client = Mock()

    with pytest.raises(ValueError, match="question cannot be empty"):
        generate_answer("   ", make_retrieved_chunks(), client)

    client.responses.create.assert_not_called()


def test_generate_answer_rejects_empty_chunks_without_calling_api():
    client = Mock()

    with pytest.raises(ValueError, match="retrieved_chunks cannot be empty"):
        generate_answer("What is artificial intelligence?", [], client)

    client.responses.create.assert_not_called()
