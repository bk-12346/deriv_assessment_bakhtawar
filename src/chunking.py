"""Document chunking stage."""

from __future__ import annotations

import re
from typing import Any

MAX_CHUNK_WORDS = 150


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple punctuation boundaries."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return re.split(r"(?<=[.!?])\s+", normalized)


def word_count(text: str) -> int:
    """Count words using whitespace tokenization."""
    return len(text.split())


def chunk_documents(
    documents: list[dict[str, Any]],
    max_words: int = MAX_CHUNK_WORDS,
) -> list[dict[str, str]]:
    """Create deterministic sentence-grouped chunks from loaded documents."""
    chunks: list[dict[str, str]] = []

    for document in documents:
        doc_id = _required_string(document, "doc_id")
        title = _required_string(document, "title")
        text = _required_string(document, "text")

        current_sentences: list[str] = []
        current_words = 0
        chunk_index = 1

        for sentence in split_sentences(text):
            sentence_words = word_count(sentence)
            if current_sentences and current_words + sentence_words > max_words:
                chunks.append(_make_chunk(doc_id, title, chunk_index, current_sentences))
                chunk_index += 1
                current_sentences = []
                current_words = 0

            current_sentences.append(sentence)
            current_words += sentence_words

        if current_sentences:
            chunks.append(_make_chunk(doc_id, title, chunk_index, current_sentences))

    return chunks


def _make_chunk(
    doc_id: str,
    title: str,
    chunk_index: int,
    sentences: list[str],
) -> dict[str, str]:
    return {
        "chunk_id": f"{doc_id}_chunk_{chunk_index}",
        "doc_id": doc_id,
        "title": title,
        "text": " ".join(sentences),
    }


def _required_string(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"document field '{field}' must be a non-empty string")
    return value.strip()
