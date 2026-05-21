"""TF-IDF and cosine similarity retrieval stage."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

TOP_K = 3
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def build_tfidf_index(chunks: list[dict[str, str]]) -> dict[str, Any]:
    """Build a minimal TF-IDF index over chunk text."""
    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
    doc_count = len(tokenized_chunks)
    document_frequencies: Counter[str] = Counter()

    for tokens in tokenized_chunks:
        document_frequencies.update(set(tokens))

    idf = {
        token: math.log((1 + doc_count) / (1 + frequency)) + 1
        for token, frequency in document_frequencies.items()
    }

    vectors = [_tfidf_vector(tokens, idf) for tokens in tokenized_chunks]

    return {
        "chunks": chunks,
        "idf": idf,
        "vectors": vectors,
    }


def retrieve_for_questions(
    index: dict[str, Any],
    questions: list[dict[str, Any]],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """Retrieve the top chunks for each question."""
    results = []

    for question in questions:
        question_id = question.get("question_id")
        question_text = _required_string(question, "question")
        retrieved_chunks = retrieve(index, question_text, top_k=top_k)
        results.append(
            {
                "question_id": question_id,
                "question": question_text,
                "retrieved_chunks": retrieved_chunks,
            }
        )

    return results


def retrieve(index: dict[str, Any], question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Retrieve chunks by TF-IDF cosine similarity."""
    chunks = index["chunks"]
    idf = index["idf"]
    chunk_vectors = index["vectors"]
    query_vector = _tfidf_vector(tokenize(question), idf)

    scored_chunks = []
    for position, (chunk, chunk_vector) in enumerate(zip(chunks, chunk_vectors)):
        score = cosine_similarity(query_vector, chunk_vector)
        scored_chunks.append((score, position, chunk))

    scored_chunks.sort(key=lambda item: (-item[0], item[1]))

    return [
        {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "score": float(score),
            "text": chunk["text"],
        }
        for score, _position, chunk in scored_chunks[: min(top_k, len(scored_chunks))]
    ]


def tokenize(text: str) -> list[str]:
    """Tokenize text for keyword retrieval."""
    return TOKEN_PATTERN.findall(text.lower())


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """Compute cosine similarity for sparse vectors."""
    if not left or not right:
        return 0.0

    dot_product = sum(weight * right.get(token, 0.0) for token, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = len(tokens)
    return {
        token: (count / total) * idf[token]
        for token, count in counts.items()
        if token in idf
    }


def _required_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"question field '{field}' must be a non-empty string")
    return value.strip()
