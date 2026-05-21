"""Answer generation stage."""

from __future__ import annotations

import re
from typing import Any

from src.constants import ANSWER_STATUSES, ROUTING_ACTIONS, SUPPORT_RATINGS
from src.retrieval import tokenize

MIN_TOP_SCORE = 0.25
MIN_SENTENCE_OVERLAP = 1
MAX_EVIDENCE_SENTENCES = 2

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "for",
    "in",
    "is",
    "it",
    "of",
    "or",
    "should",
    "that",
    "the",
    "their",
    "to",
    "what",
    "when",
    "where",
    "who",
    "will",
    "with",
}

ROUTE_KEYWORDS = [
    ("account security", "account_security"),
    ("billing operations", "billing_operations"),
    ("risk and payments", "risk_and_payments"),
    ("compliance operations", "compliance_operations"),
    ("frontend investigation", "frontend_investigation"),
]


def generate_answers(retrieval_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate conservative extractive answers from retrieved chunks only."""
    return [generate_answer(record) for record in retrieval_results]


def generate_answer(retrieval_record: dict[str, Any]) -> dict[str, Any]:
    """Generate one deterministic answer from one retrieval result."""
    question = _required_string(retrieval_record, "question")
    retrieved_chunks = retrieval_record.get("retrieved_chunks", [])
    if not isinstance(retrieved_chunks, list) or not retrieved_chunks:
        return _abstain(retrieval_record, "No retrieved chunks were available.")

    top_score = _score(retrieved_chunks[0])
    evidence_sentences = _select_evidence_sentences(question, retrieved_chunks)
    route_action = _route_action(evidence_sentences)
    citations = _citations_for_sentences(evidence_sentences)

    if _should_abstain(question, retrieved_chunks, top_score, evidence_sentences):
        if route_action == "none":
            route_action = _route_action_from_chunks(retrieved_chunks)
        return {
            "question_id": retrieval_record.get("question_id"),
            "question": question,
            "status": "insufficient_evidence",
            "answer": "",
            "support_rating": "none",
            "citations": citations,
            "route_action": _allowed(route_action, ROUTING_ACTIONS, "route_action"),
            "reason": "Retrieved evidence was weak or did not clearly answer the question.",
        }

    answer = " ".join(sentence["text"] for sentence in evidence_sentences)
    status = "answered" if answer else "insufficient_evidence"
    support_rating = _support_rating(top_score, len(evidence_sentences), status)

    return {
        "question_id": retrieval_record.get("question_id"),
        "question": question,
        "status": _allowed(status, ANSWER_STATUSES, "status"),
        "answer": answer,
        "support_rating": _allowed(support_rating, SUPPORT_RATINGS, "support_rating"),
        "citations": citations,
        "route_action": _allowed(route_action, ROUTING_ACTIONS, "route_action"),
    }


def _select_evidence_sentences(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    question_terms = _content_terms(question)
    candidates = []

    for chunk_position, chunk in enumerate(retrieved_chunks):
        chunk_text = _required_string(chunk, "text")
        for sentence_position, sentence in enumerate(_split_sentences(chunk_text)):
            sentence_terms = _content_terms(sentence)
            overlap = len(question_terms.intersection(sentence_terms))
            if {"maximum", "size"}.intersection(question_terms) and "mb" in sentence_terms:
                overlap += 2
            if overlap >= MIN_SENTENCE_OVERLAP:
                candidates.append(
                    {
                        "overlap": overlap,
                        "chunk_position": chunk_position,
                        "sentence_position": sentence_position,
                        "doc_id": _required_string(chunk, "doc_id"),
                        "text": sentence,
                    }
                )

    candidates.sort(
        key=lambda item: (
            -int(item["overlap"]),
            int(item["chunk_position"]),
            int(item["sentence_position"]),
        )
    )

    if not candidates:
        return []

    best_overlap = int(candidates[0]["overlap"])
    focused_candidates = [
        candidate for candidate in candidates if int(candidate["overlap"]) == best_overlap
    ]

    return [
        {"doc_id": str(candidate["doc_id"]), "text": str(candidate["text"])}
        for candidate in focused_candidates[:MAX_EVIDENCE_SENTENCES]
    ]


def _should_abstain(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    top_score: float,
    evidence_sentences: list[dict[str, str]],
) -> bool:
    if top_score < MIN_TOP_SCORE or not evidence_sentences:
        return True

    question_terms = _content_terms(question)
    top_text = _required_string(retrieved_chunks[0], "text").lower()
    if "available" in question_terms or "availability" in question_terms:
        missing_terms = [term for term in question_terms if term not in top_text]
        if missing_terms and "must not guess" in top_text:
            return True

    return False


def _support_rating(top_score: float, evidence_count: int, status: str) -> str:
    if status != "answered":
        return "none"
    if top_score >= 0.45:
        return "strong"
    if top_score >= 0.25 and evidence_count > 0:
        return "partial"
    return "weak"


def _route_action(evidence_sentences: list[dict[str, str]]) -> str:
    for sentence in evidence_sentences:
        sentence_text = sentence["text"].lower()
        for keyword, action in ROUTE_KEYWORDS:
            if keyword in sentence_text:
                return action
    return "none"


def _route_action_from_chunks(retrieved_chunks: list[dict[str, Any]]) -> str:
    for chunk in retrieved_chunks:
        chunk_text = _required_string(chunk, "text").lower()
        for keyword, action in ROUTE_KEYWORDS:
            if keyword in chunk_text:
                return action
    return "none"


def _citations_for_sentences(evidence_sentences: list[dict[str, str]]) -> list[str]:
    citations = []
    for sentence in evidence_sentences:
        doc_id = sentence["doc_id"]
        if doc_id not in citations:
            citations.append(doc_id)
    return citations


def _content_terms(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1}


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return re.split(r"(?<=[.!?])\s+", normalized)


def _score(chunk: dict[str, Any]) -> float:
    score = chunk.get("score", 0.0)
    if isinstance(score, int | float):
        return float(score)
    return 0.0


def _abstain(retrieval_record: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "question_id": retrieval_record.get("question_id"),
        "question": _required_string(retrieval_record, "question"),
        "status": "insufficient_evidence",
        "answer": "",
        "support_rating": "none",
        "citations": [],
        "route_action": "none",
        "reason": reason,
    }


def _required_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"field '{field}' must be a non-empty string")
    return value.strip()


def _allowed(value: str, allowed_values: list[str], field: str) -> str:
    if value not in allowed_values:
        raise ValueError(f"{field} must be one of {allowed_values}")
    return value
