"""Citation validation stage."""

from __future__ import annotations

from typing import Any

from src.retrieval import tokenize

MIN_OVERLAP = 1

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


def validate_citations(
    generated_answers: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate generated answer citations against retrieved chunks."""
    retrieval_by_question_id = {
        record.get("question_id"): record for record in retrieval_results
    }

    return [
        validate_answer_citations(answer, retrieval_by_question_id.get(answer.get("question_id")))
        for answer in generated_answers
    ]


def validate_answer_citations(
    answer_record: dict[str, Any],
    retrieval_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate one generated answer."""
    question_id = answer_record.get("question_id")
    status = answer_record.get("status")
    answer_text = str(answer_record.get("answer", "")).strip()
    question_text = str(answer_record.get("question", "")).strip()
    citations = answer_record.get("citations", [])

    unsupported_claims: list[str] = []
    validated_citations: list[str] = []
    notes: list[str] = []

    if retrieval_record is None:
        return _record(question_id, False, ["No retrieval record found."], [], "Missing retrieval record.")

    retrieved_chunks = retrieval_record.get("retrieved_chunks", [])
    if not isinstance(retrieved_chunks, list):
        return _record(question_id, False, ["Retrieved chunks were malformed."], [], "Malformed retrieval record.")

    chunks_by_doc_id = _chunks_by_doc_id(retrieved_chunks)
    retrieved_doc_ids = set(chunks_by_doc_id)

    if status == "answered" and not answer_text:
        unsupported_claims.append("Answered output has an empty answer.")

    if status == "answered" and not citations:
        unsupported_claims.append("Answered output has no citations.")

    if not isinstance(citations, list):
        unsupported_claims.append("Citations field is not a list.")
        citations = []

    for citation in citations:
        if not isinstance(citation, str):
            unsupported_claims.append("Citation value is not a doc_id string.")
            continue
        if citation not in retrieved_doc_ids:
            unsupported_claims.append(f"Cited doc_id was not retrieved: {citation}")
            continue

        if _citation_has_evidence(citation, chunks_by_doc_id, answer_text, question_text):
            validated_citations.append(citation)
        else:
            unsupported_claims.append(f"Cited doc_id lacks lexical support: {citation}")

    if status == "answered" and not validated_citations:
        unsupported_claims.append("Answered output has no validated citations.")

    if unsupported_claims:
        notes.append("Citation validation failed.")
    elif validated_citations:
        notes.append("Citations are retrieved and lexically supported.")
    else:
        notes.append("No citations required for non-answered output.")

    return _record(
        question_id,
        not unsupported_claims,
        unsupported_claims,
        validated_citations,
        " ".join(notes),
    )


def _chunks_by_doc_id(retrieved_chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    chunks: dict[str, list[dict[str, Any]]] = {}
    for chunk in retrieved_chunks:
        doc_id = chunk.get("doc_id")
        if isinstance(doc_id, str):
            chunks.setdefault(doc_id, []).append(chunk)
    return chunks


def _citation_has_evidence(
    citation: str,
    chunks_by_doc_id: dict[str, list[dict[str, Any]]],
    answer_text: str,
    question_text: str,
) -> bool:
    answer_terms = _content_terms(answer_text)
    question_terms = _content_terms(question_text)
    required_terms = answer_terms or question_terms

    if not required_terms:
        return False

    for chunk in chunks_by_doc_id.get(citation, []):
        chunk_terms = _content_terms(str(chunk.get("text", "")))
        if len(chunk_terms.intersection(required_terms)) >= MIN_OVERLAP:
            return True

    return False


def _content_terms(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1}


def _record(
    question_id: Any,
    passed: bool,
    unsupported_claims: list[str],
    validated_citations: list[str],
    notes: str,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "citation_validation_passed": passed,
        "unsupported_claims": unsupported_claims,
        "validated_citations": validated_citations,
        "notes": notes,
    }
