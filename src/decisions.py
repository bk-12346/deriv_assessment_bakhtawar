"""Final answer decision stage."""

from __future__ import annotations

from typing import Any

from src.constants import ANSWER_STATUSES, ROUTING_ACTIONS, SUPPORT_RATINGS


def make_final_decisions(
    generated_answers: list[dict[str, Any]],
    citation_validation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine generated answers and citation checks into final QA results."""
    validation_by_question_id = {
        record.get("question_id"): record for record in citation_validation
    }

    return [
        make_final_decision(answer, validation_by_question_id.get(answer.get("question_id")))
        for answer in generated_answers
    ]


def make_final_decision(
    answer: dict[str, Any],
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create one final decision record."""
    validation_passed = bool(validation and validation.get("citation_validation_passed") is True)
    validated_citations = _validated_citations(validation)
    original_status = _allowed(
        str(answer.get("status", "error")),
        ANSWER_STATUSES,
        "status",
    )

    fallback_applied = original_status == "answered" and not validation_passed

    if fallback_applied:
        final_status = "insufficient_evidence"
        final_answer = ""
        support_rating = "none"
    else:
        final_status = original_status
        final_answer = str(answer.get("answer", "")).strip()
        support_rating = _allowed(
            str(answer.get("support_rating", "none")),
            SUPPORT_RATINGS,
            "support_rating",
        )

    if final_status == "answered" and not final_answer:
        final_status = "insufficient_evidence"
        support_rating = "none"
        fallback_applied = True

    return {
        "question_id": answer.get("question_id"),
        "final_status": _allowed(final_status, ANSWER_STATUSES, "final_status"),
        "final_answer": final_answer if final_status == "answered" else "",
        "final_citations": validated_citations,
        "support_rating": support_rating,
        "routing_action": _allowed(
            str(answer.get("route_action", "none")),
            ROUTING_ACTIONS,
            "routing_action",
        ),
        "citation_validation_passed": validation_passed,
        "fallback_applied": fallback_applied,
    }


def _validated_citations(validation: dict[str, Any] | None) -> list[str]:
    if not validation:
        return []

    citations = validation.get("validated_citations", [])
    if not isinstance(citations, list):
        return []

    valid_citations = []
    for citation in citations:
        if isinstance(citation, str) and citation not in valid_citations:
            valid_citations.append(citation)
    return valid_citations


def _allowed(value: str, allowed_values: list[str], field: str) -> str:
    if value not in allowed_values:
        raise ValueError(f"{field} must be one of {allowed_values}")
    return value
