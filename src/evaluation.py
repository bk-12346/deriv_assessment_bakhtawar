"""Evaluation stage."""

from __future__ import annotations

from collections import Counter
from typing import Any


def evaluate_results(
    questions: list[dict[str, Any]],
    qa_results: list[dict[str, Any]],
    citation_validation: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic evaluation summary."""
    results_by_question_id = {result.get("question_id"): result for result in qa_results}

    expected_behavior = {
        "answerable_as_answered": 0,
        "answerable_not_answered": 0,
        "insufficient_evidence_not_answered": 0,
        "insufficient_evidence_as_answered": 0,
        "unknown_expected_behavior": 0,
    }

    for question in questions:
        result = results_by_question_id.get(question.get("question_id"), {})
        final_status = result.get("final_status")
        expected = question.get("expected_behavior")

        if expected == "answerable" and final_status == "answered":
            expected_behavior["answerable_as_answered"] += 1
        elif expected == "answerable":
            expected_behavior["answerable_not_answered"] += 1
        elif expected == "insufficient_evidence" and final_status != "answered":
            expected_behavior["insufficient_evidence_not_answered"] += 1
        elif expected == "insufficient_evidence":
            expected_behavior["insufficient_evidence_as_answered"] += 1
        else:
            expected_behavior["unknown_expected_behavior"] += 1

    status_counts = Counter(result.get("final_status") for result in qa_results)
    citation_failures = sum(
        1
        for record in citation_validation
        if record.get("citation_validation_passed") is not True
    )

    return {
        "total_questions": len(questions),
        "answered_count": status_counts.get("answered", 0),
        "insufficient_evidence_count": status_counts.get("insufficient_evidence", 0),
        "citation_validation_failure_count": citation_failures,
        "routing_action_distribution": dict(
            sorted(Counter(result.get("routing_action") for result in qa_results).items())
        ),
        "support_rating_distribution": dict(
            sorted(Counter(result.get("support_rating") for result in qa_results).items())
        ),
        "expected_behavior_comparison": expected_behavior,
    }
