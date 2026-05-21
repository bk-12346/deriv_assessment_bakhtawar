"""Lightweight support signal heuristics."""

from __future__ import annotations

from typing import Any


def build_support_signals(
    retrieval_results: list[dict[str, Any]],
    qa_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create deterministic confidence signals from retrieval and final answers."""
    results_by_question_id = {result.get("question_id"): result for result in qa_results}
    signals = []

    for retrieval in retrieval_results:
        question_id = retrieval.get("question_id")
        chunks = retrieval.get("retrieved_chunks", [])
        if not isinstance(chunks, list):
            chunks = []

        top_score = _chunk_score(chunks, 0)
        second_score = _chunk_score(chunks, 1)
        score_gap = top_score - second_score
        final_result = results_by_question_id.get(question_id, {})
        final_status = final_result.get("final_status")
        support_rating = str(final_result.get("support_rating", "none"))
        likely_answerable = final_status != "insufficient_evidence"

        signals.append(
            {
                "question_id": question_id,
                "top_score": top_score,
                "second_score": second_score,
                "score_gap": score_gap,
                "retrieved_chunk_count": len(chunks),
                "support_rating": support_rating,
                "likely_answerable": likely_answerable,
                "notes": _notes(top_score, score_gap, len(chunks), final_status),
            }
        )

    return signals


def _chunk_score(chunks: list[dict[str, Any]], index: int) -> float:
    if index >= len(chunks):
        return 0.0
    score = chunks[index].get("score", 0.0)
    if isinstance(score, (int, float)):
        return float(score)
    return 0.0


def _notes(top_score: float, score_gap: float, chunk_count: int, final_status: Any) -> str:
    if final_status == "insufficient_evidence":
        return "Final status is insufficient_evidence; likely_answerable is false."
    if chunk_count < 3:
        return "Fewer than three chunks were retrieved; confidence is lower."
    if top_score < 0.25:
        return "Low top retrieval score; confidence is weaker."
    if score_gap >= 0.15:
        return "Top retrieval score is clearly separated from the second result."
    return "Retrieval support is present but score separation is limited."
