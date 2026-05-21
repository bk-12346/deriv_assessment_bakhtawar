"""Validation entry point for generated artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.constants import ANSWER_STATUSES, DOCUMENTS_PATH, QUESTIONS_PATH, ROUTING_ACTIONS
from src.constants import SUPPORT_RATINGS
from src.io_utils import load_json, require_list

OUTPUTS_DIR = Path("outputs")
ARTIFACTS = {
    "chunks": OUTPUTS_DIR / "chunks.json",
    "index_metadata": OUTPUTS_DIR / "index_metadata.json",
    "retrieval_results": OUTPUTS_DIR / "retrieval_results.json",
    "generated_answers": OUTPUTS_DIR / "generated_answers.json",
    "citation_validation": OUTPUTS_DIR / "citation_validation.json",
    "qa_results": OUTPUTS_DIR / "qa_results.json",
    "evaluation_summary": OUTPUTS_DIR / "evaluation_summary.json",
}


def main() -> None:
    """Validate pipeline outputs."""
    data = _load_required_data()

    _validate_chunks(data["documents"], data["chunks"])
    _validate_question_coverage(data["questions"], data)
    _validate_retrieval(data["retrieval_results"])
    _validate_generated_answers(data["generated_answers"], data["retrieval_results"])
    _validate_citation_validation(data["citation_validation"], data["retrieval_results"])
    _validate_final_results(data["qa_results"], data["citation_validation"])
    _validate_evaluation_summary(data["evaluation_summary"], data["questions"], data["qa_results"], data["citation_validation"])

    print("Validation passed.")


def _load_required_data() -> dict[str, Any]:
    for path in [Path(DOCUMENTS_PATH), Path(QUESTIONS_PATH), *ARTIFACTS.values()]:
        _require(path.exists(), f"Missing required file: {path}")

    return {
        "documents": require_list(load_json(DOCUMENTS_PATH), "documents"),
        "questions": require_list(load_json(QUESTIONS_PATH), "questions"),
        "chunks": require_list(load_json(ARTIFACTS["chunks"]), "chunks"),
        "index_metadata": load_json(ARTIFACTS["index_metadata"]),
        "retrieval_results": require_list(load_json(ARTIFACTS["retrieval_results"]), "retrieval_results"),
        "generated_answers": require_list(load_json(ARTIFACTS["generated_answers"]), "generated_answers"),
        "citation_validation": require_list(load_json(ARTIFACTS["citation_validation"]), "citation_validation"),
        "qa_results": require_list(load_json(ARTIFACTS["qa_results"]), "qa_results"),
        "evaluation_summary": load_json(ARTIFACTS["evaluation_summary"]),
    }


def _validate_chunks(documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    document_ids = {_required_string(document, "doc_id") for document in documents}
    chunk_doc_ids = {_required_string(chunk, "doc_id") for chunk in chunks}

    _require(document_ids == chunk_doc_ids, "Not all documents were chunked with doc_id preserved.")

    for chunk in chunks:
        for field in ["chunk_id", "doc_id", "title", "text"]:
            _required_string(chunk, field)


def _validate_question_coverage(questions: list[dict[str, Any]], data: dict[str, Any]) -> None:
    question_ids = {question.get("question_id") for question in questions}
    for key in ["retrieval_results", "generated_answers", "citation_validation", "qa_results"]:
        processed_ids = {record.get("question_id") for record in data[key]}
        _require(processed_ids == question_ids, f"{key} does not process all questions.")


def _validate_retrieval(retrieval_results: list[dict[str, Any]]) -> None:
    for result in retrieval_results:
        chunks = result.get("retrieved_chunks")
        _require(isinstance(chunks, list), "retrieved_chunks must be a list.")
        _require(len(chunks) <= 3, "retrieval results contain more than 3 chunks.")
        for chunk in chunks:
            score = chunk.get("score")
            _require(isinstance(score, (int, float)), "retrieval score must be numeric.")
            for field in ["chunk_id", "doc_id", "text"]:
                _required_string(chunk, field)


def _validate_generated_answers(
    generated_answers: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
) -> None:
    retrieved_doc_ids = _retrieved_doc_ids_by_question(retrieval_results)

    for answer in generated_answers:
        question_id = answer.get("question_id")
        status = _required_allowed(answer, "status", ANSWER_STATUSES)
        _required_allowed(answer, "support_rating", SUPPORT_RATINGS)
        _required_allowed(answer, "route_action", ROUTING_ACTIONS)

        answer_text = str(answer.get("answer", "")).strip()
        if status == "answered":
            _require(answer_text, f"Answered output has empty answer: {question_id}")

        citations = answer.get("citations")
        _require(isinstance(citations, list), f"citations must be a list: {question_id}")
        for citation in citations:
            _require(citation in retrieved_doc_ids[question_id], f"Citation was not retrieved: {citation}")


def _validate_citation_validation(
    citation_validation: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
) -> None:
    retrieved_doc_ids = _retrieved_doc_ids_by_question(retrieval_results)

    for record in citation_validation:
        question_id = record.get("question_id")
        _require(isinstance(record.get("citation_validation_passed"), bool), "citation pass flag must be boolean.")
        _require(isinstance(record.get("unsupported_claims"), list), "unsupported_claims must be a list.")
        validated = record.get("validated_citations")
        _require(isinstance(validated, list), "validated_citations must be a list.")
        _require(isinstance(record.get("notes"), str), "notes must be a string.")
        for citation in validated:
            _require(citation in retrieved_doc_ids[question_id], f"Validated citation was not retrieved: {citation}")


def _validate_final_results(
    qa_results: list[dict[str, Any]],
    citation_validation: list[dict[str, Any]],
) -> None:
    validation_by_id = {record.get("question_id"): record for record in citation_validation}

    for result in qa_results:
        question_id = result.get("question_id")
        final_status = _required_allowed(result, "final_status", ANSWER_STATUSES)
        _required_allowed(result, "support_rating", SUPPORT_RATINGS)
        _required_allowed(result, "routing_action", ROUTING_ACTIONS)

        _require(isinstance(result.get("citation_validation_passed"), bool), "final citation flag must be boolean.")
        _require(isinstance(result.get("fallback_applied"), bool), "fallback_applied must be boolean.")
        _require(isinstance(result.get("final_citations"), list), "final_citations must be a list.")

        if final_status == "answered":
            _require(str(result.get("final_answer", "")).strip(), f"Final answered output is empty: {question_id}")

        validation = validation_by_id[question_id]
        if validation.get("citation_validation_passed") is not True:
            _require(
                result.get("final_status") != "answered" and result.get("fallback_applied") is True,
                f"Citation validation failure not reflected in final output: {question_id}",
            )

        expected_citations = validation.get("validated_citations", [])
        _require(result.get("final_citations") == expected_citations, f"Final citations are not validated only: {question_id}")


def _validate_evaluation_summary(
    summary: dict[str, Any],
    questions: list[dict[str, Any]],
    qa_results: list[dict[str, Any]],
    citation_validation: list[dict[str, Any]],
) -> None:
    _require(isinstance(summary, dict), "evaluation summary must be an object.")

    answered_count = sum(1 for result in qa_results if result.get("final_status") == "answered")
    insufficient_count = sum(1 for result in qa_results if result.get("final_status") == "insufficient_evidence")
    failure_count = sum(1 for record in citation_validation if record.get("citation_validation_passed") is not True)

    _require(summary.get("total_questions") == len(questions), "total_questions is inconsistent.")
    _require(summary.get("answered_count") == answered_count, "answered_count is inconsistent.")
    _require(
        summary.get("insufficient_evidence_count") == insufficient_count,
        "insufficient_evidence_count is inconsistent.",
    )
    _require(
        summary.get("citation_validation_failure_count") == failure_count,
        "citation_validation_failure_count is inconsistent.",
    )
    _require(
        sum(summary.get("routing_action_distribution", {}).values()) == len(qa_results),
        "routing action distribution is inconsistent.",
    )
    _require(
        sum(summary.get("support_rating_distribution", {}).values()) == len(qa_results),
        "support rating distribution is inconsistent.",
    )
    _require(
        sum(summary.get("expected_behavior_comparison", {}).values()) == len(questions),
        "expected behavior comparison is inconsistent.",
    )


def _retrieved_doc_ids_by_question(retrieval_results: list[dict[str, Any]]) -> dict[Any, set[str]]:
    retrieved_doc_ids: dict[Any, set[str]] = {}
    for result in retrieval_results:
        retrieved_doc_ids[result.get("question_id")] = {
            chunk.get("doc_id")
            for chunk in result.get("retrieved_chunks", [])
            if isinstance(chunk.get("doc_id"), str)
        }
    return retrieved_doc_ids


def _required_allowed(record: dict[str, Any], field: str, allowed: list[str]) -> str:
    value = record.get(field)
    _require(value in allowed, f"{field} has invalid value: {value}")
    return str(value)


def _required_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must be a non-empty string.")
    return value.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
