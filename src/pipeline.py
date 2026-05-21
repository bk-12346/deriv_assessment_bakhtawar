"""Pipeline orchestration."""

from __future__ import annotations

from src.chunking import chunk_documents
from src.citation_validation import validate_citations
from src.constants import DOCUMENTS_PATH, QUESTIONS_PATH
from src.decisions import make_final_decisions
from src.evaluation import evaluate_results
from src.generation import generate_answers
from src.io_utils import ensure_outputs_dir, iso_timestamp, load_json, require_list, write_json
from src.retrieval import build_tfidf_index, retrieve_for_questions
from src.support_signals import build_support_signals


def run_pipeline() -> dict[str, object]:
    """Run the full deterministic QA pipeline."""
    outputs_dir = ensure_outputs_dir()
    artifacts = [
        "chunks.json",
        "index_metadata.json",
        "retrieval_results.json",
        "generated_answers.json",
        "citation_validation.json",
        "qa_results.json",
        "evaluation_summary.json",
        "support_signals.json",
    ]

    documents = require_list(load_json(DOCUMENTS_PATH), "documents")
    chunks = chunk_documents(documents)
    index = build_tfidf_index(chunks)
    questions = require_list(load_json(QUESTIONS_PATH), "questions")
    retrieval_results = retrieve_for_questions(index, questions)
    generated_answers = generate_answers(retrieval_results)
    citation_validation = validate_citations(generated_answers, retrieval_results)
    qa_results = make_final_decisions(generated_answers, citation_validation)
    evaluation_summary = evaluate_results(questions, qa_results, citation_validation)
    support_signals = build_support_signals(retrieval_results, qa_results)

    write_json(outputs_dir / "chunks.json", chunks)
    write_json(
        outputs_dir / "index_metadata.json",
        {
            "retrieval_method": "keyword",
            "chunk_count": len(chunks),
            "build_timestamp": iso_timestamp(),
        },
    )
    write_json(outputs_dir / "retrieval_results.json", retrieval_results)
    write_json(outputs_dir / "generated_answers.json", generated_answers)
    write_json(outputs_dir / "citation_validation.json", citation_validation)
    write_json(outputs_dir / "qa_results.json", qa_results)
    write_json(outputs_dir / "evaluation_summary.json", evaluation_summary)
    write_json(outputs_dir / "support_signals.json", support_signals)

    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "question_count": len(questions),
        "answered_count": evaluation_summary["answered_count"],
        "insufficient_evidence_count": evaluation_summary["insufficient_evidence_count"],
        "citation_validation_failure_count": evaluation_summary[
            "citation_validation_failure_count"
        ],
        "artifacts": [str(outputs_dir / artifact) for artifact in artifacts],
    }


def answer_adhoc_question(question_text: str) -> dict[str, object]:
    """Answer one ad-hoc question using the same pipeline stages."""
    run_pipeline()

    documents = require_list(load_json(DOCUMENTS_PATH), "documents")
    chunks = chunk_documents(documents)
    index = build_tfidf_index(chunks)
    question = {"question_id": "adhoc_1", "question": question_text}
    retrieval_results = retrieve_for_questions(index, [question])
    generated_answers = generate_answers(retrieval_results)
    citation_validation = validate_citations(generated_answers, retrieval_results)
    qa_results = make_final_decisions(generated_answers, citation_validation)

    return qa_results[0]
