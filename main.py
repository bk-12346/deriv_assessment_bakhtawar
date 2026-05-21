"""Entry point for the retrieval-backed QA pipeline."""

from __future__ import annotations

import argparse

from src.pipeline import answer_adhoc_question, run_pipeline
from validate import main as validate_artifacts


def main() -> None:
    """Run the pipeline CLI."""
    parser = argparse.ArgumentParser(description="Retrieval-backed policy QA")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("run", help="Run the full pipeline")
    subparsers.add_parser("validate", help="Validate generated artifacts")
    ask_parser = subparsers.add_parser("ask", help="Answer one ad-hoc question")
    ask_parser.add_argument("question", help="Question text")

    args = parser.parse_args()

    if args.command in (None, "run"):
        _run_and_print_summary()
    elif args.command == "validate":
        validate_artifacts()
    elif args.command == "ask":
        result = answer_adhoc_question(args.question)
        _print_adhoc_result(result)


def _run_and_print_summary() -> None:
    summary = run_pipeline()
    print("Pipeline complete.")
    print(f"Documents: {summary['document_count']}")
    print(f"Chunks: {summary['chunk_count']}")
    print(f"Questions: {summary['question_count']}")
    print(f"Answered: {summary['answered_count']}")
    print(f"Insufficient evidence: {summary['insufficient_evidence_count']}")
    print(f"Citation validation failures: {summary['citation_validation_failure_count']}")
    print("Artifacts written to outputs/.")


def _print_adhoc_result(result: dict[str, object]) -> None:
    print(f"Status: {result['final_status']}")
    print(f"Answer: {result['final_answer']}")
    print(f"Citations: {', '.join(result['final_citations'])}")
    print(f"Support rating: {result['support_rating']}")
    print(f"Routing action: {result['routing_action']}")


if __name__ == "__main__":
    main()
