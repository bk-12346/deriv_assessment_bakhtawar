"""Entry point for the retrieval-backed QA pipeline."""

from src.pipeline import run_pipeline


def main() -> None:
    """Run the pipeline."""
    summary = run_pipeline()
    print("Pipeline complete.")
    print(f"Documents: {summary['document_count']}")
    print(f"Chunks: {summary['chunk_count']}")
    print(f"Questions: {summary['question_count']}")
    print(f"Answered: {summary['answered_count']}")
    print(f"Insufficient evidence: {summary['insufficient_evidence_count']}")
    print(f"Citation validation failures: {summary['citation_validation_failure_count']}")
    print("Artifacts written to outputs/.")


if __name__ == "__main__":
    main()
