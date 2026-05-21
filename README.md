# Retrieval-Backed Policy QA

## Project Overview

Replayable retrieval-backed QA pipeline for internal policy documents. The
pipeline loads local fixtures, chunks policy text deterministically, retrieves
relevant chunks with TF-IDF, generates grounded extractive answers, validates
citations, and applies safe fallback behavior when evidence is insufficient.

Evaluators can replace `documents.json` and `questions.json`, rerun the
pipeline, and inspect deterministic artifacts in `outputs/`.

## Features

- Deterministic document chunking
- TF-IDF and cosine similarity keyword retrieval
- Separate retrieval, generation, citation validation, decision, and evaluation stages
- Grounded extractive answers using retrieved chunks only
- Controlled vocabularies for statuses, support ratings, and routing actions
- Citation validation against retrieved evidence
- Safe fallback to `insufficient_evidence`

## Repository Structure

```text
documents.json
questions.json
main.py
validate.py
README.md
outputs/
src/
  constants.py
  io_utils.py
  chunking.py
  retrieval.py
  generation.py
  citation_validation.py
  decisions.py
  evaluation.py
  pipeline.py
```

## Pipeline Stages

```text
INIT
INPUTS_LOADED
DOCUMENTS_CHUNKED
INDEX_BUILT
QUESTIONS_LOADED
RETRIEVAL_COMPLETE
ANSWERS_GENERATED
CITATION_VALIDATION_COMPLETE
FINAL_DECISIONS_COMPLETE
EVALUATION_COMPLETE
VALIDATION_COMPLETE
RESULTS_FINALISED
```

## Installation

Requires Python 3.10+.

No external dependencies are required.

## Usage

Run the full pipeline:

```bash
python main.py
python main.py run
```

Validate generated artifacts:

```bash
python main.py validate
```

Ask one ad-hoc question without modifying `questions.json`:

```bash
python main.py ask "What team handles chargeback threats?"
```

## Validation

Validation checks required artifacts, JSON validity, question coverage,
retrieval result shape, controlled vocabularies, citation grounding, fallback
behavior, and evaluation count consistency.

```bash
python validate.py
```

## Output Artifacts

Artifacts are written to `outputs/`:

- `chunks.json`
- `index_metadata.json`
- `retrieval_results.json`
- `generated_answers.json`
- `citation_validation.json`
- `qa_results.json`
- `evaluation_summary.json`

## Design Decisions

- Uses deterministic TF-IDF retrieval instead of embeddings.
- Uses sentence-based chunking with stable chunk IDs.
- Keeps retrieval, generation, validation, decisions, and evaluation separate.
- Generates answers only from retrieved chunks.
- Preserves only validated citations in final results.
- Downgrades unsupported answered outputs to `insufficient_evidence`.

## Limitations

- Answer generation is heuristic and extractive.
- Retrieval is keyword-based and may miss semantic matches.
- Citation validation uses lexical overlap, not deep entailment.
- The pipeline is designed for small local policy fixtures.
