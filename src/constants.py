"""Shared constants for pipeline stages and artifact paths."""

ANSWER_STATUSES = [
    "answered",
    "insufficient_evidence",
    "conflicting_evidence",
    "error",
]

SUPPORT_RATINGS = [
    "strong",
    "partial",
    "weak",
    "none",
]

ROUTING_ACTIONS = [
    "none",
    "account_security",
    "billing_operations",
    "risk_and_payments",
    "compliance_operations",
    "frontend_investigation",
]

PIPELINE_STAGES = [
    "INIT",
    "INPUTS_LOADED",
    "DOCUMENTS_CHUNKED",
    "INDEX_BUILT",
    "QUESTIONS_LOADED",
    "RETRIEVAL_COMPLETE",
    "ANSWERS_GENERATED",
    "CITATION_VALIDATION_COMPLETE",
    "FINAL_DECISIONS_COMPLETE",
    "EVALUATION_COMPLETE",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]

STAGES = PIPELINE_STAGES

DOCUMENTS_PATH = "documents.json"
QUESTIONS_PATH = "questions.json"
OUTPUTS_DIR = "outputs"
