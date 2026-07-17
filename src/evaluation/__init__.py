"""
Evaluation module for insurance LLM fine-tuning pipeline.

Provides text similarity metrics (ROUGE, BLEU), domain-specific
evaluation (keyword coverage, format compliance), catastrophic
forgetting analysis, and report generation.
"""

from .metrics import (
    compute_rouge,
    compute_rouge_l,
    compute_rouge_n,
    compute_bleu,
    compute_task_metrics,
    compute_keyword_coverage,
    compute_format_compliance,
    run_inference,
)
from .forgetting import (
    evaluate_multiple_choice,
    compute_forgetting_metrics,
    save_forgetting_report,
)
from .reporter import (
    EvaluationReporter,
    generate_before_after_report,
)

__all__ = [
    "compute_rouge",
    "compute_rouge_l",
    "compute_rouge_n",
    "compute_bleu",
    "compute_task_metrics",
    "compute_keyword_coverage",
    "compute_format_compliance",
    "run_inference",
    "evaluate_multiple_choice",
    "compute_forgetting_metrics",
    "save_forgetting_report",
    "EvaluationReporter",
    "generate_before_after_report",
]

__version__ = "0.1.0"