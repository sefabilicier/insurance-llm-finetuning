"""
Before/After report generator for fine-tuning evaluation.

Produces:
- JSON report with all metrics
- Markdown summary for RESULTS.md
- Side-by-side comparison examples
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate comprehensive evaluation reports."""

    def __init__(self, output_dir: str = "./outputs/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        task_metrics: Dict[str, Any],
        forgetting_metrics: Optional[Dict[str, Any]] = None,
        comparisons: Optional[List[Dict[str, Any]]] = None,
        training_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """
        Generate full evaluation report.

        Args:
            task_metrics: Results from compute_task_metrics
            forgetting_metrics: Results from compute_forgetting_metrics
            comparisons: Before/after example comparisons
            training_info: Training config and metadata

        Returns:
            Dict of output file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build full report
        report = {
            "metadata": {
                "timestamp": timestamp,
                "model": training_info.get("model", "unknown") if training_info else "unknown",
                "phase": training_info.get("phase", "unknown") if training_info else "unknown",
            },
            "task_metrics": task_metrics,
            "forgetting": forgetting_metrics,
            "comparisons": comparisons,
            "training_info": training_info,
        }

        # Save JSON report
        json_path = self.output_dir / f"eval_report_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Generate Markdown report
        md_path = self.output_dir / f"eval_report_{timestamp}.md"
        self._write_markdown_report(report, md_path)

        logger.info(f"✓ Reports saved:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  Markdown: {md_path}")

        return {"json": json_path, "markdown": md_path}

    def _write_markdown_report(
        self,
        report: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """Write Markdown-formatted evaluation report."""
        lines = []

        # Header
        lines.append("# Evaluation Report")
        lines.append("")
        lines.append(f"**Date:** {report['metadata']['timestamp']}")
        lines.append(f"**Model:** {report['metadata']['model']}")
        lines.append(f"**Phase:** {report['metadata']['phase']}")
        lines.append("")

        # Overall metrics
        lines.append("## Overall Metrics")
        lines.append("")

        overall = report.get("task_metrics", {}).get("overall", {})
        if overall:
            lines.append("| Metric | Score |")
            lines.append("|---|---|")
            lines.append(f"| ROUGE-1 | {overall.get('rouge_1', 0):.4f} |")
            lines.append(f"| ROUGE-2 | {overall.get('rouge_2', 0):.4f} |")
            lines.append(f"| ROUGE-L | {overall.get('rouge_l', 0):.4f} |")
            lines.append(f"| BLEU | {overall.get('bleu', 0):.4f} |")
            lines.append(f"| Keyword Coverage | {overall.get('keyword_coverage', 0):.4f} |")
            lines.append(f"| Format Compliance | {overall.get('format_compliance', 0):.4f} |")
            lines.append(f"| Examples | {overall.get('num_examples', 0)} |")
            lines.append("")

        # Per-category metrics
        per_cat = report.get("task_metrics", {}).get("per_category", {})
        if per_cat:
            lines.append("## Per-Category Metrics")
            lines.append("")
            lines.append("| Category | ROUGE-L | BLEU | Keywords | Format | N |")
            lines.append("|---|---|---|---|---|---|")

            for cat, metrics in sorted(per_cat.items()):
                lines.append(
                    f"| {cat} "
                    f"| {metrics.get('rouge_l', 0):.4f} "
                    f"| {metrics.get('bleu', 0):.4f} "
                    f"| {metrics.get('keyword_coverage', 0):.4f} "
                    f"| {metrics.get('format_compliance', 0):.4f} "
                    f"| {metrics.get('num_examples', 0)} |"
                )
            lines.append("")

        # Forgetting analysis
        forgetting = report.get("forgetting")
        if forgetting:
            lines.append("## Catastrophic Forgetting Analysis")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            lines.append(f"| Baseline Accuracy | {forgetting.get('baseline_accuracy', 0):.2%} |")
            lines.append(f"| Fine-tuned Accuracy | {forgetting.get('finetuned_accuracy', 0):.2%} |")
            lines.append(f"| Absolute Drop | {forgetting.get('absolute_drop', 0):.2%} |")
            lines.append(f"| Relative Drop | {forgetting.get('relative_drop', 0):.2%} |")
            lines.append(f"| Severity | {forgetting.get('severity', 'unknown')} |")
            lines.append(f"| Within 5% Threshold | {'✅ Yes' if forgetting.get('threshold_5pct') else '❌ No'} |")
            lines.append("")
            lines.append(f"**Assessment:** {forgetting.get('message', '')}")
            lines.append("")

        # Comparisons
        comparisons = report.get("comparisons", [])
        if comparisons:
            lines.append("## Before/After Comparisons")
            lines.append("")

            for i, comp in enumerate(comparisons[:5], 1):  # Show top 5
                lines.append(f"### Example {i}: {comp.get('category', 'unknown')}")
                lines.append("")
                lines.append(f"**User:** {comp.get('user', '')}")
                lines.append("")
                lines.append(f"**Baseline:** {comp.get('baseline_response', 'N/A')}")
                lines.append("")
                lines.append(f"**Fine-tuned:** {comp.get('finetuned_response', 'N/A')}")
                lines.append("")
                lines.append(f"**Reference:** {comp.get('reference', 'N/A')}")
                lines.append("")
                lines.append("---")
                lines.append("")

        # Write file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def generate_before_after_report(
    test_examples: List[Dict[str, str]],
    baseline_responses: List[str],
    finetuned_responses: List[str],
    output_dir: str = "./outputs/evaluation",
    training_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Generate a complete before/after comparison report.

    Args:
        test_examples: Test set examples with 'user', 'assistant', 'category'
        baseline_responses: Responses from base model
        finetuned_responses: Responses from fine-tuned model
        output_dir: Output directory
        training_info: Training metadata

    Returns:
        Dict of report file paths
    """
    from .metrics import compute_task_metrics

    references = [ex["assistant"] for ex in test_examples]
    categories = [ex.get("category", "unknown") for ex in test_examples]

    # Compute metrics for both
    baseline_metrics = compute_task_metrics(
        baseline_responses, references, categories
    )
    finetuned_metrics = compute_task_metrics(
        finetuned_responses, references, categories
    )

    # Build comparisons
    comparisons = []
    for i, ex in enumerate(test_examples):
        comparisons.append({
            "user": ex["user"],
            "category": ex.get("category", "unknown"),
            "reference": ex["assistant"],
            "baseline_response": baseline_responses[i] if i < len(baseline_responses) else "N/A",
            "finetuned_response": finetuned_responses[i] if i < len(finetuned_responses) else "N/A",
        })

    # Generate report
    reporter = EvaluationReporter(output_dir=output_dir)

    combined_metrics = {
        "overall": {
            "baseline": baseline_metrics["overall"],
            "finetuned": finetuned_metrics["overall"],
        },
        "per_category": {
            "baseline": baseline_metrics["per_category"],
            "finetuned": finetuned_metrics["per_category"],
        },
    }

    return reporter.generate_report(
        task_metrics=finetuned_metrics,
        comparisons=comparisons,
        training_info=training_info,
    )