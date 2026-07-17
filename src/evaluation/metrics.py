"""
Evaluation metrics for insurance LLM fine-tuning.

Provides:
- ROUGE (L, 1, 2) for text similarity
- BLEU for n-gram precision
- Task-specific metrics (format compliance, keyword accuracy)
- Inference utilities for generating predictions
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# TEXT SIMILARITY METRICS
# ============================================================

def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from token list."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L (Longest Common Subsequence) F1 score.

    Args:
        prediction: Generated text
        reference: Reference text

    Returns:
        ROUGE-L F1 score (0-1)
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    precision = lcs_length / m if m > 0 else 0
    recall = lcs_length / n if n > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_rouge_n(prediction: str, reference: str, n: int = 1) -> float:
    """
    Compute ROUGE-N F1 score.

    Args:
        prediction: Generated text
        reference: Reference text
        n: N-gram size (1 for ROUGE-1, 2 for ROUGE-2)

    Returns:
        ROUGE-N F1 score (0-1)
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0

    pred_ngrams = _get_ngrams(pred_tokens, n)
    ref_ngrams = _get_ngrams(ref_tokens, n)

    overlap = sum((pred_ngrams & ref_ngrams).values())
    pred_count = sum(pred_ngrams.values())
    ref_count = sum(ref_ngrams.values())

    precision = overlap / pred_count if pred_count > 0 else 0
    recall = overlap / ref_count if ref_count > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """Compute all ROUGE variants."""
    return {
        "rouge_1": compute_rouge_n(prediction, reference, n=1),
        "rouge_2": compute_rouge_n(prediction, reference, n=2),
        "rouge_l": compute_rouge_l(prediction, reference),
    }


def compute_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """
    Compute BLEU score with brevity penalty.

    Args:
        prediction: Generated text
        reference: Reference text
        max_n: Maximum n-gram order

    Returns:
        BLEU score (0-1)
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))

    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        if len(pred_tokens) < n:
            precisions.append(0.0)
            continue

        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())

        precisions.append(overlap / total if total > 0 else 0.0)

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(np.log(p) for p in precisions) / len(precisions)
    bleu = bp * np.exp(log_avg)

    return float(bleu)


# ============================================================
# TASK-SPECIFIC METRICS (Insurance Domain)
# ============================================================

# Keywords expected in professional insurance responses
INSURANCE_KEYWORDS = {
    "policy_inquiry": [
        "policy", "coverage", "deductible", "premium", "limit",
        "insured", "terms", "renewal", "benefit",
    ],
    "claim_processing": [
        "claim", "file", "submit", "document", "adjuster",
        "process", "approval", "status", "evidence",
    ],
    "coverage_questions": [
        "covered", "coverage", "protection", "exclude",
        "liability", "comprehensive", "collision", "benefit",
    ],
    "premium_billing": [
        "premium", "payment", "billing", "invoice", "due",
        "discount", "installment", "renewal", "rate",
    ],
    "policy_modifications": [
        "change", "modify", "update", "cancel", "add",
        "remove", "adjust", "transfer", "endorsement",
    ],
}


def compute_keyword_coverage(
    prediction: str,
    category: str,
    min_keywords: int = 2,
) -> Dict[str, Any]:
    """
    Check if response contains expected domain keywords.

    Args:
        prediction: Generated response
        category: Insurance category
        min_keywords: Minimum keywords expected

    Returns:
        Dict with coverage score and found keywords
    """
    keywords = INSURANCE_KEYWORDS.get(category, [])
    if not keywords:
        return {"score": 1.0, "found": [], "missing": [], "total": 0}

    pred_lower = prediction.lower()
    found = [kw for kw in keywords if kw in pred_lower]
    missing = [kw for kw in keywords if kw not in pred_lower]

    score = len(found) / max(min_keywords, 1)
    score = min(score, 1.0)  # Cap at 1.0

    return {
        "score": score,
        "found": found,
        "missing": missing,
        "total": len(keywords),
    }


def compute_format_compliance(prediction: str) -> Dict[str, Any]:
    """
    Check if response follows professional format guidelines.

    Checks:
    - Minimum length (at least 20 words)
    - No excessive repetition
    - Contains greeting or professional tone
    - Ends with closing or follow-up offer
    - No placeholder text

    Returns:
        Dict with compliance score and details
    """
    checks = {}

    # Length check
    word_count = len(prediction.split())
    checks["adequate_length"] = word_count >= 20

    # Repetition check
    words = prediction.lower().split()
    if words:
        word_freq = Counter(words)
        max_freq = max(word_freq.values())
        checks["no_excessive_repetition"] = max_freq / len(words) < 0.15

    # Professional tone indicators
    professional_phrases = [
        "thank you", "i can", "happy to help", "assist",
        "let me", "please", "your policy", "your claim",
        "i understand", "certainly", "of course",
    ]
    pred_lower = prediction.lower()
    checks["professional_tone"] = any(p in pred_lower for p in professional_phrases)

    # Closing check
    closing_phrases = [
        "anything else", "further assistance", "don't hesitate",
        "contact us", "help you", "let me know", "questions",
    ]
    checks["has_closing"] = any(p in pred_lower for p in closing_phrases)

    # No placeholders
    placeholder_patterns = ["[placeholder]", "[todo]", "[fill in]", "lorem ipsum"]
    checks["no_placeholders"] = not any(p in pred_lower for p in placeholder_patterns)

    # Overall score
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    score = passed / total if total > 0 else 0.0

    return {
        "score": score,
        "checks": checks,
        "passed": passed,
        "total": total,
    }


def compute_task_metrics(
    predictions: List[str],
    references: List[str],
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute all metrics across a dataset.

    Args:
        predictions: List of generated responses
        references: List of reference responses
        categories: Optional list of category labels

    Returns:
        Aggregated metrics dict
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    n = len(predictions)
    if categories is None:
        categories = ["unknown"] * n

    # Collect per-example metrics
    all_rouge_1, all_rouge_2, all_rouge_l = [], [], []
    all_bleu = []
    all_keyword_scores = []
    all_format_scores = []
    category_metrics = defaultdict(lambda: defaultdict(list))

    for pred, ref, cat in zip(predictions, references, categories):
        # ROUGE
        rouge = compute_rouge(pred, ref)
        all_rouge_1.append(rouge["rouge_1"])
        all_rouge_2.append(rouge["rouge_2"])
        all_rouge_l.append(rouge["rouge_l"])

        # BLEU
        bleu = compute_bleu(pred, ref)
        all_bleu.append(bleu)

        # Keyword coverage
        kw = compute_keyword_coverage(pred, cat)
        all_keyword_scores.append(kw["score"])

        # Format compliance
        fmt = compute_format_compliance(pred)
        all_format_scores.append(fmt["score"])

        # Per-category
        category_metrics[cat]["rouge_l"].append(rouge["rouge_l"])
        category_metrics[cat]["bleu"].append(bleu)
        category_metrics[cat]["keyword"].append(kw["score"])
        category_metrics[cat]["format"].append(fmt["score"])

    # Aggregate
    results = {
        "overall": {
            "rouge_1": float(np.mean(all_rouge_1)),
            "rouge_2": float(np.mean(all_rouge_2)),
            "rouge_l": float(np.mean(all_rouge_l)),
            "bleu": float(np.mean(all_bleu)),
            "keyword_coverage": float(np.mean(all_keyword_scores)),
            "format_compliance": float(np.mean(all_format_scores)),
            "num_examples": n,
        },
        "per_category": {},
    }

    for cat, metrics in category_metrics.items():
        results["per_category"][cat] = {
            "rouge_l": float(np.mean(metrics["rouge_l"])),
            "bleu": float(np.mean(metrics["bleu"])),
            "keyword_coverage": float(np.mean(metrics["keyword"])),
            "format_compliance": float(np.mean(metrics["format"])),
            "num_examples": len(metrics["rouge_l"]),
        }

    return results


# ============================================================
# INFERENCE HELPER
# ============================================================

def run_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    batch_size: int = 4,
    device: str = "cuda",
) -> List[str]:
    """
    Run batch inference on a list of prompts.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of input prompts
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size for inference
        device: Device to use

    Returns:
        List of generated responses
    """
    import torch

    model.eval()
    responses = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True,
            ).strip()
            responses.append(response)

        logger.info(f"  Inference batch {i // batch_size + 1}: {len(batch)} examples")

    return responses