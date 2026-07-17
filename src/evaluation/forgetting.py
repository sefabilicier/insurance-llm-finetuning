"""
Catastrophic forgetting analysis for fine-tuned models.

Measures how much general knowledge is lost after domain fine-tuning
by comparing performance on general benchmarks before and after training.

Methods:
- MMLU-style multiple choice evaluation (lightweight)
- Perplexity comparison on general text
- Domain shift detection
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# LIGHTWEIGHT MMLU-STYLE QUESTIONS
# ============================================================
# Subset of general knowledge questions to detect catastrophic forgetting
# without needing the full MMLU benchmark (which requires GPU inference)

GENERAL_KNOWLEDGE_QUESTIONS = [
    {
        "question": "What is the capital of France?",
        "choices": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": 2,
        "category": "geography",
    },
    {
        "question": "What is the chemical symbol for water?",
        "choices": ["CO2", "H2O", "NaCl", "O2"],
        "answer": 1,
        "category": "science",
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
        "answer": 1,
        "category": "literature",
    },
    {
        "question": "What is the largest planet in our solar system?",
        "choices": ["Mars", "Saturn", "Jupiter", "Neptune"],
        "answer": 2,
        "category": "science",
    },
    {
        "question": "In what year did World War II end?",
        "choices": ["1943", "1944", "1945", "1946"],
        "answer": 2,
        "category": "history",
    },
    {
        "question": "What is the square root of 144?",
        "choices": ["10", "11", "12", "14"],
        "answer": 2,
        "category": "math",
    },
    {
        "question": "Which element has the atomic number 1?",
        "choices": ["Helium", "Hydrogen", "Lithium", "Carbon"],
        "answer": 1,
        "category": "science",
    },
    {
        "question": "What is the speed of light approximately?",
        "choices": ["300,000 km/s", "150,000 km/s", "500,000 km/s", "100,000 km/s"],
        "answer": 0,
        "category": "science",
    },
    {
        "question": "Which programming language was created by Guido van Rossum?",
        "choices": ["Java", "C++", "Python", "Ruby"],
        "answer": 2,
        "category": "technology",
    },
    {
        "question": "What is the largest ocean on Earth?",
        "choices": ["Atlantic", "Indian", "Arctic", "Pacific"],
        "answer": 3,
        "category": "geography",
    },
    {
        "question": "Who developed the theory of general relativity?",
        "choices": ["Isaac Newton", "Albert Einstein", "Niels Bohr", "Stephen Hawking"],
        "answer": 1,
        "category": "science",
    },
    {
        "question": "What is the primary function of the CPU in a computer?",
        "choices": ["Store data", "Display graphics", "Process instructions", "Connect to internet"],
        "answer": 2,
        "category": "technology",
    },
    {
        "question": "How many continents are there on Earth?",
        "choices": ["5", "6", "7", "8"],
        "answer": 2,
        "category": "geography",
    },
    {
        "question": "What is photosynthesis?",
        "choices": [
            "Converting light to electricity",
            "Converting light energy to chemical energy in plants",
            "A type of chemical reaction in animals",
            "The process of cell division",
        ],
        "answer": 1,
        "category": "science",
    },
    {
        "question": "In mathematics, what is the value of pi (π) approximately?",
        "choices": ["2.14", "3.14", "4.14", "1.14"],
        "answer": 1,
        "category": "math",
    },
    {
        "question": "Which country has the largest population in the world?",
        "choices": ["United States", "India", "China", "Indonesia"],
        "answer": 1,
        "category": "geography",
    },
    {
        "question": "What is the boiling point of water at sea level in Celsius?",
        "choices": ["90°C", "95°C", "100°C", "110°C"],
        "answer": 2,
        "category": "science",
    },
    {
        "question": "What does HTML stand for?",
        "choices": [
            "Hyper Text Markup Language",
            "High Tech Modern Language",
            "Hyper Transfer Markup Language",
            "Home Tool Markup Language",
        ],
        "answer": 0,
        "category": "technology",
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Venus", "Mars", "Jupiter", "Mercury"],
        "answer": 1,
        "category": "science",
    },
    {
        "question": "What is the main language spoken in Brazil?",
        "choices": ["Spanish", "Portuguese", "French", "English"],
        "answer": 1,
        "category": "geography",
    },
]


def evaluate_multiple_choice(
    model,
    tokenizer,
    questions: Optional[List[Dict]] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate model on multiple choice questions.

    Uses log-probability scoring: for each question, compute the
    log-probability of each answer choice and pick the most likely.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        questions: List of MC questions (uses built-in if None)
        device: Device

    Returns:
        Results dict with accuracy per category
    """
    import torch

    if questions is None:
        questions = GENERAL_KNOWLEDGE_QUESTIONS

    model.eval()
    results = []
    category_correct = {}
    category_total = {}

    for q in questions:
        prompt = (
            f"Question: {q['question']}\n"
            f"A) {q['choices'][0]}\n"
            f"B) {q['choices'][1]}\n"
            f"C) {q['choices'][2]}\n"
            f"D) {q['choices'][3]}\n"
            f"Answer:"
        )

        choice_labels = ["A", "B", "C", "D"]

        # Score each choice
        scores = []
        for label in choice_labels:
            full_text = f"{prompt} {label}"
            inputs = tokenizer(full_text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # Get log-prob of the last token (the choice label)
                logits = outputs.logits[0, -1, :]
                label_token_id = tokenizer.encode(f" {label}", add_special_tokens=False)[-1]
                score = logits[label_token_id].item()
                scores.append(score)

        predicted = int(np.argmax(scores))
        correct = predicted == q["answer"]

        cat = q.get("category", "unknown")
        category_correct[cat] = category_correct.get(cat, 0) + int(correct)
        category_total[cat] = category_total.get(cat, 0) + 1

        results.append({
            "question": q["question"],
            "predicted": choice_labels[predicted],
            "correct_answer": choice_labels[q["answer"]],
            "is_correct": correct,
            "category": cat,
        })

    # Aggregate
    total_correct = sum(r["is_correct"] for r in results)
    total = len(results)

    per_category = {}
    for cat in category_correct:
        per_category[cat] = {
            "correct": category_correct[cat],
            "total": category_total[cat],
            "accuracy": category_correct[cat] / category_total[cat],
        }

    return {
        "overall_accuracy": total_correct / total if total > 0 else 0,
        "correct": total_correct,
        "total": total,
        "per_category": per_category,
        "details": results,
    }


def compute_forgetting_metrics(
    baseline_results: Dict[str, Any],
    finetuned_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare baseline vs fine-tuned model performance.

    Args:
        baseline_results: Results from base model
        finetuned_results: Results from fine-tuned model

    Returns:
        Forgetting analysis dict
    """
    baseline_acc = baseline_results["overall_accuracy"]
    finetuned_acc = finetuned_results["overall_accuracy"]

    absolute_drop = baseline_acc - finetuned_acc
    relative_drop = absolute_drop / baseline_acc if baseline_acc > 0 else 0

    # Per-category comparison
    category_comparison = {}
    all_cats = set(
        list(baseline_results.get("per_category", {}).keys())
        + list(finetuned_results.get("per_category", {}).keys())
    )

    for cat in all_cats:
        base_cat = baseline_results.get("per_category", {}).get(cat, {})
        ft_cat = finetuned_results.get("per_category", {}).get(cat, {})

        base_acc = base_cat.get("accuracy", 0)
        ft_acc = ft_cat.get("accuracy", 0)

        category_comparison[cat] = {
            "baseline": base_acc,
            "finetuned": ft_acc,
            "drop": base_acc - ft_acc,
        }

    # Severity assessment
    if relative_drop <= 0:
        severity = "none"
        message = "No forgetting detected — model maintained or improved general knowledge."
    elif relative_drop < 0.05:
        severity = "minimal"
        message = "Minimal forgetting (<5%) — acceptable for domain fine-tuning."
    elif relative_drop < 0.10:
        severity = "moderate"
        message = "Moderate forgetting (5-10%) — consider adjusting training parameters."
    else:
        severity = "significant"
        message = "Significant forgetting (>10%) — review LoRA rank, learning rate, or epochs."

    return {
        "baseline_accuracy": baseline_acc,
        "finetuned_accuracy": finetuned_acc,
        "absolute_drop": absolute_drop,
        "relative_drop": relative_drop,
        "severity": severity,
        "message": message,
        "per_category": category_comparison,
        "threshold_5pct": relative_drop < 0.05,
    }


def save_forgetting_report(
    forgetting_metrics: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Save forgetting analysis to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(forgetting_metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Forgetting report saved: {output_path}")
    return output_path