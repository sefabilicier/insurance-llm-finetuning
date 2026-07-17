#!/usr/bin/env python3
"""
Evaluation orchestration for insurance LLM fine-tuning.

Usage:
    python scripts/evaluate.py --phase sft                  # Evaluate SFT model
    python scripts/evaluate.py --phase dpo                  # Evaluate DPO model
    python scripts/evaluate.py --phase all                  # Full evaluation pipeline
    python scripts/evaluate.py --phase offline               # Offline metrics (no GPU)

Examples:
    python scripts/evaluate.py --phase offline --test-file ./data/splits/test.json
    python scripts/evaluate.py --phase sft --adapter-path ./outputs/checkpoints/sft_*/final_adapter
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def phase_offline(args):
    """
    Run offline evaluation (no GPU/model required).

    Uses template-generated data to compute metrics against references.
    Useful for testing the evaluation pipeline before training.
    """
    from src.evaluation.metrics import compute_task_metrics, compute_rouge, compute_format_compliance
    from src.evaluation.reporter import EvaluationReporter

    logger.info("=" * 60)
    logger.info("OFFLINE EVALUATION (No GPU Required)")
    logger.info("=" * 60)

    # Load test data
    test_path = Path(args.test_file)
    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}")
        sys.exit(1)

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"Loaded {len(test_data)} test examples")

    # Use references as "predictions" for baseline scoring
    # This gives us perfect scores as a sanity check
    references = [ex["assistant"] for ex in test_data]
    categories = [ex.get("category", "unknown") for ex in test_data]

    # Scenario 1: Self-evaluation (reference vs reference = perfect)
    logger.info("\n[1] Self-evaluation (sanity check)...")
    self_metrics = compute_task_metrics(references, references, categories)

    logger.info(f"  ROUGE-L: {self_metrics['overall']['rouge_l']:.4f} (expected ~1.0)")
    logger.info(f"  BLEU: {self_metrics['overall']['bleu']:.4f} (expected ~1.0)")
    logger.info(f"  Keywords: {self_metrics['overall']['keyword_coverage']:.4f}")
    logger.info(f"  Format: {self_metrics['overall']['format_compliance']:.4f}")

    # Scenario 2: Simulated baseline (generic responses)
    logger.info("\n[2] Simulated baseline (generic responses)...")
    baseline_responses = [
        "I can help you with that. Please check your policy documents for more details."
    ] * len(test_data)

    baseline_metrics = compute_task_metrics(baseline_responses, references, categories)

    logger.info(f"  ROUGE-L: {baseline_metrics['overall']['rouge_l']:.4f}")
    logger.info(f"  BLEU: {baseline_metrics['overall']['bleu']:.4f}")
    logger.info(f"  Keywords: {baseline_metrics['overall']['keyword_coverage']:.4f}")
    logger.info(f"  Format: {baseline_metrics['overall']['format_compliance']:.4f}")

    # Scenario 3: Per-category breakdown
    logger.info("\n[3] Per-category analysis...")
    for cat, metrics in sorted(self_metrics["per_category"].items()):
        logger.info(
            f"  {cat:25s} | ROUGE-L: {metrics['rouge_l']:.4f} | "
            f"Keywords: {metrics['keyword_coverage']:.4f} | "
            f"Format: {metrics['format_compliance']:.4f} | "
            f"N: {metrics['num_examples']}"
        )

    # Generate report
    logger.info("\n[4] Generating report...")
    reporter = EvaluationReporter(output_dir=args.output_dir)

    comparisons = []
    for i, ex in enumerate(test_data[:5]):
        comparisons.append({
            "user": ex["user"],
            "category": ex.get("category", "unknown"),
            "reference": ex["assistant"],
            "baseline_response": baseline_responses[i],
            "finetuned_response": "(model not yet trained)",
        })

    paths = reporter.generate_report(
        task_metrics=self_metrics,
        comparisons=comparisons,
        training_info={
            "model": "offline-evaluation",
            "phase": "offline",
            "note": "Sanity check using test set references",
        },
    )

    logger.info(f"\n✓ Offline evaluation complete!")
    logger.info(f"  Report: {paths['markdown']}")

    return self_metrics


def phase_model_eval(args, phase_name: str = "sft"):
    """
    Run model evaluation (requires GPU + trained model).

    Loads the fine-tuned model and evaluates against test set.
    """
    from src.evaluation.metrics import compute_task_metrics, run_inference
    from src.evaluation.forgetting import evaluate_multiple_choice, compute_forgetting_metrics
    from src.evaluation.reporter import EvaluationReporter, generate_before_after_report
    from src.training.utils import load_config
    from src.data.dataset import ChatMLFormatter

    import torch

    logger.info("=" * 60)
    logger.info(f"MODEL EVALUATION: {phase_name.upper()}")
    logger.info("=" * 60)

    # Load config
    model_config = load_config(args.model_config)
    model_name = model_config["model"]["model_name"]

    # Load test data
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"Test examples: {len(test_data)}")

    # Determine adapter path
    adapter_path = args.adapter_path
    if adapter_path is None:
        from src.training.utils import find_latest_checkpoint
        adapter_path = find_latest_checkpoint(args.checkpoint_dir, phase_name)
        if adapter_path is None:
            logger.error(f"No {phase_name} checkpoint found")
            sys.exit(1)

    logger.info(f"Adapter: {adapter_path}")

    # --- Step 1: Load base model (for baseline) ---
    logger.info("\n[1] Loading base model for baseline...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Build prompts
    system_prompt = ChatMLFormatter.SYSTEM_PROMPT
    prompts = []
    for ex in test_data:
        prompt = (
            f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
            f"<|im_start|>user\n{ex['user']}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append(prompt)

    # Baseline inference
    logger.info("[2] Running baseline inference...")
    baseline_responses = run_inference(
        base_model, tokenizer, prompts, max_new_tokens=256
    )

    # Baseline forgetting test
    logger.info("[3] Baseline forgetting test...")
    baseline_forgetting = evaluate_multiple_choice(base_model, tokenizer)
    logger.info(f"  Baseline accuracy: {baseline_forgetting['overall_accuracy']:.2%}")

    # --- Step 2: Load fine-tuned model ---
    logger.info(f"\n[4] Loading fine-tuned model ({phase_name})...")
    from peft import PeftModel

    ft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    # Fine-tuned inference
    logger.info("[5] Running fine-tuned inference...")
    finetuned_responses = run_inference(
        ft_model, tokenizer, prompts, max_new_tokens=256
    )

    # Fine-tuned forgetting test
    logger.info("[6] Fine-tuned forgetting test...")
    ft_forgetting = evaluate_multiple_choice(ft_model, tokenizer)
    logger.info(f"  Fine-tuned accuracy: {ft_forgetting['overall_accuracy']:.2%}")

    # --- Step 3: Compute all metrics ---
    logger.info("\n[7] Computing metrics...")
    references = [ex["assistant"] for ex in test_data]
    categories = [ex.get("category", "unknown") for ex in test_data]

    finetuned_metrics = compute_task_metrics(
        finetuned_responses, references, categories
    )

    forgetting = compute_forgetting_metrics(baseline_forgetting, ft_forgetting)

    # --- Step 4: Generate report ---
    logger.info("\n[8] Generating report...")
    paths = generate_before_after_report(
        test_examples=test_data,
        baseline_responses=baseline_responses,
        finetuned_responses=finetuned_responses,
        output_dir=args.output_dir,
        training_info={
            "model": model_name,
            "phase": phase_name,
            "adapter_path": str(adapter_path),
        },
    )

    # Save forgetting report separately
    from src.evaluation.forgetting import save_forgetting_report
    forgetting_path = Path(args.output_dir) / "forgetting_analysis.json"
    save_forgetting_report(forgetting, forgetting_path)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  ROUGE-L: {finetuned_metrics['overall']['rouge_l']:.4f}")
    logger.info(f"  BLEU: {finetuned_metrics['overall']['bleu']:.4f}")
    logger.info(f"  Keywords: {finetuned_metrics['overall']['keyword_coverage']:.4f}")
    logger.info(f"  Format: {finetuned_metrics['overall']['format_compliance']:.4f}")
    logger.info(f"  Forgetting: {forgetting['severity']} ({forgetting['relative_drop']:.2%} drop)")
    logger.info(f"  Report: {paths['markdown']}")

    return finetuned_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation orchestration for insurance LLM"
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["sft", "dpo", "all", "offline"],
        default="offline",
        help="Evaluation phase",
    )

    parser.add_argument("--test-file", default="./data/splits/test.json")
    parser.add_argument("--model-config", default="./config/model_config.yaml")
    parser.add_argument("--output-dir", default="./outputs/evaluation")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--checkpoint-dir", default="./outputs/checkpoints")

    args = parser.parse_args()

    try:
        if args.phase == "offline":
            phase_offline(args)
        elif args.phase in ["sft", "dpo"]:
            phase_model_eval(args, phase_name=args.phase)
        elif args.phase == "all":
            phase_model_eval(args, phase_name="sft")
            phase_model_eval(args, phase_name="dpo")

    except Exception as e:
        logger.error(f"\n✗ Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()