#!/usr/bin/env python3
"""
Training orchestration for insurance LLM fine-tuning.

Usage:
    python scripts/train.py --phase sft                     # Run SFT training
    python scripts/train.py --phase dpo                     # Run DPO training
    python scripts/train.py --phase all                     # Run SFT → DPO
    python scripts/train.py --phase merge                   # Merge LoRA into base
    python scripts/train.py --phase sft --resume <path>     # Resume from checkpoint

Examples:
    python scripts/train.py --phase sft --epochs 3 --batch-size 8
    python scripts/train.py --phase dpo --sft-adapter ./outputs/checkpoints/sft_*/final_adapter
    python scripts/train.py --phase merge --adapter-path ./outputs/checkpoints/dpo_*/final_adapter
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def phase_sft(args):
    """Run Supervised Fine-Tuning."""
    from src.training.sft_trainer import SFTTrainerWrapper

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: SUPERVISED FINE-TUNING (SFT)")
    logger.info("=" * 80)

    trainer = SFTTrainerWrapper(
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        output_dir=args.output_dir,
    )

    metrics = trainer.train(
        train_file=args.train_file,
        val_file=args.val_file,
        run_name=args.run_name or "insurance-sft",
        resume_from_checkpoint=args.resume,
    )

    logger.info(f"\n✓ SFT complete: {metrics}")
    return metrics


def phase_dpo(args):
    """Run Direct Preference Optimization."""
    from src.training.dpo_trainer import DPOTrainerWrapper

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: DIRECT PREFERENCE OPTIMIZATION (DPO)")
    logger.info("=" * 80)

    trainer = DPOTrainerWrapper(
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        output_dir=args.output_dir,
    )

    # Generate preference data if needed
    pref_train = args.train_pref_file
    pref_val = args.val_pref_file

    if not Path(pref_train).exists():
        logger.info("Generating preference pairs from SFT data...")
        paths = trainer.prepare_preference_data(
            sft_train_file=args.train_file,
            sft_val_file=args.val_file,
        )
        pref_train = str(paths["train"])
        pref_val = str(paths["validation"])

    metrics = trainer.train(
        train_file=pref_train,
        val_file=pref_val,
        sft_adapter_path=args.sft_adapter,
        run_name=args.run_name or "insurance-dpo",
    )

    logger.info(f"\n✓ DPO complete: {metrics}")
    return metrics


def phase_merge(args):
    """Merge LoRA adapter into base model."""
    from src.training.utils import merge_lora_weights, load_config

    logger.info("\n" + "=" * 80)
    logger.info("MERGING LoRA WEIGHTS INTO BASE MODEL")
    logger.info("=" * 80)

    model_config = load_config(args.model_config)
    base_model_name = model_config["model"]["model_name"]

    adapter_path = args.adapter_path
    if adapter_path is None:
        # Try to find latest DPO adapter, then SFT
        from src.training.utils import find_latest_checkpoint
        adapter_path = find_latest_checkpoint(args.output_dir, "dpo")
        if adapter_path is None:
            adapter_path = find_latest_checkpoint(args.output_dir, "sft")
        if adapter_path is None:
            logger.error("No adapter found. Specify --adapter-path")
            sys.exit(1)
        adapter_path = str(adapter_path)

    output_path = args.merge_output or "./outputs/merged_models/insurance-model-final"

    merge_lora_weights(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        output_path=output_path,
        torch_dtype="bfloat16",
    )

    logger.info(f"\n✓ Merged model saved: {output_path}")


def phase_all(args):
    """Run full pipeline: SFT → DPO → Merge."""
    logger.info("\n" + "=" * 80)
    logger.info("FULL TRAINING PIPELINE: SFT → DPO → MERGE")
    logger.info("=" * 80)

    # Phase 1: SFT
    sft_metrics = phase_sft(args)

    # Phase 2: DPO
    dpo_metrics = phase_dpo(args)

    # Phase 3: Merge
    phase_merge(args)

    logger.info("\n" + "=" * 80)
    logger.info("FULL PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"  SFT metrics: {sft_metrics}")
    logger.info(f"  DPO metrics: {dpo_metrics}")


def main():
    parser = argparse.ArgumentParser(
        description="Training orchestration for insurance LLM fine-tuning"
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["sft", "dpo", "merge", "all"],
        required=True,
        help="Training phase to run",
    )

    # Data paths
    parser.add_argument("--train-file", default="./data/splits/train.json")
    parser.add_argument("--val-file", default="./data/splits/validation.json")
    parser.add_argument("--train-pref-file", default="./data/splits/train_preferences.json")
    parser.add_argument("--val-pref-file", default="./data/splits/validation_preferences.json")

    # Config paths
    parser.add_argument("--model-config", default="./config/model_config.yaml")
    parser.add_argument("--training-config", default="./config/training_config.yaml")

    # Output
    parser.add_argument("--output-dir", default="./outputs/checkpoints")
    parser.add_argument("--merge-output", default=None)

    # Model paths
    parser.add_argument("--sft-adapter", default=None, help="Path to SFT adapter for DPO")
    parser.add_argument("--adapter-path", default=None, help="Path to adapter for merging")

    # Resume
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")

    # W&B
    parser.add_argument("--run-name", default=None, help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()

    # Disable W&B if requested
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    try:
        phases = {
            "sft": phase_sft,
            "dpo": phase_dpo,
            "merge": phase_merge,
            "all": phase_all,
        }
        phases[args.phase](args)

    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()