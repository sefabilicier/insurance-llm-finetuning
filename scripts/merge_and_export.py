#!/usr/bin/env python3
"""
Merge LoRA adapter weights into base model and export.

Usage:
    python scripts/merge_and_export.py --adapter-path ./outputs/checkpoints/dpo_*/final_adapter
    python scripts/merge_and_export.py --adapter-path ./outputs/checkpoints/sft_*/final_adapter --output ./outputs/merged_models/sft-only
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and export model")

    parser.add_argument("--adapter-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", default="./outputs/merged_models/insurance-model-final")
    parser.add_argument("--model-config", default="./config/model_config.yaml")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])

    args = parser.parse_args()

    try:
        from src.training.utils import merge_lora_weights, load_config

        config = load_config(args.model_config)
        base_model = config["model"]["model_name"]

        merge_lora_weights(
            base_model_name=base_model,
            adapter_path=args.adapter_path,
            output_path=args.output,
            torch_dtype=args.dtype,
        )

        logger.info(f"\n✓ Model exported to: {args.output}")
        logger.info("Next: python scripts/train.py --phase serve")

    except Exception as e:
        logger.error(f"Merge failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
