#!/usr/bin/env python3
"""
Data preparation pipeline orchestration.

Usage:
    python scripts/prepare_data.py [--step STEP] [--num-examples NUM]

Steps:
    1. generate - Generate synthetic data via Ollama
    2. process - Clean, deduplicate, and quality filter
    3. format - Convert to ChatML format and split
    4. all - Run all steps (default)

Examples:
    python scripts/prepare_data.py --step all --num-examples 1000
    python scripts/prepare_data.py --step generate --num-examples 500
    python scripts/prepare_data.py --step process
    python scripts/prepare_data.py --step format
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.data.loader import DataLoader
from src.data.generator import OllamaGenerator
from src.data.processor import DataProcessor
from src.data.dataset import DatasetBuilder, ChatMLFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config():
    """Load config from YAML files."""
    from pathlib import Path
    import yaml

    config_dir = Path("./config")

    with open(config_dir / "data_config.yaml", "r") as f:
        return yaml.safe_load(f)


def step_generate(args):
    """Step 1: Generate synthetic data using Ollama."""
    logger.info("\n" + "="*80)
    logger.info("STEP 1: SYNTHETIC DATA GENERATION")
    logger.info("="*80)

    # Setup paths
    raw_dir = Path("./data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "synthetic_insurance_data.json"

    # Initialize generator
    try:
        generator = OllamaGenerator(
            ollama_url="http://localhost:11434",
            model_name="llama2",
            timeout=120,
            max_retries=3,
            verbose=True
        )
    except ConnectionError as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        logger.error("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    # Generate data
    logger.info(f"\nGenerating {args.num_examples} examples with balanced distribution...")
    examples = generator.generate_all_categories(
        total_examples=args.num_examples,
        balanced=True,
        temperature=0.7,
        delay=0.5
    )

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Step 1 complete!")
    logger.info(f"  Generated: {len(examples)} examples")
    logger.info(f"  Saved to: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def step_process(args):
    """Step 2: Process data (clean, deduplicate, quality filter)."""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: DATA PROCESSING")
    logger.info("="*80)

    # Load raw data
    raw_path = Path("./data/raw/synthetic_insurance_data.json")
    if not raw_path.exists():
        logger.error(f"Raw data not found at {raw_path}")
        logger.error("Run step 1 (generate) first")
        sys.exit(1)

    loader = DataLoader(verbose=True)
    raw_data = loader.load_json(raw_path, is_jsonl=False)

    logger.info(f"\nLoaded {len(raw_data)} raw examples")

    # Process
    processor = DataProcessor(
        min_quality_score=0.6,
        dedup_threshold=0.95,
        mask_pii=True,
        verbose=True
    )

    processed_data, stats = processor.process(
        raw_data,
        remove_duplicates=True,
        filter_by_quality=True
    )

    # Save processed
    processed_path = Path("./data/processed/insurance_data_processed.json")
    processor.save_processed(processed_data, processed_path)

    logger.info(f"\n✓ Step 2 complete!")
    logger.info(f"  Original: {stats['original_count']}")
    logger.info(f"  Processed: {len(processed_data)}")
    logger.info(f"  Retention: {len(processed_data) / stats['original_count'] * 100:.1f}%")
    logger.info(f"  Saved to: {processed_path}")

    return processed_path


def step_format_and_split(args):
    """Step 3: Format to ChatML and create train/val/test splits."""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: FORMATTING & SPLITTING")
    logger.info("="*80)

    # Load processed data
    processed_path = Path("./data/processed/insurance_data_processed.json")
    if not processed_path.exists():
        logger.error(f"Processed data not found at {processed_path}")
        logger.error("Run step 2 (process) first")
        sys.exit(1)

    loader = DataLoader(verbose=True)
    examples = loader.load_json(processed_path, is_jsonl=False)

    logger.info(f"\nLoaded {len(examples)} processed examples")

    # Build and save splits
    builder = DatasetBuilder(verbose=True)
    splits_dir = Path("./data/splits")

    splits = builder.build_and_save(
        examples,
        splits_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        system_prompt=ChatMLFormatter.SYSTEM_PROMPT,
        random_seed=42
    )

    # Create formatted versions
    formatted_dir = Path("./data/formatted")
    formatted_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nCreating ChatML-formatted versions...")
    for split_name, split_path in splits.items():
        formatted_path = formatted_dir / f"{split_name}_formatted.json"
        builder.create_formatted_copy(split_path, formatted_path)

    logger.info(f"\n✓ Step 3 complete!")
    logger.info(f"  Train: {splits['train']}")
    logger.info(f"  Val: {splits['validation']}")
    logger.info(f"  Test: {splits['test']}")
    logger.info(f"  Formatted dir: {formatted_dir}")

    return splits


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Data preparation pipeline for insurance LLM fine-tuning"
    )

    parser.add_argument(
        "--step",
        type=str,
        choices=["generate", "process", "format", "all"],
        default="all",
        help="Which step(s) to run"
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=1000,
        help="Number of examples to generate (default: 1000)"
    )

    parser.add_argument(
        "--skip-pii",
        action="store_true",
        help="Skip PII masking"
    )

    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip deduplication"
    )

    args = parser.parse_args()

    try:
        if args.step in ["generate", "all"]:
            step_generate(args)

        if args.step in ["process", "all"]:
            step_process(args)

        if args.step in ["format", "all"]:
            step_format_and_split(args)

        logger.info("\n" + "="*80)
        logger.info("✓ DATA PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("  1. Review data in ./data/splits/")
        logger.info("  2. Run: python scripts/train.py --phase sft")
        logger.info("  3. Run: python scripts/train.py --phase dpo")
        logger.info("\nFor details, see RESULTS.md")

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()