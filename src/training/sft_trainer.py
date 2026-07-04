"""
SFT (Supervised Fine-Tuning) Trainer for insurance domain.

Wraps HuggingFace TRL's SFTTrainer with:
- LoRA adapter configuration
- ChatML response template masking
- W&B experiment tracking
- Custom callbacks for loss monitoring
- Checkpoint management
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from datasets import Dataset, load_from_disk
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

from .callbacks import get_default_callbacks
from .utils import (
    load_model_and_tokenizer,
    create_lora_config,
    apply_lora,
    get_checkpoint_dir,
    load_config,
)

logger = logging.getLogger(__name__)


class SFTTrainerWrapper:
    """
    Supervised Fine-Tuning trainer for insurance domain.

    Trains Qwen2.5-7B + LoRA on ChatML-formatted insurance conversations
    using HuggingFace TRL's SFTTrainer.
    """

    def __init__(
        self,
        model_config_path: str = "./config/model_config.yaml",
        training_config_path: str = "./config/training_config.yaml",
        output_dir: str = "./outputs/checkpoints",
    ):
        """
        Initialize SFT trainer.

        Args:
            model_config_path: Path to model configuration
            training_config_path: Path to training configuration
            output_dir: Base directory for checkpoints
        """
        self.model_config = load_config(model_config_path)
        self.training_config = load_config(training_config_path)
        self.output_dir = Path(output_dir)

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self) -> None:
        """Load model, tokenizer, and apply LoRA."""
        logger.info("=" * 60)
        logger.info("SFT TRAINER SETUP")
        logger.info("=" * 60)

        model_cfg = self.model_config["model"]
        peft_cfg = self.model_config["peft"]
        opt_cfg = self.model_config.get("optimization", {})

        # Load base model + tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=model_cfg["model_name"],
            torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
            device_map=model_cfg.get("device_map", "auto"),
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            use_flash_attention=opt_cfg.get("use_flash_attention_2", True),
            load_in_4bit=self.model_config.get("quantization", {}).get("load_in_4bit", False),
            max_seq_length=model_cfg.get("max_seq_length", 2048),
        )

        # Create and apply LoRA
        lora_config = create_lora_config(
            r=peft_cfg.get("lora_r", 16),
            lora_alpha=peft_cfg.get("lora_alpha", 32),
            lora_dropout=peft_cfg.get("lora_dropout", 0.05),
            target_modules=peft_cfg.get("target_modules"),
            bias=peft_cfg.get("bias", "none"),
            task_type=peft_cfg.get("task_type", "CAUSAL_LM"),
        )

        self.model = apply_lora(self.model, lora_config)

        logger.info("✓ SFT setup complete")

    def _load_dataset(self, filepath: str) -> Dataset:
        """Load dataset from JSON file and convert to HF Dataset."""
        filepath = Path(filepath)

        # Try loading HF format first
        hf_path = filepath.parent / f"{filepath.stem}_hf"
        if hf_path.exists():
            logger.info(f"Loading HF dataset: {hf_path}")
            return load_from_disk(str(hf_path))

        # Fall back to JSON
        logger.info(f"Loading JSON dataset: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Format to ChatML text
        from src.data.dataset import ChatMLFormatter

        texts = [ChatMLFormatter.format_example(ex) for ex in data]
        dataset = Dataset.from_dict({"text": texts})

        logger.info(f"  Loaded {len(dataset)} examples")
        return dataset

    def _build_training_args(self, run_name: str) -> SFTConfig:
        """Build SFTConfig from config files."""
        sft_cfg = self.training_config["sft"]
        common_cfg = self.training_config["common"]

        checkpoint_dir = get_checkpoint_dir(str(self.output_dir), "sft")

        return SFTConfig(
            # Output
            output_dir=str(checkpoint_dir),
            run_name=run_name,

            # Training duration
            num_train_epochs=sft_cfg.get("num_train_epochs", 3),
            max_steps=sft_cfg.get("max_steps", -1),

            # Batch sizes
            per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=sft_cfg.get("per_device_eval_batch_size", 16),

            # Optimization
            learning_rate=sft_cfg.get("learning_rate", 2e-4),
            lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=sft_cfg.get("warmup_ratio", 0.05),
            weight_decay=sft_cfg.get("weight_decay", 0.01),
            max_grad_norm=common_cfg.get("max_grad_norm", 1.0),
            gradient_accumulation_steps=common_cfg.get("gradient_accumulation_steps", 4),

            # Precision
            fp16=common_cfg.get("fp16", False),
            bf16=common_cfg.get("bf16", True),

            # Memory
            gradient_checkpointing=common_cfg.get("gradient_checkpointing", True),

            # Evaluation
            eval_strategy=sft_cfg.get("evaluation_strategy", "steps"),
            eval_steps=sft_cfg.get("eval_steps", 250),

            # Saving
            save_strategy=sft_cfg.get("save_strategy", "steps"),
            save_steps=sft_cfg.get("save_steps", 500),
            save_total_limit=sft_cfg.get("save_total_limit", 3),
            load_best_model_at_end=sft_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=sft_cfg.get("metric_for_best_model", "eval_loss"),
            greater_is_better=sft_cfg.get("greater_is_better", False),

            # Logging
            logging_steps=common_cfg.get("logging_steps", 50),
            logging_dir=common_cfg.get("logging_dir", "./outputs/logs"),
            report_to=common_cfg.get("report_to", ["wandb"]),

            # SFT specific
            max_seq_length=sft_cfg.get("max_seq_length", 2048),
            packing=sft_cfg.get("packing", False),

            # Seed
            seed=common_cfg.get("seed", 42),

            # Dataset
            dataset_text_field="text",
        )

    def train(
        self,
        train_file: str = "./data/splits/train.json",
        val_file: str = "./data/splits/validation.json",
        run_name: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run SFT training.

        Args:
            train_file: Path to training data
            val_file: Path to validation data
            run_name: W&B run name
            resume_from_checkpoint: Path to resume from

        Returns:
            Training metrics dict
        """
        if self.model is None:
            self.setup()

        if run_name is None:
            run_name = "insurance-sft"

        logger.info("=" * 60)
        logger.info("SFT TRAINING")
        logger.info("=" * 60)

        # Load datasets
        train_dataset = self._load_dataset(train_file)
        eval_dataset = self._load_dataset(val_file)

        logger.info(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

        # Build training arguments
        training_args = self._build_training_args(run_name)

        # Get response template for masking
        response_template = self.training_config["sft"].get(
            "response_template", "<|im_start|>assistant\n"
        )

        # Create SFT trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            callbacks=[
                *get_default_callbacks(),
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config["sft"].get(
                        "early_stopping_patience", 3
                    ),
                    early_stopping_threshold=self.training_config["sft"].get(
                        "early_stopping_threshold", 0.001
                    ),
                ),
            ],
        )

        # Train
        logger.info("\nStarting SFT training...")
        result = self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )

        # Save final adapter
        final_path = Path(training_args.output_dir) / "final_adapter"
        self.trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))

        logger.info(f"\n✓ SFT training complete!")
        logger.info(f"  Final adapter: {final_path}")
        logger.info(f"  Metrics: {result.metrics}")

        return result.metrics

    def save_adapter(self, path: str) -> Path:
        """Save current LoRA adapter."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        logger.info(f"✓ Adapter saved: {path}")
        return path