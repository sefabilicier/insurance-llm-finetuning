"""
DPO (Direct Preference Optimization) Trainer for insurance domain.

Wraps HuggingFace TRL's DPOTrainer with:
- Loading from SFT checkpoint (continues from Phase 1)
- Preference pair dataset handling
- Synthetic preference data generation
- W&B experiment tracking
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import DPOTrainer, DPOConfig

from .callbacks import get_default_callbacks
from .utils import (
    load_model_and_tokenizer,
    get_checkpoint_dir,
    find_latest_checkpoint,
    load_config,
)

logger = logging.getLogger(__name__)


class PreferenceDataBuilder:
    """
    Build preference pairs for DPO training.

    Creates (prompt, chosen, rejected) triplets from insurance data.
    'Chosen' = professional, policy-compliant response
    'Rejected' = poor quality, vague, or non-compliant response
    """

    REJECTION_STRATEGIES = [
        # Strategy 1: Vague/unhelpful response
        lambda chosen: "I'm not sure about that. You should check your policy documents or call us back later.",

        # Strategy 2: Missing key information
        lambda chosen: "Yes, that should be covered. Let me know if you have other questions.",

        # Strategy 3: Incorrect procedure
        lambda chosen: "Just send us an email about it and we'll figure it out eventually.",

        # Strategy 4: Unprofessional tone
        lambda chosen: "Look, I don't know the details of your policy off the top of my head. You'll need to check that yourself.",

        # Strategy 5: Overpromising
        lambda chosen: "Don't worry, everything is definitely covered under your policy. We'll take care of everything no matter what.",

        # Strategy 6: Too short
        lambda chosen: "Please check your policy.",

        # Strategy 7: Generic non-answer
        lambda chosen: "Thank you for your question. Our policies vary and I would recommend reviewing your specific policy documentation for more details.",

        # Strategy 8: Redirect without help
        lambda chosen: "That's handled by a different department. You'll need to call them directly during business hours.",
    ]

    @staticmethod
    def build_preference_pairs(
        examples: List[Dict[str, str]],
        seed: int = 42,
    ) -> List[Dict[str, str]]:
        """
        Build preference pairs from SFT examples.

        Each example gets:
        - prompt: the user question (with system prompt)
        - chosen: the original high-quality response
        - rejected: a synthetically degraded response

        Args:
            examples: List of SFT examples with 'user' and 'assistant' keys
            seed: Random seed

        Returns:
            List of preference pair dicts
        """
        import random
        rng = random.Random(seed)

        pairs = []
        strategies = PreferenceDataBuilder.REJECTION_STRATEGIES

        system_prompt = (
            "You are an expert insurance support agent for a Turkish insurance company.\n\n"
            "You help customers with:\n"
            "- Policy inquiries and explanations\n"
            "- Claims processing guidance\n"
            "- Coverage questions\n"
            "- Premium and billing information\n"
            "- Policy modifications and renewals\n\n"
            "Respond professionally, accurately, and within company policies. "
            "Always be helpful and clear."
        )

        for example in examples:
            user_msg = example["user"]
            chosen_msg = example["assistant"]

            # Build prompt (system + user in ChatML)
            prompt = (
                f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            # Select rejection strategy
            strategy = rng.choice(strategies)
            rejected_msg = strategy(chosen_msg)

            pairs.append({
                "prompt": prompt,
                "chosen": chosen_msg + "\n<|im_end|>",
                "rejected": rejected_msg + "\n<|im_end|>",
                "category": example.get("category", "unknown"),
            })

        logger.info(f"✓ Built {len(pairs)} preference pairs")
        return pairs

    @staticmethod
    def save_preference_data(
        pairs: List[Dict[str, str]],
        output_path: Path,
    ) -> Path:
        """Save preference pairs to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved preference data: {output_path} ({len(pairs)} pairs)")
        return output_path


class DPOTrainerWrapper:
    """
    Direct Preference Optimization trainer for insurance domain.

    Takes SFT-trained model and optimizes for preference alignment
    using (prompt, chosen, rejected) triplets.
    """

    def __init__(
        self,
        model_config_path: str = "./config/model_config.yaml",
        training_config_path: str = "./config/training_config.yaml",
        output_dir: str = "./outputs/checkpoints",
    ):
        self.model_config = load_config(model_config_path)
        self.training_config = load_config(training_config_path)
        self.output_dir = Path(output_dir)

        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None

    def setup(
        self,
        sft_adapter_path: Optional[str] = None,
    ) -> None:
        """
        Load SFT-trained model for DPO training.

        Args:
            sft_adapter_path: Path to SFT LoRA adapter.
                If None, searches for latest SFT checkpoint.
        """
        logger.info("=" * 60)
        logger.info("DPO TRAINER SETUP")
        logger.info("=" * 60)

        model_cfg = self.model_config["model"]

        # Find SFT adapter
        if sft_adapter_path is None:
            sft_adapter_path = find_latest_checkpoint(
                str(self.output_dir), phase="sft"
            )
            if sft_adapter_path is None:
                raise FileNotFoundError(
                    "No SFT checkpoint found. Run SFT training first."
                )
            sft_adapter_path = str(sft_adapter_path)

        logger.info(f"  SFT adapter: {sft_adapter_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["model_name"],
            trust_remote_code=True,
            padding_side="left",  # DPO requires left padding
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with SFT adapter
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        dtype = dtype_map.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_name"],
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load SFT adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            sft_adapter_path,
            is_trainable=True,
        )

        logger.info("✓ DPO setup complete (loaded SFT model)")

    def _load_preference_dataset(self, filepath: str) -> Dataset:
        """Load preference pairs from JSON."""
        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = Dataset.from_dict({
            "prompt": [d["prompt"] for d in data],
            "chosen": [d["chosen"] for d in data],
            "rejected": [d["rejected"] for d in data],
        })

        logger.info(f"  Loaded {len(dataset)} preference pairs from {filepath}")
        return dataset

    def prepare_preference_data(
        self,
        sft_train_file: str = "./data/splits/train.json",
        sft_val_file: str = "./data/splits/validation.json",
        output_dir: str = "./data/splits",
    ) -> Dict[str, Path]:
        """
        Generate preference data from SFT training data.

        Args:
            sft_train_file: SFT training data
            sft_val_file: SFT validation data
            output_dir: Where to save preference files

        Returns:
            Dict with paths to train/val preference files
        """
        logger.info("Generating preference pairs...")

        output_dir = Path(output_dir)

        # Load SFT data
        with open(sft_train_file, "r") as f:
            train_data = json.load(f)
        with open(sft_val_file, "r") as f:
            val_data = json.load(f)

        # Build preference pairs
        train_pairs = PreferenceDataBuilder.build_preference_pairs(train_data, seed=42)
        val_pairs = PreferenceDataBuilder.build_preference_pairs(val_data, seed=43)

        # Save
        train_path = PreferenceDataBuilder.save_preference_data(
            train_pairs, output_dir / "train_preferences.json"
        )
        val_path = PreferenceDataBuilder.save_preference_data(
            val_pairs, output_dir / "validation_preferences.json"
        )

        return {"train": train_path, "validation": val_path}

    def _build_training_args(self, run_name: str) -> DPOConfig:
        """Build DPOConfig from config files."""
        dpo_cfg = self.training_config["dpo"]
        common_cfg = self.training_config["common"]

        checkpoint_dir = get_checkpoint_dir(str(self.output_dir), "dpo")

        return DPOConfig(
            # Output
            output_dir=str(checkpoint_dir),
            run_name=run_name,

            # Training
            num_train_epochs=dpo_cfg.get("num_train_epochs", 1),
            per_device_train_batch_size=dpo_cfg.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=dpo_cfg.get("per_device_eval_batch_size", 8),

            # Optimization
            learning_rate=dpo_cfg.get("learning_rate", 5e-5),
            lr_scheduler_type=dpo_cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=dpo_cfg.get("warmup_ratio", 0.1),
            weight_decay=dpo_cfg.get("weight_decay", 0.01),
            max_grad_norm=common_cfg.get("max_grad_norm", 1.0),
            gradient_accumulation_steps=common_cfg.get("gradient_accumulation_steps", 4),

            # DPO specific
            beta=dpo_cfg.get("beta", 0.1),
            loss_type=dpo_cfg.get("loss_type", "sigmoid"),
<<<<<<< HEAD
            max_length=dpo_cfg.get("max_seq_length", 2048),
=======
            max_length=dpo_cfg.get("max_length", 2048),
>>>>>>> 355bf7f
            max_prompt_length=dpo_cfg.get("max_prompt_length", 1024),

            # Precision
            fp16=common_cfg.get("fp16", False),
            bf16=common_cfg.get("bf16", True),

            # Memory
            gradient_checkpointing=common_cfg.get("gradient_checkpointing", True),

            # Evaluation
            eval_strategy=dpo_cfg.get("evaluation_strategy", "steps"),
            eval_steps=dpo_cfg.get("eval_steps", 100),

            # Saving
            save_strategy=dpo_cfg.get("save_strategy", "steps"),
            save_steps=dpo_cfg.get("save_steps", 200),
            save_total_limit=dpo_cfg.get("save_total_limit", 3),
            load_best_model_at_end=dpo_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=dpo_cfg.get("metric_for_best_model", "eval_loss"),
            greater_is_better=dpo_cfg.get("greater_is_better", False),

            # Logging
            logging_steps=common_cfg.get("logging_steps", 50),
            logging_dir=common_cfg.get("logging_dir", "./outputs/logs"),
            report_to=common_cfg.get("report_to", ["wandb"]),

            # Seed
            seed=common_cfg.get("seed", 42),
        )

    def train(
        self,
        train_file: str = "./data/splits/train_preferences.json",
        val_file: str = "./data/splits/validation_preferences.json",
        sft_adapter_path: Optional[str] = None,
        run_name: Optional[str] = None,
        auto_generate_preferences: bool = True,
    ) -> Dict[str, Any]:
        """
        Run DPO training.

        Args:
            train_file: Path to training preference pairs
            val_file: Path to validation preference pairs
            sft_adapter_path: Path to SFT adapter (optional)
            run_name: W&B run name
            auto_generate_preferences: Auto-generate if preference files missing

        Returns:
            Training metrics dict
        """
        # Setup model if needed
        if self.model is None:
            self.setup(sft_adapter_path)

        if run_name is None:
            run_name = "insurance-dpo"

        logger.info("=" * 60)
        logger.info("DPO TRAINING")
        logger.info("=" * 60)

        # Check if preference data exists, generate if needed
        if auto_generate_preferences and not Path(train_file).exists():
            logger.info("Preference data not found, generating from SFT data...")
            pref_paths = self.prepare_preference_data()
            train_file = str(pref_paths["train"])
            val_file = str(pref_paths["validation"])

        # Load datasets
        train_dataset = self._load_preference_dataset(train_file)
        eval_dataset = self._load_preference_dataset(val_file)

        logger.info(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

        # Build training arguments
        training_args = self._build_training_args(run_name)

        # Create DPO trainer
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Uses implicit reference (PEFT)
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            callbacks=get_default_callbacks(),
        )

        # Train
        logger.info("\nStarting DPO training...")
        result = self.trainer.train()

        # Save final adapter
        final_path = Path(training_args.output_dir) / "final_adapter"
        self.trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))

        logger.info(f"\n✓ DPO training complete!")
        logger.info(f"  Final adapter: {final_path}")
        logger.info(f"  Metrics: {result.metrics}")

        return result.metrics

    def save_adapter(self, path: str) -> Path:
        """Save current adapter."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        logger.info(f"✓ DPO adapter saved: {path}")
<<<<<<< HEAD
        return path
=======
        return path
>>>>>>> 355bf7f
