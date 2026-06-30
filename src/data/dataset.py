"""
ChatML format dataset for fine-tuning.

Converts raw examples (user, assistant) to ChatML format:
<|im_start|>system
You are a helpful insurance support agent...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
...
<|im_end|>
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


class ChatMLFormatter:
    """Convert examples to ChatML format."""

    SYSTEM_PROMPT = """You are an expert insurance support agent for a Turkish insurance company.

You help customers with:
- Policy inquiries and explanations
- Claims processing guidance
- Coverage questions
- Premium and billing information
- Policy modifications and renewals

Respond professionally, accurately, and within company policies. Always be helpful and clear."""

    @staticmethod
    def format_example(
        example: Dict[str, str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format single example to ChatML.

        Args:
            example: Dict with 'user' and 'assistant' keys
            system_prompt: Custom system prompt

        Returns:
            ChatML formatted string
        """
        system = system_prompt or ChatMLFormatter.SYSTEM_PROMPT
        user_msg = example["user"]
        assistant_msg = example["assistant"]

        text = (
            f"<|im_start|>system\n{system}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}\n<|im_end|>"
        )

        return text

    @staticmethod
    def format_batch(
        examples: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Format batch of examples to ChatML.

        Args:
            examples: List of dicts
            system_prompt: Custom system prompt

        Returns:
            List of ChatML formatted strings
        """
        return [
            ChatMLFormatter.format_example(ex, system_prompt)
            for ex in examples
        ]


class InsuranceDataset:
    """PyTorch-compatible dataset for insurance conversations."""

    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer=None,
        max_length: int = 2048,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize dataset.

        Args:
            examples: List of conversation dicts
            tokenizer: HuggingFace tokenizer (optional, for preprocessing)
            max_length: Max sequence length
            system_prompt: Custom system prompt
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt

        # Format all examples to ChatML
        self.formatted_texts = [
            ChatMLFormatter.format_example(ex, system_prompt)
            for ex in examples
        ]

        logger.info(f"Initialized dataset with {len(examples)} examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get single example."""
        example = self.examples[idx]
        text = self.formatted_texts[idx]

        result = {
            "input_ids": None,
            "attention_mask": None,
            "text": text,
        }

        # Tokenize if tokenizer available
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            result["input_ids"] = encoded["input_ids"][0]
            result["attention_mask"] = encoded["attention_mask"][0]

        return result

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset."""
        data = {
            "text": self.formatted_texts,
            "category": [ex.get("category", "unknown") for ex in self.examples],
        }

        dataset = Dataset.from_dict(data)
        logger.info(f"Converted to HF Dataset: {len(dataset)} examples")

        return dataset


class DatasetBuilder:
    """Build datasets with splitting and formatting."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def build_and_save(
        self,
        examples: List[Dict[str, str]],
        output_dir: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        system_prompt: Optional[str] = None,
        random_seed: int = 42
    ) -> Dict[str, Path]:
        """
        Build train/val/test splits and save to files.

        Args:
            examples: All examples
            output_dir: Output directory
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            system_prompt: Custom system prompt
            random_seed: Random seed for reproducibility

        Returns:
            Dict with keys 'train', 'val', 'test' mapping to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        # Shuffle and split
        import random
        rng = random.Random(random_seed)
        shuffled = examples.copy()
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_examples = shuffled[:n_train]
        val_examples = shuffled[n_train:n_train + n_val]
        test_examples = shuffled[n_train + n_val:]

        if self.verbose:
            logger.info(f"\n{'='*60}")
            logger.info("DATASET SPLIT SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Total examples: {n_total}")
            logger.info(f"Train: {len(train_examples)} ({len(train_examples)/n_total*100:.1f}%)")
            logger.info(f"Val: {len(val_examples)} ({len(val_examples)/n_total*100:.1f}%)")
            logger.info(f"Test: {len(test_examples)} ({len(test_examples)/n_total*100:.1f}%)")

        # Save splits
        splits = {}

        for name, data in [("train", train_examples), ("validation", val_examples), ("test", test_examples)]:
            # Save raw JSON
            json_path = output_dir / f"{name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            splits[name] = json_path

            if self.verbose:
                logger.info(f"Saved {name}: {json_path}")

            # Create HF dataset version
            dataset = InsuranceDataset(
                data,
                system_prompt=system_prompt,
                max_length=2048
            )

            hf_dataset = dataset.to_hf_dataset()

            # Save HF format
            hf_path = output_dir / f"{name}_hf"
            hf_dataset.save_to_disk(str(hf_path))

            if self.verbose:
                logger.info(f"  └─ HF format: {hf_path}")

        if self.verbose:
            logger.info(f"{'='*60}\n")

        return splits

    def load_split(self, filepath: Path) -> List[Dict[str, str]]:
        """Load split from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_formatted_copy(
        self,
        input_path: Path,
        output_path: Path,
        system_prompt: Optional[str] = None
    ) -> Path:
        """Create ChatML-formatted copy of dataset."""
        # Load original
        examples = self.load_split(input_path)

        # Format
        dataset = InsuranceDataset(examples, system_prompt=system_prompt)

        # Save formatted
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        formatted_data = [
            {"text": text, "category": ex.get("category", "unknown")}
            for text, ex in zip(dataset.formatted_texts, examples)
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)

        if self.verbose:
            logger.info(f"Created formatted dataset: {output_path}")

        return output_path