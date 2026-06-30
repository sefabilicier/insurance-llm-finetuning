"""
Data loading utilities for insurance LLM fine-tuning.

Supports:
- CSV files
- JSON/JSONL files
- Hugging Face datasets
- Directory-based loading
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import pandas as pd
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


class DataLoader:
    """Load data from various formats."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def load_csv(
        self,
        filepath: Union[str, Path],
        text_column: str = "text",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file
            text_column: Column name containing text
            **kwargs: Additional pandas read_csv arguments

        Returns:
            List of dictionaries
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, **kwargs)
        
        if self.verbose:
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            logger.info(f"Columns: {list(df.columns)}")

        return df.to_dict(orient="records")

    def load_json(
        self,
        filepath: Union[str, Path],
        is_jsonl: bool = False
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load data from JSON or JSONL file.

        Args:
            filepath: Path to JSON/JSONL file
            is_jsonl: If True, treat as JSONL (one JSON per line)

        Returns:
            List of dictionaries (JSONL) or dict (JSON)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            if is_jsonl:
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)

        if self.verbose:
            if isinstance(data, list):
                logger.info(f"Loaded {len(data)} items from {filepath}")
            else:
                logger.info(f"Loaded JSON object from {filepath}")

        return data

    def load_huggingface_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        **kwargs
    ) -> Dataset:
        """
        Load from Hugging Face datasets.

        Args:
            dataset_name: Dataset identifier (e.g., "wmt14")
            split: Dataset split (e.g., "train", "validation")
            **kwargs: Additional load_dataset arguments

        Returns:
            Hugging Face Dataset
        """
        dataset = load_dataset(dataset_name, split=split, **kwargs)

        if self.verbose:
            logger.info(f"Loaded {len(dataset)} examples from {dataset_name}/{split}")

        return dataset

    def load_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.json",
        is_jsonl: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Load all files matching pattern from directory.

        Args:
            directory: Path to directory
            pattern: Glob pattern (e.g., "*.json", "*.jsonl")
            is_jsonl: If True, treat files as JSONL

        Returns:
            Combined list of dictionaries
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        all_data = []
        files = list(directory.glob(pattern))

        if self.verbose:
            logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")

        for filepath in sorted(files):
            data = self.load_json(filepath, is_jsonl=is_jsonl)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

        if self.verbose:
            logger.info(f"Loaded {len(all_data)} total items from directory")

        return all_data

    def save_json(
        self,
        data: Union[List[Dict], Dict],
        filepath: Union[str, Path],
        is_jsonl: bool = False,
        indent: Optional[int] = 2
    ) -> None:
        """
        Save data to JSON/JSONL file.

        Args:
            data: Data to save
            filepath: Output file path
            is_jsonl: If True, save as JSONL
            indent: JSON indentation (None for no formatting)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            if is_jsonl:
                if not isinstance(data, list):
                    data = [data]
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                json.dump(data, f, indent=indent, ensure_ascii=False)

        if self.verbose:
            logger.info(f"Saved {len(data) if isinstance(data, list) else 1} items to {filepath}")

    def save_csv(
        self,
        data: List[Dict[str, Any]],
        filepath: Union[str, Path],
        **kwargs
    ) -> None:
        """
        Save data to CSV file.

        Args:
            data: List of dictionaries
            filepath: Output file path
            **kwargs: Additional pandas to_csv arguments
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, **kwargs)

        if self.verbose:
            logger.info(f"Saved {len(data)} rows to {filepath}")