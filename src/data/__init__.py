"""
Data module for insurance LLM fine-tuning pipeline.

Handles:
- Loading raw datasets (CSV, JSON, Hugging Face)
- Data cleaning and deduplication
- PII masking and quality scoring
- ChatML formatting
- Train/Val/Test splitting
"""

from .loader import DataLoader
from .processor import DataProcessor
from .quality_scorer import QualityScorer
from .dataset import InsuranceDataset

__all__ = [
    "DataLoader",
    "DataProcessor",
    "QualityScorer",
    "InsuranceDataset",
]

__version__ = "0.1.0"