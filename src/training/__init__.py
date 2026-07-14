"""
Training module for insurance LLM fine-tuning pipeline.

Provides SFT and DPO trainers with LoRA adapter support,
experiment tracking, and checkpoint management.
"""

from .sft_trainer import SFTTrainerWrapper
from .dpo_trainer import DPOTrainerWrapper, PreferenceDataBuilder
from .utils import (
    load_model_and_tokenizer,
    create_lora_config,
    apply_lora,
    merge_lora_weights,
    export_model,
    load_checkpoint,
    find_latest_checkpoint,
)
from .callbacks import get_default_callbacks

__all__ = [
    "SFTTrainerWrapper",
    "DPOTrainerWrapper",
    "PreferenceDataBuilder",
    "load_model_and_tokenizer",
    "create_lora_config",
    "apply_lora",
    "merge_lora_weights",
    "export_model",
    "load_checkpoint",
    "find_latest_checkpoint",
    "get_default_callbacks",
]

<<<<<<< HEAD
__version__ = "0.1.0"
=======
__version__ = "0.1.0"
>>>>>>> 355bf7f
