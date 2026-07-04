"""
Training utilities for model loading, LoRA merging, and checkpoint management.

Provides:
- Base model + tokenizer loading with LoRA config
- LoRA adapter merging into base model
- Checkpoint save/load/resume
- Model export (safetensors)
- Config loading from YAML
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    use_flash_attention: bool = True,
    load_in_4bit: bool = False,
    max_seq_length: int = 2048,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base model and tokenizer from Hugging Face.

    Args:
        model_name: HF model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
        torch_dtype: Precision (bfloat16, float16, float32)
        device_map: Device mapping strategy
        trust_remote_code: Allow custom model code
        use_flash_attention: Use Flash Attention 2
        load_in_4bit: Load in 4-bit quantization (for QLoRA)
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    logger.info(f"  dtype: {torch_dtype}")
    logger.info(f"  device_map: {device_map}")
    logger.info(f"  flash_attention: {use_flash_attention}")
    logger.info(f"  4bit: {load_in_4bit}")

    # Resolve dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # Quantization config (if QLoRA)
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    # Model kwargs
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Prepare for k-bit training if quantized
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    logger.info(f"✓ Model loaded: {model_name}")
    logger.info(f"  Parameters: {model.num_parameters():,}")
    logger.info(f"  Vocab size: {len(tokenizer)}")
    logger.info(f"  Max seq length: {max_seq_length}")

    return model, tokenizer


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    Create LoRA configuration.

    Args:
        r: Rank of LoRA matrices
        lora_alpha: Scaling factor
        lora_dropout: Dropout rate
        target_modules: Which modules to apply LoRA to
        bias: Bias configuration
        task_type: Task type

    Returns:
        LoraConfig
    """
    if target_modules is None:
        # Default for Qwen2.5
        target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )

    logger.info(f"✓ LoRA config created:")
    logger.info(f"  rank={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"  targets: {target_modules}")

    return config


def apply_lora(model: PreTrainedModel, lora_config: LoraConfig) -> PeftModel:
    """Apply LoRA adapters to model."""
    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    pct = trainable_params / total_params * 100

    logger.info(f"✓ LoRA applied:")
    logger.info(f"  Trainable: {trainable_params:,} ({pct:.2f}%)")
    logger.info(f"  Total: {total_params:,}")

    return peft_model


def merge_lora_weights(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
) -> Path:
    """
    Merge LoRA adapter weights into base model.

    Args:
        base_model_name: HF model identifier
        adapter_path: Path to saved LoRA adapter
        output_path: Where to save merged model
        torch_dtype: Output precision
        push_to_hub: Whether to push to HF Hub
        hub_model_id: HF Hub model identifier

    Returns:
        Path to merged model
    """
    logger.info("=" * 60)
    logger.info("MERGING LoRA WEIGHTS")
    logger.info("=" * 60)
    logger.info(f"  Base: {base_model_name}")
    logger.info(f"  Adapter: {adapter_path}")
    logger.info(f"  Output: {output_path}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="cpu",  # Merge on CPU to avoid OOM
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    # Load and merge adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()

    # Save merged model
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    logger.info(f"✓ Merged model saved: {output_path}")

    # Push to Hub if requested
    if push_to_hub and hub_model_id:
        merged_model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        logger.info(f"✓ Pushed to Hub: {hub_model_id}")

    return output_path


def export_model(
    model_path: str,
    output_path: str,
    export_format: str = "safetensors",
) -> Path:
    """Export model in specified format."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    safe_serialization = export_format == "safetensors"
    model.save_pretrained(output_path, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(output_path)

    logger.info(f"✓ Exported ({export_format}): {output_path}")
    return output_path


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[PreTrainedModel] = None,
) -> Dict[str, Any]:
    """
    Load training checkpoint metadata.

    Args:
        checkpoint_path: Path to checkpoint directory
        model: Optional model to load weights into

    Returns:
        Checkpoint metadata dict
    """
    checkpoint_path = Path(checkpoint_path)

    # Load trainer state
    state_path = checkpoint_path / "trainer_state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        logger.info(f"✓ Loaded checkpoint: {checkpoint_path}")
        logger.info(f"  Step: {state.get('global_step', 'unknown')}")
        logger.info(f"  Epoch: {state.get('epoch', 'unknown')}")
        return state

    logger.warning(f"No trainer_state.json found at {checkpoint_path}")
    return {}


def get_checkpoint_dir(
    base_dir: str,
    phase: str,
    timestamp: Optional[str] = None,
) -> Path:
    """
    Generate timestamped checkpoint directory.

    Args:
        base_dir: Base output directory
        phase: Training phase (sft, dpo)
        timestamp: Optional timestamp string

    Returns:
        Path to checkpoint directory
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = Path(base_dir) / f"{phase}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)

    return path


def find_latest_checkpoint(checkpoint_dir: str, phase: str = "sft") -> Optional[Path]:
    """
    Find the latest checkpoint for a given phase.

    Args:
        checkpoint_dir: Base checkpoint directory
        phase: Training phase prefix

    Returns:
        Path to latest checkpoint, or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith(phase)],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if checkpoints:
        # Look for numbered sub-checkpoints (checkpoint-500, etc.)
        latest = checkpoints[0]
        sub_checkpoints = sorted(
            [d for d in latest.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[-1]),
            reverse=True,
        )
        if sub_checkpoints:
            logger.info(f"✓ Found latest checkpoint: {sub_checkpoints[0]}")
            return sub_checkpoints[0]

        return latest

    return None


def cleanup_checkpoints(
    checkpoint_dir: str,
    keep_n: int = 3,
    phase: str = "sft",
) -> None:
    """Remove old checkpoints, keeping only the N most recent."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    phase_dirs = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith(phase)],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    for old_dir in phase_dirs[keep_n:]:
        shutil.rmtree(old_dir)
        logger.info(f"  Removed old checkpoint: {old_dir}")