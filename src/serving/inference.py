"""
Inference engine for insurance LLM.

Loads model from HuggingFace Hub (SFT or DPO adapter) and provides
generation capabilities. Works on both CPU and GPU.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

# Default system prompt
SYSTEM_PROMPT = """You are an expert insurance support agent for a Turkish insurance company.

You help customers with:
- Policy inquiries and explanations
- Claims processing guidance
- Coverage questions
- Premium and billing information
- Policy modifications and renewals

Respond professionally, accurately, and within company policies. Always be helpful and clear."""


class InsuranceInferenceEngine:
    """
    Inference engine for the fine-tuned insurance model.

    Supports:
    - Loading from HF Hub (adapter) or local path
    - CPU and GPU inference
    - QLoRA 4-bit quantization (GPU only)
    - ChatML prompt formatting
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(
        self,
        base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_repo: str = "sefabilicier/insurance-qwen3b-dpo",
        hf_token: Optional[str] = None,
        local_adapter_path: Optional[str] = None,
        use_4bit: Optional[bool] = None,
    ) -> None:
        """
        Load base model + LoRA adapter.

        Args:
            base_model_name: HF base model identifier
            adapter_repo: HF Hub adapter repo (ignored if local_adapter_path set)
            hf_token: HF token for private repos
            local_adapter_path: Local path to adapter (skips download)
            use_4bit: Force 4-bit quantization (auto-detected if None)
        """
        logger.info("=" * 50)
        logger.info("LOADING MODEL")
        logger.info(f"  Base: {base_model_name}")
        logger.info(f"  Adapter: {adapter_repo}")
        logger.info(f"  Device: {self.device}")
        logger.info("=" * 50)

        start = time.time()

        # Auto-detect quantization
        if use_4bit is None:
            use_4bit = self.device == "cuda"

        # Model kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if use_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["torch_dtype"] = torch.float16
            logger.info("  Mode: QLoRA 4-bit (GPU)")
        else:
            model_kwargs["torch_dtype"] = torch.float32
            logger.info("  Mode: FP32 (CPU)")

        # Load base model
        logger.info("  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, **model_kwargs
        )

        # Download adapter if needed
        if local_adapter_path is None:
            logger.info(f"  Downloading adapter from Hub...")
            local_adapter_path = "/tmp/insurance_adapter"
            snapshot_download(
                repo_id=adapter_repo,
                local_dir=local_adapter_path,
                token=hf_token,
            )

        # Load adapter
        logger.info("  Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            base_model, local_adapter_path, is_trainable=False
        )
        self.model.eval()

        # Move to device if CPU (device_map handles GPU)
        if self.device == "cpu":
            self.model = self.model.to("cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_adapter_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_id = adapter_repo
        self._loaded = True

        elapsed = time.time() - start
        logger.info(f"\n✓ Model loaded in {elapsed:.1f}s")
        if self.device == "cuda":
            logger.info(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    def format_prompt(
        self,
        messages: list,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Format messages to ChatML.

        Args:
            messages: List of {"role": ..., "content": ...}
            system_prompt: Override system prompt

        Returns:
            ChatML formatted string
        """
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        parts = [f"<|im_start|>system\n{system_prompt}\n<|im_end|>"]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role != "system":  # System already added
                parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")

        # Add assistant prompt
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def generate(
        self,
        messages: list,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Generate response.

        Args:
            messages: List of {"role": ..., "content": ...}
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            system_prompt: Override system prompt

        Returns:
            Dict with response text and usage info
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompt = self.format_prompt(messages, system_prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        output_len = outputs.shape[1] - input_len
        elapsed = time.time() - start

        response = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        return {
            "response": response,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": output_len,
                "total_tokens": input_len + output_len,
                "latency_seconds": round(elapsed, 2),
                "tokens_per_second": round(output_len / elapsed, 1) if elapsed > 0 else 0,
            },
        }


# Singleton instance
engine = InsuranceInferenceEngine()