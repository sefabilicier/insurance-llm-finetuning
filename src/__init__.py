"""
Insurance LLM Fine-Tuning Pipeline

A production-ready fine-tuning pipeline for adapting LLMs to insurance domain
customer support tasks using LoRA + SFT + DPO.

Modules:
- src.data: Data loading, processing, and formatting
- src.training: SFT and DPO trainer implementations
- src.evaluation: Metrics and evaluation utilities
- src.serving: vLLM and inference infrastructure
- src.api: FastAPI application and endpoints
"""

__version__ = "0.1.0"
__author__ = "Sefa's"