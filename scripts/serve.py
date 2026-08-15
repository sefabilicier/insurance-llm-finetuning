#!/usr/bin/env python3
"""
Start the Insurance LLM API server.

Usage:
    python scripts/serve.py                          # Defaults (DPO model from Hub)
    python scripts/serve.py --adapter-repo sefabilicier/insurance-qwen3b-sft  # SFT model
    python scripts/serve.py --local-adapter ./outputs/checkpoints/dpo/final_adapter  # Local
    python scripts/serve.py --port 8080              # Custom port
"""

import argparse
import os
import logging

import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Start Insurance LLM API server")

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-repo", default="sefabilicier/insurance-qwen3b-dpo")
    parser.add_argument("--local-adapter", default=None, help="Local adapter path (skips Hub download)")
    parser.add_argument("--hf-token", default=None)

    args = parser.parse_args()

    # Set env vars for FastAPI lifespan
    os.environ["BASE_MODEL"] = args.base_model
    os.environ["ADAPTER_REPO"] = args.adapter_repo
    if args.local_adapter:
        os.environ["LOCAL_ADAPTER_PATH"] = args.local_adapter
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    import uvicorn
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()