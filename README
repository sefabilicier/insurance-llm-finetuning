# Insurance LLM Fine-Tuning Pipeline

> **A production-ready, end-to-end LLM fine-tuning pipeline that adapts a general-purpose language model to the insurance domain using Parameter-Efficient Fine-Tuning (PEFT) and human preference alignment.**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Theoretical Foundation](#theoretical-foundation)
- [Solution Architecture](#solution-architecture)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Phase 1: Project Setup](#phase-1-project-setup)
  - [Phase 2: Data Pipeline](#phase-2-data-pipeline)
  - [Phase 3: Training Infrastructure](#phase-3-training-infrastructure)
  - [Phase 4: Evaluation Framework](#phase-4-evaluation-framework)
  - [Phase 5: Serving Infrastructure](#phase-5-serving-infrastructure)
- [System Diagrams](#system-diagrams)
- [Services & Ports](#services--ports)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Evaluation Results](#evaluation-results)
- [References & Theoretical Background](#references--theoretical-background)

---

## Problem Statement

An insurance company needs to automate customer support across five core domains: policy inquiries, claim processing, coverage questions, premium/billing, and policy modifications. General-purpose LLMs fail in this context because they:

- **Misinterpret insurance terminology** — "deductible", "premium", "endorsement" carry domain-specific meanings that general models conflate with everyday usage
- **Generate incorrect claim procedures** — Steps are hallucinated or presented out of order, creating compliance risk
- **Violate company policy** — Responses may promise coverage that doesn't exist or misrepresent policy terms
- **Produce inconsistent formats** — Response structure varies unpredictably, making downstream integration unreliable

These failures are not solvable through prompt engineering alone. When domain knowledge must be internalized rather than retrieved, and when response format must be structurally consistent, fine-tuning becomes the appropriate intervention. This aligns with IBM's AI Ladder methodology, which positions fine-tuning as the recommended approach when "the model must reason within domain constraints, not merely reference domain documents" (IBM Think 2024).

---

## Theoretical Foundation

### Why Fine-Tuning Over Alternatives?

The decision to fine-tune follows a structured evaluation framework drawn from Google's Vertex AI documentation and AWS Bedrock best practices:

<img width="6000" height="5000" alt="Image" src="https://github.com/user-attachments/assets/a4952af7-a428-44c0-ad27-779c6e665b4d" />

### LoRA: Low-Rank Adaptation

Full fine-tuning updates all model parameters (3 billion for Qwen2.5-3B), which is computationally expensive and risks catastrophic forgetting. LoRA (Hu et al., 2021) addresses this by decomposing weight updates into low-rank matrices:

```
    Standard Fine-Tuning          LoRA Adaptation
    ┌─────────────────┐          ┌─────────────────┐
    │                 │          │                 │
    │  W (3B params)  │          │  W (frozen)     │
    │  ALL updated    │          │  + A·B (0.1%    │
    │                 │          │    params)       │
    └─────────────────┘          └─────────────────┘
    
    Where: W' = W + A·B
    A: d × r  (down-projection)
    B: r × d  (up-projection)
    r = 16   (rank, our config)
    
    Result: Only 0.58% of parameters trained
    Memory: ~60% reduction vs full fine-tuning
```

With rank r=16 and alpha=32, the scaling factor is α/r = 2.0, meaning LoRA updates are scaled by 2× before adding to frozen weights. This configuration follows the recommendations from the original LoRA paper and Microsoft Research's practical guidelines for 1B-7B parameter models.

### QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) extends LoRA by loading the base model in 4-bit NormalFloat (NF4) quantization, reducing memory from ~12GB to ~5GB for a 3B model. This enables fine-tuning on consumer GPUs (T4 16GB):

```
    Memory Comparison (Qwen2.5-3B):
    
    Full Fine-Tuning:     ~24 GB  ████████████████████████
    LoRA (FP16):          ~12 GB  ████████████
    QLoRA (NF4 + LoRA):   ~5 GB   █████
    
    Our setup: QLoRA on T4 (16GB VRAM) — 5GB model + 3GB training overhead = ~8GB
```

### SFT → DPO: Two-Phase Alignment

The training follows a two-phase alignment strategy based on the InstructGPT methodology (Ouyang et al., 2022) and refined by Rafailov et al.'s Direct Preference Optimization:

<img width="6000" height="4000" alt="Image" src="https://github.com/user-attachments/assets/8fed3b6e-adfe-4465-9010-708c3b0dd009" />

DPO eliminates the need for a separate reward model (unlike RLHF), directly optimizing the policy from preference pairs. With β=0.1, the model strongly enforces preference ordering while maintaining generation diversity. This approach is recommended by Anthropic and HuggingFace for production fine-tuning where reward model training is infeasible.

### ChatML: Conversation Format Standard

All training data uses the ChatML format, which is the native conversation template for Qwen models and adopted as a standard by the OpenAI ecosystem:

```
    <|im_start|>system
    You are an expert insurance support agent...
    <|im_end|>
    <|im_start|>user
    What is my deductible on my auto insurance?
    <|im_end|>
    <|im_start|>assistant
    Thank you for contacting us. Your auto insurance 
    deductible is currently set at $500...
    <|im_end|>
```

This format provides explicit role boundaries, preventing role confusion during generation and enabling the response template masking strategy in SFT (only assistant tokens contribute to the loss).

---

## Solution Architecture

### High-Level System Architecture

<img width="7000" height="5500" alt="Image" src="https://github.com/user-attachments/assets/069799ae-b02e-47df-a62e-e9bcc79c4cc1" />

### Data Pipeline Architecture

<img width="7000" height="4500" alt="Image" src="https://github.com/user-attachments/assets/eea4dde2-5b3a-4658-9965-a245bbf670bc" />

### Training Architecture

<img width="6000" height="5000" alt="Image" src="https://github.com/user-attachments/assets/a09b54bd-437a-4597-bbaf-f1bacaf3be93" />

### Evaluation Architecture

<img width="6000" height="4500" alt="Image" src="https://github.com/user-attachments/assets/bee3bb53-b54d-4ca0-a51e-fccb46f44587" />

---

## Technical Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Base Model** | Qwen2.5-3B-Instruct | Multilingual (Turkish support), instruction-tuned, fits T4 with QLoRA. Selected over Llama-3.2-3B for stronger multilingual capabilities (Qwen technical report, 2024) |
| **PEFT** | LoRA (r=16, α=32) | 0.58% parameters trained; zero inference overhead after merge. Follows Microsoft Research LoRA guidelines for sub-7B models |
| **Quantization** | QLoRA NF4 4-bit | Reduces VRAM from ~12GB to ~5GB. NormalFloat4 provides better quantization fidelity than INT4 (Dettmers et al., 2023) |
| **SFT Framework** | HuggingFace TRL SFTTrainer | Production-tested, integrates with PEFT natively, supports ChatML response masking |
| **Alignment** | DPO (β=0.1, sigmoid loss) | No reward model needed; direct preference optimization. More stable than PPO for small models (Rafailov et al., 2023) |
| **Data Generation** | Hybrid: Templates (80%) + Ollama Llama 3.1 (20%) | Templates ensure consistency and speed (~3 seconds for 800 examples); Ollama adds natural variation for edge cases |
| **Experiment Storage** | HuggingFace Hub | Free, persistent across sessions, versioned. Adapters auto-pushed after each training phase |
| **Training GPU** | Kaggle T4 (16GB VRAM, free) | 30 hours/week free quota. T4 supports FP16 mixed precision; QLoRA keeps memory under 8GB |
| **Evaluation** | Custom metrics suite | ROUGE, BLEU (text similarity), keyword coverage (domain relevance), format compliance (professional quality), catastrophic forgetting (knowledge retention) |
| **Serving** | FastAPI + HF Transformers | Lightweight API with OpenAI-compatible chat endpoint. Supports both CPU and GPU inference |
| **Containerization** | Docker + Docker Compose (WSL2) | Reproducible serving environment. Separate profiles for CPU-only and GPU deployments |

---

## Project Structure

```
insurance-llm-finetuning/
│
├── config/                              # All hyperparameters (YAML)
│   ├── model_config.yaml                #   Qwen2.5-3B + QLoRA + LoRA settings
│   ├── data_config.yaml                 #   Data source, cleaning, splitting config
│   └── training_config.yaml             #   SFT + DPO training parameters
│
├── src/                                 # Source code (5 modules)
│   ├── data/                            #   Data pipeline
│   │   ├── loader.py                    #     Load CSV, JSON, HF datasets
│   │   ├── generator.py                 #     Hybrid template + Ollama generation
│   │   ├── processor.py                 #     PII masking, dedup, quality scoring
│   │   ├── quality_scorer.py            #     Quality scoring utilities
│   │   └── dataset.py                   #     ChatML formatting, train/val/test split
│   │
│   ├── training/                        #   Training infrastructure
│   │   ├── sft_trainer.py               #     SFT trainer (TRL wrapper)
│   │   ├── dpo_trainer.py               #     DPO trainer + preference pair builder
│   │   ├── callbacks.py                 #     Loss logging, GPU memory, best model
│   │   └── utils.py                     #     Model loading, LoRA merge, checkpoints
│   │
│   ├── evaluation/                      #   Evaluation framework
│   │   ├── metrics.py                   #     ROUGE, BLEU, keyword, format metrics
│   │   ├── forgetting.py               #     Catastrophic forgetting analysis (20 MC)
│   │   └── reporter.py                 #     JSON + Markdown report generation
│   │
│   ├── serving/                         #   Inference engine
│   │   └── inference.py                 #     Model loading from Hub, generation
│   │
│   └── api/                             #   REST API
│       ├── main.py                      #     FastAPI app with lifespan management
│       └── schemas.py                   #     Request/Response Pydantic models
│
├── scripts/                             # CLI entry points
│   ├── prepare_data.py                  #   Data pipeline orchestration
│   ├── train.py                         #   Training orchestration (sft/dpo/merge/all)
│   ├── evaluate.py                      #   Evaluation orchestration (offline/model)
│   ├── merge_and_export.py              #   LoRA merge into base model
│   └── serve.py                         #   Start FastAPI server
│
├── notebooks/                           # Jupyter notebooks
│   └── kaggle_training.ipynb            #   Full pipeline for Kaggle (SFT+DPO+Eval)
│
├── data/                                # Data directory
│   ├── raw/                             #   Generated synthetic data
│   ├── processed/                       #   Cleaned, deduplicated
│   ├── formatted/                       #   ChatML format
│   └── splits/                          #   train.json, validation.json, test.json
│
├── outputs/                             # Training outputs
│   ├── checkpoints/                     #   SFT + DPO checkpoints
│   ├── merged_models/                   #   Final merged model (if applicable)
│   ├── logs/                            #   Training logs
│   └── evaluation/                      #   Metrics JSON + reports
│
├── tests/                               # Unit tests
├── Dockerfile                           #   Training container (CUDA)
├── Dockerfile.serve                     #   Serving container (CPU)
├── docker-compose.yml                   #   Multi-service orchestration
├── requirements.txt                     #   Python dependencies
├── .env.example                         #   Environment variables template
├── .gitignore                           #   Git ignore patterns
├── README.md                            #   This file
└── RESULTS.md                           #   Experiment results template
```

---

## Pipeline Walkthrough

### Phase 1: Project Setup

Establishes the repository structure, configuration files, dependency management, and containerization. All hyperparameters are externalized to YAML configs, following the 12-factor app methodology for ML systems (Google MLOps Level 2).

**Key files:** `config/*.yaml`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.env.example`

### Phase 2: Data Pipeline

Generates 1000 synthetic insurance customer support conversations using a hybrid approach.

**Template Generator (80%):** 50 handcrafted templates across 5 insurance categories, each with 15+ randomization pools (customer names, policy numbers, amounts, vehicle types, etc.). Produces 800 diverse examples in approximately 3 seconds.

**Ollama Generator (20%):** Uses locally-hosted Llama 3.1 to generate 200 additional examples with natural language variation and edge cases that templates cannot cover.

**Processing pipeline:** Three-stage cleaning ensures data quality:

1. **PII Masking** — Regex-based detection and replacement of emails, phone numbers, SSNs, credit card numbers, and IP addresses. Replacements use bracketed tokens (`[EMAIL]`, `[PHONE]`) to preserve semantic meaning while removing sensitive data.

2. **Deduplication** — N-gram Jaccard similarity with threshold 0.95. Compares bigram sets between all example pairs and removes near-duplicates. This approach catches paraphrases that exact-match deduplication would miss.

3. **Quality Scoring** — Heuristic scoring (0–1) based on response length, word repetition ratio, language detection, and placeholder detection. Examples below 0.6 are filtered.

**Output:** 997 cleaned examples split into train (797), validation (99), and test (101) with seed=42 for reproducibility.

**Key files:** `src/data/generator.py`, `src/data/processor.py`, `src/data/dataset.py`, `scripts/prepare_data.py`

### Phase 3: Training Infrastructure

Two-phase training pipeline executing on Kaggle's free T4 GPU (16GB VRAM).

**SFT Phase:** Trains the QLoRA adapter for 3 epochs (early stopping at epoch 2) on instruction-response pairs. The SFTTrainer uses response template masking (`<|im_start|>assistant\n`), ensuring that only assistant tokens contribute to the cross-entropy loss. Training uses cosine learning rate scheduling with 5% warmup, gradient accumulation of 4 (effective batch size 8), and gradient checkpointing for memory efficiency.

**DPO Phase:** Takes the SFT-trained adapter and optimizes it for 1 epoch on preference pairs. Each training example consists of a prompt, a "chosen" response (the original high-quality answer), and a "rejected" response (one of 8 degradation strategies: vague answers, missing information, unprofessional tone, overpromising, generic non-answers, etc.). With β=0.1 and sigmoid loss, the model learns to strongly prefer professional, accurate responses.

**Persistent Storage:** Both adapters are automatically pushed to HuggingFace Hub during training (`hub_strategy="checkpoint"`), ensuring that no training progress is lost if the Kaggle session expires. This addresses a real failure mode encountered during development: Colab/Kaggle sessions can terminate unexpectedly, and adapter weights stored only in ephemeral storage are permanently lost.

**Key files:** `src/training/sft_trainer.py`, `src/training/dpo_trainer.py`, `src/training/utils.py`, `scripts/train.py`, `notebooks/kaggle_training.ipynb`

### Phase 4: Evaluation Framework

Comprehensive evaluation suite measuring four dimensions of model quality.

**Text Similarity (ROUGE, BLEU):** Measures lexical overlap between generated responses and reference answers. ROUGE-L (longest common subsequence) is the primary metric, capturing both content coverage and response ordering. BLEU measures n-gram precision with brevity penalty.

**Domain Keyword Coverage:** For each insurance category, a curated set of expected domain terms is checked against the model's response. This ensures the model actually uses insurance terminology rather than generating generic text.

**Format Compliance:** Five-point professional format checklist: adequate length (≥20 words), no excessive repetition, professional tone indicators, closing/follow-up offer, and no placeholder text. A perfect score of 1.0 indicates all responses meet professional standards.

**Catastrophic Forgetting:** 20 multiple-choice questions spanning science, geography, math, technology, history, and literature. The model's log-probability is computed for each answer choice, and accuracy is compared between the base model and fine-tuned model. A relative accuracy drop exceeding 5% triggers a warning. This methodology follows AWS SageMaker's recommended approach for monitoring knowledge retention during domain adaptation.

**Key files:** `src/evaluation/metrics.py`, `src/evaluation/forgetting.py`, `src/evaluation/reporter.py`, `scripts/evaluate.py`

### Phase 5: Serving Infrastructure

Production-ready API serving the fine-tuned model via FastAPI.

**Inference Engine:** Loads the base model (Qwen2.5-3B) and applies the DPO adapter from HuggingFace Hub. On GPU systems, the model loads in 4-bit quantization for optimal throughput. On CPU systems (32GB RAM), the model loads in FP32 with acceptable latency for demonstration purposes.

**API Design:** OpenAI-compatible chat completions endpoint (`/v1/chat/completions`) accepts conversation messages and returns generated responses with token usage statistics. The health endpoint (`/health`) reports model loading status, device type, and model identifier.

**Containerization:** Two deployment profiles via Docker Compose: `serving` (CPU-only, uses `Dockerfile.serve`) and `gpu-serving` (requires NVIDIA GPU, uses vLLM for high-throughput inference). Both profiles use a shared HuggingFace cache volume to avoid re-downloading model weights.

**Key files:** `src/serving/inference.py`, `src/api/main.py`, `src/api/schemas.py`, `scripts/serve.py`, `Dockerfile.serve`, `docker-compose.yml`

---

## Services & Ports

| Service | Port | Profile | Description |
|---|---|---|---|
| **FastAPI (CPU)** | `8001` | `serving` | Insurance chat API, CPU inference, auto-downloads adapter from HF Hub |
| **Swagger UI** | `8001/docs` | `serving` | Interactive API documentation |
| **vLLM** | `8000` | `gpu-serving` | High-throughput GPU inference server (requires NVIDIA GPU) |
| **FastAPI (GPU)** | `8001` | `gpu-serving` | API gateway proxying to vLLM backend |
| **TensorBoard** | `6006` | `monitoring` | Training loss curves and metric visualization |
| **Jupyter** | `8888` | `dev` | Development notebook server |

**Running services:**

```bash
# CPU serving (no GPU required)
docker compose --profile serving up --build

# GPU serving (NVIDIA GPU required)
docker compose --profile gpu-serving up --build

# Training monitoring
docker compose --profile monitoring up

# Development environment
docker compose --profile dev up
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- Git
- Docker + Docker Compose (WSL2 on Windows)
- HuggingFace account with write token

### 1. Clone & Setup

```bash
git clone https://github.com/sefabilicier/insurance-llm-finetuning.git
cd insurance-llm-finetuning

python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
cp .env.example .env
# Edit .env: set HF_TOKEN
```

### 2. Generate Data

```bash
python scripts/prepare_data.py --step all --num-examples 1000
```

### 3. Train (on Kaggle)

Upload project to Kaggle, open `notebooks/kaggle_training.ipynb`, enable T4 GPU, and run all cells. Adapters are automatically pushed to HuggingFace Hub.

### 4. Evaluate

```bash
# Offline evaluation (no GPU needed)
python scripts/evaluate.py --phase offline

# Model evaluation (requires GPU + trained model)
python scripts/evaluate.py --phase dpo
```

### 5. Serve

```bash
# Direct (CPU)
HF_TOKEN=hf_xxx python scripts/serve.py --port 8001

# Docker (WSL2)
echo "HF_TOKEN=hf_xxx" > .env
docker compose --profile serving up --build

# Test
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is my deductible?"}]}'

#Response example
{
    "response": "I understand your concern. Your policy deductible is currently set at $250. This is the amount you would need to pay out-of-pocket before your insurance coverage kicks in for any covered claim. Would you like me to help you with anything else?",
    "model": "sefabilicier/insurance-qwen3b-dpo",
    "usage": {
        "prompt_tokens": 80,
        "completion_tokens": 52,
        "total_tokens": 132,
        "latency_seconds": 49.93,
        "tokens_per_second": 1.0
    }
}
```

---

## Configuration Reference

### model_config.yaml

```yaml
model:
  model_name: "Qwen/Qwen2.5-3B-Instruct"
  torch_dtype: "float16"          # T4 compatible
peft:
  lora_r: 16                      # Rank (capacity vs efficiency)
  lora_alpha: 32                   # Scaling factor (2× rank)
  lora_dropout: 0.05               # Regularization
  target_modules: [q,v,k,o,gate,up,down]_proj
quantization:
  load_in_4bit: true               # QLoRA enabled
  bnb_4bit_quant_type: "nf4"       # NormalFloat4
```

### training_config.yaml

```yaml
sft:
  num_train_epochs: 3
  per_device_train_batch_size: 2   # ×4 accumulation = 8 effective
  learning_rate: 2e-4              # Standard for LoRA
  max_seq_length: 1024             # Reduced for T4 VRAM
dpo:
  num_train_epochs: 1
  per_device_train_batch_size: 1   # DPO pairs need more memory
  learning_rate: 5e-5              # Lower for preference learning
  beta: 0.1                        # Preference enforcement strength
```

---

## Evaluation Results

### Overall Performance

| Metric | Score | Interpretation |
|---|---|---|
| **ROUGE-1** | 0.7690 | Strong unigram overlap with reference responses |
| **ROUGE-2** | 0.6974 | Good bigram-level content reproduction |
| **ROUGE-L** | 0.7544 | Strong structural similarity to reference answers |
| **BLEU** | 0.6665 | Good n-gram precision across 1-4 grams |
| **Keyword Coverage** | 0.8911 | Model uses 89% of expected domain terminology |
| **Format Compliance** | 1.0000 | Every response meets professional format standards |

### Per-Category Performance

| Category | ROUGE-L | Keywords | Format | N |
|---|---|---|---|---|
| Claim Processing | 0.7887 | 1.0000 | 1.0000 | 18 |
| Policy Modifications | 0.7986 | 0.6905 | 1.0000 | 21 |
| Coverage Questions | 0.7773 | 0.8947 | 1.0000 | 19 |
| Premium/Billing | 0.7615 | 0.9722 | 1.0000 | 18 |
| Policy Inquiry | 0.6703 | 0.9200 | 1.0000 | 25 |

### Training Metrics

| Phase | Final Loss | Duration | Early Stopping |
|---|---|---|---|
| SFT | 0.099 (eval) | ~90 min | Yes (epoch 2/3) |
| DPO | 0.000 (converged) | ~60 min | No |

### Model Artifacts (HuggingFace Hub)

| Artifact | Repository | Type |
|---|---|---|
| SFT Adapter | `sefabilicier/insurance-qwen3b-sft` | LoRA weights + tokenizer |
| DPO Adapter | `sefabilicier/insurance-qwen3b-dpo` | LoRA weights + tokenizer |
| Evaluation | `sefabilicier/insurance-llm-eval-results` | Metrics JSON + predictions |

---

## References & Theoretical Background

### Core Papers

| Paper | Contribution | Our Usage |
|---|---|---|
| Hu et al. (2021) — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | Introduced low-rank decomposition for efficient fine-tuning | LoRA configuration (r=16, α=32) applied to all attention + MLP projections |
| Dettmers et al. (2023) — [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 4-bit NormalFloat quantization with LoRA | NF4 quantization enabling 3B model training on T4 16GB |
| Rafailov et al. (2023) — [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | Eliminated reward model from RLHF, direct policy optimization | DPO alignment with β=0.1 sigmoid loss for preference learning |
| Ouyang et al. (2022) — [Training Language Models to Follow Instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) | SFT → RLHF pipeline for instruction following | Two-phase SFT → DPO alignment strategy |

### Industry Best Practices

| Source | Practice | Our Application |
|---|---|---|
| **IBM Think 2024** — Enterprise AI Fine-Tuning | "Fine-tune when the model must reason within domain constraints, not merely reference domain documents" | Decision to fine-tune vs RAG for insurance policy logic |
| **Google Vertex AI** — [Tuning Guide](https://cloud.google.com/vertex-ai/docs/generative-ai/models/tune-models) | Structured evaluation framework: prompt engineering → RAG → fine-tuning decision tree | Problem assessment methodology and evaluation strategy |
| **AWS Bedrock** — [Custom Model Training](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html) | Catastrophic forgetting monitoring with general benchmarks before/after training | 20-question MMLU-style evaluation with <5% drop threshold |
| **HuggingFace TRL** — [SFT Best Practices](https://huggingface.co/docs/trl/sft_trainer) | Response template masking, packing strategies, gradient checkpointing | ChatML response masking, gradient checkpointing for VRAM optimization |
| **Microsoft Research** — [LoRA Guidelines](https://github.com/microsoft/LoRA) | Rank selection: r=8-16 for domain adaptation, target all projection matrices for best quality | r=16 with all 7 projection targets (q,k,v,o,gate,up,down) |
| **Google MLOps** — [Level 2 ML Systems](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) | Externalized configuration, reproducible pipelines, experiment tracking | YAML configs, seed=42, HF Hub versioned storage |

### Key Design Decisions

| Decision | Alternatives Considered | Rationale |
|---|---|---|
| Qwen2.5-3B over Llama-3.2-3B | Llama 3.2, Phi-3-mini | Stronger multilingual support (Turkish), native ChatML format |
| QLoRA over full LoRA | Full LoRA (FP16), full fine-tuning | VRAM constraint (T4 16GB); QLoRA reduces from ~12GB to ~5GB |
| DPO over RLHF/ORPO | RLHF (PPO), ORPO, SimPO | No reward model needed; stable training; well-documented |
| Hybrid data generation | Pure LLM, pure template, public datasets | Templates ensure speed and consistency; LLM adds naturality |
| HF Hub over local storage | Local checkpoints, Google Drive, S3 | Free, versioned, persistent across ephemeral Kaggle sessions |
| Kaggle over Colab | Google Colab, Lightning AI, Lambda | 30hr/week GPU quota; Colab session expired during first training |

---