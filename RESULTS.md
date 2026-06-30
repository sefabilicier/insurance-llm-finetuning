# Experiment Results & Analysis

**Project:** Insurance LLM Fine-Tuning  
**Model:** Qwen2.5-7B + LoRA  
**Training Method:** SFT → DPO  
**Date:** TBD  
**Status:** `🔴 Not yet started`

---

## 📋 Experiment Summary

| Metric | Value | Status |
|---|---|---|
| **Total Training Time** | — | ⏳ Pending |
| **SFT Completion** | — | ⏳ Pending |
| **DPO Completion** | — | ⏳ Pending |
| **Final Model Size** | ~14GB | ✅ |
| **LoRA Parameters** | ~0.1M | ✅ |

---

## 🎯 Phase 1: SFT Training

**Goal:** Teach domain knowledge and response format

### Training Configuration
```yaml
Model: Qwen2.5-7B-Instruct + LoRA(r=16)
Duration: 3 epochs
Batch Size: 8
Learning Rate: 2e-4
Optimizer: AdamW
Scheduler: Cosine
```

### Metrics

| Metric | Baseline | After SFT | Target | Status |
|---|---|---|---|---|
| **Train Loss** | — | — | < 0.5 | ⏳ |
| **Eval Loss** | — | — | < 1.0 | ⏳ |
| **ROUGE-L** | — | — | > 0.65 | ⏳ |
| **MMLU Score** | — | — | > 75% | ⏳ |

### Checkpoints

| Epoch | Checkpoint Path | Eval Loss | Status |
|---|---|---|---|
| 1 | `outputs/checkpoints/sft_epoch1` | — | ⏳ |
| 2 | `outputs/checkpoints/sft_epoch2` | — | ⏳ |
| 3 | `outputs/checkpoints/sft_epoch3` | — | ⏳ |

---

## 🎯 Phase 2: DPO Training

**Goal:** Align responses with insurance-specific quality criteria

### Training Configuration
```yaml
Model: SFT Model + LoRA(r=16)
Duration: 1 epoch
Batch Size: 4
Learning Rate: 5e-5
Beta (preference strength): 0.1
Optimizer: AdamW
Scheduler: Cosine
```

### Metrics

| Metric | After SFT | After DPO | Target | Status |
|---|---|---|---|---|
| **Train Loss** | — | — | < 0.3 | ⏳ |
| **Eval Loss** | — | — | < 0.8 | ⏳ |
| **ROUGE-L** | — | — | > 0.72 | ⏳ |
| **Instruction Following** | — | — | > 85% | ⏳ |

### Checkpoints

| Iteration | Checkpoint Path | Eval Loss | Status |
|---|---|---|---|
| Final | `outputs/checkpoints/dpo_final` | — | ⏳ |

---

## 📊 Catastrophic Forgetting Analysis

**Goal:** Ensure general knowledge isn't degraded by domain fine-tuning

### Methodology
- Test on MMLU (5-shot, 100 examples from each category)
- Compare: Baseline vs. SFT vs. DPO
- Threshold: < 5% performance drop

### Results

| Benchmark | Baseline | After SFT | After DPO | Δ Drop |
|---|---|---|---|---|
| **MMLU Overall** | — | — | — | — |
| STEM | — | — | — | — |
| Humanities | — | — | — | — |
| Social Sciences | — | — | — | — |

**Status:** ⏳ Pending

---

## 🎯 Task-Specific Evaluation

### Claim Processing Correctness

**Test:** Given a claim scenario, does the model list processing steps in correct order?

| Scenario | Baseline | SFT | DPO | Target |
|---|---|---|---|---|
| Auto insurance claim | — | — | — | 100% |
| Health insurance claim | — | — | — | 100% |
| Property insurance claim | — | — | — | 100% |

**Status:** ⏳ Pending

### Policy Compliance

**Test:** Does the model generate responses within company policy?

| Policy Rule | Baseline | SFT | DPO | Target |
|---|---|---|---|---|
| No unauthorized promises | — | — | — | 95%+ |
| Deductible explanation accuracy | — | — | — | 90%+ |
| Coverage limit clarity | — | — | — | 90%+ |

**Status:** ⏳ Pending

### Response Format Consistency

**Test:** Is every response in standardized format? (Title + Body + Next Steps)

| Metric | Baseline | SFT | DPO |
|---|---|---|---|
| **Format Compliance** | — | — | — |
| **Average Response Length** | — | — | — |

**Status:** ⏳ Pending

---

## 🔍 Before/After Comparison

### Example 1: Policy Inquiry

**User Query:** "What is my deductible on my auto insurance policy?"

#### Baseline (Qwen2.5-7B unmodified)
```
[Response pending — will add after inference test]
```

#### After SFT
```
[Response pending — will add after SFT completion]
```

#### After DPO
```
[Response pending — will add after DPO completion]
```

**Analysis:** [To be filled after results]

---

### Example 2: Claim Processing

**User Query:** "I was in a car accident. How do I file a claim?"

#### Baseline
```
[Response pending]
```

#### After SFT
```
[Response pending]
```

#### After DPO
```
[Response pending]
```

**Analysis:** [To be filled after results]

---

### Example 3: Coverage Question

**User Query:** "Does my policy cover roadside assistance?"

#### Baseline
```
[Response pending]
```

#### After SFT
```
[Response pending]
```

#### After DPO
```
[Response pending]
```

**Analysis:** [To be filled after results]

---

## 📈 Training Curves

### SFT Training Loss

```
[Training curve chart will be generated from W&B logs]
```

**Observations:** (To be filled)

### DPO Training Loss

```
[Training curve chart will be generated from W&B logs]
```

**Observations:** (To be filled)

---

## 💡 Key Findings

### What Worked Well

1. **[To be filled after experiments]**
2. **[To be filled after experiments]**
3. **[To be filled after experiments]**

### Challenges & Lessons

1. **[To be filled after experiments]**
2. **[To be filled after experiments]**
3. **[To be filled after experiments]**

---

## 📝 Recommendations for Production

### Model Selection

- **Chosen for Deployment:** [SFT / DPO]
- **Reason:** [To be explained after results]
- **Performance:** [Key metrics]

### Hyperparameter Tuning

Potential improvements for next iteration:

1. **Learning rate:** Consider [value] instead of [current]
2. **LoRA rank:** Could increase to [value] for better capacity
3. **Training duration:** Extend to [value] epochs

### Data Augmentation

- Current dataset size: ~500 examples
- Recommendation: Generate [value] more examples for:
  - Edge cases in claim processing
  - Policy exceptions
  - Regional variations

### Serving Strategy

- **Model Size:** ~14GB (loaded as bfloat16)
- **Inference Speed:** [X tokens/second] via vLLM
- **Cost Estimation:** [Cost per 1M tokens]

---

## 🎓 Learnings & Technical Insights

### LoRA vs. Full Fine-Tuning

**Decision:** LoRA with r=16
- **Reasoning:** [To be explained]
- **Trade-off:** [Accuracy vs. efficiency]
- **Result:** [Actual performance]

### SFT → DPO Strategy

**Decision:** Two-phase training
- **SFT Phase:** Establish baseline understanding
- **DPO Phase:** Refine preference alignment
- **Why:** [To be explained]
- **Result:** [Performance improvement from SFT → DPO]

### Data Quality Impact

**Observations:**
- Effect of PII masking on model behavior: [TBD]
- Optimal deduplication threshold: [TBD]
- Quality score filtering impact: [TBD]

---

## 📚 Appendix

### A. Hardware Configuration

```
GPU: [Model]
Memory: [VRAM]
Training Environment: WSL2 Docker
Peak Memory Usage: [TBD]
Training Time per Epoch: [TBD]
```

### B. Data Statistics

```
Total Examples Generated: [TBD]
After Deduplication: [TBD]
After Quality Filtering: [TBD]
Final Train/Val/Test Split: [TBD]

Domain Distribution:
- Claim Processing: [%]
- Policy Inquiry: [%]
- Coverage Questions: [%]
- Premium/Billing: [%]
- Modifications: [%]
```

### C. Experiment Tracking

- **W&B Project:** [insurance-llm-finetuning]
- **Run IDs:** 
  - SFT: [TBD]
  - DPO: [TBD]
- **Dashboard:** [Link to W&B]

---

**Last Updated:** [TBD]  
**Status:** 🔴 In Progress