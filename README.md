# subset-kl

Memory-efficient KL divergence for large vocabulary models.

[![PyPI version](https://badge.fury.io/py/subset-kl.svg)](https://badge.fury.io/py/subset-kl)

## Overview

Computing KL divergence over full vocabularies can dominate memory:

```
Full KL: [batch, seq, vocab] = [2, 1024, 128000] = 1 GB (fp32)
```

This is often the bottleneck during distillation or lens training. `subset-kl`
reduces memory by restricting KL to a subset of tokens:

```
Subset KL: [batch, seq, k] = [2, 1024, 256] = 2 MB (500x reduction)
```

## Design Scope

This package provides KL math only. It does not:
- Hold model weights
- Compute student logits from hidden states
- Provide CUDA kernels

Your model/lens is responsible for computing student logits for selected indices. This separation of concerns means subset-kl works with any architecture.

## Installation

```bash
pip install subset-kl
```

## Usage

### Efficient Workflow

```python
from subset_kl import select_topk_indices, subset_kl_from_gathered

# Step 1: Select indices from teacher distribution
indices, teacher_k = select_topk_indices(teacher_logits, k=256)
# indices: [B, T, k] - which tokens to compute
# teacher_k: [B, T, k] - teacher logits for those tokens

# Step 2: Compute student logits only for those indices
# Use indexed_logits, your lens's vocab_indices param, etc.
student_k = your_lens.forward(hidden_states, vocab_indices=indices).logits

# Step 3: Compute KL on the subsets
loss = subset_kl_from_gathered(student_k, teacher_k, attention_mask)
```

This never materializes `[B, T, V]` student logits.

### With a LoRA Lens

```python
from subset_kl import select_topk_indices, subset_kl_from_gathered

class EfficientLensTrainer:
    def __init__(self, lens, k=256):
        self.lens = lens
        self.k = k
    
    def compute_loss(self, hidden_states, teacher_logits, attention_mask):
        # Get top-k indices from teacher
        indices, teacher_k = select_topk_indices(teacher_logits, k=self.k)
        
        # Lens computes logits only for selected indices
        student_k = self.lens.forward(
            hidden_states, 
            vocab_indices=indices  # [B, T, k]
        ).logits  # [B, T, k]
        
        # Compute KL
        return subset_kl_from_gathered(student_k, teacher_k, attention_mask)
```

### Convenience: When You Have Full Logits

If you already have full student logits (no memory savings, but convenient):

```python
from subset_kl import SubsetKLLoss

loss_fn = SubsetKLLoss(k=256)
loss = loss_fn(student_logits, teacher_logits, attention_mask)
```

Or functional:

```python
from subset_kl import compute_subset_kl

loss = compute_subset_kl(student_logits, teacher_logits, k=256)
```

## API Reference

### Core Functions (Recommended)

```python
# Index selection
indices, teacher_k = select_topk_indices(teacher_logits, k=256)

# KL on pre-gathered tensors (the efficient path)
loss = subset_kl_from_gathered(student_k, teacher_k, attention_mask)

# Full-vocab KL (baseline for comparison)
loss = full_kl(student_logits, teacher_logits)
```

### Class Interface

```python
# Subset KL (top-k)
loss_fn = SubsetKLLoss(k=256, reduction="mean")
loss = loss_fn(student_logits, teacher_logits)  # Convenience path

# Full KL (baseline for comparison)
loss_fn = KLDivergenceLoss(reduction="mean", temperature=1.0)

# Exact top-k head + Monte Carlo teacher-tail estimate
loss_fn = SubsetMonteCarloKLLoss(k_head=256, k_tail=256)
```

Two estimators are provided:
- **Top-k** — KL renormalized over the teacher's top-k tokens. Fast and the
  recommended default.
- **Monte Carlo (MC)** — exact KL on the top-k head plus an importance-weighted
  estimate of the tail, sampled from the (normalized) teacher tail. Use when the
  renormalized top-k underestimates the tail.

### Monte Carlo Tail Estimator (Advanced)

For the memory-efficient MC path, select a head plus sampled tail, compute
student logits only for those indices, and supply the student full-vocabulary
log-normalizer (the tail term needs true student log-probabilities):

```python
from subset_kl import select_head_tail_indices, subset_mc_kl_from_gathered

indices, teacher_log_probs, p_head = select_head_tail_indices(
    teacher_logits, k_head=256, k_tail=256
)
student_selected = your_model.forward_subset(hidden, indices)
student_log_normalizer = your_model.full_vocab_logsumexp(hidden)  # logsumexp over V

loss = subset_mc_kl_from_gathered(
    student_selected,
    teacher_log_probs,
    k_head=256,
    k_tail=256,
    p_head=p_head,
    student_log_normalizer=student_log_normalizer,
    attention_mask=attention_mask,
)
```

Estimators at a glance:
- Head only (renormalized): `select_topk_indices` + `subset_kl_from_gathered`
- Exact head + sampled tail (MC): `select_head_tail_indices` + `subset_mc_kl_from_gathered`

## Memory Comparison

| Vocab Size | Full KL | Subset (k=256) | Reduction |
|------------|---------|----------------|-----------|
| 32,000     | 256 MB  | 2 MB           | 128x      |
| 50,257     | 402 MB  | 2 MB           | 200x      |
| 128,000    | 1 GB    | 2 MB           | 512x      |
| 256,000    | 2 GB    | 2 MB           | 1024x     |

*(B=2, T=1024, fp32)*

## Integration with Other Packages

subset-kl is designed to work with:

| Package | Role |
|---------|------|
| **indexed_logits** | CUDA kernel for efficient `H @ W[indices].T` |
| **hookbox** | Activation capture from base model |
| **your lens** | Computes student logits for selected indices |

The workflow is:
- Select indices from the teacher logits.
- Compute student logits only for those indices.
- Compute KL on the subset.

## Notes

The full KL divergence is:

```
KL(P || Q) = Σ_{i=1}^{V} P_i * (log P_i - log Q_i)
```

The top-k approximation restricts to k tokens and renormalizes:

```
KL_k(P || Q) = Σ_{i ∈ top-k(P)} P'_i * (log P'_i - log Q'_i)
```

For large vocabularies, `k` in the low hundreds often captures most mass,
but the right value depends on the model and task.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
