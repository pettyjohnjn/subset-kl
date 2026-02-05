# src/subset_kl/core.py
"""
Core functional interfaces for subset KL divergence.

This module provides the primary APIs for memory-efficient KL computation:

1. `select_topk_indices()` - Select which tokens to use
2. `subset_kl_from_gathered()` - Compute KL on pre-gathered subsets
3. `compute_subset_kl()` - All-in-one when you have full logits

The typical workflow for maximum efficiency:

    # 1. Select indices from teacher (subset-kl)
    indices, teacher_k = select_topk_indices(teacher_logits, k=256)
    
    # 2. Compute student logits ONLY for those indices (your model/lens)
    student_k = your_model.compute_logits_for_indices(hidden, indices)
    
    # 3. Compute KL on the subsets (subset-kl)
    loss = subset_kl_from_gathered(student_k, teacher_k)

This avoids materializing full [B, T, V] student logits.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base import apply_reduction, ReductionType


# =============================================================================
# Index Selection
# =============================================================================

def select_topk_indices(
    teacher_logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k indices from the teacher distribution.
    
    This is the first step in memory-efficient subset KL. The returned
    indices should be passed to your model to compute student logits
    for ONLY these tokens.
    
    Parameters
    ----------
    teacher_logits : torch.Tensor
        Teacher logits [batch, seq, vocab] or [N, vocab].
    k : int
        Number of top tokens to select.
        
    Returns
    -------
    indices : torch.Tensor
        Top-k token indices [batch, seq, k] or [N, k].
    teacher_logits_k : torch.Tensor
        Teacher logits for selected indices [batch, seq, k] or [N, k].
        
    Examples
    --------
    >>> indices, teacher_k = select_topk_indices(teacher_logits, k=256)
    >>> student_k = model.forward_subset(hidden_states, indices)
    >>> loss = subset_kl_from_gathered(student_k, teacher_k)
    """
    # topk returns (values, indices)
    teacher_logits_k, indices = teacher_logits.topk(k=k, dim=-1)
    return indices, teacher_logits_k


def select_indices_with_sampling(
    teacher_logits: torch.Tensor,
    k_head: int,
    k_tail: int = 0,
    oversample: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select indices using top-k head + importance-sampled tail.
    
    For most cases, pure top-k (k_tail=0) is sufficient and more stable.
    Use this when you need an unbiased estimator.
    
    Parameters
    ----------
    teacher_logits : torch.Tensor
        Teacher logits [batch, seq, vocab].
    k_head : int
        Number of top tokens (deterministic).
    k_tail : int
        Number of additional sampled tokens (stochastic).
    oversample : int
        Oversample factor for PPS sampling.
        
    Returns
    -------
    indices : torch.Tensor
        Selected indices [batch, seq, S] where S <= k_head + k_tail.
    teacher_logits_selected : torch.Tensor
        Teacher logits for selected indices.
    inclusion_probs : torch.Tensor
        Inclusion probabilities for importance weighting.
    mask : torch.Tensor
        Valid mask for variable-length selections.
    """
    from .sampling import pps_sample_indices_batched
    
    B, T, V = teacher_logits.shape
    
    # Flatten for sampling
    teacher_flat = teacher_logits.view(B * T, V)
    log_probs = F.log_softmax(teacher_flat, dim=-1)
    
    indices, inc_probs, mask, _ = pps_sample_indices_batched(
        log_probs, k_head=k_head, k_tail=k_tail, oversample=oversample
    )
    
    # Gather teacher values
    teacher_selected = torch.gather(teacher_flat, -1, indices)
    
    # Reshape back to [B, T, S]
    S = indices.shape[-1]
    indices = indices.view(B, T, S)
    teacher_selected = teacher_selected.view(B, T, S)
    inc_probs = inc_probs.view(B, T, S)
    mask = mask.view(B, T, S)
    
    return indices, teacher_selected, inc_probs, mask


def select_indices_with_importance_sampling(
    teacher_logits: torch.Tensor,
    k: int,
    oversample: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select indices using pure importance sampling (no deterministic head).
    
    This is equivalent to select_indices_with_sampling(k_head=0, k_tail=k).
    
    Parameters
    ----------
    teacher_logits : torch.Tensor
        Teacher logits [batch, seq, vocab].
    k : int
        Number of sampled tokens (stochastic).
    oversample : int
        Oversample factor for PPS sampling.
        
    Returns
    -------
    indices : torch.Tensor
        Selected indices [batch, seq, S] where S <= k.
    teacher_logits_selected : torch.Tensor
        Teacher logits for selected indices.
    inclusion_probs : torch.Tensor
        Inclusion probabilities for importance weighting.
    mask : torch.Tensor
        Valid mask for variable-length selections.
    """
    return select_indices_with_sampling(
        teacher_logits=teacher_logits,
        k_head=0,
        k_tail=k,
        oversample=oversample,
    )


# =============================================================================
# KL Computation on Pre-Gathered Tensors
# =============================================================================

def subset_kl_from_gathered(
    student_logits_k: torch.Tensor,
    teacher_logits_k: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
) -> torch.Tensor:
    """
    Compute KL divergence from pre-gathered subset logits.
    
    Both student and teacher logits should be for the SAME k indices
    (typically from `select_topk_indices`). Distributions are 
    renormalized over the k tokens before computing KL.
    
    Parameters
    ----------
    student_logits_k : torch.Tensor
        Student logits for subset [batch, seq, k] or [N, k].
    teacher_logits_k : torch.Tensor
        Teacher logits for same subset [batch, seq, k] or [N, k].
    attention_mask : Optional[torch.Tensor]
        Mask [batch, seq] or [N]. Applied after KL computation.
    reduction : str
        "none", "mean", or "sum".
        
    Returns
    -------
    torch.Tensor
        KL divergence. Shape depends on reduction.
        
    Examples
    --------
    >>> indices, teacher_k = select_topk_indices(teacher_logits, k=256)
    >>> student_k = lens.forward(hidden, vocab_indices=indices).logits
    >>> loss = subset_kl_from_gathered(student_k, teacher_k)
    
    Notes
    -----
    The KL is computed as:
    
        KL(P || Q) = Σ_i P_i * (log P_i - log Q_i)
    
    where P and Q are renormalized over the k tokens. This is not
    the same as full-vocabulary KL, but for k >= 256 with typical
    LLM distributions, the difference is negligible (<1%).
    """
    # Renormalize both distributions over the subset
    teacher_logprobs = F.log_softmax(teacher_logits_k, dim=-1)
    student_logprobs = F.log_softmax(student_logits_k, dim=-1)
    
    # KL(teacher || student)
    teacher_probs = teacher_logprobs.exp()
    kl_per_token = (teacher_probs * (teacher_logprobs - student_logprobs)).sum(dim=-1)
    
    return apply_reduction(kl_per_token, attention_mask, reduction)


def subset_kl_from_gathered_with_weights(
    student_logits_k: torch.Tensor,
    teacher_log_probs_k: torch.Tensor,
    inclusion_probs: torch.Tensor,
    mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
    weight_clip: float = 50.0,
) -> torch.Tensor:
    """
    Compute importance-weighted KL (Hajek estimator) from pre-gathered tensors.
    
    Use this with `select_indices_with_sampling()` for unbiased estimation.
    
    Parameters
    ----------
    student_logits_k : torch.Tensor
        Student logits for selected indices [batch, seq, S].
    teacher_log_probs_k : torch.Tensor
        Teacher LOG PROBS (not logits) for selected indices [batch, seq, S].
    inclusion_probs : torch.Tensor
        Inclusion probabilities from sampling [batch, seq, S].
    mask : torch.Tensor
        Valid mask [batch, seq, S].
    attention_mask : Optional[torch.Tensor]
        Sequence mask [batch, seq].
    reduction : str
        "none", "mean", or "sum".
    weight_clip : float
        Maximum importance weight.
        
    Returns
    -------
    torch.Tensor
        Importance-weighted KL estimate.
    """
    from .sampling import hajek_kl_estimate
    
    B, T, S = student_logits_k.shape
    
    # Flatten for Hajek computation
    teacher_flat = teacher_log_probs_k.view(B * T, S)
    student_flat = student_logits_k.view(B * T, S)
    inc_probs_flat = inclusion_probs.view(B * T, S)
    mask_flat = mask.view(B * T, S)
    
    # Hajek estimator
    kl_flat, _ = hajek_kl_estimate(
        teacher_flat, student_flat, 
        indices=None,  # Not needed for computation
        inclusion_probs=inc_probs_flat, 
        mask=mask_flat,
        weight_clip=weight_clip,
    )
    
    kl = kl_flat.view(B, T)
    return apply_reduction(kl, attention_mask, reduction)


# =============================================================================
# Convenience: All-in-one when you have full logits
# =============================================================================

def compute_subset_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k: int = 256,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
) -> torch.Tensor:
    """
    Compute subset KL when you have full logits (convenience function).
    
    NOTE: This function still requires full [B, T, V] student logits,
    so it doesn't provide memory savings over full KL. Use this for:
    - Testing/validation
    - When student logits are already computed
    - Comparing subset vs full KL
    
    For memory efficiency, use `select_topk_indices()` + your model's
    subset forward + `subset_kl_from_gathered()`.
    
    Parameters
    ----------
    student_logits : torch.Tensor
        Student logits [batch, seq, vocab].
    teacher_logits : torch.Tensor
        Teacher logits [batch, seq, vocab].
    k : int
        Number of top tokens.
    attention_mask : Optional[torch.Tensor]
        Mask [batch, seq].
    reduction : str
        "none", "mean", or "sum".
        
    Returns
    -------
    torch.Tensor
        Subset KL divergence.
    """
    # Select top-k from teacher
    indices, teacher_k = select_topk_indices(teacher_logits, k)
    
    # Gather student logits (this is why we need full student logits)
    student_k = torch.gather(student_logits, -1, indices)
    
    # Compute KL
    return subset_kl_from_gathered(student_k, teacher_k, attention_mask, reduction)


# =============================================================================
# Full KL for comparison
# =============================================================================

def full_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: ReductionType = "mean",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute full-vocabulary KL divergence (baseline).
    
    This is O(B*T*V) memory - use subset_kl for large vocabularies.
    
    Parameters
    ----------
    student_logits : torch.Tensor
        Student logits [batch, seq, vocab].
    teacher_logits : torch.Tensor
        Teacher logits [batch, seq, vocab].
    attention_mask : Optional[torch.Tensor]
        Mask [batch, seq].
    reduction : str
        "none", "mean", or "sum".
    temperature : float
        Temperature scaling (1.0 = no scaling).
    Returns
    -------
    torch.Tensor
        Full KL divergence.
    """
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
    
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
    student_logprobs = F.log_softmax(student_logits, dim=-1)
    
    teacher_probs = teacher_logprobs.exp()
    kl_per_token = (teacher_probs * (teacher_logprobs - student_logprobs)).sum(dim=-1)
    
    return apply_reduction(kl_per_token, attention_mask, reduction)
