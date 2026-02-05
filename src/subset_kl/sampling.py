# src/subset_kl/sampling.py
"""
Sampling utilities for subset KL divergence estimation.

Implements Probability Proportional to Size (PPS) sampling and
Hajek (self-normalized) importance sampling estimators for
unbiased estimation of full-vocabulary KL divergence.

For most use cases, the simple top-k approach via
`select_topk_indices()` is sufficient and more stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingDiagnostics:
    """Diagnostics from PPS sampling for monitoring estimator quality."""
    num_unique_indices: int
    mean_inclusion_prob: float
    min_inclusion_prob: float
    max_importance_weight: float
    variance_proxy: Optional[float] = None
    
    def __repr__(self) -> str:
        return (
            f"SamplingDiagnostics(unique={self.num_unique_indices}, "
            f"mean_π={self.mean_inclusion_prob:.4f}, "
            f"min_π={self.min_inclusion_prob:.6f}, "
            f"max_w={self.max_importance_weight:.1f})"
        )


def pps_sample_indices_batched(
    log_probs: torch.Tensor,
    k_head: int,
    k_tail: int,
    oversample: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SamplingDiagnostics]:
    """
    Probability Proportional to Size sampling with deterministic head.
    
    Combines top-k (deterministic head) with PPS sampling (stochastic tail)
    for a hybrid subset that captures both high-probability tokens and
    provides unbiased coverage of the tail distribution.
    
    Parameters
    ----------
    log_probs : torch.Tensor
        Log probabilities [N, V] where N = B*T (flattened positions).
    k_head : int
        Number of top tokens to include deterministically. Set to 0
        for pure importance sampling (no deterministic head).
    k_tail : int
        Number of additional tokens to sample from tail.
    oversample : int
        Oversample factor for PPS (draw oversample*k_tail then deduplicate).
        
    Returns
    -------
    indices : torch.Tensor
        Selected indices [N, S] where S <= k_head + k_tail.
    inclusion_probs : torch.Tensor
        Inclusion probabilities for each selected index [N, S].
    mask : torch.Tensor
        Valid mask for variable-length selections [N, S].
    diagnostics : SamplingDiagnostics
        Monitoring statistics.
    """
    N, V = log_probs.shape
    device = log_probs.device

    if k_head < 0 or k_tail < 0:
        raise ValueError("k_head and k_tail must be non-negative")
    if k_head + k_tail == 0:
        raise ValueError("k_head + k_tail must be > 0")
    
    # Ensure numerical stability
    log_probs = log_probs.float()
    probs = F.softmax(log_probs, dim=-1)
    
    # HEAD: Top-k deterministic indices
    if k_head > 0:
        _, top_idx = log_probs.topk(k_head, dim=-1)  # [N, k_head]
    else:
        top_idx = torch.empty(N, 0, device=device, dtype=torch.long)
    
    # Create mask for tail (exclude head indices)
    head_mask = torch.zeros(N, V, device=device, dtype=torch.bool)
    if k_head > 0:
        head_mask.scatter_(1, top_idx, True)
    
    # Renormalize probabilities over tail
    tail_probs = probs.clone()
    tail_probs[head_mask] = 0.0
    tail_sum = tail_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    tail_probs = tail_probs / tail_sum
    
    # PPS sampling from tail with replacement
    num_draws = k_tail * oversample
    if num_draws > 0 and (tail_sum > 1e-10).any():
        # Sample indices proportional to tail_probs
        sampled = torch.multinomial(
            tail_probs,
            num_samples=min(num_draws, V - k_head),
            replacement=True
        )  # [N, num_draws]
        
        # Combine head and tail
        all_idx = torch.cat([top_idx, sampled], dim=-1)
        
        # Deduplicate via sort + unique detection
        all_idx_sorted, _ = all_idx.sort(dim=-1)
        diff = torch.diff(all_idx_sorted, dim=-1, prepend=all_idx_sorted[:, :1] - 1)
        keep_mask = diff != 0
        
        # Count unique per row
        counts = keep_mask.sum(dim=-1)
        max_unique = counts.max().item()
        
        # Extract unique indices with padding
        S_max = min(k_head + k_tail, max_unique)
        indices = torch.zeros(N, S_max, device=device, dtype=torch.long)
        mask = torch.zeros(N, S_max, device=device, dtype=torch.bool)
        
        for i in range(N):
            unique_i = all_idx_sorted[i][keep_mask[i]]
            n_i = min(len(unique_i), S_max)
            indices[i, :n_i] = unique_i[:n_i]
            mask[i, :n_i] = True
    else:
        # No tail sampling
        indices = top_idx
        mask = torch.ones(N, k_head, device=device, dtype=torch.bool)
    
    # Compute inclusion probabilities
    inclusion_probs = torch.ones_like(indices, dtype=torch.float32)
    
    if k_tail > 0 and num_draws > 0:
        # Gather probabilities for selected indices
        p_sel = torch.gather(probs, 1, indices)
        
        # Identify which are head (prob >= top-k threshold)
        if k_head > 0:
            p_head_min = torch.gather(probs, 1, top_idx[:, -1:])
            is_head = p_sel >= p_head_min
        else:
            is_head = torch.zeros_like(p_sel, dtype=torch.bool)
        
        # For tail items: π_i = 1 - (1-p_i)^m
        tail_inclusion = 1.0 - torch.pow(1.0 - p_sel, num_draws)
        inclusion_probs = torch.where(is_head, torch.ones_like(p_sel), tail_inclusion)
        inclusion_probs = inclusion_probs.clamp_min(1e-8)
    
    # Compute diagnostics
    valid_inclusion = inclusion_probs[mask]
    diagnostics = SamplingDiagnostics(
        num_unique_indices=int(mask.sum().item() / N) if N > 0 else 0,
        mean_inclusion_prob=valid_inclusion.mean().item() if valid_inclusion.numel() > 0 else 1.0,
        min_inclusion_prob=valid_inclusion.min().item() if valid_inclusion.numel() > 0 else 1.0,
        max_importance_weight=(1.0 / valid_inclusion.min()).item() if valid_inclusion.numel() > 0 else 1.0,
    )
    
    return indices, inclusion_probs, mask, diagnostics


def hajek_kl_estimate(
    teacher_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    indices: Optional[torch.Tensor],
    inclusion_probs: torch.Tensor,
    mask: torch.Tensor,
    weight_clip: float = 50.0,
) -> Tuple[torch.Tensor, float]:
    """
    Hajek (self-normalized) importance sampling estimator for KL divergence.
    
    Provides an approximately unbiased estimate of the full-vocabulary
    KL divergence using only a subset of vocabulary indices.
    
    KL(P || Q) ≈ Σ_i∈S w_i * P_i * (log P_i - log Q_i) / Σ_i∈S w_i * P_i
    
    where w_i = 1/π_i are importance weights.
    
    Parameters
    ----------
    teacher_log_probs : torch.Tensor
        Teacher log probabilities [N, S] for selected indices.
    student_logits : torch.Tensor
        Student logits [N, S] for selected indices.
    indices : Optional[torch.Tensor]
        Selected vocabulary indices [N, S]. Not used in computation,
        kept for API consistency.
    inclusion_probs : torch.Tensor
        Inclusion probabilities π_i for each index [N, S].
    mask : torch.Tensor
        Valid mask [N, S].
    weight_clip : float
        Maximum importance weight to prevent variance explosion.
        
    Returns
    -------
    kl : torch.Tensor
        KL divergence estimate [N].
    variance_proxy : float
        Proxy for estimator variance (for monitoring).
    """
    # Importance weights: w_i = 1/π_i
    weights = (1.0 / inclusion_probs).clamp(max=weight_clip)
    weights = weights * mask.float()
    
    # Teacher probabilities
    teacher_probs = teacher_log_probs.exp()
    
    # Student log-probabilities via log-softmax over subset
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    
    # Weighted KL terms
    kl_terms = teacher_probs * (teacher_log_probs - student_log_probs) * weights
    
    # Self-normalize (Hajek estimator)
    numerator = (kl_terms * mask.float()).sum(dim=-1)
    denominator = (teacher_probs * weights * mask.float()).sum(dim=-1).clamp_min(1e-8)
    
    kl = numerator / denominator
    
    # Variance proxy
    w_normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    variance_proxy = (w_normalized ** 2 * mask.float()).sum().item() / max(mask.shape[0], 1)
    
    return kl, variance_proxy
