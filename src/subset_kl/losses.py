# src/subset_kl/losses.py
"""
Class-based loss interfaces for subset KL divergence.

These wrap the functional interfaces in `core.py` for use in training loops
where you want a stateful loss object.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseLoss, ReductionType
from .core import (
    select_topk_indices,
    subset_kl_from_gathered,
    full_kl,
)


class SubsetKLLoss(BaseLoss):
    """
    Memory-efficient subset KL loss (class interface).
    
    This class provides two usage patterns:
    
    **Pattern 1: Full student logits (convenience, no memory savings)**
    
        >>> loss_fn = SubsetKLLoss(k=256)
        >>> loss = loss_fn(student_logits, teacher_logits)  # [B,T,V] inputs
    
    **Pattern 2: Pre-gathered tensors (memory-efficient)**
    
        >>> loss_fn = SubsetKLLoss(k=256)
        >>> indices, teacher_k = loss_fn.select_indices(teacher_logits)
        >>> student_k = your_model.forward_subset(hidden, indices)  # You compute this!
        >>> loss = loss_fn.forward_gathered(student_k, teacher_k)
    
    Parameters
    ----------
    k : int
        Number of top tokens to use.
    reduction : str
        "none", "mean", or "sum".
        
    Examples
    --------
    Memory-efficient usage with a lens:
    
        >>> loss_fn = SubsetKLLoss(k=256)
        >>> 
        >>> # Step 1: Get indices from teacher
        >>> indices, teacher_k = loss_fn.select_indices(teacher_logits)
        >>> 
        >>> # Step 2: Compute student logits only for those indices
        >>> # (Use indexed_logits or your lens's vocab_indices.)
        >>> student_k = lens.forward(hidden, vocab_indices=indices).logits
        >>> 
        >>> # Step 3: Compute KL
        >>> loss = loss_fn.forward_gathered(student_k, teacher_k, attention_mask)
    """

    def __init__(
        self,
        k: int = 256,
        reduction: ReductionType = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.k = k
        
        # Cache last indices for debugging/inspection
        self._last_indices: Optional[torch.Tensor] = None

    def select_indices(
        self,
        teacher_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k indices from teacher distribution.
        
        Parameters
        ----------
        teacher_logits : torch.Tensor
            Teacher logits [batch, seq, vocab].
            
        Returns
        -------
        indices : torch.Tensor
            Top-k indices [batch, seq, k].
        teacher_logits_k : torch.Tensor
            Teacher logits for those indices [batch, seq, k].
        """
        indices, teacher_k = select_topk_indices(teacher_logits, self.k)
        self._last_indices = indices
        return indices, teacher_k

    def forward_gathered(
        self,
        student_logits_k: torch.Tensor,
        teacher_logits_k: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL from pre-gathered subset logits.
        
        Parameters
        ----------
        student_logits_k : torch.Tensor
            Student logits for subset [batch, seq, k].
        teacher_logits_k : torch.Tensor
            Teacher logits for same subset [batch, seq, k].
        attention_mask : Optional[torch.Tensor]
            Mask [batch, seq].
            
        Returns
        -------
        torch.Tensor
            Loss value.
        """
        return subset_kl_from_gathered(
            student_logits_k, teacher_logits_k, attention_mask, self.reduction
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute subset KL from full logits (convenience, no memory savings).
        
        This requires full [B, T, V] student logits. For efficiency, use
        `select_indices()` + your model's subset forward + `forward_gathered()`.
        
        Parameters
        ----------
        student_logits : torch.Tensor
            Student logits [batch, seq, vocab].
        teacher_logits : torch.Tensor
            Teacher logits [batch, seq, vocab].
        attention_mask : Optional[torch.Tensor]
            Mask [batch, seq].
        """
        indices, teacher_k = self.select_indices(teacher_logits)
        student_k = torch.gather(student_logits, -1, indices)
        return self.forward_gathered(student_k, teacher_k, attention_mask)

    @property
    def last_indices(self) -> Optional[torch.Tensor]:
        """Last selected indices (for debugging)."""
        return self._last_indices

    def __repr__(self) -> str:
        return f"SubsetKLLoss(k={self.k}, reduction={self.reduction!r})"


class KLDivergenceLoss(BaseLoss):
    """
    Full-vocabulary KL divergence loss (baseline).
    
    Use this for comparison with subset KL, or when memory isn't a concern.
    
    Parameters
    ----------
    reduction : str
        "none", "mean", or "sum".
    temperature : float
        Temperature for softmax (1.0 = no scaling).
    chunk_size : Optional[int]
        If set, compute KL in chunks to reduce peak memory.
    """

    def __init__(
        self,
        reduction: ReductionType = "mean",
        temperature: float = 1.0,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.chunk_size = chunk_size

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute full-vocabulary KL divergence."""
        if self.chunk_size is not None:
            return self._forward_chunked(student_logits, teacher_logits, attention_mask)
        
        return full_kl(
            student_logits, teacher_logits, attention_mask,
            self.reduction, self.temperature
        )

    def _forward_chunked(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Chunked computation for memory efficiency."""
        batch, seq, vocab = student_logits.shape
        chunk_size = self.chunk_size
        
        total_loss = torch.zeros((), device=student_logits.device, dtype=torch.float32)
        total_count = torch.zeros((), device=student_logits.device, dtype=torch.float32)
        
        for t0 in range(0, seq, chunk_size):
            t1 = min(t0 + chunk_size, seq)
            
            s_chunk = student_logits[:, t0:t1, :]
            t_chunk = teacher_logits[:, t0:t1, :]
            m_chunk = attention_mask[:, t0:t1] if attention_mask is not None else None
            
            # Apply temperature
            if self.temperature != 1.0:
                s_chunk = s_chunk / self.temperature
                t_chunk = t_chunk / self.temperature
            
            s_logprobs = F.log_softmax(s_chunk, dim=-1)
            t_logprobs = F.log_softmax(t_chunk, dim=-1)
            t_probs = t_logprobs.exp()
            
            kl_chunk = (t_probs * (t_logprobs - s_logprobs)).sum(dim=-1)
            
            if m_chunk is not None:
                kl_chunk = kl_chunk * m_chunk.to(kl_chunk.dtype)
                total_count += m_chunk.sum()
            else:
                total_count += kl_chunk.numel()
            
            total_loss += kl_chunk.sum()
        
        if self.reduction == "sum":
            return total_loss
        elif self.reduction == "mean":
            return total_loss / total_count.clamp_min(1.0)
        else:
            raise ValueError("Chunked KL only supports 'mean' or 'sum' reduction")

    def __repr__(self) -> str:
        parts = [f"reduction={self.reduction!r}"]
        if self.temperature != 1.0:
            parts.append(f"temperature={self.temperature}")
        if self.chunk_size is not None:
            parts.append(f"chunk_size={self.chunk_size}")
        return f"KLDivergenceLoss({', '.join(parts)})"


class HajekKLLoss(BaseLoss):
    """
    Importance-weighted KL loss using Hajek estimator.
    
    This provides an unbiased estimator but has higher variance.
    For most cases, `SubsetKLLoss` (pure top-k) is preferred.
    Set k_head=0 for pure importance sampling (no deterministic head).
    
    Parameters
    ----------
    k_head : int
        Number of top tokens (deterministic).
    k_tail : int
        Number of additional sampled tokens (stochastic).
    reduction : str
        "none", "mean", or "sum".
    weight_clip : float
        Maximum importance weight.
    oversample : int
        Oversample factor for PPS sampling.
    """

    def __init__(
        self,
        k_head: int = 128,
        k_tail: int = 64,
        reduction: ReductionType = "mean",
        weight_clip: float = 50.0,
        oversample: int = 4,
    ) -> None:
        super().__init__(reduction=reduction)
        self.k_head = k_head
        self.k_tail = k_tail
        self.weight_clip = weight_clip
        self.oversample = oversample
        
        self._last_variance_proxy: Optional[float] = None
        self._last_diagnostics = None

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute importance-weighted KL."""
        from .sampling import pps_sample_indices_batched, hajek_kl_estimate
        
        B, T, V = student_logits.shape
        
        # Flatten for sampling
        teacher_flat = teacher_logits.view(B * T, V)
        student_flat = student_logits.view(B * T, V)
        
        with torch.no_grad():
            teacher_log_probs = F.log_softmax(teacher_flat, dim=-1)
            
            indices, inc_probs, mask, diagnostics = pps_sample_indices_batched(
                teacher_log_probs,
                k_head=self.k_head,
                k_tail=self.k_tail,
                oversample=self.oversample,
            )
            self._last_diagnostics = diagnostics
        
        # Gather values
        teacher_sel = torch.gather(teacher_log_probs, -1, indices)
        student_sel = torch.gather(student_flat, -1, indices)
        
        # Hajek estimator
        kl_flat, variance_proxy = hajek_kl_estimate(
            teacher_sel, student_sel, indices, inc_probs, mask, self.weight_clip
        )
        self._last_variance_proxy = variance_proxy
        
        kl = kl_flat.view(B, T)
        return self._apply_reduction(kl, attention_mask)

    @property
    def variance_proxy(self) -> Optional[float]:
        """Variance proxy from last computation."""
        return self._last_variance_proxy

    @property
    def diagnostics(self):
        """Sampling diagnostics from last computation."""
        return self._last_diagnostics

    def __repr__(self) -> str:
        return (
            f"HajekKLLoss(k_head={self.k_head}, k_tail={self.k_tail}, "
            f"reduction={self.reduction!r})"
        )


class ImportanceKLLoss(HajekKLLoss):
    """
    Pure importance-sampling KL loss (no deterministic head).
    
    This is a convenience wrapper around HajekKLLoss with k_head=0.
    """

    def __init__(
        self,
        k: int = 128,
        reduction: ReductionType = "mean",
        weight_clip: float = 50.0,
        oversample: int = 4,
    ) -> None:
        super().__init__(
            k_head=0,
            k_tail=k,
            reduction=reduction,
            weight_clip=weight_clip,
            oversample=oversample,
        )
        self.k = k

    def __repr__(self) -> str:
        return f"ImportanceKLLoss(k={self.k}, reduction={self.reduction!r})"
