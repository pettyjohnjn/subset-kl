"""Experiment helpers for head/tail subset KL estimators."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import torch

from .core import TailProposalType, compute_subset_hajek_kl, compute_subset_mc_kl, full_kl

TailEstimator = Literal["mc", "hajek"]


def evaluate_tail_proposal_grid(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k_head: int,
    k_tail: int,
    estimator: TailEstimator = "hajek",
    alpha_values: Iterable[float] = (0.6, 0.8, 0.9, 1.0),
    tau_values: Iterable[float] = (0.5, 0.7, 0.9),
    sample_counts: Iterable[int] | None = None,
    num_trials: int = 32,
    base_seed: int = 0,
) -> list[dict[str, float | int | str]]:
    """
    Repeated-trial comparison for target, tempered, mixed, and top-k baselines.

    This convenience path requires full student logits and is intended for
    estimator diagnostics, not memory-efficient training.
    """
    if sample_counts is None:
        sample_counts = (k_tail, 2 * k_tail, 4 * k_tail)
    if estimator not in {"mc", "hajek"}:
        raise ValueError("estimator must be 'mc' or 'hajek'")
    if num_trials <= 0:
        raise ValueError("num_trials must be positive")

    exact = full_kl(student_logits, teacher_logits, reduction="none")
    topk = compute_subset_hajek_kl(
        student_logits,
        teacher_logits,
        k_head=k_head,
        k_tail=0,
        reduction="none",
    )
    rows: list[dict[str, float | int | str]] = [
        _summarize_trials(
            name="topk_only",
            estimates=topk.unsqueeze(0),
            exact=exact,
            ess_values=None,
            max_weights=None,
            weight_variances=None,
            extreme_frequencies=None,
            sample_count=0,
            alpha=float("nan"),
            tau=float("nan"),
            proposal="topk_only",
        )
    ]

    proposal_specs: list[tuple[str, TailProposalType, float, float]] = [
        ("target", "target", 1.0, 1.0),
    ]
    proposal_specs.extend(
        (f"tempered_tau{tau:g}", "tempered", 0.0, float(tau))
        for tau in tau_values
    )
    proposal_specs.extend(
        (f"mixed_alpha{alpha:g}_tau{tau:g}", "mixed", float(alpha), float(tau))
        for alpha in alpha_values
        for tau in tau_values
    )

    compute = compute_subset_hajek_kl if estimator == "hajek" else compute_subset_mc_kl
    for sample_count in sample_counts:
        for name, proposal, alpha, tau in proposal_specs:
            estimates = []
            ess_values = []
            max_weights = []
            weight_variances = []
            extreme_frequencies = []
            for trial in range(num_trials):
                generator = torch.Generator(device=teacher_logits.device)
                generator.manual_seed(base_seed + trial)
                estimate, diagnostics = compute(
                    student_logits,
                    teacher_logits,
                    k_head=k_head,
                    k_tail=int(sample_count),
                    tail_proposal=proposal,
                    tail_proposal_alpha=alpha,
                    tail_proposal_tau=tau,
                    reduction="none",
                    generator=generator,
                    return_diagnostics=True,
                )
                estimates.append(estimate)
                ess_values.append(diagnostics.get("tail_ess_mean", float(sample_count)))
                max_weights.append(diagnostics.get("tail_max_weight", 1.0))
                weight_variances.append(diagnostics.get("tail_weight_variance", 0.0))
                extreme_frequencies.append(
                    diagnostics.get("tail_extreme_weight_frequency", 0.0)
                )

            rows.append(
                _summarize_trials(
                    name=name,
                    estimates=torch.stack(estimates),
                    exact=exact,
                    ess_values=ess_values,
                    max_weights=max_weights,
                    weight_variances=weight_variances,
                    extreme_frequencies=extreme_frequencies,
                    sample_count=int(sample_count),
                    alpha=alpha,
                    tau=tau,
                    proposal=proposal,
                )
            )
    return rows


def _summarize_trials(
    name: str,
    estimates: torch.Tensor,
    exact: torch.Tensor,
    ess_values: list[float] | None,
    max_weights: list[float] | None,
    weight_variances: list[float] | None,
    extreme_frequencies: list[float] | None,
    sample_count: int,
    alpha: float,
    tau: float,
    proposal: str,
) -> dict[str, float | int | str]:
    estimates_f = estimates.float()
    mean_estimate = estimates_f.mean(dim=0)
    trial_means = estimates_f.reshape(estimates.shape[0], -1).mean(dim=-1)
    err = mean_estimate - exact.float()
    row: dict[str, float | int | str] = {
        "name": name,
        "proposal": proposal,
        "sample_count": sample_count,
        "alpha": alpha,
        "tau": tau,
        "num_trials": estimates.shape[0],
        "mean_estimate": mean_estimate.mean().item(),
        "empirical_variance": trial_means.var(unbiased=False).item(),
        "mse": err.square().mean().item(),
    }
    if ess_values is not None:
        ess = torch.tensor(ess_values, dtype=torch.float32)
        row["ess_mean"] = ess.mean().item()
        row["ess_min"] = ess.min().item()
        row["ess_max"] = ess.max().item()
    if max_weights is not None:
        row["max_weight_mean"] = torch.tensor(max_weights).float().mean().item()
    if weight_variances is not None:
        row["weight_variance_mean"] = torch.tensor(weight_variances).float().mean().item()
    if extreme_frequencies is not None:
        row["extreme_weight_frequency"] = torch.tensor(extreme_frequencies).float().mean().item()
    return row
