"""Differentiable objectives for virtual-slot assignment."""

from __future__ import annotations

import torch


def expected_assignment_cost(
    assignment_matrix: torch.Tensor,
    cost_matrix: torch.Tensor,
    real_request_count: int,
) -> torch.Tensor:
    """Mean cost of real UAV rows; ignore padded dummy UAV rows.

    `assignment_matrix` is square after Sinkhorn padding. `cost_matrix` may be
    either rectangular `[batch, M, slots]` or already square.
    """
    if assignment_matrix.ndim != 3 or cost_matrix.ndim != 3:
        raise ValueError("assignment and cost tensors must be rank three")
    if real_request_count < 1:
        raise ValueError("real_request_count must be positive")
    real_assignment = assignment_matrix[:, :real_request_count, :]
    real_cost = cost_matrix[:, :real_request_count, :]
    if real_assignment.shape != real_cost.shape:
        raise ValueError(
            f"shape mismatch: {tuple(real_assignment.shape)} versus "
            f"{tuple(real_cost.shape)}"
        )
    per_instance = (real_assignment * real_cost).sum(dim=(-2, -1))
    return per_instance.mean()


def reinforce_loss(
    log_probability: torch.Tensor,
    sampled_cost: torch.Tensor,
    baseline_cost: torch.Tensor,
) -> torch.Tensor:
    """REINFORCE objective for future closed-loop route-and-assignment training."""
    if not (
        log_probability.shape == sampled_cost.shape == baseline_cost.shape
    ):
        raise ValueError("REINFORCE tensors must have identical batch shapes")
    advantage = (sampled_cost - baseline_cost).detach()
    return (advantage * log_probability).mean()
