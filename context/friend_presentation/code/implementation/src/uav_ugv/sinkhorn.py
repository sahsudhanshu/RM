"""Numerically stable Sinkhorn and Hungarian utilities."""

from __future__ import annotations

import numpy as np
import torch


def pad_with_dummy_rows(logits: torch.Tensor, dummy_value: float = 0.0) -> torch.Tensor:
    """Pad `[batch, real_rows, slots]` logits to a square slot-by-slot matrix."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, rows, columns]")
    batch, rows, columns = logits.shape
    if rows > columns:
        raise ValueError("there must be at least as many virtual slots as UAV rows")
    if rows == columns:
        return logits
    dummy = torch.full(
        (batch, columns - rows, columns),
        fill_value=dummy_value,
        dtype=logits.dtype,
        device=logits.device,
    )
    return torch.cat((logits, dummy), dim=1)


def sample_gumbel_like(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    uniform = torch.rand_like(tensor).clamp_(eps, 1.0 - eps)
    return -torch.log(-torch.log(uniform))


def log_sinkhorn(
    log_alpha: torch.Tensor,
    iterations: int = 20,
) -> torch.Tensor:
    """Return log of an approximately doubly stochastic square matrix."""
    if log_alpha.ndim != 3:
        raise ValueError("log_alpha must have shape [batch, size, size]")
    if log_alpha.shape[-1] != log_alpha.shape[-2]:
        raise ValueError("Sinkhorn input must be square; pad dummy rows first")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    normalized = log_alpha
    for _ in range(iterations):
        normalized = normalized - torch.logsumexp(
            normalized, dim=-1, keepdim=True
        )
        normalized = normalized - torch.logsumexp(
            normalized, dim=-2, keepdim=True
        )
    return normalized


def gumbel_sinkhorn(
    logits: torch.Tensor,
    temperature: float = 0.2,
    iterations: int = 20,
    add_noise: bool = True,
) -> torch.Tensor:
    """Differentiable approximation of a permutation matrix."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    square = pad_with_dummy_rows(logits)
    if add_noise:
        square = square + sample_gumbel_like(square)
    return torch.exp(log_sinkhorn(square / temperature, iterations))


def rectangular_hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimize a rectangular cost matrix with the Hungarian algorithm.

    This shortest-augmenting-path implementation handles ``rows <= columns``
    in O(rows^2 * columns) time and avoids tying the core model to a particular
    SciPy build on deployment machines.
    """
    if cost.ndim != 2:
        raise ValueError("cost must be a two-dimensional matrix")
    row_count, column_count = cost.shape
    if row_count > column_count:
        raise ValueError("virtual-slot count must be at least the UAV count")
    if not np.isfinite(cost).all():
        raise ValueError("cost matrix contains non-finite values")
    if row_count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    row_potential = np.zeros(row_count + 1, dtype=np.float64)
    column_potential = np.zeros(column_count + 1, dtype=np.float64)
    matched_row = np.zeros(column_count + 1, dtype=np.int64)
    predecessor = np.zeros(column_count + 1, dtype=np.int64)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        current_column = 0
        minimum_slack = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        while True:
            used[current_column] = True
            current_row = matched_row[current_column]
            delta = np.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced_cost = (
                    cost[current_row - 1, column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced_cost < minimum_slack[column]:
                    minimum_slack[column] = reduced_cost
                    predecessor[column] = current_column
                if minimum_slack[column] < delta:
                    delta = minimum_slack[column]
                    next_column = column

            for column in range(column_count + 1):
                if used[column]:
                    row_potential[matched_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum_slack[column] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break

        while True:
            previous_column = predecessor[current_column]
            matched_row[current_column] = matched_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    row_to_column = np.empty(row_count, dtype=np.int64)
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            row_to_column[matched_row[column] - 1] = column - 1
    return np.arange(row_count, dtype=np.int64), row_to_column


def hard_permutation_from_soft(soft: torch.Tensor) -> torch.Tensor:
    """Project square soft assignments to permutation matrices on the CPU."""
    if soft.ndim != 3 or soft.shape[-1] != soft.shape[-2]:
        raise ValueError("soft must have shape [batch, size, size]")
    hard = torch.zeros_like(soft)
    for batch_index, matrix in enumerate(soft.detach().cpu().numpy()):
        rows, columns = rectangular_hungarian(-matrix)
        hard[batch_index, rows, columns] = 1.0
    return hard


def straight_through_hungarian(soft: torch.Tensor) -> torch.Tensor:
    """Use a hard forward assignment and soft Sinkhorn gradients backward."""
    hard = hard_permutation_from_soft(soft)
    return hard - soft.detach() + soft
