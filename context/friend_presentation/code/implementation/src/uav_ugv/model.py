"""Queue-aware cross-attention over physical UGVs and virtual slots."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional

from .sinkhorn import gumbel_sinkhorn, straight_through_hungarian


class QueueAwareCrossAttention(nn.Module):
    """Score every UAV request against every virtual UGV service slot.

    Inputs use shapes:

    - UAV features: `[batch, M, uav_feature_dim]`
    - UGV features: `[batch, N, ugv_feature_dim]`
    - normalized queue latency: `[batch, M, N, K]`

    The output is flattened to `[batch, M, N * K]` for rectangular matching.
    """

    def __init__(
        self,
        uav_feature_dim: int,
        ugv_feature_dim: int,
        embedding_dim: int = 128,
        attention_dim: int = 64,
        max_slots_per_ugv: int = 8,
        initial_queue_weight: float = 1.0,
        learn_queue_weight: bool = True,
        zero_queue_penalty: bool = False,
    ) -> None:
        super().__init__()
        if min(uav_feature_dim, ugv_feature_dim, embedding_dim, attention_dim) < 1:
            raise ValueError("feature and embedding dimensions must be positive")
        if max_slots_per_ugv < 1:
            raise ValueError("max_slots_per_ugv must be positive")
        if initial_queue_weight <= 0:
            raise ValueError("initial_queue_weight must be positive")

        self.max_slots_per_ugv = max_slots_per_ugv
        self.zero_queue_penalty = zero_queue_penalty
        self.uav_encoder = nn.Sequential(
            nn.Linear(uav_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.ugv_encoder = nn.Sequential(
            nn.Linear(ugv_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.slot_embedding = nn.Embedding(max_slots_per_ugv, embedding_dim)
        self.query_projection = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.key_projection = nn.Linear(embedding_dim, attention_dim, bias=False)
        raw_gamma = math.log(math.expm1(initial_queue_weight))
        self.raw_queue_weight = nn.Parameter(
            torch.tensor(raw_gamma, dtype=torch.float32),
            requires_grad=learn_queue_weight and not zero_queue_penalty,
        )

    @property
    def queue_weight(self) -> torch.Tensor:
        if self.zero_queue_penalty:
            return torch.zeros_like(self.raw_queue_weight)
        return functional.softplus(self.raw_queue_weight)

    def forward(
        self,
        uav_features: torch.Tensor,
        ugv_features: torch.Tensor,
        normalized_queue_latency: torch.Tensor,
    ) -> torch.Tensor:
        if uav_features.ndim != 3 or ugv_features.ndim != 3:
            raise ValueError("UAV and UGV features must be rank-three tensors")
        batch, request_count, _ = uav_features.shape
        ugv_batch, ugv_count, _ = ugv_features.shape
        if batch != ugv_batch:
            raise ValueError("UAV and UGV batch dimensions do not match")
        expected = (batch, request_count, ugv_count, request_count)
        if tuple(normalized_queue_latency.shape) != expected:
            raise ValueError(
                "normalized_queue_latency must have shape "
                f"{expected}, got {tuple(normalized_queue_latency.shape)}"
            )
        if request_count > self.max_slots_per_ugv:
            raise ValueError(
                f"request count {request_count} exceeds configured slot capacity "
                f"{self.max_slots_per_ugv}"
            )

        uav_embedding = self.uav_encoder(uav_features)
        ugv_embedding = self.ugv_encoder(ugv_features)
        slot_indices = torch.arange(request_count, device=uav_features.device)
        slot_embedding = self.slot_embedding(slot_indices)
        expanded_slots = ugv_embedding[:, :, None, :] + slot_embedding[None, None, :, :]

        query = self.query_projection(uav_embedding)
        key = self.key_projection(expanded_slots)
        attention = torch.einsum("bmd,bnkd->bmnk", query, key)
        attention = attention / math.sqrt(query.shape[-1])
        scores = attention - self.queue_weight * normalized_queue_latency
        return scores.reshape(batch, request_count, ugv_count * request_count)

    def soft_assignment(
        self,
        uav_features: torch.Tensor,
        ugv_features: torch.Tensor,
        normalized_queue_latency: torch.Tensor,
        temperature: float = 0.2,
        sinkhorn_iterations: int = 20,
        add_gumbel_noise: bool = True,
        straight_through: bool = False,
    ) -> torch.Tensor:
        """Return a square matrix that includes dummy UAV rows."""
        logits = self(
            uav_features,
            ugv_features,
            normalized_queue_latency,
        )
        soft = gumbel_sinkhorn(
            logits,
            temperature=temperature,
            iterations=sinkhorn_iterations,
            add_noise=add_gumbel_noise,
        )
        if straight_through:
            return straight_through_hungarian(soft)
        return soft
