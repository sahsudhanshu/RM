"""Portable trainer for assignment-head pretraining on fixed-size batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..model import QueueAwareCrossAttention
from .losses import expected_assignment_cost


@dataclass(frozen=True, slots=True)
class AssignmentBatch:
    """A same-size batch prepared by an external dataset generator."""

    uav_features: torch.Tensor
    ugv_features: torch.Tensor
    normalized_queue_latency: torch.Tensor
    normalized_assignment_cost: torch.Tensor

    @property
    def request_count(self) -> int:
        return int(self.uav_features.shape[1])

    def to(self, device: torch.device | str) -> "AssignmentBatch":
        return AssignmentBatch(
            uav_features=self.uav_features.to(device),
            ugv_features=self.ugv_features.to(device),
            normalized_queue_latency=self.normalized_queue_latency.to(device),
            normalized_assignment_cost=self.normalized_assignment_cost.to(device),
        )


@dataclass(frozen=True, slots=True)
class TrainerSettings:
    learning_rate: float = 1e-4
    temperature: float = 0.2
    sinkhorn_iterations: int = 20
    gradient_clip_norm: float = 1.0
    straight_through: bool = True


class AssignmentTrainer:
    """Train only the virtual-slot head before full policy-gradient coupling.

    This trainer minimizes differentiable expected slot cost using the
    Gumbel-Sinkhorn relaxation. Full mission REINFORCE belongs in the later
    route-environment integration and is intentionally not faked here.
    """

    def __init__(
        self,
        model: QueueAwareCrossAttention,
        settings: TrainerSettings = TrainerSettings(),
        device: torch.device | str | None = None,
    ) -> None:
        self.settings = settings
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=settings.learning_rate
        )

    def train_step(self, batch: AssignmentBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        assignment = self.model.soft_assignment(
            batch.uav_features,
            batch.ugv_features,
            batch.normalized_queue_latency,
            temperature=self.settings.temperature,
            sinkhorn_iterations=self.settings.sinkhorn_iterations,
            add_gumbel_noise=True,
            straight_through=self.settings.straight_through,
        )
        loss = expected_assignment_cost(
            assignment,
            batch.normalized_assignment_cost,
            real_request_count=batch.request_count,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.settings.gradient_clip_norm
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(torch.as_tensor(gradient_norm).cpu()),
            "queue_weight": float(self.model.queue_weight.detach().cpu()),
        }

    def evaluation_step(self, batch: AssignmentBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        self.model.eval()
        with torch.no_grad():
            assignment = self.model.soft_assignment(
                batch.uav_features,
                batch.ugv_features,
                batch.normalized_queue_latency,
                temperature=self.settings.temperature,
                sinkhorn_iterations=self.settings.sinkhorn_iterations,
                add_gumbel_noise=False,
                straight_through=False,
            )
            loss = expected_assignment_cost(
                assignment,
                batch.normalized_assignment_cost,
                real_request_count=batch.request_count,
            )
        return {"loss": float(loss.cpu())}

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "settings": asdict(self.settings),
            },
            destination,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return int(checkpoint["epoch"])
