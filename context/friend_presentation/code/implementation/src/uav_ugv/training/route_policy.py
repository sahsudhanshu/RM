"""Trainable, fuel-masked sortie routing on the shared mission simulator."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from ..mission import MissionScenario, TaskPoint, UAVMissionState


class RoutingTransformer(nn.Module):
    """Three-layer, eight-head encoder and state-conditioned attention decoder."""

    def __init__(self, embedding_dim: int = 128, heads: int = 8, layers: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.node_projection = nn.Linear(3, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=4 * embedding_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.state_projection = nn.Sequential(
            nn.Linear(6, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.glimpse = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=heads, batch_first=True
        )
        self.query_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.ugv_state_projection = nn.Sequential(
            nn.Linear(4, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.recharge_projection = nn.Sequential(
            nn.Linear(5, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)
        )

    def encode(self, scenario: MissionScenario, device: torch.device) -> torch.Tensor:
        rows = [
            (task.x / scenario.width_km, task.y / scenario.height_km,
             float(task.kind.value == "ground"))
            for task in scenario.tasks
        ]
        nodes = torch.tensor(rows, dtype=torch.float32, device=device).unsqueeze(0)
        return self.encoder(self.node_projection(nodes))

    def score(
        self,
        embeddings: torch.Tensor,
        scenario: MissionScenario,
        uav: UAVMissionState,
        visited: set[str],
        allowed_ids: set[str],
        fuel_capacity_j: float,
    ) -> torch.Tensor:
        state = torch.tensor(
            [uav.x / scenario.width_km, uav.y / scenario.height_km,
             uav.fuel_j / fuel_capacity_j, uav.elapsed_time / 500.0,
             uav.sorties_completed / 20.0, len(visited) / len(scenario.tasks)],
            dtype=torch.float32,
            device=embeddings.device,
        ).reshape(1, 1, 6)
        query = self.state_projection(state) + embeddings.mean(dim=1, keepdim=True)
        glimpse, _ = self.glimpse(query, embeddings, embeddings, need_weights=False)
        query = self.query_projection(glimpse)
        keys = self.key_projection(embeddings)
        logits = torch.bmm(query, keys.transpose(1, 2)).flatten()
        logits = 10.0 * torch.tanh(logits / self.embedding_dim**0.5)
        mask = torch.tensor(
            [task.task_id in allowed_ids for task in scenario.tasks],
            dtype=torch.bool,
            device=embeddings.device,
        )
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def score_service_order(
        self,
        ugv_features: torch.Tensor,
        request_features: torch.Tensor,
    ) -> torch.Tensor:
        query = self.ugv_state_projection(ugv_features)
        keys = self.recharge_projection(request_features)
        return 10.0 * torch.tanh(
            (keys * query.unsqueeze(0)).sum(dim=-1) / self.embedding_dim**0.5
        )


class LearnedRoutePolicy:
    """Episode wrapper with teacher, sampled, and greedy action selection."""

    def __init__(
        self,
        model: RoutingTransformer,
        device: torch.device | str,
        fuel_capacity_j: float = 287_700.0,
        mode: str = "sample",
    ) -> None:
        if mode not in {"teacher", "sample", "greedy"}:
            raise ValueError("mode must be teacher, sample, or greedy")
        self.model = model
        self.device = torch.device(device)
        self.fuel_capacity_j = fuel_capacity_j
        self.mode = mode
        self.log_probabilities: list[torch.Tensor] = []
        self.scenario: MissionScenario | None = None
        self.embeddings: torch.Tensor | None = None

    def start_episode(self, scenario: MissionScenario) -> None:
        self.scenario = scenario
        self.log_probabilities = []
        self.embeddings = self.model.encode(scenario, self.device)

    def choose_next(
        self,
        uav: UAVMissionState,
        visited: set[str],
        feasible_tasks: Sequence[TaskPoint],
        reachable_grounds: Sequence[TaskPoint],
        teacher_choice: TaskPoint,
    ) -> TaskPoint:
        if self.scenario is None or self.embeddings is None:
            raise RuntimeError("start_episode must be called before choosing actions")
        allowed = {task.task_id for task in feasible_tasks}
        allowed.update(task.task_id for task in reachable_grounds)
        logits = self.model.score(
            self.embeddings, self.scenario, uav, visited, allowed,
            self.fuel_capacity_j,
        )
        distribution = Categorical(logits=logits)
        if self.mode == "teacher":
            index = next(
                i for i, task in enumerate(self.scenario.tasks)
                if task.task_id == teacher_choice.task_id
            )
            selected = torch.tensor(index, device=self.device)
        elif self.mode == "sample":
            selected = distribution.sample()
        else:
            selected = logits.argmax()
        if self.mode != "greedy":
            self.log_probabilities.append(distribution.log_prob(selected))
        return self.scenario.tasks[int(selected.item())]

    def episode_log_probability(self) -> torch.Tensor:
        if not self.log_probabilities:
            return torch.zeros((), device=self.device)
        return torch.stack(self.log_probabilities).sum()
