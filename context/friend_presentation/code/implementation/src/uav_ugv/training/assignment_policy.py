"""A valid discrete assignment distribution for joint policy-gradient training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.distributions import Categorical

from ..domain import Assignment, RechargeRequest, UGVState
from ..features import build_model_inputs
from ..model import QueueAwareCrossAttention
from ..policies import AssignmentPolicy, NearestEuclideanPolicy, compact_slot_positions
from ..road_network import RoadNetwork
from ..sinkhorn import rectangular_hungarian
from ..virtual_slots import build_slot_matrices
from .route_policy import RoutingTransformer


class NearestWithLearnedOrderPolicy(AssignmentPolicy):
    """Fixed nearest physical UGV; learned serial service order per UGV."""

    name = "nearest_euclidean_learned_order"

    def __init__(
        self,
        route_model: RoutingTransformer,
        device: torch.device | str,
        mode: str = "sample",
    ) -> None:
        if mode not in {"teacher", "sample", "greedy"}:
            raise ValueError("mode must be teacher, sample, or greedy")
        self.route_model = route_model
        self.device = torch.device(device)
        self.mode = mode
        self.log_probabilities: list[torch.Tensor] = []
        self.nearest = NearestEuclideanPolicy()

    def start_episode(self) -> None:
        self.log_probabilities = []

    def assign(
        self,
        requests: Sequence[RechargeRequest],
        ugvs: Sequence[UGVState],
        road_network: RoadNetwork,
        ugv_speed: float,
        recharge_time: float,
    ) -> list[Assignment]:
        physical = self.nearest.assign(
            requests, ugvs, road_network, ugv_speed, recharge_time
        )
        assigned_ugv = {item.uav_id: item.ugv_id for item in physical}
        by_ugv: dict[int, list[RechargeRequest]] = {
            ugv.ugv_id: [] for ugv in ugvs
        }
        for request in requests:
            by_ugv[assigned_ugv[request.uav_id]].append(request)
        output: list[Assignment] = []
        for ugv in ugvs:
            pending = by_ugv[ugv.ugv_id]
            current_node = ugv.current_node
            ready_time = ugv.ready_time
            for position in range(1, len(pending) + 1):
                x, y = road_network.coordinates(current_node)
                state = torch.tensor(
                    [x / 20.0, y / 20.0, ready_time / 500.0,
                     (ugv.completed_services + position - 1) / 20.0],
                    device=self.device, dtype=torch.float32,
                )
                features = []
                for request in pending:
                    rx, ry = road_network.coordinates(request.landing_node)
                    features.append([
                        rx / 20.0, ry / 20.0,
                        request.touchdown_time / 500.0,
                        request.remaining_energy_fraction,
                        request.sortie_index / 20.0,
                    ])
                candidates = torch.tensor(
                    features, device=self.device, dtype=torch.float32
                )
                scores = self.route_model.score_service_order(state, candidates)
                distribution = Categorical(logits=scores)
                if self.mode == "teacher":
                    selected_index = min(
                        range(len(pending)),
                        key=lambda index: (
                            pending[index].touchdown_time, pending[index].uav_id
                        ),
                    )
                    selected = torch.tensor(selected_index, device=self.device)
                elif self.mode == "sample":
                    selected = distribution.sample()
                else:
                    selected = scores.argmax()
                if self.mode != "greedy":
                    self.log_probabilities.append(distribution.log_prob(selected))
                request = pending.pop(int(selected.item()))
                distance = road_network.road_distance(
                    current_node, request.landing_node
                )
                arrival = ready_time + distance / ugv_speed
                ready_time = max(arrival, request.touchdown_time) + recharge_time
                current_node = request.landing_node
                output.append(Assignment(
                    uav_id=request.uav_id,
                    ugv_id=ugv.ugv_id,
                    slot_position=position,
                    objective_cost=distance,
                ))
        return sorted(output, key=lambda item: item.uav_id)

    def episode_log_probability(self) -> torch.Tensor:
        if not self.log_probabilities:
            return torch.zeros((), device=self.device)
        return torch.stack(self.log_probabilities).sum()


class SampledAssignmentPolicy(AssignmentPolicy):
    """Sample slots without replacement; use rectangular Hungarian when greedy."""

    name = "learned_queue_aware_hungarian"

    def __init__(
        self,
        model: QueueAwareCrossAttention,
        device: torch.device | str,
        mode: str = "sample",
    ) -> None:
        if mode not in {"sample", "greedy"}:
            raise ValueError("mode must be sample or greedy")
        self.model = model
        self.device = torch.device(device)
        self.mode = mode
        self.log_probabilities: list[torch.Tensor] = []

    def start_episode(self) -> None:
        self.log_probabilities = []

    def assign(
        self,
        requests: Sequence[RechargeRequest],
        ugvs: Sequence[UGVState],
        road_network: RoadNetwork,
        ugv_speed: float,
        recharge_time: float,
    ) -> list[Assignment]:
        if not requests:
            return []
        matrices = build_slot_matrices(
            requests, ugvs, road_network, ugv_speed, recharge_time
        )
        inputs = build_model_inputs(
            requests, ugvs, road_network, matrices, recharge_time, device=self.device
        )
        scores = self.model(
            inputs.uav_features, inputs.ugv_features,
            inputs.normalized_queue_latency,
        )[0]
        if self.mode == "greedy":
            row_indices, columns = rectangular_hungarian(
                -scores.detach().cpu().numpy()
            )
            selected = {int(row): int(column) for row, column in zip(row_indices, columns)}
        else:
            selected: dict[int, int] = {}
            available = torch.ones(scores.shape[1], device=self.device, dtype=torch.bool)
            order = sorted(
                range(len(requests)),
                key=lambda row: (requests[row].touchdown_time, requests[row].uav_id),
            )
            for row in order:
                logits = scores[row].masked_fill(
                    ~available, torch.finfo(scores.dtype).min
                )
                distribution = Categorical(logits=logits)
                column = distribution.sample()
                self.log_probabilities.append(distribution.log_prob(column))
                selected[row] = int(column.item())
                available[selected[row]] = False
        raw = [
            Assignment(
                uav_id=request.uav_id,
                ugv_id=matrices.slots[selected[row]].ugv_id,
                slot_position=matrices.slots[selected[row]].position,
                objective_cost=float(-scores[row, selected[row]].detach().item()),
            )
            for row, request in enumerate(requests)
        ]
        return compact_slot_positions(raw, requests)

    def episode_log_probability(self) -> torch.Tensor:
        if not self.log_probabilities:
            return torch.zeros((), device=self.device)
        return torch.stack(self.log_probabilities).sum()
