"""Discrete UAV-to-UGV assignment policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence

import torch

from .domain import Assignment, RechargeRequest, UGVState
from .features import FeatureScales, build_model_inputs
from .model import QueueAwareCrossAttention
from .road_network import RoadNetwork
from .sinkhorn import rectangular_hungarian
from .virtual_slots import build_slot_matrices


class AssignmentPolicy(ABC):
    """Interface shared by deterministic and learned assignment policies."""

    name = "abstract"

    @abstractmethod
    def assign(
        self,
        requests: Sequence[RechargeRequest],
        ugvs: Sequence[UGVState],
        road_network: RoadNetwork,
        ugv_speed: float,
        recharge_time: float,
    ) -> list[Assignment]:
        raise NotImplementedError


def _independent_nearest_assignments(
    requests: Sequence[RechargeRequest],
    ugvs: Sequence[UGVState],
    distance,
) -> list[Assignment]:
    grouped: dict[int, list[tuple[RechargeRequest, float]]] = defaultdict(list)
    for request in requests:
        ugv, value = min(
            ((ugv, float(distance(ugv, request))) for ugv in ugvs),
            key=lambda pair: (pair[1], pair[0].ugv_id),
        )
        grouped[ugv.ugv_id].append((request, value))

    assignments: list[Assignment] = []
    for ugv_id, items in grouped.items():
        items.sort(key=lambda pair: (pair[0].touchdown_time, pair[0].uav_id))
        for slot_position, (request, value) in enumerate(items, start=1):
            assignments.append(
                Assignment(
                    uav_id=request.uav_id,
                    ugv_id=ugv_id,
                    slot_position=slot_position,
                    objective_cost=value,
                )
            )
    return sorted(assignments, key=lambda item: item.uav_id)


class NearestEuclideanPolicy(AssignmentPolicy):
    """Base-paper baseline: independently choose the spatially nearest UGV."""

    name = "nearest_euclidean"

    def assign(
        self,
        requests: Sequence[RechargeRequest],
        ugvs: Sequence[UGVState],
        road_network: RoadNetwork,
        ugv_speed: float,
        recharge_time: float,
    ) -> list[Assignment]:
        del ugv_speed, recharge_time
        return _independent_nearest_assignments(
            requests,
            ugvs,
            lambda ugv, request: road_network.euclidean_distance(
                ugv.current_node, request.landing_node
            ),
        )


class NearestRoadPolicy(AssignmentPolicy):
    """Assign independently using shortest road distance but ignore queues."""

    name = "nearest_road"

    def assign(
        self,
        requests: Sequence[RechargeRequest],
        ugvs: Sequence[UGVState],
        road_network: RoadNetwork,
        ugv_speed: float,
        recharge_time: float,
    ) -> list[Assignment]:
        del ugv_speed, recharge_time
        return _independent_nearest_assignments(
            requests,
            ugvs,
            lambda ugv, request: road_network.road_distance(
                ugv.current_node, request.landing_node
            ),
        )


class QueueAwareHungarianPolicy(AssignmentPolicy):
    """Minimize predicted service completion over virtual charging slots."""

    name = "queue_aware_hungarian"

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
            requests=requests,
            ugvs=ugvs,
            road_network=road_network,
            ugv_speed=ugv_speed,
            recharge_time=recharge_time,
        )
        rows, columns = rectangular_hungarian(matrices.predicted_completion)
        raw = [
            Assignment(
                uav_id=requests[row].uav_id,
                ugv_id=matrices.slots[column].ugv_id,
                slot_position=matrices.slots[column].position,
                objective_cost=float(matrices.predicted_completion[row, column]),
            )
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True)
        ]
        return compact_slot_positions(raw, requests)


def compact_slot_positions(
    assignments: Sequence[Assignment],
    requests: Sequence[RechargeRequest],
) -> list[Assignment]:
    """Remove harmless Hungarian slot gaps without changing parent UGVs."""
    touchdown = {request.uav_id: request.touchdown_time for request in requests}
    grouped: dict[int, list[Assignment]] = defaultdict(list)
    for assignment in assignments:
        grouped[assignment.ugv_id].append(assignment)

    compacted: list[Assignment] = []
    for ugv_id, group in grouped.items():
        group.sort(
            key=lambda item: (
                item.slot_position,
                touchdown[item.uav_id],
                item.uav_id,
            )
        )
        for position, item in enumerate(group, start=1):
            compacted.append(
                Assignment(
                    uav_id=item.uav_id,
                    ugv_id=ugv_id,
                    slot_position=position,
                    objective_cost=item.objective_cost,
                )
            )
    return sorted(compacted, key=lambda item: item.uav_id)


class LearnedHungarianPolicy(AssignmentPolicy):
    """Run a trained cross-attention head, then discretize with Hungarian."""

    name = "learned_queue_aware_hungarian"

    def __init__(
        self,
        model: QueueAwareCrossAttention,
        feature_scales: FeatureScales = FeatureScales(),
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.feature_scales = feature_scales
        self.device = torch.device(device)

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
            requests=requests,
            ugvs=ugvs,
            road_network=road_network,
            ugv_speed=ugv_speed,
            recharge_time=recharge_time,
        )
        inputs = build_model_inputs(
            requests=requests,
            ugvs=ugvs,
            road_network=road_network,
            slot_matrices=matrices,
            recharge_time=recharge_time,
            scales=self.feature_scales,
            device=self.device,
        )
        self.model.eval()
        with torch.no_grad():
            scores = self.model(
                inputs.uav_features,
                inputs.ugv_features,
                inputs.normalized_queue_latency,
            )[0]
        rows, columns = rectangular_hungarian(-scores.cpu().numpy())
        raw = [
            Assignment(
                uav_id=requests[row].uav_id,
                ugv_id=matrices.slots[column].ugv_id,
                slot_position=matrices.slots[column].position,
                objective_cost=float(-scores[row, column].item()),
            )
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True)
        ]
        return compact_slot_positions(raw, requests)
