"""Convert physical assignment state into normalized neural-network features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .domain import RechargeRequest, UGVState
from .road_network import RoadNetwork
from .virtual_slots import SlotMatrices


@dataclass(frozen=True, slots=True)
class FeatureScales:
    """Fixed normalization constants shared by training and evaluation."""

    coordinate: float = 20.0
    time: float = 500.0
    sortie_index: float = 20.0
    completed_services: float = 20.0

    def __post_init__(self) -> None:
        if min(
            self.coordinate,
            self.time,
            self.sortie_index,
            self.completed_services,
        ) <= 0:
            raise ValueError("all feature scales must be positive")


@dataclass(frozen=True, slots=True)
class ModelInputs:
    uav_features: torch.Tensor
    ugv_features: torch.Tensor
    normalized_queue_latency: torch.Tensor


def build_model_inputs(
    requests: Sequence[RechargeRequest],
    ugvs: Sequence[UGVState],
    road_network: RoadNetwork,
    slot_matrices: SlotMatrices,
    recharge_time: float,
    scales: FeatureScales = FeatureScales(),
    device: torch.device | str = "cpu",
) -> ModelInputs:
    """Build one batch item using the feature order documented below.

    UAV features: ``x, y, touchdown time, remaining energy, sortie index``.
    UGV features: ``x, y, ready time, completed service count``.
    """
    if not requests:
        raise ValueError("at least one recharge request is required")
    if not ugvs:
        raise ValueError("at least one UGV is required")
    if recharge_time <= 0:
        raise ValueError("recharge_time must be positive for queue normalization")
    expected_slots = len(requests) * len(ugvs)
    if len(slot_matrices.slots) != expected_slots:
        raise ValueError(
            "slot matrices must contain one slot per request for every UGV"
        )

    uav_rows: list[list[float]] = []
    for request in requests:
        x, y = road_network.coordinates(request.landing_node)
        uav_rows.append(
            [
                x / scales.coordinate,
                y / scales.coordinate,
                request.touchdown_time / scales.time,
                request.remaining_energy_fraction,
                request.sortie_index / scales.sortie_index,
            ]
        )

    ugv_rows: list[list[float]] = []
    for ugv in ugvs:
        x, y = road_network.coordinates(ugv.current_node)
        ugv_rows.append(
            [
                x / scales.coordinate,
                y / scales.coordinate,
                ugv.ready_time / scales.time,
                ugv.completed_services / scales.completed_services,
            ]
        )

    request_count = len(requests)
    ugv_count = len(ugvs)
    queue = slot_matrices.predicted_wait.reshape(
        request_count,
        ugv_count,
        request_count,
    ) / recharge_time
    return ModelInputs(
        uav_features=torch.as_tensor(
            np.asarray(uav_rows), dtype=torch.float32, device=device
        ).unsqueeze(0),
        ugv_features=torch.as_tensor(
            np.asarray(ugv_rows), dtype=torch.float32, device=device
        ).unsqueeze(0),
        normalized_queue_latency=torch.as_tensor(
            queue, dtype=torch.float32, device=device
        ).unsqueeze(0),
    )
