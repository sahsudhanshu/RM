"""Virtual charging slots and queue-aware cost construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .domain import RechargeRequest, UGVState, VirtualSlot
from .road_network import RoadNetwork


@dataclass(frozen=True, slots=True)
class SlotMatrices:
    """Pairwise quantities aligned as [request, flattened virtual slot]."""

    slots: tuple[VirtualSlot, ...]
    road_distance: np.ndarray
    road_travel_time: np.ndarray
    predicted_wait: np.ndarray
    predicted_completion: np.ndarray


def build_virtual_slots(
    ugvs: Sequence[UGVState],
    request_count: int,
    slots_per_ugv: int | None = None,
) -> tuple[VirtualSlot, ...]:
    """Give every UGV enough sequential slots to accept every request."""
    if request_count < 0:
        raise ValueError("request_count must be non-negative")
    if not ugvs and request_count:
        raise ValueError("at least one UGV is required")
    capacity = request_count if slots_per_ugv is None else slots_per_ugv
    if capacity < request_count:
        raise ValueError(
            "slots_per_ugv must be at least request_count so one UGV can "
            "legally receive every request"
        )
    return tuple(
        VirtualSlot(ugv_id=ugv.ugv_id, position=position)
        for ugv in ugvs
        for position in range(1, capacity + 1)
    )


def build_slot_matrices(
    requests: Sequence[RechargeRequest],
    ugvs: Sequence[UGVState],
    road_network: RoadNetwork,
    ugv_speed: float,
    recharge_time: float,
    slots_per_ugv: int | None = None,
) -> SlotMatrices:
    """Construct the approximation used by the assignment head.

    ``UGVState.ready_time`` is defined as the completion time of all work that
    was committed before this decision round. Consequently, the slot term only
    adds ``(k - 1) * recharge_time`` and does not double count an existing queue.
    Exact travel between consecutively assigned landing nodes is calculated by
    the event simulator after matching.
    """
    if ugv_speed <= 0:
        raise ValueError("ugv_speed must be positive")
    if recharge_time < 0:
        raise ValueError("recharge_time must be non-negative")
    if len({request.uav_id for request in requests}) != len(requests):
        raise ValueError("uav_id values must be unique within an assignment round")
    if len({ugv.ugv_id for ugv in ugvs}) != len(ugvs):
        raise ValueError("ugv_id values must be unique")

    slots = build_virtual_slots(ugvs, len(requests), slots_per_ugv)
    ugv_by_id = {ugv.ugv_id: ugv for ugv in ugvs}
    shape = (len(requests), len(slots))
    distance = np.zeros(shape, dtype=np.float64)
    travel = np.zeros(shape, dtype=np.float64)
    wait = np.zeros(shape, dtype=np.float64)
    completion = np.zeros(shape, dtype=np.float64)

    for row, request in enumerate(requests):
        for column, slot in enumerate(slots):
            ugv = ugv_by_id[slot.ugv_id]
            distance[row, column] = road_network.road_distance(
                ugv.current_node, request.landing_node
            )
            travel[row, column] = distance[row, column] / ugv_speed
            base_arrival = ugv.ready_time + travel[row, column]
            slot_delay = (slot.position - 1) * recharge_time
            wait[row, column] = max(
                0.0,
                base_arrival - request.touchdown_time,
            ) + slot_delay
            completion[row, column] = (
                request.touchdown_time + wait[row, column] + recharge_time
            )

    return SlotMatrices(
        slots=slots,
        road_distance=distance,
        road_travel_time=travel,
        predicted_wait=wait,
        predicted_completion=completion,
    )


def assert_prefix_slots(slots: Sequence[VirtualSlot]) -> None:
    """Validate that occupied positions for each UGV are 1..k without gaps."""
    positions: dict[int, list[int]] = {}
    for slot in slots:
        positions.setdefault(slot.ugv_id, []).append(slot.position)
    for ugv_id, values in positions.items():
        actual = sorted(values)
        expected = list(range(1, len(values) + 1))
        if actual != expected:
            raise ValueError(
                f"UGV {ugv_id} has non-prefix virtual slots {actual}; "
                f"expected {expected}"
            )
