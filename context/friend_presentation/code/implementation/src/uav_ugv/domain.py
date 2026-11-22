"""Typed domain objects shared by assignment, simulation, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RechargeRequest:
    """A UAV waiting (or expected to wait) at a road-accessible landing node."""

    uav_id: int
    landing_node: int
    touchdown_time: float
    remaining_energy_fraction: float = 0.0
    sortie_index: int = 0

    def __post_init__(self) -> None:
        if self.touchdown_time < 0:
            raise ValueError("touchdown_time must be non-negative")
        if not 0.0 <= self.remaining_energy_fraction <= 1.0:
            raise ValueError("remaining_energy_fraction must be in [0, 1]")


@dataclass(slots=True)
class UGVState:
    """State of a physical UGV at the end of its committed service schedule."""

    ugv_id: int
    current_node: int
    ready_time: float = 0.0
    completed_services: int = 0
    total_distance: float = 0.0

    def __post_init__(self) -> None:
        if self.ready_time < 0:
            raise ValueError("ready_time must be non-negative")


@dataclass(frozen=True, slots=True)
class VirtualSlot:
    """The one-indexed queue position of a physical UGV."""

    ugv_id: int
    position: int

    def __post_init__(self) -> None:
        if self.position < 1:
            raise ValueError("virtual-slot position must be at least one")


@dataclass(frozen=True, slots=True)
class Assignment:
    """A discrete match from one UAV request to one virtual UGV slot."""

    uav_id: int
    ugv_id: int
    slot_position: int
    objective_cost: float


@dataclass(frozen=True, slots=True)
class ServiceRecord:
    """Exact event timings obtained after simulating a discrete assignment."""

    uav_id: int
    ugv_id: int
    slot_position: int
    landing_node: int
    touchdown_time: float
    departure_time: float
    ugv_arrival_time: float
    charging_start_time: float
    charging_end_time: float
    waiting_time: float
    road_distance: float


@dataclass(slots=True)
class RoundResult:
    """Output of one batch of recharge requests."""

    assignments: list[Assignment]
    services: list[ServiceRecord]
    final_ugv_states: list[UGVState]
    makespan: float
    metadata: dict[str, float | str] = field(default_factory=dict)

    @property
    def mean_waiting_time(self) -> float:
        if not self.services:
            return 0.0
        return sum(item.waiting_time for item in self.services) / len(self.services)

    @property
    def max_waiting_time(self) -> float:
        return max((item.waiting_time for item in self.services), default=0.0)

    @property
    def total_road_distance(self) -> float:
        return sum(item.road_distance for item in self.services)
