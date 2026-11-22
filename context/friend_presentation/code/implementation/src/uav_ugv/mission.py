"""Domain objects for complete 2D fuel-constrained missions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .domain import RoundResult
from .road_network import RoadNetwork


class TaskKind(Enum):
    AERIAL = "aerial"
    GROUND = "ground"


@dataclass(frozen=True, slots=True)
class TaskPoint:
    task_id: str
    x: float
    y: float
    kind: TaskKind
    road_node: int | None = None

    def __post_init__(self) -> None:
        if self.kind is TaskKind.GROUND and self.road_node is None:
            raise ValueError("ground task points require a road_node")
        if self.kind is TaskKind.AERIAL and self.road_node is not None:
            raise ValueError("aerial task points cannot have a road_node")


@dataclass(frozen=True, slots=True)
class MissionScenario:
    width_km: float
    height_km: float
    road_network: RoadNetwork
    depot_node: int
    aerial_tasks: tuple[TaskPoint, ...]
    ground_tasks: tuple[TaskPoint, ...]
    seed: int

    @property
    def tasks(self) -> tuple[TaskPoint, ...]:
        return self.ground_tasks + self.aerial_tasks

    @property
    def depot_coordinates(self) -> tuple[float, float]:
        return self.road_network.coordinates(self.depot_node)


@dataclass(slots=True)
class UAVMissionState:
    uav_id: int
    x: float
    y: float
    elapsed_time: float
    fuel_j: float
    sorties_completed: int = 0
    waiting_time: float = 0.0


@dataclass(frozen=True, slots=True)
class FlightLeg:
    uav_id: int
    sortie_index: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    start_time: float
    end_time: float
    fuel_used_j: float
    target_id: str
    is_rendezvous: bool


@dataclass(slots=True)
class MissionResult:
    scenario: MissionScenario
    visited_task_ids: set[str]
    uav_states: list[UAVMissionState]
    charging_rounds: list[RoundResult]
    flight_legs: list[FlightLeg]
    makespan: float
    failed: bool
    failure_reason: str | None

    @property
    def completed(self) -> bool:
        return not self.failed and len(self.visited_task_ids) == len(self.scenario.tasks)

    @property
    def mean_uav_waiting_time(self) -> float:
        if not self.uav_states:
            return 0.0
        return sum(state.waiting_time for state in self.uav_states) / len(self.uav_states)
