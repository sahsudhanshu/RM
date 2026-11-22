"""Complete 2D sortie, fuel, rendezvous, queue, and recharge simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from .domain import RechargeRequest, UGVState
from .mission import (
    FlightLeg,
    MissionResult,
    MissionScenario,
    TaskKind,
    TaskPoint,
    UAVMissionState,
)
from .policies import AssignmentPolicy
from .simulator import ChargingRoundSimulator


@dataclass(frozen=True, slots=True)
class MissionParameters:
    uav_speed_m_per_s: float = 10.0
    ugv_speed_km_per_min: float = 0.27
    uav_fuel_capacity_j: float = 287_700.0
    recharge_time_min: float = 15.0
    max_sortie_rounds: int = 100

    def __post_init__(self) -> None:
        if min(
            self.uav_speed_m_per_s,
            self.ugv_speed_km_per_min,
            self.uav_fuel_capacity_j,
            self.recharge_time_min,
            self.max_sortie_rounds,
        ) <= 0:
            raise ValueError("mission parameters must be positive")

    @property
    def uav_power_w(self) -> float:
        speed = self.uav_speed_m_per_s
        return (
            0.0461 * speed**3
            - 0.5834 * speed**2
            - 1.8761 * speed
            + 229.6
        )


class RouteDecisionPolicy(Protocol):
    """Select a feasible task or a reachable landing point for one UAV."""

    def start_episode(self, scenario: MissionScenario) -> None: ...

    def choose_next(
        self,
        uav: UAVMissionState,
        visited: set[str],
        feasible_tasks: Sequence[TaskPoint],
        reachable_grounds: Sequence[TaskPoint],
        teacher_choice: TaskPoint,
    ) -> TaskPoint: ...


class TwoDMissionSimulator:
    """Run complete missions using a deterministic nearest-feasible route policy.

    The route policy is intentionally pluggable later. This implementation gives
    the assignment head a physically coherent 2D environment now: UAVs consume
    aerodynamic energy, retain enough reserve to reach a ground rendezvous, and
    wait for road-constrained UGV service before starting another sortie.
    """

    def __init__(
        self,
        scenario: MissionScenario,
        number_of_uavs: int,
        number_of_ugvs: int,
        assignment_policy: AssignmentPolicy,
        parameters: MissionParameters = MissionParameters(),
        route_policy: RouteDecisionPolicy | None = None,
    ) -> None:
        if number_of_uavs < 1 or number_of_ugvs < 1:
            raise ValueError("at least one UAV and UGV are required")
        self.scenario = scenario
        self.number_of_uavs = number_of_uavs
        self.number_of_ugvs = number_of_ugvs
        self.assignment_policy = assignment_policy
        self.parameters = parameters
        self.route_policy = route_policy

    def run(self) -> MissionResult:
        if self.route_policy is not None:
            self.route_policy.start_episode(self.scenario)
        depot_x, depot_y = self.scenario.depot_coordinates
        uavs = [
            UAVMissionState(
                uav_id=index,
                x=depot_x,
                y=depot_y,
                elapsed_time=0.0,
                fuel_j=self.parameters.uav_fuel_capacity_j,
            )
            for index in range(self.number_of_uavs)
        ]
        ugvs = [
            UGVState(ugv_id=index, current_node=self.scenario.depot_node)
            for index in range(self.number_of_ugvs)
        ]
        depot_tasks = {
            task.task_id
            for task in self.scenario.ground_tasks
            if task.road_node == self.scenario.depot_node
        }
        visited = set(depot_tasks)
        legs: list[FlightLeg] = []
        charging_rounds = []
        failed = False
        failure_reason: str | None = None

        charging_simulator = ChargingRoundSimulator(
            road_network=self.scenario.road_network,
            ugv_speed=self.parameters.ugv_speed_km_per_min,
            recharge_time=self.parameters.recharge_time_min,
        )

        for _round in range(self.parameters.max_sortie_rounds):
            if len(visited) == len(self.scenario.tasks):
                break
            visited_before = len(visited)
            positions_before = {(uav.uav_id, uav.x, uav.y) for uav in uavs}
            requests: list[RechargeRequest] = []
            rendezvous_by_uav: dict[int, TaskPoint] = {}

            for uav in sorted(uavs, key=lambda state: state.elapsed_time):
                rendezvous = self._fly_task_segment(uav, visited, legs)
                if rendezvous is None:
                    failed = True
                    failure_reason = f"UAV {uav.uav_id} cannot reach a rendezvous"
                    break
                self._move_uav(
                    uav,
                    rendezvous.x,
                    rendezvous.y,
                    target_id=rendezvous.task_id,
                    is_rendezvous=True,
                    legs=legs,
                )
                visited.add(rendezvous.task_id)
                rendezvous_by_uav[uav.uav_id] = rendezvous
                requests.append(
                    RechargeRequest(
                        uav_id=uav.uav_id,
                        landing_node=int(rendezvous.road_node),
                        touchdown_time=uav.elapsed_time,
                        remaining_energy_fraction=max(
                            0.0,
                            uav.fuel_j / self.parameters.uav_fuel_capacity_j,
                        ),
                        sortie_index=uav.sorties_completed,
                    )
                )
            if failed:
                break

            charging = charging_simulator.run(requests, ugvs, self.assignment_policy)
            charging_rounds.append(charging)
            ugvs = charging.final_ugv_states
            service_by_uav = {service.uav_id: service for service in charging.services}
            for uav in uavs:
                service = service_by_uav[uav.uav_id]
                rendezvous = rendezvous_by_uav[uav.uav_id]
                uav.elapsed_time = service.charging_end_time
                uav.waiting_time += service.waiting_time
                uav.fuel_j = self.parameters.uav_fuel_capacity_j
                uav.x, uav.y = rendezvous.x, rendezvous.y
                uav.sorties_completed += 1

            positions_after = {(uav.uav_id, uav.x, uav.y) for uav in uavs}
            if len(visited) == visited_before and positions_after == positions_before:
                failed = True
                failure_reason = "no task or repositioning progress was possible"
                break

        if len(visited) != len(self.scenario.tasks) and not failed:
            failed = True
            failure_reason = "maximum sortie rounds exceeded"
        makespan = max(
            [state.elapsed_time for state in uavs]
            + [state.ready_time for state in ugvs],
            default=0.0,
        )
        return MissionResult(
            scenario=self.scenario,
            visited_task_ids=visited,
            uav_states=uavs,
            charging_rounds=charging_rounds,
            flight_legs=legs,
            makespan=makespan,
            failed=failed,
            failure_reason=failure_reason,
        )

    def _fly_task_segment(
        self,
        uav: UAVMissionState,
        visited: set[str],
        legs: list[FlightLeg],
    ) -> TaskPoint | None:
        sortie_has_task = False
        while True:
            feasible = [
                task
                for task in self.scenario.tasks
                if task.task_id not in visited and self._can_visit_and_land(uav, task)
            ]
            reachable = [
                task for task in self.scenario.ground_tasks
                if self._fuel_for_distance(
                    self._distance(uav.x, uav.y, task.x, task.y)
                ) <= uav.fuel_j + 1e-6
            ]
            if not feasible and not reachable:
                return None
            teacher = (
                min(
                    feasible,
                    key=lambda task: self._distance(uav.x, uav.y, task.x, task.y),
                )
                if feasible else self._best_reachable_rendezvous(uav, visited)
            )
            if teacher is None:
                return None
            if self.route_policy is None:
                if not feasible:
                    return teacher
                target = teacher
            else:
                eligible_grounds = reachable
                if feasible and not sortie_has_task:
                    # A learned sortie must make task progress before
                    # voluntarily landing for recharge.
                    eligible_grounds = [
                        ground for ground in reachable
                        if ground.task_id not in visited
                    ]
                elif not feasible and len(reachable) > 1:
                    # Full-battery, zero-distance recharge cannot reposition
                    # the UAV when another landing site is reachable.
                    alternatives = [
                        ground for ground in reachable
                        if self._distance(uav.x, uav.y, ground.x, ground.y) > 1e-9
                    ]
                    if alternatives:
                        eligible_grounds = alternatives
                if teacher.kind is TaskKind.GROUND and teacher not in eligible_grounds:
                    teacher = min(
                        eligible_grounds,
                        key=lambda ground: self._distance(
                            uav.x, uav.y, ground.x, ground.y
                        ),
                    )
                target = self.route_policy.choose_next(
                    uav, visited, feasible, eligible_grounds, teacher
                )
                allowed = {task.task_id for task in feasible + eligible_grounds}
                if target.task_id not in allowed:
                    raise ValueError("route policy selected an infeasible action")
                if target.kind is TaskKind.GROUND:
                    return target
            self._move_uav(
                uav,
                target.x,
                target.y,
                target_id=target.task_id,
                is_rendezvous=False,
                legs=legs,
            )
            visited.add(target.task_id)
            sortie_has_task = True

    def _can_visit_and_land(self, uav: UAVMissionState, task: TaskPoint) -> bool:
        to_task = self._fuel_for_distance(
            self._distance(uav.x, uav.y, task.x, task.y)
        )
        reserve = min(
            self._fuel_for_distance(
                self._distance(task.x, task.y, ground.x, ground.y)
            )
            for ground in self.scenario.ground_tasks
        )
        return to_task + reserve <= uav.fuel_j + 1e-6

    def _best_reachable_rendezvous(
        self,
        uav: UAVMissionState,
        visited: set[str],
    ) -> TaskPoint | None:
        candidates = [
            task
            for task in self.scenario.ground_tasks
            if self._fuel_for_distance(
                self._distance(uav.x, uav.y, task.x, task.y)
            )
            <= uav.fuel_j + 1e-6
        ]
        if not candidates:
            return None
        remaining = [
            task for task in self.scenario.tasks if task.task_id not in visited
        ]
        if remaining:
            def future_access_cost(ground: TaskPoint) -> float:
                return min(
                    self._distance(ground.x, ground.y, task.x, task.y)
                    + min(
                        self._distance(task.x, task.y, other.x, other.y)
                        for other in self.scenario.ground_tasks
                    )
                    for task in remaining
                )

            return min(
                candidates,
                key=lambda task: (
                    future_access_cost(task),
                    self._distance(uav.x, uav.y, task.x, task.y),
                ),
            )
        return min(
            candidates,
            key=lambda task: self._distance(uav.x, uav.y, task.x, task.y),
        )

    def _move_uav(
        self,
        uav: UAVMissionState,
        x: float,
        y: float,
        target_id: str,
        is_rendezvous: bool,
        legs: list[FlightLeg],
    ) -> None:
        distance = self._distance(uav.x, uav.y, x, y)
        fuel = self._fuel_for_distance(distance)
        if fuel > uav.fuel_j + 1e-6:
            raise RuntimeError(f"UAV {uav.uav_id} depleted fuel in flight")
        travel_minutes = distance * 1000.0 / self.parameters.uav_speed_m_per_s / 60.0
        start_time = uav.elapsed_time
        start_x, start_y = uav.x, uav.y
        uav.elapsed_time += travel_minutes
        uav.fuel_j -= fuel
        uav.x, uav.y = x, y
        legs.append(
            FlightLeg(
                uav_id=uav.uav_id,
                sortie_index=uav.sorties_completed,
                start_x=start_x,
                start_y=start_y,
                end_x=x,
                end_y=y,
                start_time=start_time,
                end_time=uav.elapsed_time,
                fuel_used_j=fuel,
                target_id=target_id,
                is_rendezvous=is_rendezvous,
            )
        )

    def _fuel_for_distance(self, distance_km: float) -> float:
        seconds = distance_km * 1000.0 / self.parameters.uav_speed_m_per_s
        return self.parameters.uav_power_w * seconds

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)
