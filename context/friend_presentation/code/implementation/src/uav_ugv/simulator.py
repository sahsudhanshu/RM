"""Event-driven simulation of UGV travel and sequential UAV charging."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Sequence

from .domain import (
    Assignment,
    RechargeRequest,
    RoundResult,
    ServiceRecord,
    UGVState,
    VirtualSlot,
)
from .policies import AssignmentPolicy
from .road_network import RoadNetwork
from .virtual_slots import assert_prefix_slots


class ChargingRoundSimulator:
    """Simulate the exact consequences of one assignment decision round."""

    def __init__(
        self,
        road_network: RoadNetwork,
        ugv_speed: float,
        recharge_time: float,
    ) -> None:
        if ugv_speed <= 0:
            raise ValueError("ugv_speed must be positive")
        if recharge_time < 0:
            raise ValueError("recharge_time must be non-negative")
        self.road_network = road_network
        self.ugv_speed = ugv_speed
        self.recharge_time = recharge_time

    def run(
        self,
        requests: Sequence[RechargeRequest],
        ugvs: Sequence[UGVState],
        policy: AssignmentPolicy,
    ) -> RoundResult:
        assignments = policy.assign(
            requests=requests,
            ugvs=ugvs,
            road_network=self.road_network,
            ugv_speed=self.ugv_speed,
            recharge_time=self.recharge_time,
        )
        self._validate_assignments(requests, ugvs, assignments)

        request_by_id = {request.uav_id: request for request in requests}
        state_by_id = {state.ugv_id: deepcopy(state) for state in ugvs}
        grouped: dict[int, list[Assignment]] = defaultdict(list)
        for assignment in assignments:
            grouped[assignment.ugv_id].append(assignment)

        services: list[ServiceRecord] = []
        for ugv_id, group in grouped.items():
            state = state_by_id[ugv_id]
            for assignment in sorted(group, key=lambda item: item.slot_position):
                request = request_by_id[assignment.uav_id]
                departure = state.ready_time
                distance = self.road_network.road_distance(
                    state.current_node, request.landing_node
                )
                arrival = departure + distance / self.ugv_speed
                charging_start = max(arrival, request.touchdown_time)
                charging_end = charging_start + self.recharge_time
                services.append(
                    ServiceRecord(
                        uav_id=request.uav_id,
                        ugv_id=ugv_id,
                        slot_position=assignment.slot_position,
                        landing_node=request.landing_node,
                        touchdown_time=request.touchdown_time,
                        departure_time=departure,
                        ugv_arrival_time=arrival,
                        charging_start_time=charging_start,
                        charging_end_time=charging_end,
                        waiting_time=charging_start - request.touchdown_time,
                        road_distance=distance,
                    )
                )
                state.current_node = request.landing_node
                state.ready_time = charging_end
                state.completed_services += 1
                state.total_distance += distance

        final_states = sorted(state_by_id.values(), key=lambda state: state.ugv_id)
        makespan = max(
            [state.ready_time for state in final_states]
            + [request.touchdown_time for request in requests],
            default=0.0,
        )
        return RoundResult(
            assignments=sorted(assignments, key=lambda item: item.uav_id),
            services=sorted(services, key=lambda item: item.uav_id),
            final_ugv_states=final_states,
            makespan=makespan,
            metadata={"policy": policy.name},
        )

    @staticmethod
    def _validate_assignments(
        requests: Sequence[RechargeRequest],
        ugvs: Sequence[UGVState],
        assignments: Sequence[Assignment],
    ) -> None:
        expected = {request.uav_id for request in requests}
        actual = [assignment.uav_id for assignment in assignments]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("every UAV request must be assigned exactly once")
        valid_ugvs = {ugv.ugv_id for ugv in ugvs}
        if any(assignment.ugv_id not in valid_ugvs for assignment in assignments):
            raise ValueError("assignment references an unknown UGV")
        slots = [
            VirtualSlot(assignment.ugv_id, assignment.slot_position)
            for assignment in assignments
        ]
        if len(set(slots)) != len(slots):
            raise ValueError("a virtual slot was assigned more than once")
        assert_prefix_slots(slots)
