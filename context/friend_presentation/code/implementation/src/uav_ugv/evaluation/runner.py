"""Paired seeded evaluation across assignment policies."""

from __future__ import annotations

from collections.abc import Sequence

from ..policies import AssignmentPolicy
from ..road_network import RoadNetwork
from ..scenarios import random_assignment_round
from ..simulator import ChargingRoundSimulator
from .metrics import aggregate_results


def run_paired_benchmark(
    road_network: RoadNetwork,
    policies: Sequence[AssignmentPolicy],
    number_of_instances: int,
    number_of_uavs: int,
    number_of_ugvs: int,
    ugv_speed: float,
    recharge_time: float,
    first_seed: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate every policy on precisely the same generated instances."""
    if number_of_instances < 1:
        raise ValueError("number_of_instances must be positive")
    names = [policy.name for policy in policies]
    if len(names) != len(set(names)):
        raise ValueError("policy names must be unique")

    simulator = ChargingRoundSimulator(
        road_network=road_network,
        ugv_speed=ugv_speed,
        recharge_time=recharge_time,
    )
    results = {policy.name: [] for policy in policies}
    for seed in range(first_seed, first_seed + number_of_instances):
        requests, ugvs = random_assignment_round(
            road_network=road_network,
            number_of_uavs=number_of_uavs,
            number_of_ugvs=number_of_ugvs,
            seed=seed,
        )
        for policy in policies:
            results[policy.name].append(simulator.run(requests, ugvs, policy))
    return {
        policy_name: aggregate_results(policy_results)
        for policy_name, policy_results in results.items()
    }
