"""Command-line entry points for smoke testing and local demonstrations."""

from __future__ import annotations

import argparse

from .base_map import BasePaperScenarioGenerator
from .evaluation.runner import run_paired_benchmark
from .evaluation.metrics import result_metrics
from .mission_simulator import TwoDMissionSimulator
from .policies import (
    NearestEuclideanPolicy,
    NearestRoadPolicy,
    QueueAwareHungarianPolicy,
)
from .scenarios import grid_road_network, random_assignment_round
from .simulator import ChargingRoundSimulator
from .visualization import plot_scenario


def run_demo(seed: int) -> None:
    road = grid_road_network(rows=4, columns=4, spacing=1.0)
    requests, ugvs = random_assignment_round(
        road_network=road,
        number_of_uavs=4,
        number_of_ugvs=2,
        seed=seed,
    )
    simulator = ChargingRoundSimulator(
        road_network=road,
        ugv_speed=0.27,
        recharge_time=10.0,
    )
    policies = [
        NearestEuclideanPolicy(),
        NearestRoadPolicy(),
        QueueAwareHungarianPolicy(),
    ]
    print("policy,makespan,mean_wait,max_wait,total_road_distance,load_std")
    for policy in policies:
        result = simulator.run(requests, ugvs, policy)
        metrics = result_metrics(result)
        print(
            f"{policy.name},{metrics['makespan']:.3f},"
            f"{metrics['mean_wait']:.3f},{metrics['max_wait']:.3f},"
            f"{metrics['total_road_distance']:.3f},"
            f"{metrics['ugv_service_load_std']:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run a deterministic 4-UAV/2-UGV round")
    demo.add_argument("--seed", type=int, default=7)
    benchmark = subparsers.add_parser(
        "benchmark", help="run a paired synthetic assignment benchmark"
    )
    benchmark.add_argument("--instances", type=int, default=100)
    benchmark.add_argument("--uavs", type=int, default=4)
    benchmark.add_argument("--ugvs", type=int, default=2)
    benchmark.add_argument("--first-seed", type=int, default=0)
    scenario = subparsers.add_parser(
        "scenario", help="generate and plot a base-paper-style 20 km map"
    )
    scenario.add_argument("--aerial-points", type=int, default=15)
    scenario.add_argument("--ground-points", type=int, default=5)
    scenario.add_argument("--seed", type=int, default=1234)
    scenario.add_argument(
        "--output", default="outputs/scenario_U15G5_seed1234.png"
    )
    mission = subparsers.add_parser(
        "mission", help="run and plot a complete 2D fuel-constrained mission"
    )
    mission.add_argument("--aerial-points", type=int, default=15)
    mission.add_argument("--ground-points", type=int, default=5)
    mission.add_argument("--uavs", type=int, default=2)
    mission.add_argument("--ugvs", type=int, default=2)
    mission.add_argument("--seed", type=int, default=1234)
    mission.add_argument(
        "--policy",
        choices=("nearest", "road", "queue"),
        default="queue",
    )
    mission.add_argument(
        "--output", default="outputs/mission_U15G5_2U2G_seed1234.png"
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "demo":
        run_demo(arguments.seed)
    elif arguments.command == "benchmark":
        road = grid_road_network(rows=8, columns=8, spacing=2.5)
        policies = [
            NearestEuclideanPolicy(),
            NearestRoadPolicy(),
            QueueAwareHungarianPolicy(),
        ]
        summaries = run_paired_benchmark(
            road_network=road,
            policies=policies,
            number_of_instances=arguments.instances,
            number_of_uavs=arguments.uavs,
            number_of_ugvs=arguments.ugvs,
            ugv_speed=0.27,
            recharge_time=10.0,
            first_seed=arguments.first_seed,
        )
        print("policy,metric,mean,std,ci95_low,ci95_high")
        for policy_name, metrics in summaries.items():
            for metric_name, summary in metrics.items():
                print(
                    f"{policy_name},{metric_name},{summary['mean']:.6f},"
                    f"{summary['std']:.6f},{summary['ci95_low']:.6f},"
                    f"{summary['ci95_high']:.6f}"
                )
    elif arguments.command == "scenario":
        generated = BasePaperScenarioGenerator().generate(
            aerial_task_count=arguments.aerial_points,
            ground_task_count=arguments.ground_points,
            seed=arguments.seed,
        )
        destination = plot_scenario(generated, arguments.output)
        print(
            f"wrote {destination} ({generated.width_km:.0f} km x "
            f"{generated.height_km:.0f} km, "
            f"{len(generated.aerial_tasks)} aerial tasks, "
            f"{len(generated.ground_tasks)} ground tasks)"
        )
    elif arguments.command == "mission":
        generated = BasePaperScenarioGenerator().generate(
            aerial_task_count=arguments.aerial_points,
            ground_task_count=arguments.ground_points,
            seed=arguments.seed,
        )
        policies = {
            "nearest": NearestEuclideanPolicy(),
            "road": NearestRoadPolicy(),
            "queue": QueueAwareHungarianPolicy(),
        }
        result = TwoDMissionSimulator(
            scenario=generated,
            number_of_uavs=arguments.uavs,
            number_of_ugvs=arguments.ugvs,
            assignment_policy=policies[arguments.policy],
        ).run()
        destination = plot_scenario(generated, arguments.output, result=result)
        print(
            f"completed={result.completed} failed={result.failed} "
            f"makespan_min={result.makespan:.3f} "
            f"mean_uav_wait_min={result.mean_uav_waiting_time:.3f} "
            f"sortie_rounds={len(result.charging_rounds)} "
            f"visited={len(result.visited_task_ids)}/{len(generated.tasks)} "
            f"plot={destination}"
        )
        if result.failure_reason:
            print(f"failure_reason={result.failure_reason}")


if __name__ == "__main__":
    main()
