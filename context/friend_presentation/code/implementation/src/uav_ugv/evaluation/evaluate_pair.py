"""Evaluate both frozen trained policies on the same unopened test scenarios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..base_map import BasePaperScenarioGenerator
from ..mission_simulator import TwoDMissionSimulator
from ..model import QueueAwareCrossAttention
from ..training.assignment_policy import (
    NearestWithLearnedOrderPolicy,
    SampledAssignmentPolicy,
)
from ..training.route_policy import LearnedRoutePolicy, RoutingTransformer
from .timing import TimedAssignmentPolicy, TimedRoutePolicy, planning_metrics


def _scenario_digest(scenario) -> str:
    road = scenario.road_network.graph
    payload = {
        "seed": scenario.seed,
        "depot": scenario.depot_node,
        "tasks": [
            (task.task_id, task.x, task.y, task.kind.value, task.road_node)
            for task in scenario.tasks
        ],
        "road_nodes": sorted(
            (node, *scenario.road_network.coordinates(node)) for node in road.nodes
        ),
        "road_edges": sorted(
            (min(a, b), max(a, b), data["distance"])
            for a, b, data in road.edges(data=True)
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _metrics(result) -> dict[str, float | int | str]:
    services = [service for round_result in result.charging_rounds
                for service in round_result.services]
    waits = [service.waiting_time for service in services]
    ugv_loads = []
    ugv_busy_times = []
    ugv_counts = []
    for ugv_id in range(len(result.charging_rounds[0].final_ugv_states)
                        if result.charging_rounds else 0):
        group = [service for service in services if service.ugv_id == ugv_id]
        ugv_loads.append(sum(
            service.charging_end_time - service.departure_time for service in group
        ))
        ugv_busy_times.append(sum(
            (service.ugv_arrival_time - service.departure_time)
            + (service.charging_end_time - service.charging_start_time)
            for service in group
        ))
        ugv_counts.append(len(group))
    return {
        "status": "completed" if result.completed else "failed",
        "failure_reason": result.failure_reason or "",
        "makespan_min": result.makespan,
        "penalized_cost_min": result.makespan + (800.0 if result.failed else 0.0),
        "mean_wait_per_event_min": statistics.fmean(waits) if waits else 0.0,
        "mean_total_wait_per_uav_min": result.mean_uav_waiting_time,
        "max_wait_per_event_min": max(waits, default=0.0),
        "ugv_service_time_std_min": statistics.pstdev(ugv_loads)
        if ugv_loads else 0.0,
        "ugv_busy_time_std_min": statistics.pstdev(ugv_busy_times)
        if ugv_busy_times else 0.0,
        "ugv_service_count_std": statistics.pstdev(ugv_counts)
        if ugv_counts else 0.0,
        "total_road_distance_km": sum(
            service.road_distance for service in services
        ),
        "visited_tasks": len(result.visited_task_ids),
        "total_tasks": len(result.scenario.tasks),
        "fuel_violations": 0,
    }


def evaluate_pair(run_dir: Path, count: int = 256) -> Path:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "training_complete_test_pending":
        raise RuntimeError("both frozen checkpoints must exist before the test is opened")
    if (run_dir / "evaluation.csv").exists():
        raise FileExistsError("the frozen test has already been evaluated")
    source_root = Path(__file__).resolve().parents[1]
    for relative, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"source changed since training: {relative}; use the archived source"
            )
    config = json.loads((run_dir / "resolved_config.json").read_text())
    selection_path = run_dir / "checkpoint_selection.json"
    if not selection_path.exists():
        raise RuntimeError("select both checkpoints on validation data first")
    selection = json.loads(selection_path.read_text())
    checkpoints = {
        name: run_dir / selection["methods"][name]["selected"]["checkpoint"]
        for name in ("baseline", "proposed")
    }
    seeds = json.loads((run_dir / "dataset_manifest.json").read_text())[
        "test_seeds"
    ][:count]
    if count < 1 or count > 256:
        raise ValueError("count must be between one and 256")
    device = torch.device("cpu")
    torch.set_num_threads(4)
    baseline_payload = torch.load(
        checkpoints["baseline"],
        map_location=device, weights_only=False,
    )
    proposed_payload = torch.load(
        checkpoints["proposed"],
        map_location=device, weights_only=False,
    )
    baseline_route_model = RoutingTransformer().to(device)
    baseline_route_model.load_state_dict(baseline_payload["route_state"])
    proposed_route_model = RoutingTransformer().to(device)
    proposed_route_model.load_state_dict(proposed_payload["route_state"])
    assignment_model = QueueAwareCrossAttention(
        5, 4, max_slots_per_ugv=max(8, config["uavs"]),
        zero_queue_penalty=config.get("queue_penalty") == "zero",
    ).to(device)
    assignment_model.load_state_dict(proposed_payload["assignment_state"])
    baseline_route_model.eval()
    proposed_route_model.eval()
    assignment_model.eval()
    methods = {
        "baseline": (
            LearnedRoutePolicy(baseline_route_model, device, mode="greedy"),
            NearestWithLearnedOrderPolicy(
                baseline_route_model, device, mode="greedy"
            ),
        ),
        "proposed": (
            LearnedRoutePolicy(proposed_route_model, device, mode="greedy"),
            SampledAssignmentPolicy(assignment_model, device, mode="greedy"),
        ),
    }
    checkpoint_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in checkpoints.items()
    }
    generator = BasePaperScenarioGenerator()
    rows: list[dict] = []
    event_waits: dict[str, list[float]] = {name: [] for name in methods}
    events_path = run_dir / "events.jsonl"
    with events_path.open("x", encoding="utf-8") as events:
        with torch.no_grad():
            for seed in seeds:
                scenario = generator.generate(
                    config["aerial"], config["ground"], seed
                )
                digest = _scenario_digest(scenario)
                for name, (route, assignment) in methods.items():
                    timed_route = TimedRoutePolicy(route)
                    timed_assignment = TimedAssignmentPolicy(assignment)
                    result = TwoDMissionSimulator(
                        scenario, config["uavs"], config["ugvs"],
                        timed_assignment, route_policy=timed_route,
                    ).run()
                    event_waits[name].extend(
                        service.waiting_time
                        for round_result in result.charging_rounds
                        for service in round_result.services
                    )
                    rows.append({
                        "run_id": run_dir.name,
                        "instance_seed": seed,
                        "scenario_sha256": digest,
                        "method": name,
                        "checkpoint_sha256": checkpoint_hashes[name],
                        **_metrics(result),
                        **planning_metrics(
                            timed_route, timed_assignment,
                            len(result.charging_rounds),
                        ),
                    })
                    events.write(json.dumps({
                        "instance_seed": seed,
                        "method": name,
                        "flight_legs": [asdict(leg) for leg in result.flight_legs],
                        "charging_services": [
                            asdict(service)
                            for round_result in result.charging_rounds
                            for service in round_result.services
                        ],
                    }) + "\n")
    output = run_dir / "evaluation.csv"
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_method = {
        name: [row for row in rows if row["method"] == name]
        for name in methods
    }
    baseline_cost = np.array([
        row["penalized_cost_min"] for row in by_method["baseline"]
    ])
    proposed_cost = np.array([
        row["penalized_cost_min"] for row in by_method["proposed"]
    ])
    differences = proposed_cost - baseline_cost
    rng = np.random.default_rng(37)
    indices = rng.integers(0, len(differences), size=(2000, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    summary = {
        "instance_count": len(seeds),
        "checkpoint_selection": selection["methods"],
        "paired_cost_difference_proposed_minus_baseline_min": float(
            differences.mean()
        ),
        "paired_bootstrap_95_ci_min": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "bootstrap_resamples": 2000,
        "methods": {
            name: {
                "mean_makespan_min": statistics.fmean(
                    row["makespan_min"] for row in method_rows
                ),
                "mean_penalized_cost_min": statistics.fmean(
                    row["penalized_cost_min"] for row in method_rows
                ),
                "mean_wait_per_event_min": statistics.fmean(
                    event_waits[name]
                ) if event_waits[name] else 0.0,
                "mean_instance_wait_per_event_min": statistics.fmean(
                    row["mean_wait_per_event_min"] for row in method_rows
                ),
                "mean_ugv_busy_time_std_min": statistics.fmean(
                    row["ugv_busy_time_std_min"] for row in method_rows
                ),
                "mean_tplan_ms_per_replanning_round": statistics.fmean(
                    row["tplan_ms_per_replanning_round"] for row in method_rows
                ),
                "completion_rate": statistics.fmean(
                    row["status"] == "completed" for row in method_rows
                ),
            }
            for name, method_rows in by_method.items()
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    manifest["status"] = "evaluated"
    manifest["test_status"] = "evaluated_once"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--instances", type=int, default=256)
    arguments = parser.parse_args()
    print(evaluate_pair(arguments.run_dir, arguments.instances))


if __name__ == "__main__":
    main()
