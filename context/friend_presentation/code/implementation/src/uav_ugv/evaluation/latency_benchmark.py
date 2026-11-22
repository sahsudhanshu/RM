"""Measure frozen-policy compute time on a dedicated scenario split."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from ..base_map import BasePaperScenarioGenerator
from ..mission_simulator import TwoDMissionSimulator
from ..model import QueueAwareCrossAttention
from ..training.assignment_policy import (
    NearestWithLearnedOrderPolicy, SampledAssignmentPolicy,
)
from ..training.route_policy import LearnedRoutePolicy, RoutingTransformer
from .timing import TimedAssignmentPolicy, TimedRoutePolicy, planning_metrics


def benchmark(run_dir: Path, count: int = 64,
              source_run: Path | None = None) -> Path:
    if count < 1:
        raise ValueError("count must be positive")
    output = run_dir / "latency_benchmark.csv"
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((run_dir / "resolved_config.json").read_text())
    checkpoint_run = source_run or run_dir
    selected = json.loads((checkpoint_run / "checkpoint_selection.json").read_text())
    torch.set_num_threads(4)
    methods = {}
    for name in ("baseline", "proposed"):
        checkpoint = checkpoint_run / selected["methods"][name]["selected"]["checkpoint"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        route_model = RoutingTransformer()
        route_model.load_state_dict(payload["route_state"])
        route_model.eval()
        if name == "baseline":
            assignment = NearestWithLearnedOrderPolicy(
                route_model, "cpu", mode="greedy"
            )
        else:
            head = QueueAwareCrossAttention(
                5, 4, max_slots_per_ugv=max(8, config["uavs"]),
                zero_queue_penalty=config.get("queue_penalty") == "zero",
            )
            head.load_state_dict(payload["assignment_state"])
            head.eval()
            assignment = SampledAssignmentPolicy(head, "cpu", mode="greedy")
        methods[name] = (
            LearnedRoutePolicy(route_model, "cpu", mode="greedy"), assignment
        )
    rows = []
    generator = BasePaperScenarioGenerator()
    with torch.no_grad():
        warmup = generator.generate(
            config["aerial"], config["ground"], 3_299_999_999
        )
        for route, assignment in methods.values():
            TwoDMissionSimulator(
                warmup, config["uavs"], config["ugvs"],
                assignment, route_policy=route,
            ).run()
        for seed in range(3_300_000_000, 3_300_000_000 + count):
            scenario = generator.generate(
                config["aerial"], config["ground"], seed
            )
            for name, (route, assignment) in methods.items():
                timed_route = TimedRoutePolicy(route)
                timed_assignment = TimedAssignmentPolicy(assignment)
                result = TwoDMissionSimulator(
                    scenario, config["uavs"], config["ugvs"],
                    timed_assignment, route_policy=timed_route,
                ).run()
                rows.append({
                    "run_id": run_dir.name,
                    "instance_seed": seed,
                    "method": name,
                    "completed": result.completed,
                    "charging_rounds": len(result.charging_rounds),
                    **planning_metrics(
                        timed_route, timed_assignment,
                        len(result.charging_rounds),
                    ),
                })
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "dedicated_inference_latency_split",
        "model_source_run": checkpoint_run.name,
        "seed_start": 3_300_000_000,
        "instance_count": count,
        "warmup_seed": 3_299_999_999,
        "device": "cpu",
        "torch_cpu_threads": 4,
        "definition": "policy compute milliseconds per completed recharge/replanning round; excludes scenario generation and simulator event arithmetic",
        "methods": {
            name: {
                "mean_tplan_ms_per_replanning_round": statistics.fmean(
                    float(row["tplan_ms_per_replanning_round"])
                    for row in rows if row["method"] == name
                ),
                "median_tplan_ms_per_replanning_round": statistics.median(
                    float(row["tplan_ms_per_replanning_round"])
                    for row in rows if row["method"] == name
                ),
                "p95_tplan_ms_per_replanning_round": float(np.quantile([
                    float(row["tplan_ms_per_replanning_round"])
                    for row in rows if row["method"] == name
                ], 0.95)),
            }
            for name in methods
        },
    }
    (run_dir / "latency_summary.json").write_text(json.dumps(summary, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--instances", type=int, default=64)
    parser.add_argument("--source-run", type=Path)
    args = parser.parse_args()
    print(benchmark(args.run_dir, args.instances, args.source_run))


if __name__ == "__main__":
    main()
