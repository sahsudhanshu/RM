"""Select frozen checkpoints using validation scenarios only."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from ..base_map import BasePaperScenarioGenerator
from ..mission_simulator import TwoDMissionSimulator
from ..model import QueueAwareCrossAttention
from ..training.assignment_policy import (
    NearestWithLearnedOrderPolicy,
    SampledAssignmentPolicy,
)
from ..training.route_policy import LearnedRoutePolicy, RoutingTransformer


def select_checkpoints(run_dir: Path, interval: int = 500) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "training_complete_test_pending":
        raise RuntimeError("training must finish before checkpoint selection")
    if (run_dir / "evaluation.csv").exists():
        raise RuntimeError("cannot select checkpoints after opening the test set")
    destination = run_dir / "checkpoint_selection.json"
    if destination.exists():
        raise FileExistsError("checkpoints have already been selected")
    config = json.loads((run_dir / "resolved_config.json").read_text())
    validation_seeds = json.loads(
        (run_dir / "dataset_manifest.json").read_text()
    )["validation_seeds"]
    generator = BasePaperScenarioGenerator()
    device = torch.device("cpu")
    torch.set_num_threads(4)
    scenarios = [
        generator.generate(config["aerial"], config["ground"], seed)
        for seed in validation_seeds
    ]
    selected: dict[str, dict] = {}
    for method, final_steps in (
        ("baseline", config["baseline_steps"]),
        ("proposed", config["proposed_steps"]),
    ):
        candidates = [
            run_dir / "checkpoints" / f"{method}_step{step:06d}.pt"
            for step in range(interval, final_steps + 1, interval)
        ]
        final_path = run_dir / "checkpoints" / f"{method}_final_step{final_steps:06d}.pt"
        if final_steps % interval != 0 or not candidates:
            candidates.append(final_path)
        measurements = []
        for checkpoint in candidates:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            route_model = RoutingTransformer().to(device)
            route_model.load_state_dict(payload["route_state"])
            route_model.eval()
            route = LearnedRoutePolicy(route_model, device, mode="greedy")
            if method == "baseline":
                assignment = NearestWithLearnedOrderPolicy(
                    route_model, device, mode="greedy"
                )
            else:
                assignment_model = QueueAwareCrossAttention(
                    5, 4, max_slots_per_ugv=max(8, config["uavs"]),
                    zero_queue_penalty=config.get("queue_penalty") == "zero",
                ).to(device)
                assignment_model.load_state_dict(payload["assignment_state"])
                assignment_model.eval()
                assignment = SampledAssignmentPolicy(
                    assignment_model, device, mode="greedy"
                )
            costs, completed = [], []
            with torch.no_grad():
                for scenario in scenarios:
                    result = TwoDMissionSimulator(
                        scenario, config["uavs"], config["ugvs"],
                        assignment, route_policy=route,
                    ).run()
                    costs.append(
                        result.makespan + (800.0 if result.failed else 0.0)
                    )
                    completed.append(result.completed)
            record = {
                "checkpoint": str(checkpoint.relative_to(run_dir)),
                "validation_mean_penalized_cost_min": statistics.fmean(costs),
                "validation_completion_rate": statistics.fmean(completed),
                "validation_instance_count": len(validation_seeds),
            }
            measurements.append(record)
            print(method, checkpoint.name, record["validation_mean_penalized_cost_min"], flush=True)
        selected[method] = {
            "selected": min(
                measurements,
                key=lambda item: item["validation_mean_penalized_cost_min"],
            ),
            "candidates": measurements,
        }
    destination.write_text(json.dumps({
        "selection_rule": "lowest mean penalized mission cost on 256 frozen validation seeds",
        "candidate_interval_updates": interval,
        "methods": selected,
        "test_status": "frozen_unopened",
    }, indent=2))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=int, default=500)
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("interval must be positive")
    print(select_checkpoints(args.run_dir, args.interval))


if __name__ == "__main__":
    main()
