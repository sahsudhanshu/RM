"""Choose a baseline warm start from an interrupted pilot using validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from ..base_map import BasePaperScenarioGenerator
from ..mission_simulator import TwoDMissionSimulator
from ..training.assignment_policy import NearestWithLearnedOrderPolicy
from ..training.route_policy import LearnedRoutePolicy, RoutingTransformer


def select(run_dir: Path) -> Path:
    output = run_dir / "baseline_validation_selection.json"
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((run_dir / "resolved_config.json").read_text())
    seeds = json.loads((run_dir / "dataset_manifest.json").read_text())[
        "validation_seeds"
    ]
    candidates = sorted((run_dir / "checkpoints").glob("baseline_step*.pt"))
    teacher = sorted((run_dir / "checkpoints").glob("teacher_step*.pt"))
    if teacher:
        candidates.insert(0, teacher[-1])
    if not candidates:
        raise FileNotFoundError("no baseline checkpoints")
    torch.set_num_threads(4)
    generator = BasePaperScenarioGenerator()
    scenarios = [
        generator.generate(config["aerial"], config["ground"], seed)
        for seed in seeds
    ]
    records = []
    for path in candidates:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = RoutingTransformer()
        model.load_state_dict(payload["route_state"])
        model.eval()
        route = LearnedRoutePolicy(model, "cpu", mode="greedy")
        assignment = NearestWithLearnedOrderPolicy(model, "cpu", mode="greedy")
        costs, completed = [], []
        with torch.no_grad():
            for scenario in scenarios:
                result = TwoDMissionSimulator(
                    scenario, config["uavs"], config["ugvs"],
                    assignment, route_policy=route,
                ).run()
                costs.append(result.makespan + (800.0 if result.failed else 0.0))
                completed.append(result.completed)
        record = {
            "checkpoint": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "update": payload["step"],
            "validation_mean_penalized_cost_min": statistics.fmean(costs),
            "validation_completion_rate": statistics.fmean(completed),
            "validation_instance_count": len(seeds),
        }
        records.append(record)
        print(record, flush=True)
    result = {
        "selection_rule": "lowest penalized mission cost on all frozen validation seeds",
        "selected": min(
            records,
            key=lambda row: row["validation_mean_penalized_cost_min"],
        ),
        "candidates": records,
        "test_status": "frozen_unopened",
    }
    output.write_text(json.dumps(result, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(select(args.run_dir))


if __name__ == "__main__":
    main()
