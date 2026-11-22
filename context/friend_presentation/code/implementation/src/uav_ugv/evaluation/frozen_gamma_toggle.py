"""Isolate the direct queue-penalty effect with all learned weights frozen."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..base_map import BasePaperScenarioGenerator
from ..mission_simulator import TwoDMissionSimulator
from ..model import QueueAwareCrossAttention
from ..training.assignment_policy import SampledAssignmentPolicy
from ..training.route_policy import LearnedRoutePolicy, RoutingTransformer
from .evaluate_pair import _metrics, _scenario_digest


def frozen_toggle(run_dir: Path, count: int = 256,
                  target_aerial: int | None = None,
                  target_ground: int | None = None,
                  output_dir: Path | None = None) -> Path:
    archive = output_dir or run_dir
    destination = archive / "frozen_gamma_toggle.csv"
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "evaluated":
        raise RuntimeError("select and evaluate the proposed policy first")
    config = json.loads((run_dir / "resolved_config.json").read_text())
    aerial = target_aerial or config["aerial"]
    ground = target_ground or config["ground"]
    if config.get("queue_penalty", "learned") != "learned":
        raise ValueError("source must have a learned queue coefficient")
    selected = json.loads((run_dir / "checkpoint_selection.json").read_text())
    checkpoint = run_dir / selected["methods"]["proposed"]["selected"]["checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    torch.set_num_threads(4)
    route_model = RoutingTransformer()
    route_model.load_state_dict(payload["route_state"])
    route_model.eval()
    route = LearnedRoutePolicy(route_model, "cpu", mode="greedy")
    heads = {}
    for mode, zero in (("learned_gamma", False), ("fixed_zero_same_weights", True)):
        head = QueueAwareCrossAttention(
            5, 4, max_slots_per_ugv=max(8, config["uavs"]),
            zero_queue_penalty=zero,
        )
        head.load_state_dict(payload["assignment_state"])
        head.eval()
        heads[mode] = SampledAssignmentPolicy(head, "cpu", mode="greedy")
    rows = []
    generator = BasePaperScenarioGenerator()
    with torch.no_grad():
        for seed in range(2_100_000_000, 2_100_000_000 + count):
            scenario = generator.generate(aerial, ground, seed)
            digest = _scenario_digest(scenario)
            for mode, assignment in heads.items():
                result = TwoDMissionSimulator(
                    scenario, config["uavs"], config["ugvs"],
                    assignment, route_policy=route,
                ).run()
                rows.append({
                    "instance_seed": seed,
                    "scenario_sha256": digest,
                    "mode": mode,
                    **_metrics(result),
                })
    with destination.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_mode = {mode: [row for row in rows if row["mode"] == mode] for mode in heads}
    differences = np.array([
        learned["penalized_cost_min"] - zero["penalized_cost_min"]
        for learned, zero in zip(
            by_mode["learned_gamma"], by_mode["fixed_zero_same_weights"]
        )
    ])
    rng = np.random.default_rng(37)
    indices = rng.integers(0, count, (2000, count))
    summary = {
        "status": "frozen_weight_gamma_toggle",
        "source_checkpoint": str(checkpoint.relative_to(run_dir)),
        "source_run_id": run_dir.name,
        "target_aerial": aerial,
        "target_ground": ground,
        "learned_gamma_value": float(F.softplus(
            payload["assignment_state"]["raw_queue_weight"]
        )),
        "scenario_seed_start": 2_100_000_000,
        "instance_count": count,
        "learned_minus_zero_cost_min": float(differences.mean()),
        "paired_bootstrap_95_ci_min": np.quantile(
            differences[indices].mean(axis=1), [.025, .975]
        ).tolist(),
        "methods": {
            mode: {
                "mean_penalized_cost_min": statistics.fmean(
                    row["penalized_cost_min"] for row in method_rows
                ),
                "mean_instance_wait_per_event_min": statistics.fmean(
                    row["mean_wait_per_event_min"] for row in method_rows
                ),
                "completion_rate": statistics.fmean(
                    row["status"] == "completed" for row in method_rows
                ),
            }
            for mode, method_rows in by_mode.items()
        },
    }
    (archive / "frozen_gamma_toggle_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--instances", type=int, default=256)
    parser.add_argument("--target-aerial", type=int)
    parser.add_argument("--target-ground", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(frozen_toggle(args.run_dir, args.instances,
                        args.target_aerial, args.target_ground, args.output_dir))


if __name__ == "__main__":
    main()
