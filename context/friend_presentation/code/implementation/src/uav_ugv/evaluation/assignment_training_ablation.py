"""Compare soft Sinkhorn and row-wise hard ArgMax head training stability.

This is a controlled assignment-head surrogate ablation on baseline-generated
recharge states. It is separate from full mission policy-gradient training.
Both methods use the same states, initial weights, batches, and evaluation
decoder; row-wise ArgMax may select the same slot twice during training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..base_map import BasePaperScenarioGenerator
from ..features import build_model_inputs
from ..mission_simulator import TwoDMissionSimulator
from ..model import QueueAwareCrossAttention
from ..sinkhorn import gumbel_sinkhorn, rectangular_hungarian
from ..training.assignment_policy import NearestWithLearnedOrderPolicy
from ..training.route_policy import LearnedRoutePolicy, RoutingTransformer
from ..virtual_slots import build_slot_matrices


class CaptureAssignments:
    def __init__(self, delegate, records: list[tuple[torch.Tensor, ...]]) -> None:
        self.delegate = delegate
        self.records = records
        self.name = delegate.name

    def assign(self, requests, ugvs, road_network, ugv_speed, recharge_time):
        matrices = build_slot_matrices(
            requests, ugvs, road_network, ugv_speed, recharge_time
        )
        inputs = build_model_inputs(
            requests, ugvs, road_network, matrices, recharge_time
        )
        cost = torch.as_tensor(
            matrices.predicted_completion / 500.0,
            dtype=torch.float32,
        ).unsqueeze(0)
        self.records.append((
            inputs.uav_features[0], inputs.ugv_features[0],
            inputs.normalized_queue_latency[0], cost[0],
        ))
        return self.delegate.assign(
            requests, ugvs, road_network, ugv_speed, recharge_time
        )


def collect_states(config: dict, checkpoint: Path, count: int,
                   seed_start: int) -> list[tuple[torch.Tensor, ...]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    route_model = RoutingTransformer()
    route_model.load_state_dict(payload["route_state"])
    route_model.eval()
    route = LearnedRoutePolicy(route_model, "cpu", mode="greedy")
    order = NearestWithLearnedOrderPolicy(route_model, "cpu", mode="greedy")
    records: list[tuple[torch.Tensor, ...]] = []
    generator = BasePaperScenarioGenerator()
    with torch.no_grad():
        for seed in range(seed_start, seed_start + count):
            scenario = generator.generate(config["aerial"], config["ground"], seed)
            TwoDMissionSimulator(
                scenario, config["uavs"], config["ugvs"],
                CaptureAssignments(order, records), route_policy=route,
            ).run()
    return records


def _batch(records: list[tuple[torch.Tensor, ...]], indices) -> tuple[torch.Tensor, ...]:
    return tuple(torch.stack([records[int(i)][field] for i in indices])
                 for field in range(4))


def _validation_cost(model, records: list[tuple[torch.Tensor, ...]]) -> float:
    values = []
    model.eval()
    with torch.no_grad():
        for inputs in records:
            uav, ugv, queue, cost = (value.unsqueeze(0) for value in inputs)
            scores = model(uav, ugv, queue)[0].numpy()
            rows, columns = rectangular_hungarian(-scores)
            values.append(float(cost[0, rows, columns].sum()))
    return float(np.mean(values))


def ablate(run_dir: Path, output_dir: Path, updates: int = 100,
           train_scenarios: int = 64, validation_scenarios: int = 16,
           target_aerial: int | None = None,
           target_ground: int | None = None) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    config = json.loads((run_dir / "resolved_config.json").read_text())
    config["aerial"] = target_aerial or config["aerial"]
    config["ground"] = target_ground or config["ground"]
    configuration = (
        f"U{config['aerial']}G{config['ground']}_"
        f"{config['uavs']}U{config['ugvs']}G"
    )
    selection = json.loads((run_dir / "checkpoint_selection.json").read_text())
    checkpoint = run_dir / selection["methods"]["baseline"]["selected"]["checkpoint"]
    torch.set_num_threads(4)
    train = collect_states(config, checkpoint, train_scenarios, 3_100_000_000)
    validation = collect_states(
        config, checkpoint, validation_scenarios, 3_200_000_000
    )
    if not train or not validation:
        raise RuntimeError("no recharge states were collected")
    rows = []
    summaries = []
    for seed in (11, 22, 33):
        torch.manual_seed(seed)
        initial = QueueAwareCrossAttention(
            5, 4, max_slots_per_ugv=max(8, config["uavs"])
        ).state_dict()
        batches = np.random.default_rng(seed).integers(
            0, len(train), size=(updates, 16)
        )
        for method in ("sinkhorn_soft", "hard_argmax"):
            model = QueueAwareCrossAttention(
                5, 4, max_slots_per_ugv=max(8, config["uavs"])
            )
            model.load_state_dict(initial)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            losses, gradients, collisions = [], [], []
            nonfinite = 0
            for update, indices in enumerate(batches, start=1):
                uav, ugv, queue, cost = _batch(train, indices)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                scores = model(uav, ugv, queue)
                if method == "sinkhorn_soft":
                    matrix = gumbel_sinkhorn(
                        scores, temperature=0.5, iterations=20,
                        add_noise=False,
                    )[:, :uav.shape[1], :]
                    collision = 0.0
                else:
                    probabilities = F.softmax(scores / 0.5, dim=-1)
                    columns = probabilities.argmax(dim=-1)
                    hard = F.one_hot(
                        columns, num_classes=scores.shape[-1]
                    ).to(probabilities.dtype)
                    matrix = hard - probabilities.detach() + probabilities
                    collision = float((
                        columns.unique(dim=1).shape[1] < columns.shape[1]
                    )) if columns.shape[0] == 1 else float(np.mean([
                        len(set(row.tolist())) < len(row)
                        for row in columns.detach().cpu()
                    ]))
                loss = (matrix * cost).sum(dim=(-2, -1)).mean()
                if not torch.isfinite(loss):
                    nonfinite += 1
                    continue
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    nonfinite += 1
                    continue
                optimizer.step()
                value = float(loss.detach())
                grad = float(norm.detach())
                losses.append(value)
                gradients.append(grad)
                collisions.append(collision)
                rows.append({
                    "run_id": run_dir.name, "configuration": configuration,
                    "seed": seed, "method": method,
                    "update": update, "train_surrogate_loss": value,
                    "gradient_norm_before_clip": grad,
                    "argmax_collision_rate": collision,
                })
            summaries.append({
                "run_id": run_dir.name, "configuration": configuration,
                "seed": seed, "method": method,
                "updates": updates, "nonfinite_updates": nonfinite,
                "tail_loss_std": float(np.std(losses[-20:])) if losses else math.nan,
                "p95_gradient_norm": float(np.quantile(gradients, .95))
                if gradients else math.nan,
                "mean_argmax_collision_rate": float(np.mean(collisions))
                if collisions else math.nan,
                "validation_hungarian_cost": _validation_cost(model, validation),
            })
    with (output_dir / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output_dir / "manifest.json").write_text(json.dumps({
        "status": "completed", "source_run": run_dir.name,
        "target_aerial": config["aerial"],
        "target_ground": config["ground"],
        "baseline_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "ablation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_scenario_seed_start": 3_100_000_000,
        "train_scenario_count": train_scenarios,
        "validation_scenario_seed_start": 3_200_000_000,
        "validation_scenario_count": validation_scenarios,
        "train_recharge_states": len(train),
        "validation_recharge_states": len(validation),
        "methods": ["sinkhorn_soft", "hard_argmax"],
        "hard_argmax_note": "Row-wise straight-through ArgMax can collide across slots; collision rate is reported.",
        "objective": "mean predicted slot completion time divided by 500 min",
    }, indent=2) + "\n")
    return output_dir / "summary.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--target-aerial", type=int)
    parser.add_argument("--target-ground", type=int)
    args = parser.parse_args()
    print(ablate(args.run_dir, args.output_dir, updates=args.updates,
                 target_aerial=args.target_aerial,
                 target_ground=args.target_ground))


if __name__ == "__main__":
    main()
