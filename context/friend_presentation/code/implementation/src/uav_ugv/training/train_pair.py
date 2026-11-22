"""Train a shared routing baseline and its queue-aware assignment extension.

This command trains on generated instances and writes checkpoints. It does not
evaluate on the frozen test set; that comparison is a separate final step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..base_map import BasePaperScenarioGenerator
from ..mission_simulator import TwoDMissionSimulator
from ..model import QueueAwareCrossAttention
from .assignment_policy import NearestWithLearnedOrderPolicy, SampledAssignmentPolicy
from .route_policy import LearnedRoutePolicy, RoutingTransformer


@dataclass(frozen=True)
class PairTrainingConfig:
    aerial: int = 15
    ground: int = 5
    uavs: int = 2
    ugvs: int = 2
    seed: int = 11
    teacher_steps: int = 100
    baseline_steps: int = 200
    proposed_steps: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-4
    checkpoint_interval: int = 25
    device: str = "cuda"
    cpu_threads: int = 4
    gpu_memory_fraction: float = 0.65
    imitation_weight: float = 0.2
    warm_start_route: str | None = None
    baseline_only: bool = False
    queue_penalty: str = "learned"
    validation_seed_start: int = 1_000_000_000
    test_seed_start: int = 2_000_000_000
    proposed_seed_offset: int = 12_000_000
    save_initial_proposed: bool = False
    save_initial_baseline: bool = False
    self_greedy_proposed_reference: bool = False


def _atomic_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _cost(result) -> float:
    return result.makespan + (800.0 if result.failed else 0.0)


def _run_mission(scenario, config, route, assignment):
    return TwoDMissionSimulator(
        scenario=scenario,
        number_of_uavs=config.uavs,
        number_of_ugvs=config.ugvs,
        assignment_policy=assignment,
        route_policy=route,
    ).run()


def _record(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _snapshot(
    run_dir: Path, stage: str, step: int, route_model,
    assignment_model, optimizer, config,
) -> Path:
    destination = run_dir / "checkpoints" / f"{stage}_step{step:06d}.pt"
    payload = {
        "stage": stage,
        "step": step,
        "config": asdict(config),
        "route_state": route_model.state_dict(),
        "assignment_state": (
            assignment_model.state_dict() if assignment_model is not None else None
        ),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    _atomic_save(payload, destination)
    return destination


def _resume_stage(stage: str, run_dir: Path, route_model, assignment_model, optimizer, device) -> int:
    snapshots = sorted((run_dir / "checkpoints").glob(f"{stage}_step*.pt"))
    if not snapshots:
        return 0
    payload = torch.load(snapshots[-1], map_location=device, weights_only=False)
    route_model.load_state_dict(payload["route_state"])
    if assignment_model is not None:
        assignment_model.load_state_dict(payload["assignment_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    if device.type == "cuda" and "cuda_rng_state" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return int(payload["step"])


def train_pair(config: PairTrainingConfig, run_dir: Path, resume: bool = False) -> None:
    if min(config.aerial, config.ground, config.uavs, config.ugvs,
           config.batch_size, config.checkpoint_interval) < 1:
        raise ValueError("scenario, fleet, batch size and checkpoint interval must be positive")
    if min(config.teacher_steps, config.baseline_steps, config.proposed_steps) < 0:
        raise ValueError("training step counts cannot be negative")
    if max(config.teacher_steps, config.baseline_steps, config.proposed_steps) * config.batch_size >= 6_000_000:
        raise ValueError("each stage must use fewer than six million unique instance seeds")
    if config.cpu_threads < 1 or not 0 < config.gpu_memory_fraction <= 1:
        raise ValueError("CPU threads and GPU memory fraction must be valid")
    if config.imitation_weight < 0:
        raise ValueError("imitation weight cannot be negative")
    if config.queue_penalty not in {"learned", "zero"}:
        raise ValueError("queue_penalty must be learned or zero")
    if config.validation_seed_start < 0 or config.test_seed_start < 0:
        raise ValueError("split seed starts must be non-negative")
    if config.proposed_seed_offset < 0:
        raise ValueError("proposed seed offset must be non-negative")
    training_ranges = [
        (config.seed * 20_000_000 + offset,
         config.seed * 20_000_000 + offset + steps * config.batch_size)
        for offset, steps in (
            (0, config.teacher_steps),
            (6_000_000, config.baseline_steps),
            (config.proposed_seed_offset, config.proposed_steps),
        ) if steps
    ]
    for split_start in (config.validation_seed_start, config.test_seed_start):
        if any(start < split_start + 256 and split_start < end
               for start, end in training_ranges):
            raise ValueError("training instance seeds overlap a frozen split")
    if abs(config.validation_seed_start - config.test_seed_start) < 256:
        raise ValueError("validation and test seeds overlap")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a GPU-accessible environment")
    torch.set_num_threads(config.cpu_threads)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    if resume:
        saved_config = json.loads((run_dir / "resolved_config.json").read_text())
        if saved_config != asdict(config):
            raise ValueError("resume configuration differs from the original run")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir()
        (run_dir / "resolved_config.json").write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )
        (run_dir / "dataset_manifest.json").write_text(
        json.dumps({
            "train_seed_rule": "seed*20000000 + stage_offset + update*batch_size + item",
            "stage_offsets": {"teacher": 0, "baseline": 6000000,
                              "proposed": config.proposed_seed_offset},
            "validation_seeds": list(range(
                config.validation_seed_start, config.validation_seed_start + 256
            )),
            "test_seeds": list(range(
                config.test_seed_start, config.test_seed_start + 256
            )),
            "test_status": "frozen_unopened",
        }, indent=2), encoding="utf-8"
        )
        (run_dir / "environment.txt").write_text(
            f"python_torch={torch.__version__}\ndevice={device}\n"
            f"gpu={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'none'}\n",
            encoding="utf-8",
        )
        source_root = Path(__file__).resolve().parents[1]
        source_paths = [
            source_root / name for name in (
                "base_map.py", "mission_simulator.py", "model.py",
                "policies.py", "simulator.py", "sinkhorn.py",
                "virtual_slots.py", "training/train_pair.py",
                "training/route_policy.py", "training/assignment_policy.py",
            )
        ]
        source_copy = run_dir / "source"
        hashes = {}
        for path in source_paths:
            relative = path.relative_to(source_root)
            destination = source_copy / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (run_dir / "manifest.json").write_text(
            json.dumps({
                "run_id": run_dir.name,
                "status": "training",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": hashes,
                "units": {"distance": "km", "time": "min", "energy": "joule"},
                "methods": [
                    "transformer_route_nearest_ugv_learned_order",
                    "transformer_route_queue_aware_matching",
                ],
                "test_status": "frozen_unopened",
            }, indent=2), encoding="utf-8",
        )
    generator = BasePaperScenarioGenerator()
    baseline_model = RoutingTransformer().to(device)
    if config.warm_start_route:
        warm_start_path = Path(config.warm_start_route)
        warm_start = torch.load(
            warm_start_path, map_location=device, weights_only=False
        )
        baseline_model.load_state_dict(warm_start["route_state"])
        if not resume:
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["warm_start_checkpoint"] = str(warm_start_path.resolve())
            manifest["warm_start_sha256"] = hashlib.sha256(
                warm_start_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2))
    baseline_optimizer = torch.optim.Adam(
        baseline_model.parameters(), lr=config.learning_rate
    )
    history = run_dir / "training.jsonl"

    def train_stage(
        stage: str, steps: int, route_model, assignment_model, optimizer,
        sample_route, greedy_route, sample_assignment, greedy_assignment,
        offset: int, start_update: int,
    ) -> None:
        route_model.train()
        if assignment_model is not None:
            assignment_model.train()
        anchor_route = LearnedRoutePolicy(route_model, device, mode="teacher")
        anchor_order = (
            NearestWithLearnedOrderPolicy(route_model, device, mode="teacher")
            if assignment_model is None else None
        )
        for update in range(start_update, steps):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            costs, baseline_costs, action_counts = [], [], []
            completed, baseline_completed = [], []
            for item in range(config.batch_size):
                instance_seed = (
                    config.seed * 20_000_000 + offset
                    + update * config.batch_size + item
                )
                scenario = generator.generate(config.aerial, config.ground, instance_seed)
                if hasattr(sample_assignment, "start_episode"):
                    sample_assignment.start_episode()
                result = _run_mission(
                    scenario, config, sample_route, sample_assignment
                )
                route_log_probability = sample_route.episode_log_probability()
                if stage == "teacher":
                    count = max(
                        1, len(sample_route.log_probabilities)
                        + len(sample_assignment.log_probabilities)
                    )
                    loss = -(
                        route_log_probability
                        + sample_assignment.episode_log_probability()
                    ) / count / config.batch_size
                else:
                    with torch.no_grad():
                        baseline_result = _run_mission(
                            scenario, config, greedy_route, greedy_assignment
                        )
                    baseline_cost = _cost(baseline_result)
                    baseline_costs.append(baseline_cost)
                    baseline_completed.append(float(baseline_result.completed))
                    log_probability = route_log_probability
                    if hasattr(sample_assignment, "episode_log_probability"):
                        log_probability = (
                            log_probability
                            + sample_assignment.episode_log_probability()
                        )
                    # Scale the advantage for gradient stability; this does not
                    # change the expected policy-gradient direction.
                    loss = (
                        (_cost(result) - baseline_cost) / 100.0
                        * log_probability / config.batch_size
                    )
                loss.backward()
                if stage != "teacher" and config.imitation_weight > 0:
                    if anchor_order is not None:
                        anchor_order.start_episode()
                    _run_mission(
                        scenario, config, anchor_route,
                        anchor_order if anchor_order is not None else greedy_assignment,
                    )
                    anchor_log_probability = anchor_route.episode_log_probability()
                    anchor_count = len(anchor_route.log_probabilities)
                    if anchor_order is not None:
                        anchor_log_probability = (
                            anchor_log_probability
                            + anchor_order.episode_log_probability()
                        )
                        anchor_count += len(anchor_order.log_probabilities)
                    anchor_loss = -(
                        config.imitation_weight * anchor_log_probability
                        / max(1, anchor_count) / config.batch_size
                    )
                    anchor_loss.backward()
                costs.append(_cost(result))
                completed.append(float(result.completed))
                action_counts.append(len(sample_route.log_probabilities))
            parameters = list(route_model.parameters())
            if assignment_model is not None:
                parameters += list(assignment_model.parameters())
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            elapsed = time.monotonic() - started
            row = {
                "stage": stage, "update": update + 1,
                "instances": config.batch_size,
                "mean_cost_min": float(np.mean(costs)),
                "completion_rate": float(np.mean(completed)),
                "mean_baseline_cost_min": (
                    float(np.mean(baseline_costs)) if baseline_costs else None
                ),
                "baseline_completion_rate": (
                    float(np.mean(baseline_completed)) if baseline_completed else None
                ),
                "mean_route_actions": float(np.mean(action_counts)),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "wall_seconds": elapsed,
                "peak_cuda_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else 0
                ),
            }
            if not np.isfinite(row["gradient_norm"]):
                raise FloatingPointError("non-finite gradient norm")
            _record(history, row)
            if (update + 1) % config.checkpoint_interval == 0 or update + 1 == steps:
                checkpoint = _snapshot(
                    run_dir, stage, update + 1, route_model,
                    assignment_model, optimizer, config,
                )
                print(
                    f"{stage} {update + 1}/{steps} cost={row['mean_cost_min']:.2f} "
                    f"seconds/update={elapsed:.2f} checkpoint={checkpoint}",
                    flush=True,
                )

    teacher_route = LearnedRoutePolicy(baseline_model, device, mode="teacher")
    teacher_order = NearestWithLearnedOrderPolicy(
        baseline_model, device, mode="teacher"
    )
    greedy_baseline_route = LearnedRoutePolicy(baseline_model, device, mode="greedy")
    greedy_baseline_order = NearestWithLearnedOrderPolicy(
        baseline_model, device, mode="greedy"
    )
    if resume and list((run_dir / "checkpoints").glob("baseline_step*.pt")):
        teacher_start = config.teacher_steps
    else:
        teacher_start = _resume_stage(
            "teacher", run_dir, baseline_model, None, baseline_optimizer, device
        ) if resume else 0
    train_stage(
        "teacher", config.teacher_steps, baseline_model, None, baseline_optimizer,
        teacher_route, greedy_baseline_route, teacher_order, greedy_baseline_order,
        0, teacher_start,
    )
    sampled_baseline_route = LearnedRoutePolicy(
        baseline_model, device, mode="sample"
    )
    sampled_baseline_order = NearestWithLearnedOrderPolicy(
        baseline_model, device, mode="sample"
    )
    if config.save_initial_baseline and not resume:
        initial_checkpoint = _snapshot(
            run_dir, "baseline", 0, baseline_model,
            None, baseline_optimizer, config,
        )
        print(f"baseline_initial={initial_checkpoint}", flush=True)
    baseline_start = _resume_stage(
        "baseline", run_dir, baseline_model, None, baseline_optimizer, device
    ) if resume else 0
    train_stage(
        "baseline", config.baseline_steps, baseline_model, None,
        baseline_optimizer, sampled_baseline_route, greedy_baseline_route,
        sampled_baseline_order, greedy_baseline_order, 6_000_000, baseline_start,
    )
    baseline_checkpoint = _snapshot(
        run_dir, "baseline_final", config.baseline_steps,
        baseline_model, None, baseline_optimizer, config,
    )
    print(f"baseline_frozen={baseline_checkpoint}", flush=True)
    if config.baseline_only:
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "baseline_ready_for_validation"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return

    proposed_route_model = RoutingTransformer().to(device)
    proposed_route_model.load_state_dict(baseline_model.state_dict())
    assignment_model = QueueAwareCrossAttention(
        uav_feature_dim=5,
        ugv_feature_dim=4,
        max_slots_per_ugv=max(8, config.uavs),
        zero_queue_penalty=config.queue_penalty == "zero",
    ).to(device)
    proposed_optimizer = torch.optim.Adam(
        list(proposed_route_model.parameters()) + list(assignment_model.parameters()),
        lr=config.learning_rate,
    )
    if config.save_initial_proposed and not resume:
        initial_checkpoint = _snapshot(
            run_dir, "proposed", 0, proposed_route_model,
            assignment_model, proposed_optimizer, config,
        )
        print(f"proposed_initial={initial_checkpoint}", flush=True)
    sampled_proposed_route = LearnedRoutePolicy(
        proposed_route_model, device, mode="sample"
    )
    sampled_assignment = SampledAssignmentPolicy(
        assignment_model, device, mode="sample"
    )
    proposed_reference_route = (
        LearnedRoutePolicy(proposed_route_model, device, mode="greedy")
        if config.self_greedy_proposed_reference else greedy_baseline_route
    )
    proposed_reference_assignment = (
        SampledAssignmentPolicy(assignment_model, device, mode="greedy")
        if config.self_greedy_proposed_reference else greedy_baseline_order
    )
    proposed_start = _resume_stage(
        "proposed", run_dir, proposed_route_model, assignment_model,
        proposed_optimizer, device,
    ) if resume else 0
    train_stage(
        "proposed", config.proposed_steps, proposed_route_model,
        assignment_model, proposed_optimizer, sampled_proposed_route,
        proposed_reference_route, sampled_assignment,
        proposed_reference_assignment,
        config.proposed_seed_offset, proposed_start,
    )
    proposed_checkpoint = _snapshot(
        run_dir, "proposed_final", config.proposed_steps,
        proposed_route_model, assignment_model, proposed_optimizer, config,
    )
    print(f"proposed_frozen={proposed_checkpoint}", flush=True)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "training_complete_test_pending"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [
        ("aerial", 15), ("ground", 5), ("uavs", 2), ("ugvs", 2),
        ("seed", 11), ("teacher-steps", 100), ("baseline-steps", 200),
        ("proposed-steps", 200), ("batch-size", 4),
        ("checkpoint-interval", 25), ("cpu-threads", 4),
    ]:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.65)
    parser.add_argument("--imitation-weight", type=float, default=0.2)
    parser.add_argument("--warm-start-route")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--queue-penalty", choices=("learned", "zero"), default="learned")
    parser.add_argument("--validation-seed-start", type=int, default=1_000_000_000)
    parser.add_argument("--test-seed-start", type=int, default=2_000_000_000)
    parser.add_argument("--proposed-seed-offset", type=int, default=12_000_000)
    parser.add_argument("--save-initial-proposed", action="store_true")
    parser.add_argument("--save-initial-baseline", action="store_true")
    parser.add_argument("--self-greedy-proposed-reference", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = PairTrainingConfig(
        aerial=args.aerial, ground=args.ground, uavs=args.uavs, ugvs=args.ugvs,
        seed=args.seed, teacher_steps=args.teacher_steps,
        baseline_steps=args.baseline_steps, proposed_steps=args.proposed_steps,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        checkpoint_interval=args.checkpoint_interval, device=args.device,
        cpu_threads=args.cpu_threads,
        gpu_memory_fraction=args.gpu_memory_fraction,
        imitation_weight=args.imitation_weight,
        warm_start_route=args.warm_start_route,
        baseline_only=args.baseline_only,
        queue_penalty=args.queue_penalty,
        validation_seed_start=args.validation_seed_start,
        test_seed_start=args.test_seed_start,
        proposed_seed_offset=args.proposed_seed_offset,
        save_initial_proposed=args.save_initial_proposed,
        save_initial_baseline=args.save_initial_baseline,
        self_greedy_proposed_reference=args.self_greedy_proposed_reference,
    )
    run_dir = args.run_dir or Path("results/runs") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_U{config.aerial}G{config.ground}_{config.uavs}U{config.ugvs}G_pair_seed{config.seed}"
    )
    train_pair(config, run_dir, resume=args.resume)


if __name__ == "__main__":
    main()
