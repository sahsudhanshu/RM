"""Finish one already-started baseline cell and its two proposed variants."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = ROOT / "implementation"
RUNS = ROOT / "results/runs"


def run(*args: str) -> None:
    env = os.environ.copy()
    env.update({
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4",
        "PYTHONPATH": "src",
    })
    print("running", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=IMPLEMENTATION, env=env, check=True)


def status(path: Path) -> str | None:
    manifest = path / "manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text())["status"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aerial", type=int)
    parser.add_argument("ground", type=int)
    parser.add_argument("uavs", type=int)
    parser.add_argument("ugvs", type=int)
    args = parser.parse_args()
    tag = f"U{args.aerial}G{args.ground}_{args.uavs}U{args.ugvs}G"
    baseline = RUNS / f"20261011_{tag}_baseline_seed11"
    while status(baseline) != "baseline_ready_for_validation":
        time.sleep(10)
    selection_path = baseline / "baseline_validation_selection.json"
    if not selection_path.exists():
        run("-m", "uav_ugv.evaluation.select_baseline_candidate", str(baseline))
    baseline_checkpoint = json.loads(selection_path.read_text())["selected"]["checkpoint"]
    def finish_variant(penalty: str) -> None:
        label = "pair" if penalty == "learned" else "gamma0"
        pair = RUNS / f"20261011_{tag}_{label}_seed11"
        if not pair.exists():
            run(
                "-m", "uav_ugv.training.train_pair",
                "--aerial", str(args.aerial), "--ground", str(args.ground),
                "--uavs", str(args.uavs), "--ugvs", str(args.ugvs),
                "--seed", "11", "--teacher-steps", "0",
                "--baseline-steps", "0", "--proposed-steps", "1000",
                "--batch-size", "4", "--learning-rate", "0.00003",
                "--imitation-weight", "0.2", "--checkpoint-interval", "100",
                "--device", "cpu", "--cpu-threads", "4",
                "--queue-penalty", penalty,
                "--warm-start-route", baseline_checkpoint,
                "--run-dir", str(pair),
            )
        elif status(pair) == "training":
            run(
                "-m", "uav_ugv.training.train_pair",
                "--aerial", str(args.aerial), "--ground", str(args.ground),
                "--uavs", str(args.uavs), "--ugvs", str(args.ugvs),
                "--seed", "11", "--teacher-steps", "0",
                "--baseline-steps", "0", "--proposed-steps", "1000",
                "--batch-size", "4", "--learning-rate", "0.00003",
                "--imitation-weight", "0.2", "--checkpoint-interval", "100",
                "--device", "cpu", "--cpu-threads", "4",
                "--queue-penalty", penalty,
                "--warm-start-route", baseline_checkpoint,
                "--run-dir", str(pair), "--resume",
            )
        if not (pair / "checkpoint_selection.json").exists():
            run("-m", "uav_ugv.evaluation.select_checkpoints", str(pair),
                "--interval", "100")
        if status(pair) == "training_complete_test_pending":
            run("-m", "uav_ugv.evaluation.evaluate_pair", str(pair))
        if status(pair) != "evaluated":
            raise RuntimeError(f"pair run did not finish: {pair}")
    if args.aerial == 75:
        # The variants write independent archives. Six four-thread CPU jobs
        # across three fleets fit the incident-aware machine limits.
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(finish_variant, penalty)
                       for penalty in ("learned", "zero")]
            for future in futures:
                future.result()
    else:
        for penalty in ("learned", "zero"):
            finish_variant(penalty)
    learned = RUNS / f"20261011_{tag}_pair_seed11"
    ablation_label = (
        "trained_assignment_training_ablation"
        if args.aerial == 75 else "assignment_training_ablation"
    )
    ablation = RUNS / f"20261011_{tag}_{ablation_label}"
    if not ablation.exists():
        run("-m", "uav_ugv.evaluation.assignment_training_ablation",
            str(learned), str(ablation))
    print(f"completed {tag}", flush=True)


if __name__ == "__main__":
    main()
