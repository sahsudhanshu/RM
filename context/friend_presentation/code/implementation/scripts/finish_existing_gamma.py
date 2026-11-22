"""Complete validation and test evaluation after an existing gamma=0 run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    fleet = sys.argv[1]
    run = ROOT / "results/runs" / f"20261011_U15G5_{fleet}_gamma0_seed11"
    manifest = run / "manifest.json"
    while json.loads(manifest.read_text())["status"] == "training":
        time.sleep(10)
    env = os.environ.copy()
    env.update({
        "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4", "PYTHONPATH": "src",
    })
    if not (run / "checkpoint_selection.json").exists():
        subprocess.run([
            sys.executable, "-m", "uav_ugv.evaluation.select_checkpoints",
            str(run), "--interval", "100",
        ], cwd=ROOT / "implementation", env=env, check=True)
    if json.loads(manifest.read_text())["status"] == "training_complete_test_pending":
        subprocess.run([
            sys.executable, "-m", "uav_ugv.evaluation.evaluate_pair", str(run),
        ], cwd=ROOT / "implementation", env=env, check=True)
    print(f"completed {run.name}", flush=True)


if __name__ == "__main__":
    main()
