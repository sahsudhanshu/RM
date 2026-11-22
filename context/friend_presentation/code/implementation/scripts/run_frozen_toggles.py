"""Run the direct gamma toggle after each listed primary run is evaluated."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results/runs"


def main() -> None:
    env = os.environ.copy()
    env.update({
        "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4", "PYTHONPATH": "src",
    })
    for name in sys.argv[1:]:
        run = RUNS / name
        manifest = run / "manifest.json"
        while not manifest.exists() or json.loads(manifest.read_text())["status"] != "evaluated":
            time.sleep(10)
        if not (run / "frozen_gamma_toggle.csv").exists():
            subprocess.run([
                sys.executable, "-m", "uav_ugv.evaluation.frozen_gamma_toggle",
                str(run),
            ], cwd=ROOT / "implementation", env=env, check=True)
        print(f"completed frozen toggle: {name}", flush=True)


if __name__ == "__main__":
    main()
