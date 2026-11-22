"""Benchmark planning latency and build plots after all matrix runs finish."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results/runs"
ENV = os.environ.copy()
ENV.update({
    "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4", "PYTHONPATH": "src",
    "MPLCONFIGDIR": "/tmp/uav_ugv_matrix_matplotlib",
})


def status(path: Path) -> str | None:
    manifest = path / "manifest.json"
    return json.loads(manifest.read_text())["status"] if manifest.exists() else None


def main() -> None:
    entries = []
    for aerial, ground in ((15, 5), (45, 15), (75, 25)):
        for uavs, ugvs in ((2, 1), (2, 2), (4, 2)):
            tag = f"U{aerial}G{ground}_{uavs}U{ugvs}G"
            if tag == "U15G5_2U2G":
                primary = RUNS / "20261010_U15G5_2U2G_pair_seed11_proposed"
                zero = RUNS / f"20261011_{tag}_gamma0_seed11"
            elif tag == "U15G5_4U2G":
                primary = RUNS / "20261010_U15G5_4U2G_pair_seed11"
                zero = RUNS / f"20261011_{tag}_gamma0_seed11"
            else:
                primary = RUNS / f"20261011_{tag}_pair_seed11"
                zero = RUNS / f"20261011_{tag}_gamma0_seed11"
            source = None
            expected = "evaluated"
            date = "20261010" if tag in {"U15G5_2U2G", "U15G5_4U2G"} else "20261011"
            ablation_label = (
                "trained_assignment_training_ablation"
                if aerial == 75 else "assignment_training_ablation"
            )
            ablation = RUNS / f"{date}_{tag}_{ablation_label}"
            entries.append((primary, zero, source, ablation, expected))
    while any(
        status(primary) != expected or status(zero) != expected
        or not (primary / "frozen_gamma_toggle_summary.json").exists()
        or not (ablation / "summary.csv").exists()
        for primary, zero, _, ablation, expected in entries
    ):
        time.sleep(30)
    print("all policy evaluations and ablations complete; benchmarking latency", flush=True)
    for primary, _, source, _, _ in entries:
        if (primary / "latency_benchmark.csv").exists():
            continue
        command = [
            sys.executable, "-m", "uav_ugv.evaluation.latency_benchmark",
            str(primary), "--instances", "64",
        ]
        if source is not None:
            command += ["--source-run", str(source)]
        subprocess.run(command, cwd=ROOT / "implementation", env=ENV, check=True)
    subprocess.run([sys.executable, "results/plot_matrix.py"],
                   cwd=ROOT, env=ENV, check=True)
    subprocess.run([sys.executable, "results/write_matrix_report.py"],
                   cwd=ROOT, env=ENV, check=True)
    subprocess.run([sys.executable, "presentation/end_term/build.py"],
                   cwd=ROOT, env=ENV, check=True)
    print("matrix figures, tables, and presentation ready", flush=True)


if __name__ == "__main__":
    main()
