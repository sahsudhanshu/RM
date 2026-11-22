#!/usr/bin/env python
"""Run the whole multi-map, multi-seed study (train + evaluate). Meant for a Kaggle CPU notebook but runs anywhere.

  python scripts/kaggle_run.py --maps border geo loop paper --seeds 0 1 2 --procs 4
  python scripts/kaggle_run.py --quick --maps border --seeds 0     # smoke test (minutes)

Phases (each phase runs up to --procs jobs in parallel; finished jobs are skipped, so re-running resumes):
  1. P0 base policy (imitation warm start + REINFORCE)      per (map, seed)
  2. fine-tuned variants: p0c p2 p3_b90 p4 p4_nb [p4_ev]    per (map, seed)
  3. evaluation: Stage A (static budgets), B (blocked/slow roads), S (stress test)   per (map, seed)
Outputs stay under results/ (models, logs, stage CSVs) and are zipped to --zip at the end.
"""
import argparse, os, subprocess, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


WAIT = "hover"


def tag(roads, uav, ugv, seed):
    base = f"U15G5_{uav}U{ugv}G" + ("" if roads == "loop" else f"_{roads}") + ("_landed" if WAIT == "landed" else "")
    return base + (f"_seed{seed}" if seed else "")


def run(cmd, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t = time.time()
    with open(log, "a") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"[{time.strftime('%H:%M:%S')}] rc={rc} {time.time()-t:6.0f}s  {' '.join(cmd[1:6])} ...", flush=True)
    return rc


def phase(name, jobs, procs):
    jobs = [j for j in jobs if j]
    print(f"== {name}: {len(jobs)} jobs, {procs} parallel", flush=True)
    with ThreadPoolExecutor(procs) as ex:
        list(ex.map(lambda j: run(*j), jobs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=["border", "geo", "loop", "paper"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--uav", type=int, default=2)
    ap.add_argument("--ugv", type=int, default=1)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--imit", type=int, default=800)
    ap.add_argument("--rl", type=int, default=1200)
    ap.add_argument("--ft", type=int, default=600)
    ap.add_argument("--n", type=int, default=100, help="test instances")
    ap.add_argument("--wait-mode", default="hover", choices=["hover", "landed"])
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--stages", nargs="+", default=["A", "B", "S"])
    ap.add_argument("--quick", action="store_true", help="tiny budgets, to test the pipeline")
    ap.add_argument("--zip", default="results_bundle.zip")
    a = ap.parse_args()
    WAIT = a.wait_mode
    if a.variants is None:      # p4_ev (trained with road events) only for the main hover study
        a.variants = ["p0c", "p2", "p3_b90", "p4", "p4_nb", "p4x"] + (["p4_ev"] if WAIT == "hover" else [])
    if a.quick:
        a.imit, a.rl, a.ft, a.n, a.variants = 20, 10, 10, 8, ["p0c", "p4x"]
    lg = lambda n: os.path.join(ROOT, "results", "logs", n)
    models = os.path.join(ROOT, "results", "models")
    t0 = time.time()

    def p0_job(m, sd):
        if os.path.exists(os.path.join(models, tag(m, a.uav, a.ugv, sd), "p0.pt")):
            return None
        return ([PY, "scripts/train.py", "p0", "--roads", m, "--uav", str(a.uav), "--ugv", str(a.ugv), "--seed", str(sd),
                 "--imit-steps", str(a.imit), "--rl-steps", str(a.rl), "--threads", "1", "--wait-mode", a.wait_mode], lg(f"p0_{tag(m, a.uav, a.ugv, sd)}.log"))

    def ft_job(m, sd, v):
        if os.path.exists(os.path.join(models, tag(m, a.uav, a.ugv, sd), f"{v}.pt")):
            return None
        return ([PY, "scripts/train.py", "ft", "--roads", m, "--uav", str(a.uav), "--ugv", str(a.ugv), "--seed", str(sd),
                 "--ft-steps", str(a.ft), "--threads", "1", "--variant", v, "--wait-mode", a.wait_mode], lg(f"ft_{v}_{tag(m, a.uav, a.ugv, sd)}.log"))

    def ev_job(m, sd, st):
        out = os.path.join(ROOT, "results", tag(m, a.uav, a.ugv, 0), f"stage{st}" + (f"_seed{sd}" if sd else "") + ".csv")
        if os.path.exists(out):
            return None
        return ([PY, "scripts/experiments.py", st, "--roads", m, "--uav", str(a.uav), "--ugv", str(a.ugv), "--seed", str(sd),
                 "--n", str(a.n), "--workers", "1", "--wait-mode", a.wait_mode], lg(f"eval{st}_{tag(m, a.uav, a.ugv, sd)}.log"))

    pairs = [(m, sd) for m in a.maps for sd in a.seeds]
    phase("P0 training", [p0_job(m, sd) for m, sd in pairs], a.procs)
    phase("variant fine-tuning", [ft_job(m, sd, v) for m, sd in pairs for v in a.variants], a.procs)
    # evaluation jobs are CPU-bound too; with --workers 1 each, run several at a time
    phase("evaluation", [ev_job(m, sd, st) for m, sd in pairs for st in a.stages], a.procs)
    with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(os.path.join(ROOT, "results")):
            for f in files:
                z.write(os.path.join(base, f), os.path.relpath(os.path.join(base, f), ROOT))
    print(f"done in {(time.time()-t0)/60:.1f} min -> {a.zip}", flush=True)
