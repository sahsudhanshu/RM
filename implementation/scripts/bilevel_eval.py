#!/usr/bin/env python
"""GLS / Tabu / SA bilevel baselines vs our policies on the same instances (paper-map Table I columns)."""
import argparse, csv, multiprocessing as mp, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from edr.bilevel import run_bilevel
from edr.rl import Setting, new_env
from edr.scenario import generate_scenario

ap = argparse.ArgumentParser()
ap.add_argument("--roads", default="paper"); ap.add_argument("--aerial", type=int, default=15); ap.add_argument("--ground", type=int, default=5)
ap.add_argument("--teams", nargs="+", default=["1,1", "2,1", "2,2", "4,2"]); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--meta", nargs="+", default=["GLS", "TS", "SA"]); ap.add_argument("--ms", type=int, default=800)
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--out", default="results/bilevel_eval.csv")
a = ap.parse_args()


def one(args):
    roads, A, Gd, m, n, k, meta, ms = args
    s = Setting(A, Gd, m, n, roads)
    scn = generate_scenario(2_000_000_000 + k, A, Gd, roads)
    r = run_bilevel(new_env(s, scn), meta, ms)
    return r.makespan, r.energy, int(r.failed), r.wait_time


rows = []
for team in a.teams:
    m, n = map(int, team.split(","))
    for meta in a.meta:
        t0 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(one, [(a.roads, a.aerial, a.ground, m, n, k, meta, a.ms) for k in range(a.n)], chunksize=1)
        ok = [r for r in res if not r[2]]
        T = np.array([r[0] for r in ok]); E = np.array([r[1] for r in ok])
        rows.append(dict(roads=a.roads, size=f"U{a.aerial}G{a.ground}", team=f"{m}U-{n}G", method=meta, T=T.mean(), T_sd=T.std(ddof=1), E=E.mean(), fails=len(res) - len(ok), n=len(res)))
        print(f"{a.roads} U{a.aerial}G{a.ground} {m}U-{n}G {meta:3s}: T {T.mean():6.1f} ± {T.std(ddof=1):5.1f}  E {E.mean():6.0f}  failed {len(res)-len(ok)}/{len(res)}  [{time.time()-t0:.0f}s]", flush=True)
os.makedirs(os.path.dirname(os.path.join(os.path.dirname(__file__), "..", a.out)), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "..", a.out), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
