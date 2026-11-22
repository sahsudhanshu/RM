#!/usr/bin/env python
"""Sampled decoding as in the base paper's DRL(1024) / DRL(10240): sample N routes per test instance from the
trained base policy and keep the one with the lowest mission time. Mean best time over the test instances
(failed samples ignored). Compare with the paper's Table I (U15G5)."""
import argparse, csv, multiprocessing as mp, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from edr.evaluate import load_model
from edr.rl import Setting, new_env, rollout
from edr.scenario import generate_scenario

ap = argparse.ArgumentParser()
ap.add_argument("--roads", default="paper"); ap.add_argument("--teams", nargs="+", default=["1,1", "2,1", "2,2", "4,2"])
ap.add_argument("--samples", type=int, nargs="+", default=[1024]); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--models-root", default="results"); ap.add_argument("--suffix", default="_paper")
ap.add_argument("--out", default="results/sampling_eval.csv")
a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(__file__), "..")
_M = {}


def one(args):
    torch.set_num_threads(1)
    path, m, n, roads, k, nsamples, chunk = args
    model = _M.setdefault(path, load_model(path))
    s = Setting(15, 5, m, n, roads)
    scn = generate_scenario(2_000_000_000 + k, 15, 5, roads)
    g = new_env(s, scn); rollout(model, [g], "greedy"); rg = g.result()
    best, done = None, 0
    torch.manual_seed(k)
    while done < nsamples:
        c = min(chunk, nsamples - done)
        envs = [new_env(s, scn) for _ in range(c)]
        rollout(model, envs, "sample")
        for e in envs:
            r = e.result()
            if not r.failed and (best is None or r.makespan < best[0]):
                best = (r.makespan, r.energy)
        done += c
    return rg.makespan, rg.failed, best[0] if best else float("nan"), best[1] if best else float("nan")


rows = []
for team in a.teams:
    m, n = map(int, team.split(","))
    path = os.path.join(ROOT, a.models_root, "models", f"U15G5_{m}U{n}G{a.suffix}", "p0.pt")
    for ns in a.samples:
        t0 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(one, [(path, m, n, a.roads, k, ns, a.chunk) for k in range(a.n)], chunksize=1)
        g = np.array([r[0] for r in res if not r[1]]); b = np.array([r[2] for r in res]); e = np.array([r[3] for r in res])
        ok = np.isfinite(b)
        rows.append(dict(roads=a.roads, team=f"{m}U-{n}G", samples=ns, greedy_T=g.mean(), best_T=b[ok].mean(), best_T_sd=b[ok].std(ddof=1),
                         best_E=e[ok].mean(), n=int(ok.sum())))
        print(f"{a.roads} {m}U-{n}G N={ns:6d}: greedy {g.mean():6.1f} | best-of-N {b[ok].mean():6.1f} ± {b[ok].std(ddof=1):4.1f} (E {e[ok].mean():.0f})  [{time.time()-t0:.0f}s]", flush=True)
os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
with open(os.path.join(ROOT, a.out), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
