#!/usr/bin/env python
"""Loop-detection fallback: does handing a stalled mission to the rule policy repair the bad seed, and does it leave the
good seeds untouched? Loop map 2U-1G, Kaggle-trained base policies, 3 seeds, three problem sizes."""
import csv, multiprocessing as mp, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from edr.evaluate import load_model
from edr.heuristics import run_policy
from edr.rl import NeuralPolicy, Setting, new_env
from edr.scenario import generate_scenario

ROOT = os.path.join(os.path.dirname(__file__), "..")
SIZES = [(15, 5), (45, 15), (75, 25)]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def one(a):
    torch.set_num_threads(1)
    seed, A, G, k, fb = a
    path = os.path.join(ROOT, "results_kaggle", "loop", "models", "U15G5_2U1G" + (f"_seed{seed}" if seed else ""), "p0.pt")
    r = run_policy(new_env(Setting(A, G, 2, 1, "loop"), generate_scenario(2_000_000_000 + k, A, G, "loop")), NeuralPolicy(load_model(path), fallback=fb))
    return r.makespan, int(r.failed)


rows = []
with mp.get_context("fork").Pool(10) as pool:
    for seed in (0, 1, 2):
        for (A, G) in SIZES:
            for fb in (False, True):
                res = pool.map(one, [(seed, A, G, k, fb) for k in range(N)], chunksize=1)
                T = np.array([r[0] for r in res]); F = sum(r[1] for r in res)
                rows.append(dict(seed=seed, size=f"U{A}G{G}", fallback=int(fb), mean_T=T.mean(), median_T=np.median(T), failed=F, n=N))
                print(f"seed {seed} U{A}G{G} fallback={'on ' if fb else 'off'}: mean T {T.mean():7.1f}  median {np.median(T):6.1f}  failed {F}/{N}", flush=True)
with open(os.path.join(ROOT, "results", "fallback_study.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
