#!/usr/bin/env python
"""Sampling + budget filter with a fixed vs adaptive number of samples (2U-1G, 4 maps x 3 seeds, static roads + events, JIT on, beta 0.9)."""
import csv, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from edr.evaluate import Spec, build_cases, evaluate, load_model
from edr.rl import Setting

ROOT = os.path.join(os.path.dirname(__file__), "..")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
EVENTS = [("block", "medium", "middle"), ("slow", "high", "early")]
VARIANTS = [("fixed 16", dict(n_samples=16)), ("fixed 64", dict(n_samples=64)), ("fixed 256", dict(n_samples=256)), ("adaptive (16..256)", dict(adaptive=True, n_max=256))]
rows = []
for roads in ("loop", "paper", "border", "geo"):
    s = Setting(15, 5, 2, 1, roads)
    for seed in (0, 1, 2):
        tag = "U15G5_2U1G" + ("" if roads == "loop" else f"_{roads}") + (f"_seed{seed}" if seed else "")
        path = os.path.join(ROOT, "results_kaggle", roads, "models", tag, "p0.pt")
        cases = build_cases(s, load_model(path), N)
        for name, kw in VARIANTS:
            for sc, grid in (("static", [None]), ("events", EVENTS)):
                t0 = time.time(); res = []
                for gg in grid:
                    ev = [None if gg is None else c.event(*gg) for c in cases]
                    res += evaluate(s, cases, Spec("x", kind="route_filter", model_path=path, mode="open", road_aware=True, jit=True, **kw), beta=0.9, events=ev, workers=10)
                ok = [x for x in res if not x["failed"]]
                rows.append(dict(map=roads, seed=seed, variant=name, scenario=sc, T=np.mean([x["T"] for x in ok]), E=np.mean([x["E"] for x in ok]), over09=np.mean([x["over"] for x in ok]),
                                 samples=np.mean([x["samples"] for x in ok]), wall_s=time.time() - t0, n=len(ok), failed=len(res) - len(ok)))
        print(roads, "seed", seed, "done", flush=True)
        with open(os.path.join(ROOT, "results", "adaptive_sampling.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote results/adaptive_sampling.csv")
