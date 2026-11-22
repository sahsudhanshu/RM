#!/usr/bin/env python
"""Just-in-time departure: plain vs re-predicted at takeoff time (env.jit_refine). Base policy, event-triggered + road layer, 2U-1G, 4 maps x 3 seeds,
all 18 road events (block/slow x level x timing)."""
import csv, itertools, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from edr.evaluate import Spec, build_cases, evaluate, load_model
from edr.events import LEVELS, TIMINGS
from edr.rl import Setting

ROOT = os.path.join(os.path.dirname(__file__), "..")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
GRID = list(itertools.product(("block", "slow"), LEVELS, TIMINGS))
rows = []
for roads in ("loop", "paper", "border", "geo"):
    s = Setting(15, 5, 2, 1, roads)
    for seed in (0, 1):
        tag = "U15G5_2U1G" + ("" if roads == "loop" else f"_{roads}") + (f"_seed{seed}" if seed else "")
        path = os.path.join(ROOT, "results_kaggle", roads, "models", tag, "p0.pt")
        cases = build_cases(s, load_model(path), N)
        for name, kw in (("no JIT", dict(jit=False)), ("JIT", dict(jit=True)), ("JIT re-predicted", dict(jit=True, jit_refine=True))):
            res = []
            for gg in GRID:
                ev = [c.event(*gg) for c in cases]
                res += evaluate(s, cases, Spec("x", model_path=path, mode="event", road_aware=True, **kw), beta=0.9, events=ev, workers=10)
            ok = [x for x in res if not x["failed"]]
            rows.append(dict(map=roads, seed=seed, variant=name, T=np.mean([x["T"] for x in ok]), E=np.mean([x["E"] for x in ok]), E_wait=np.mean([x["E_wait"] for x in ok]),
                             over09=np.mean([x["over"] for x in ok]), n=len(ok)))
        print(roads, seed, "done", flush=True)
        with open(os.path.join(ROOT, "results", "jit_refine.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote results/jit_refine.csv")
