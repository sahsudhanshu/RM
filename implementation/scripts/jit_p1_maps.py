#!/usr/bin/env python
"""Headline combination: sampling + budget filter (P1) with and without just-in-time departure, plus the base policy, on 4 maps x 3 seeds
(2U-1G, Kaggle-trained base policies). Static roads and 4 road events; budget 0.9 B0."""
import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from edr.evaluate import Spec, build_cases, evaluate, load_model
from edr.rl import Setting

ROOT = os.path.join(os.path.dirname(__file__), "..")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
EVENTS = [("block", "medium", "middle"), ("slow", "high", "early"), ("block", "high", "early"), ("slow", "medium", "middle")]
rows = []
for roads in ("loop", "paper", "border", "geo"):
    s = Setting(15, 5, 2, 1, roads)
    for seed in (0, 1, 2):
        tag = "U15G5_2U1G" + ("" if roads == "loop" else f"_{roads}") + (f"_seed{seed}" if seed else "")
        path = os.path.join(ROOT, "results_kaggle", roads, "models", tag, "p0.pt")
        cases = build_cases(s, load_model(path), N)
        for pol, kind in (("P0 base", "neural"), ("P1 sampling + budget filter", "route_filter")):
            for jit in (False, True):
                spec = Spec("x", kind=kind, model_path=path, mode="event" if kind == "neural" else "open", road_aware=True, jit=jit, n_samples=64)
                for sc, grid in (("static", [None]), ("events", EVENTS)):
                    res = []
                    for gg in grid:
                        ev = [None if gg is None else c.event(*gg) for c in cases]
                        res += evaluate(s, cases, spec, beta=0.9, events=ev, workers=10)
                    ok = [x for x in res if not x.get("failed")]
                    rows.append(dict(map=roads, seed=seed, policy=pol, jit=int(jit), scenario=sc, T=np.mean([x["T"] for x in ok]), E=np.mean([x["E"] for x in ok]),
                                     over09=np.mean([x["over"] for x in ok]), n=len(ok), failed=len(res) - len(ok)))
        print(roads, "seed", seed, "done", flush=True)
        with open(os.path.join(ROOT, "results", "jit_p1_maps.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote results/jit_p1_maps.csv")
