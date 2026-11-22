#!/usr/bin/env python
"""Just-in-time departure: does letting a UAV that expects to wait stay on the ground (free) instead of hovering
save energy without slowing the mission? Base policy P0, same instances/events with jit off vs on."""
import itertools, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, csv
from edr.evaluate import Spec, build_cases, evaluate, load_model
from edr.events import LEVELS, TIMINGS
from edr.rl import Setting

ROOT = os.path.join(os.path.dirname(__file__), "..")
MAPS = [("loop", "results_kaggle/loop", "U15G5_2U1G"), ("border", "results_kaggle/border", "U15G5_2U1G_border"),
        ("geo", "results_kaggle/geo", "U15G5_2U1G_geo"), ("paper", "results_kaggle/paper", "U15G5_2U1G_paper")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rows = []
for roads, root, tag in MAPS:
    s = Setting(15, 5, 2, 1, roads)
    for seed in (0, 1, 2):
        path = os.path.join(ROOT, root, "models", tag + (f"_seed{seed}" if seed else ""), "p0.pt")
        base = load_model(path)
        cases = build_cases(s, base, N)
        kinds = ("block", "slow") if any(not cases[0].scn.roads.is_bridge(i) for i in range(len(cases[0].scn.roads.edges))) else ("slow",)
        grid = [None] + list(itertools.product(kinds, LEVELS, TIMINGS))
        for jit in (False, True):
            for name, spec in (("open-loop", Spec("o", model_path=path, mode="open", jit=jit)),
                               ("event-triggered + road layer", Spec("e", model_path=path, mode="event", road_aware=True, jit=jit))):
                res = []
                for g in grid:
                    ev = [None if g is None else c.event(*g) for c in cases]
                    r = evaluate(s, cases, spec, beta=0.9, events=ev, workers=10)
                    res += [dict(x, scenario="static" if g is None else "events") for x in r]
                for sc in ("static", "events"):
                    rr = [x for x in res if x["scenario"] == sc]
                    rows.append(dict(map=roads, seed=seed, controller=name, jit=int(jit), scenario=sc, T=np.mean([x["T"] for x in rr]), E=np.mean([x["E"] for x in rr]),
                                     E_wait=np.mean([x["E_wait"] for x in rr]), over09=np.mean([x["over"] for x in rr]), B0=np.mean([x["B0"] for x in rr]),
                                     E_over_B0=np.mean([x["E"] / x["B0"] for x in rr])))
        print(roads, "seed", seed, "done", flush=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "jit_study.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote results/jit_study.csv")
