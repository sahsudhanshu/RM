#!/usr/bin/env python
"""Just-in-time departure beyond the base policy on one team: (a) other team sizes (loop map, Kaggle base policies, 3 seeds),
(b) the sampling + budget-filter policy P1 (loop map 2U-1G). jit off vs on, same instances/events."""
import csv, itertools, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from edr.evaluate import Spec, build_cases, evaluate, load_model
from edr.events import LEVELS, TIMINGS
from edr.rl import Setting

ROOT = os.path.join(os.path.dirname(__file__), "..")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TEAMS = [("1U-1G", 1, 1, "loop-1u1g", "U15G5_1U1G"), ("2U-1G", 2, 1, "loop", "U15G5_2U1G"), ("2U-2G", 2, 2, "loop-2u2g", "U15G5_2U2G"), ("4U-2G", 4, 2, "loop-4u2g", "U15G5_4U2G")]
rows = []
for team, u, g, root, tag in TEAMS:
    s = Setting(15, 5, u, g, "loop")
    for seed in (0, 1, 2):
        path = os.path.join(ROOT, "results_kaggle", root, "models", tag + (f"_seed{seed}" if seed else ""), "p0.pt")
        cases = build_cases(s, load_model(path), N)
        grid = [None] + list(itertools.product(("block", "slow"), LEVELS, TIMINGS))
        for jit in (False, True):
            spec = Spec("e", model_path=path, mode="event", road_aware=True, jit=jit)
            res = []
            for gg in grid:
                ev = [None if gg is None else c.event(*gg) for c in cases]
                res += [dict(x, scenario="static" if gg is None else "events") for x in evaluate(s, cases, spec, beta=0.9, events=ev, workers=10)]
            for sc in ("static", "events"):
                rr = [x for x in res if x["scenario"] == sc]
                rows.append(dict(policy="P0 event + road layer", team=team, seed=seed, jit=int(jit), scenario=sc, T=np.mean([x["T"] for x in rr]), E=np.mean([x["E"] for x in rr]),
                                 E_wait=np.mean([x["E_wait"] for x in rr]), over09=np.mean([x["over"] for x in rr])))
        print(team, "seed", seed, "done", flush=True)
# (b) P1 with and without JIT (loop 2U-1G, static roads)
s = Setting(15, 5, 2, 1, "loop")
for seed in (0, 1, 2):
    path = os.path.join(ROOT, "results_kaggle", "loop", "models", "U15G5_2U1G" + (f"_seed{seed}" if seed else ""), "p0.pt")
    cases = build_cases(s, load_model(path), 40)
    for jit in (False, True):
        r = evaluate(s, cases, Spec("p1", kind="route_filter", model_path=path, n_samples=64, jit=jit), beta=0.9, workers=10)
        rows.append(dict(policy="P1 sampling + budget filter", team="2U-1G", seed=seed, jit=int(jit), scenario="static", T=np.mean([x["T"] for x in r]), E=np.mean([x["E"] for x in r]),
                         E_wait=np.mean([x["E_wait"] for x in r]), over09=np.mean([x["over"] for x in r])))
    print("P1 seed", seed, "done", flush=True)
with open(os.path.join(ROOT, "results", "jit_teams.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote results/jit_teams.csv")
