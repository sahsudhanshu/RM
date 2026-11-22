#!/usr/bin/env python
"""Milestone-1 check: do the base environment's static-road trends match base paper Table I (U15G5)?

Compares mean mission time of (a) the nearest-feasible rule on the tree map (the base paper's exact
28-point road, reconstructed from the authors' code) and on our loop map, and (b) our trained P0 (greedy) where available, with the
paper's DRL(greedy) column. Absolute numbers are NOT expected to match (different road geometry, T_R and
training budget); the ordering across teams is.
"""
import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from edr.env import MissionEnv
from edr.evaluate import load_model
from edr.heuristics import nearest_feasible, run_policy
from edr.rl import NeuralPolicy, Setting, new_env
from edr.scenario import generate_scenario

PAPER = {(1, 1): 238.2, (2, 1): 229.7, (2, 2): 180.5, (4, 2): 188.9}      # Table I, DRL (greedy), U15G5
PAPER_S = {(1, 1): 218.6, (2, 1): 199.7, (2, 2): 154.4, (4, 2): 159.9}    # DRL (10240)
ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
rows = []
for roads in ("paper", "loop"):
    for (m, n) in PAPER:
        s = Setting(15, 5, m, n, roads)
        T, fails = [], 0
        for k in range(100):
            scn = generate_scenario(2_000_000_000 + k, 15, 5, roads)
            r = run_policy(new_env(s, scn), nearest_feasible)
            fails += r.failed
            if not r.failed:
                T.append(r.makespan)
        if fails:
            print(f"note: rule on {roads} map {m}U-{n}G: {fails} failed instance(s) excluded")
        rows.append(dict(source=f"rule, {roads} map", team=f"{m}U-{n}G", T=np.mean(T), sd=np.std(T, ddof=1)))
for roads, sfx in (("paper", "_paper"), ("loop", "")):
    for (m, n) in PAPER:
        path = os.path.join(ROOT, "models", f"U15G5_{m}U{n}G{sfx}", "p0.pt")
        if os.path.exists(path):
            s = Setting(15, 5, m, n, roads)
            pol = NeuralPolicy(load_model(path))
            T = [run_policy(new_env(s, generate_scenario(2_000_000_000 + k, 15, 5, roads)), pol).makespan for k in range(100)]
            T = [t for t in T if t < 700]                  # drop livelock failures (reported separately)
            rows.append(dict(source=f"P0 greedy (ours), {roads} map", team=f"{m}U-{n}G", T=np.mean(T), sd=np.std(T, ddof=1)))
for (m, n), v in PAPER.items():
    rows.append(dict(source="paper DRL (greedy)", team=f"{m}U-{n}G", T=v, sd=float("nan")))
    rows.append(dict(source="paper DRL (10240)", team=f"{m}U-{n}G", T=PAPER_S[(m, n)], sd=float("nan")))
os.makedirs(os.path.join(ROOT, "validation"), exist_ok=True)
with open(os.path.join(ROOT, "validation", "m1_trends.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["source", "team", "T", "sd"]); w.writeheader(); w.writerows(rows)
teams = [f"{m}U-{n}G" for (m, n) in PAPER]
print(f"{'source':30s}" + "".join(f"{t:>14s}" for t in teams))
for src in dict.fromkeys(r["source"] for r in rows):
    line = f"{src:30s}"
    for t in teams:
        r = [x for x in rows if x["source"] == src and x["team"] == t]
        line += f"{r[0]['T']:>8.1f}" + (f" ±{r[0]['sd']:4.0f}" if r and r[0]['sd'] == r[0]['sd'] else "      ") if r else f"{'-':>14s}"
    print(line)
