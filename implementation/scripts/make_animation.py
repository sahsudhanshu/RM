#!/usr/bin/env python
"""Render the demonstration simulation: the SAME instance and road event flown by the base
policy open-loop vs with the road layer + event-triggered replanning.

  python scripts/make_animation.py --kind slow --level high --timing early
Writes results/<tag>/media/mission_<kind>_<level>_<timing>.mp4 (+ .png still, + optional .gif).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edr.evaluate import build_cases, greedy_planner, load_model
from edr.plan import run_controller
from edr.rl import Setting, new_env
from edr.viz import animate_pair, plot_mission

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uav", type=int, default=2)
    ap.add_argument("--ugv", type=int, default=1)
    ap.add_argument("--kind", default="slow")
    ap.add_argument("--level", default="high")
    ap.add_argument("--timing", default="early")
    ap.add_argument("--model", default="p0")
    ap.add_argument("--roads", default=None, help="road map (loop, border, paper, ...); default: the Setting default")
    ap.add_argument("--model-path", default=None, help="explicit p0 checkpoint (e.g. results_kaggle/loop-4u2g/models/U15G5_4U2G/p0.pt); the team demo needs the Kaggle-trained 4U-2G / 2U-2G policies")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None, help="use this test seed instead of the best-gain one")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--frames", type=int, default=240)
    a = ap.parse_args()
    torch.set_num_threads(2)
    s = Setting(n_uav=a.uav, n_ugv=a.ugv, **({"roads": a.roads} if a.roads else {}))
    models = os.path.join(ROOT, "models", s.tag)
    base = load_model(a.model_path or os.path.join(models, "p0.pt"))
    model = base if a.model_path else load_model(os.path.join(models, f"{a.model}.pt"))
    cases = build_cases(s, base, a.n)
    best = None
    for c in cases:
        if a.seed is not None and c.seed != a.seed:
            continue
        ev = c.event(a.kind, a.level, a.timing)
        runs = []
        for mode, aware in (("open", False), ("event", True)):
            env = new_env(s, c.scn, events=(ev,), road_aware=aware)
            runs.append(run_controller(env, greedy_planner(model), mode))
        if a.ugv > 1 and (runs[1][0].failed or len({d[0] for d in runs[1][0].ugv_drives}) < a.ugv):
            continue                                                   # team demo: every truck must be used
        gain = runs[0][0].cost - runs[1][0].cost
        if best is None or gain > best[0]:
            best = (gain, c, ev, runs)
    gain, c, ev, ((r_open, i_open), (r_evt, i_evt)) = best
    budget = 0.9 * c.b0
    print(f"seed {c.seed}: open T={r_open.makespan:.0f} E={r_open.energy:.0f} | "
          f"event T={r_evt.makespan:.0f} E={r_evt.energy:.0f} replans={i_evt.replans} | static T0={c.t0:.0f} B0={c.b0:.0f}")
    out = os.path.join(ROOT, s.tag, "media"); os.makedirs(out, exist_ok=True)
    stem = f"mission_{a.kind}_{a.level}_{a.timing}"
    labels = (f"Open-loop (no replanning)   T = {r_open.makespan:.0f} min,  E = {r_open.energy:.0f} kJ",
              f"Event-triggered replanning ({i_evt.replans} replan)   T = {r_evt.makespan:.0f} min,  E = {r_evt.energy:.0f} kJ")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6), facecolor="white")
    for ax, r, lab in zip(axs, (r_open, r_evt), labels):
        plot_mission(ax, c.scn, r, (ev,), title=lab)
    fig.tight_layout(); fig.savefig(os.path.join(out, stem + ".png"), dpi=130); plt.close(fig)
    p = animate_pair(c.scn, r_open, r_evt, s.params, (ev,), os.path.join(out, stem + ".mp4"),
                     labels=("Open-loop (no replanning)", "Event-triggered replanning"), budget=budget,
                     frames=a.frames)
    print("wrote", p)
    if a.gif:
        print("wrote", animate_pair(c.scn, r_open, r_evt, s.params, (ev,), os.path.join(out, stem + ".gif"),
                                    labels=("Open-loop (no replanning)", "Event-triggered replanning"),
                                    budget=budget, frames=a.frames // 2, fps=12, gif=True))
