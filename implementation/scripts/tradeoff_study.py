#!/usr/bin/env python
"""Is there a time-energy trade-off to optimise? Sample many routes per instance from the trained base
policy P0, then score them under different energy models  E_k = E_fly + k * (hover kJ/min * wait):
k = 1 (hover waiting, as modelled), 0.5, 0.25, 0 (UAV waits landed: free).  Per instance we compare the
fastest route with the lowest-energy route. A real trade-off = lowest-energy route is clearly slower and
saves clearly more energy; if the two coincide there is nothing for an energy-aware method to trade."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from edr.evaluate import load_model
from edr.rl import Setting, new_env, rollout
from edr.scenario import generate_scenario
from edr.viz import BLUE, INK, MUTED, ORANGE, style_axes

ap = argparse.ArgumentParser()
ap.add_argument("--roads", default="loop"); ap.add_argument("--uav", type=int, default=2); ap.add_argument("--ugv", type=int, default=1)
ap.add_argument("--n", type=int, default=60); ap.add_argument("--samples", type=int, default=128)
ap.add_argument("--model", default=None)
a = ap.parse_args()
torch.set_num_threads(3); torch.manual_seed(0)
s = Setting(15, 5, a.uav, a.ugv, a.roads)
root = os.path.join(os.path.dirname(__file__), "..", "results")
model = load_model(a.model or os.path.join(root, "models", s.tag, "p0.pt"))
H = s.params.hover_kj_per_min
KS = (1.0, 0.5, 0.25, 0.0)
agg = {k: dict(rho=[], dT=[], dE=[], front=[]) for k in KS}
example = None
for i in range(a.n):
    scn = generate_scenario(2_000_000_000 + i, 15, 5, a.roads)
    envs = [new_env(s, scn) for _ in range(a.samples)]
    rollout(model, envs, "sample")
    res = [e.result() for e in envs if not e.result().failed]
    if len(res) < 10:
        continue
    T = np.array([r.makespan for r in res]); Ef = np.array([r.energy_fly for r in res]); W = np.array([r.wait_time for r in res])
    for k in KS:
        E = Ef + k * H * W
        agg[k]["rho"].append(stats.spearmanr(T, E)[0])
        iT, iE = int(np.argmin(T)), int(np.argmin(E))
        agg[k]["dT"].append(100 * (T[iE] - T[iT]) / T[iT])
        agg[k]["dE"].append(100 * (E[iT] - E[iE]) / E[iT])
        agg[k]["front"].append(int(sum(not np.any((T < T[j]) & (E <= E[j]) | (T <= T[j]) & (E < E[j])) for j in range(len(T)))))
    if i == 0:
        example = (T, Ef, W)
print(f"{s.tag}: {a.n} instances x {a.samples} sampled routes from P0")
print(f"{'energy model':28s} {'corr(T,E)':>10s} {'min-E route: +T %':>18s} {'saves E %':>10s} {'Pareto routes':>14s}")
names = {1.0: "hover waiting (modelled)", 0.5: "hover power x 0.5", 0.25: "hover power x 0.25", 0.0: "landed waiting (free)"}
for k in KS:
    g = agg[k]
    print(f"{names[k]:28s} {np.mean(g['rho']):10.2f} {np.mean(g['dT']):18.1f} {np.mean(g['dE']):10.1f} {np.mean(g['front']):14.1f}")
fig, axs = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")
T, Ef, W = example
for ax, k, col in ((axs[0], 1.0, BLUE), (axs[1], 0.0, ORANGE)):
    style_axes(ax); E = Ef + k * H * W
    ax.scatter(T, E, s=14, color=col, alpha=.7, edgecolor="white", lw=.3)
    ax.set_title(names[k] + f"  (rho = {stats.spearmanr(T, E)[0]:.2f})", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("mission time (min)", color=MUTED); ax.set_ylabel("UAV energy (kJ)", color=MUTED)
fig.suptitle(f"One instance, {a.samples} sampled routes from P0 ({s.tag})", x=.01, ha="left", fontsize=11)
fig.tight_layout(); out = os.path.join(root, s.tag, "figures"); os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "tradeoff_example.png"), dpi=150)
