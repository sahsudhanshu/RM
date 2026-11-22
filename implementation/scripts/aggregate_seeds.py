#!/usr/bin/env python
"""Aggregate experiment CSVs over training seeds -> results/REPORT_SEEDS.md + figure.

Per seed: mean over the 100 test instances (and the 18 events for Stage B). Across seeds: mean and
95% CI (t, n-1 dof) of those per-seed means. Paired differences are computed WITHIN each seed (same
instances/events, that seed's own P0 defines B0) and then averaged over seeds."""
import glob, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figures import read, sel
from edr.viz import BLUE, style_axes, INK, MUTED, ORANGE, AQUA

ROOT = os.path.join(os.path.dirname(__file__), "..", sys.argv[2] if len(sys.argv) > 2 else "results")
TAG = sys.argv[1] if len(sys.argv) > 1 else "U15G5_2U1G"
D = os.path.join(ROOT, TAG)


def seeds_of(stage):
    out = {}
    for f in glob.glob(os.path.join(D, f"stage{stage}*.csv")):
        m = re.search(r"stage%s(?:_seed(\d+))?\.csv$" % stage, f)
        if m:
            out[int(m.group(1) or 0)] = read(f)
    return dict(sorted(out.items()))


def tci(v):
    v = np.asarray(v, float)
    n = len(v)
    return v.mean(), (stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan"))


def per_seed(data, policy, key, flt=lambda r: True):
    return [np.mean([r[key] for r in rows if r["policy"] == policy and flt(r)]) for rows in data.values()]


def paired_seed(data, a, b, key, flt=lambda r: True):
    out = []
    for rows in data.values():
        ra = [r for r in rows if r["policy"] == a and flt(r)]
        rb = [r for r in rows if r["policy"] == b and flt(r)]
        out.append(np.mean([x[key] - y[key] for x, y in zip(ra, rb)]))
    return out


def fmt(v, d=1):
    m, c = tci(v)
    return f"{m:.{d}f} ± {c:.{d}f}"


out = [f"# Multi-seed results — {TAG}\n"]
A, B = seeds_of("A"), seeds_of("B")
n = len(B)
out.append(f"Training seeds: {list(B)} (n = {n}). Each seed = its own imitation warm start, REINFORCE and fine-tuning runs, "
           "evaluated on the same 100 test instances (its own P0 defines B0). ± = 95% CI over seeds (t, "
           f"{n-1} dof); per-seed values in brackets.\n")

out.append("## Stage A — static roads\n")
for beta in (0.9, 1.0):
    out.append(f"\n**beta = {beta}**\n\n| policy | T (min) | E (kJ) | over budget (%) | per-seed T |\n|---|---|---|---|---|")
    for pol in dict.fromkeys(r["policy"] for r in list(A.values())[0]):
        f = lambda r, b=beta: r["beta"] == b
        if not all(any(r["policy"] == pol and r["beta"] == beta for r in rows) for rows in A.values()):
            continue
        T, E = per_seed(A, pol, "T", f), per_seed(A, pol, "E", f)
        ov = [100 * x for x in per_seed(A, pol, "over", f)]
        out.append(f"| {pol} | {fmt(T)} | {fmt(E,0)} | {fmt(ov,0)} | [{', '.join(f'{x:.0f}' for x in T)}] |")
out.append("\n**Paired within seed, vs P0+ control (beta = 0.9)**\n\n| policy | dT (min) | dE (kJ) | per-seed dE |\n|---|---|---|---|")
f9 = lambda r: r["beta"] == 0.9
for pol in ("P1 route filter", "P2 fixed penalty", "P3 Lagrangian", "P4 budget-conditioned", "P4 no budget input"):
    if all(any(r["policy"] == pol and r["beta"] == 0.9 for r in rows) for rows in A.values()):
        dT = paired_seed(A, pol, "P0+ (control)", "T", f9); dE = paired_seed(A, pol, "P0+ (control)", "E", f9)
        out.append(f"| {pol} | {fmt(dT)} | {fmt(dE,0)} | [{', '.join(f'{x:+.0f}' for x in dE)}] |")

ev = lambda r: r["kind"] != "none"
out.append("\n## Stage B — blocked / slow roads (18 events), B = 0.9 B0\n\n| policy | T (min) | E (kJ) | over budget (%) | replans | per-seed T |\n|---|---|---|---|---|---|")
pols = list(dict.fromkeys(r["policy"] for r in list(B.values())[0]))
for pol in pols:
    if not all(any(r["policy"] == pol for r in rows) for rows in B.values()):
        continue
    T, E = per_seed(B, pol, "T", ev), per_seed(B, pol, "E", ev)
    out.append(f"| {pol} | {fmt(T)} | {fmt(E,0)} | {fmt([100*x for x in per_seed(B, pol, 'over', ev)],0)} | "
               f"{fmt(per_seed(B, pol, 'replans', ev),2)} | [{', '.join(f'{x:.0f}' for x in T)}] |")
out.append("\n**Paired within seed (all events)**\n\n| comparison | dT (min) | dE (kJ) | per-seed dT |\n|---|---|---|---|")
comps = [("P0: event − periodic", "P0 event + road layer", "P0 periodic + road layer"),
         ("P4: event − periodic", "P4 event + road layer", "P4 periodic + road layer"),
         ("P0: event + layer − open loop", "P0 event + road layer", "P0 open-loop (base)"),
         ("P0+ control: event + layer − open loop", "P0+ (control) event + road layer", "P0+ (control) open-loop"),
         ("P4: event + layer − open loop", "P4 event + road layer", "P4 open-loop"),
         ("P4 − P0+ control (both event + layer)", "P4 event + road layer", "P0+ (control) event + road layer"),
         ("P3(0.9) − P0+ control (both event + layer)", "P3 (beta=0.9) event + road layer", "P0+ (control) event + road layer"),
         ("P4 road-event-trained − P4", "P4 (road-event trained) event", "P4 event + road layer"),
         ("P1 route filter − P0+ control", "P1 route filter, event", "P0+ (control) event + road layer")]
for name, a, b in comps:
    if all(any(r["policy"] == a for r in rows) and any(r["policy"] == b for r in rows) for rows in B.values()):
        dT, dE = paired_seed(B, a, b, "T", ev), paired_seed(B, a, b, "E", ev)
        out.append(f"| {name} | {fmt(dT)} | {fmt(dE,0)} | [{', '.join(f'{x:+.1f}' for x in dT)}] |")
dr = paired_seed(B, "P0 event + road layer", "P0 periodic + road layer", "replans", ev)
out.append(f"\nReplans event − periodic (P0): {fmt(dr,2)} per mission.\n")
open(os.path.join(ROOT, f"REPORT_SEEDS_{TAG}.md"), "w").write("\n".join(out) + "\n")

# figure: per-seed dots + mean for the key comparisons
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
for ax, key, lab in zip(axs, ("T", "E"), ("mission time (min), Stage B, all events", "UAV energy (kJ), Stage B, all events")):
    style_axes(ax)
    show = [("P0 open-loop (base)", "#8a8984"), ("P0+ (control) event + road layer", "#b9b8b0"), ("P4 event + road layer", BLUE),
            ("P3 (beta=0.9) event + road layer", AQUA), ("P1 route filter, event", "#6250d6")]
    for i, (pol, col) in enumerate(show):
        if pol not in pols:
            continue
        v = per_seed(B, pol, key, ev)
        ax.scatter([i] * len(v), v, color=col, s=36, zorder=3, edgecolor="white")
        m, c = tci(v)
        ax.errorbar([i], [m], yerr=[c], color=INK, capsize=5, marker="_", ms=16, zorder=4)
    ax.set_xticks(range(len(show)), ["P0\nopen-loop", "P0+ control\nevent", "P4\nevent", "P3 (0.9)\nevent", "P1\nroute filter"], fontsize=8)
    ax.set_title(lab, loc="left", fontsize=10, color=INK)
fig.suptitle(f"{TAG}: one dot per training seed; bar = mean ± 95% CI over seeds", x=.01, ha="left", fontsize=11)
fig.tight_layout(); os.makedirs(os.path.join(D, "figures"), exist_ok=True)
fig.savefig(os.path.join(D, "figures", "seeds_stageB.png"), dpi=160)
print("wrote", os.path.join(ROOT, f"REPORT_SEEDS_{TAG}.md"))
