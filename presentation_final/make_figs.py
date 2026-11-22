"""Result figures for the end-sem presentation, computed from the real experiment files (no typed-in numbers
except the base-paper Table I values and the trade-off table, which are noted)."""
import csv, glob, os, re
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); IMPL = os.path.join(HERE, "..", "implementation")
OUT = os.path.join(HERE, "assets")
NAVY, TEAL, AMBER, GRAY, RED, LIGHT = "#17324D", "#159A96", "#E8A93A", "#8a93a0", "#C44536", "#D9E2E8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9aa5b1", "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "axes.titlecolor": NAVY, "axes.titlesize": 12, "axes.titleweight": "bold"})
MAPS = [("loop", "Ring road", "U15G5_2U1G", "loop"), ("border", "Border", "U15G5_2U1G_border", "border"),
        ("geo", "District", "U15G5_2U1G_geo", "geo"), ("paper", "Paper map", "U15G5_2U1G_paper", "paper")]


def read(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try: r[k] = float(v)
            except ValueError: pass
    return rows


def seeds(m, tag):
    d = os.path.join(IMPL, "results_kaggle", m, tag)
    out = {}
    for f in glob.glob(os.path.join(d, "stageB*.csv")):
        mm = re.search(r"stageB(?:_seed(\d+))?\.csv$", f)
        out[int(mm.group(1) or 0)] = read(f)
    return dict(sorted(out.items()))


def paired(data, a, b, key, flt=lambda r: r["kind"] != "none"):
    res = []
    for rows in data.values():
        ra = [r for r in rows if r["policy"] == a and flt(r)]; rb = [r for r in rows if r["policy"] == b and flt(r)]
        res.append(np.mean([x[key] - y[key] for x, y in zip(ra, rb)]))
    return np.array(res)


def per_seed(data, pol, key, flt=lambda r: r["kind"] != "none"):
    return np.array([np.mean([r[key] for r in rows if r["policy"] == pol and flt(r)]) for rows in data.values()])


D = {m: seeds(m, tag) for m, _, tag, _ in MAPS}

# ---- 1. Table I: ours vs paper (paper-map base policy)
teams = ["1U-1G", "2U-1G", "2U-2G", "4U-2G"]
paper_g, paper_s = [238.2, 229.7, 180.5, 188.9], [218.6, 199.7, 154.4, 159.9]       # base paper Table I, U15G5
ours = [228.1, 209.8, 185.1, 219.0]                                                 # scripts/validate_m1.py (paper map)
fig, ax = plt.subplots(figsize=(7.2, 3.6)); x = np.arange(4); w = 0.26
ax.bar(x - w, paper_g, w, color=GRAY, label="Paper, greedy DRL [1]")
ax.bar(x, paper_s, w, color=LIGHT, edgecolor=GRAY, label="Paper, 10240 samples [1]")
ax.bar(x + w, ours, w, color=TEAL, label="Ours, greedy (this work)")
for xi, v in zip(x + w, ours): ax.text(xi, v + 3, f"{v:.0f}", ha="center", fontsize=9, color=NAVY)
ax.set_xticks(x, teams); ax.set_ylabel("mean mission time (min)"); ax.set_ylim(0, 340); ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", columnspacing=1.0, handlelength=1.2)
ax.set_title("Base policy on the paper's own map (U15G5)", loc="left"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_tableI.png", dpi=200); plt.close(fig)

# ---- 2. how much events hurt (rule policy, map study)
rows = read(os.path.join(IMPL, "results", "map_study.csv"))
names = {"paper": "Paper map", "loop": "Ring road", "grid": "Grid", "radial": "Radial", "geo": "District", "border": "Border"}
order = ["radial", "paper", "grid", "geo", "loop", "border"]
dT = [100 * np.mean([r["To"] / r["T0"] - 1 for r in rows if r["roads"] == m]) for m in order]
dE = [100 * np.mean([r["Eo"] / r["E0"] - 1 for r in rows if r["roads"] == m]) for m in order]
fig, ax = plt.subplots(figsize=(7.2, 3.4)); x = np.arange(len(order)); w = 0.38
ax.bar(x - w / 2, dT, w, color=NAVY, label="mission time"); ax.bar(x + w / 2, dE, w, color=AMBER, label="UAV energy")
for xi, v in zip(x - w / 2, dT): ax.text(xi, v + .6, f"{v:.0f}%", ha="center", fontsize=9)
for xi, v in zip(x + w / 2, dE): ax.text(xi, v + .6, f"{v:.0f}%", ha="center", fontsize=9)
ax.set_xticks(x, [names[m] for m in order]); ax.set_ylabel("increase vs static roads (%)"); ax.legend(frameon=False)
ax.set_title("Road events hurt an open-loop mission (rule policy, 18 events)", loc="left"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_damage.png", dpi=200); plt.close(fig)

# ---- 3. replanning vs open loop: per-seed dots per map
A_, B_ = "P0 event + road layer", "P0 open-loop (base)"
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.5))
for ax, key, lab in zip(axs, ("T", "E"), ("change in mission time (min)", "change in UAV energy (kJ)")):
    for i, (m, nm, tag, _) in enumerate(MAPS):
        v = paired(D[m], A_, B_, key)
        ax.scatter([i] * len(v), v, s=46, color=TEAL, zorder=3, edgecolor="white")
        ax.plot([i - .25, i + .25], [v.mean()] * 2, color=NAVY, lw=2.2, zorder=4)
    ax.axhline(0, color=GRAY, lw=1); ax.set_xticks(range(4), [m[1] for m in MAPS]); ax.set_ylabel(lab)
axs[0].set_title("Replanning vs open loop", loc="left"); axs[1].set_title("(one dot = one training seed)", loc="left", fontsize=10, fontweight="normal")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_replan_gain.png", dpi=200); plt.close(fig)

# ---- 4. event-triggered vs periodic: replans and time
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4))
pe = [per_seed(D[m], "P0 periodic + road layer", "replans").mean() for m, *_ in MAPS]
ev = [per_seed(D[m], "P0 event + road layer", "replans").mean() for m, *_ in MAPS]
x = np.arange(4); w = 0.36
axs[0].bar(x - w / 2, pe, w, color=AMBER, label="periodic"); axs[0].bar(x + w / 2, ev, w, color=TEAL, label="event-triggered")
for xi, v in zip(x - w / 2, pe): axs[0].text(xi, v + .05, f"{v:.1f}", ha="center", fontsize=9)
for xi, v in zip(x + w / 2, ev): axs[0].text(xi, v + .05, f"{v:.1f}", ha="center", fontsize=9)
axs[0].set_xticks(x, [m[1] for m in MAPS]); axs[0].set_ylabel("replans per mission"); axs[0].set_ylim(0, 4.3); axs[0].legend(frameon=False, fontsize=9, ncol=2, loc="upper center")
axs[0].set_title("Replans needed", loc="left")
dts = [paired(D[m], "P0 event + road layer", "P0 periodic + road layer", "T") for m, *_ in MAPS]
for i, v in enumerate(dts):
    axs[1].scatter([i] * len(v), v, s=46, color=NAVY, zorder=3, edgecolor="white")
axs[1].axhline(0, color=GRAY, lw=1); axs[1].set_ylim(-1.5, 1.5); axs[1].set_xticks(range(4), [m[1] for m in MAPS])
axs[1].set_ylabel("mission time: event - periodic (min)"); axs[1].set_title("Quality is the same", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_event_vs_periodic.png", dpi=200); plt.close(fig)

# ---- 5. time-energy trade-off (scripts/tradeoff_study.py, ring road, 60 inst x 128 routes)
models = ["hover\n(as modelled)", "hover\nx 0.5", "hover\nx 0.25", "landed\n(free wait)"]
extra_t, saved_e, rho = [0.7, 0.3, 2.1, 23.2], [0.9, 0.6, 1.2, 7.9], [0.96, 0.97, 0.90, 0.39]
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4))
axs[0].bar(range(4), rho, color=[NAVY, NAVY, NAVY, TEAL]); axs[0].set_xticks(range(4), models, fontsize=9); axs[0].set_ylim(0, 1.05)
for i, v in enumerate(rho): axs[0].text(i, v + .02, f"{v:.2f}", ha="center", fontsize=9)
axs[0].set_ylabel("corr(time, energy) over sampled routes"); axs[0].set_title("How aligned are time and energy?", loc="left")
w = .36; x = np.arange(4)
axs[1].bar(x - w / 2, extra_t, w, color=AMBER, label="extra mission time"); axs[1].bar(x + w / 2, saved_e, w, color=TEAL, label="energy saved")
for xi, v in zip(x - w / 2, extra_t): axs[1].text(xi, v + .5, f"{v:.1f}%", ha="center", fontsize=9)
for xi, v in zip(x + w / 2, saved_e): axs[1].text(xi, v + .5, f"{v:.1f}%", ha="center", fontsize=9)
axs[1].set_xticks(x, models, fontsize=9); axs[1].set_ylabel("lowest-energy route vs fastest route (%)"); axs[1].legend(frameon=False, fontsize=9)
axs[1].set_title("What an energy-aware route gives", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_tradeoff.png", dpi=200); plt.close(fig)

# ---- 6. candidates vs control (+ P1), per map, per seed
C = "P0+ (control) event + road layer"
cands = [("P3 (beta=0.9) event + road layer", "A: Lagrangian (P3)", AMBER), ("P4 event + road layer", "B: budget input (P4)", NAVY),
         ("P1 route filter, event", "Sampling + budget filter (P1)", TEAL)]
fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.5), sharey=True)
for ax, (m, nm, tag, _) in zip(axs, MAPS):
    for j, (pol, lab, col) in enumerate(cands):
        v = paired(D[m], pol, C, "T")
        ax.scatter([j] * len(v), v, s=44, color=col, zorder=3, edgecolor="white"); ax.plot([j - .25, j + .25], [v.mean()] * 2, color=NAVY, lw=2)
    ax.axhline(0, color=GRAY, lw=1); ax.set_xticks(range(3), ["A", "B", "P1"]); ax.set_title(nm, loc="left", fontsize=11)
axs[0].set_ylabel("mission time vs equally trained control (min)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_candidates.png", dpi=200); plt.close(fig)
print("figures written:", sorted(os.listdir(OUT)))

# ---- 7. all six maps (2 x 3), drawn from the real map builders
import sys
sys.path.insert(0, os.path.join(IMPL, "src"))
from edr.scenario import generate_scenario
names = {"paper": "Base paper's tree", "loop": "Ring road", "grid": "City grid", "radial": "Radial town", "geo": "Irregular district", "border": "Border surveillance"}
fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.4))
for ax, key in zip(axs.ravel(), ["paper", "loop", "grid", "radial", "geo", "border"]):
    scn = generate_scenario(2_000_000_000, 15, 5, key); rm = scn.roads
    for (u, v) in rm.edges:
        ax.plot(*zip(rm.xy[u], rm.xy[v]), color="#b3bcc7", lw=2.4, solid_capstyle="round", zorder=1)
    if rm.band is not None:
        (x0, y0), (x1, y1), half = rm.band
        d = np.array([x1 - x0, y1 - y0]); nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        poly = np.array([[x0, y0] + half * nrm, [x1, y1] + half * nrm, [x1, y1] - half * nrm, [x0, y0] - half * nrm])
        ax.fill(poly[:, 0], poly[:, 1], color=TEAL, alpha=.12, zorder=0)
        ax.plot([x0, x1], [y0 + 1.8, y1 + 1.8], color=RED, lw=2, ls=(0, (4, 3)), zorder=2)
    ax.scatter(scn.xy[scn.n_ground:, 0], scn.xy[scn.n_ground:, 1], facecolors="none", edgecolors=RED, s=26, lw=1.2, zorder=4)
    ax.scatter(scn.xy[:scn.n_ground, 0], scn.xy[:scn.n_ground, 1], c=NAVY, marker="s", s=44, edgecolors="white", lw=.8, zorder=5)
    bridges = sum(rm.is_bridge(i) for i in range(len(rm.edges)))
    ax.set_title(f"{names[key]}", loc="left", fontsize=12); ax.text(0.02, 0.03, f"{rm.n} nodes, {bridges} of {len(rm.edges)} roads are bridges", transform=ax.transAxes, fontsize=8.5, color=NAVY)
    ax.set_xlim(0, 20); ax.set_ylim(0, 20); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/fig_maps.png", dpi=170); plt.close(fig)

# ======================= closing the open items (teams, Stage C, baselines) =======================
KR = os.path.join(IMPL, "results_kaggle")
def seed_files(d, stage):
    out = {}
    for f in glob.glob(os.path.join(d, f"stage{stage}*.csv")):
        mm = re.search(r"_seed(\d+)", f); out[int(mm.group(1)) if mm else 0] = read(f)
    return dict(sorted(out.items()))

# ---- 8. all team sizes (loop map, 3 seeds each)
TEAMS = [("1U-1G", "loop-1u1g", "U15G5_1U1G"), ("2U-1G", "loop", "U15G5_2U1G"), ("2U-2G", "loop-2u2g", "U15G5_2U2G"), ("4U-2G", "loop-4u2g", "U15G5_4U2G")]
TD = {t: seed_files(os.path.join(KR, d, tag), "B") for t, d, tag in TEAMS}
fig, axs = plt.subplots(1, 3, figsize=(11.6, 3.5))
for i, (t, *_ ) in enumerate(TEAMS):
    v = paired(TD[t], A_, B_, "T"); axs[0].scatter([i] * len(v), v, s=46, color=TEAL, zorder=3, edgecolor="white"); axs[0].plot([i - .25, i + .25], [v.mean()] * 2, color=NAVY, lw=2.2)
    pe = per_seed(TD[t], "P0 periodic + road layer", "replans").mean(); ev = per_seed(TD[t], "P0 event + road layer", "replans").mean()
    axs[1].bar(i - .19, pe, .36, color=AMBER, label="periodic" if i == 0 else None); axs[1].bar(i + .19, ev, .36, color=TEAL, label="event-triggered" if i == 0 else None)
    axs[1].text(i - .19, pe + .1, f"{pe:.1f}", ha="center", fontsize=8.5); axs[1].text(i + .19, ev + .1, f"{ev:.1f}", ha="center", fontsize=8.5)
    v = paired(TD[t], "P1 route filter, event", "P0+ (control) event + road layer", "T"); axs[2].scatter([i] * len(v), v, s=46, color=NAVY, zorder=3, edgecolor="white"); axs[2].plot([i - .25, i + .25], [v.mean()] * 2, color=NAVY, lw=2.2)
for ax in axs: ax.set_xticks(range(4), [t[0] for t in TEAMS])
axs[0].axhline(0, color=GRAY, lw=1); axs[0].set_ylabel("mission time (min)"); axs[0].set_title("Replanning vs open loop", loc="left")
axs[1].set_ylabel("replans per mission"); axs[1].legend(frameon=False, fontsize=9); axs[1].set_title("Replans needed", loc="left"); axs[1].set_ylim(0, 8)
axs[2].axhline(0, color=GRAY, lw=1); axs[2].set_ylabel("mission time vs control (min)"); axs[2].set_title("Sampling + filter (P1)", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_teams.png", dpi=200); plt.close(fig)

# ---- 9. Stage C: larger instances zero-shot (loop map, 2U-1G, one dot per seed)
SC = os.path.join(IMPL, "results_stageC")
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.6))
for ax, (st, pols, labs, flt, ttl) in zip(axs, [
        ("A", ["P0", "P1 route filter"], ["greedy", "search"], lambda r: r["beta"] == 0.9, "Static roads"),
        ("B", ["P0 open-loop (base)", "P0 event + road layer", "P1 route filter, event"], ["open", "event", "search"], lambda r: r["kind"] != "none", "With road events")]):
    xo = 0
    for sz, tag in (("45 tasks", "U45G15_2U1G"), ("75 tasks", "U75G25_2U1G")):
        D_ = seed_files(os.path.join(SC, tag), st)
        for j, (pol, lab) in enumerate(zip(pols, labs)):
            v = per_seed(D_, pol, "T", flt); x = xo + j
            ax.scatter([x] * len(v), v, s=40, color=[TEAL, NAVY, AMBER][j], zorder=3, edgecolor="white"); ax.plot([x - .25, x + .25], [np.median(v)] * 2, color=NAVY, lw=2)
        ax.text(xo + (len(pols) - 1) / 2, -0.2, sz, transform=ax.get_xaxis_transform(), ha="center", fontsize=10, color=NAVY)
        xo += len(pols) + 1
    ax.set_xticks([j for k in range(2) for j in [k * (len(pols) + 1) + i for i in range(len(pols))]], [l for _ in range(2) for l in labs], fontsize=9)
    ax.set_ylabel("mission time (min)"); ax.set_title(ttl + "  (line = median of 3 seeds)", loc="left", fontsize=11)
fig.tight_layout(); fig.subplots_adjust(bottom=0.27); fig.savefig(f"{OUT}/fig_stageC.png", dpi=200); plt.close(fig)

# ---- 10. baselines and sampled decoding on the paper's map
bl = {(r["team"], r["method"]): float(r["T"]) for r in read(os.path.join(IMPL, "results", "bilevel_eval_paper.csv"))}
sm = {r["team"]: float(r["best_T"]) for r in read(os.path.join(IMPL, "results", "sampling_eval_1024.csv"))}
teams4 = ["1U-1G", "2U-1G", "2U-2G", "4U-2G"]
paper_gls = [281.7, 232.8, 188.3, 219.7]; paper_g = [238.2, 229.7, 180.5, 188.9]; paper_1024 = [220.4, 203.1, 157.1, 165.6]
ours_gls = [bl[(t, "GLS")] for t in teams4]; ours_s = [sm[t] for t in teams4]
fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.5)); x = np.arange(4); w = .36
axs[0].bar(x - w / 2, paper_gls, w, color=GRAY, label="Paper, GLS [1]"); axs[0].bar(x + w / 2, ours_gls, w, color=TEAL, label="Ours, GLS (OR-Tools)")
axs[0].set_xticks(x, teams4); axs[0].set_ylabel("mean mission time (min)"); axs[0].set_ylim(0, 340); axs[0].legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
axs[0].set_title("Bilevel heuristic baseline", loc="left")
w = .2
for k, (vals, lab, col) in enumerate([(paper_g, "Paper greedy", GRAY), (ours, "Ours greedy", TEAL), (paper_1024, "Paper, 1024 samples", LIGHT), (ours_s, "Ours, best of 1024", NAVY)]):
    axs[1].bar(x + (k - 1.5) * w, vals, w, color=col, label=lab, edgecolor=GRAY if col == LIGHT else col)
axs[1].set_xticks(x, teams4); axs[1].set_ylim(0, 340); axs[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper center"); axs[1].set_title("Learned policy: greedy vs sampled", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_baselines.png", dpi=200); plt.close(fig)
print("closing-the-loop figures done")

# ---- 11. just-in-time departure (scripts/jit_study.py)
jr = read(os.path.join(IMPL, "results", "jit_study.csv"))
def javg(m, ctl, sc, j, key): return np.mean([r[key] for r in jr if r["map"] == m and r["controller"] == ctl and r["scenario"] == sc and r["jit"] == j])
ctl = "event-triggered + road layer"
fig, axs = plt.subplots(1, 3, figsize=(11.6, 3.6)); x = np.arange(4); w = .36
for ax, sc, ttl in ((axs[0], "static", "Energy, static roads"), (axs[1], "events", "Energy, with road events")):
    off = [javg(m, ctl, sc, 0, "E") for m, *_ in MAPS]; on = [javg(m, ctl, sc, 1, "E") for m, *_ in MAPS]
    ax.bar(x - w / 2, off, w, color=GRAY, label="depart immediately (hover)"); ax.bar(x + w / 2, on, w, color=TEAL, label="just-in-time departure")
    for xi, a, b in zip(x, off, on): ax.text(xi + w / 2, b + 100, f"{100 * (b / a - 1):.0f}%", ha="center", fontsize=9, color=NAVY)
    ax.set_xticks(x, [m[1] for m in MAPS], fontsize=9); ax.set_ylabel("UAV energy (kJ)"); ax.set_title(ttl, loc="left")
a_ = [100 * javg(m, ctl, "events", 0, "over09") for m, *_ in MAPS]; b_ = [100 * javg(m, ctl, "events", 1, "over09") for m, *_ in MAPS]
axs[2].bar(x - w / 2, a_, w, color=GRAY); axs[2].bar(x + w / 2, b_, w, color=TEAL)
for xi, a, b in zip(x, a_, b_): axs[2].text(xi - w / 2, a + 1.5, f"{a:.0f}%", ha="center", fontsize=9); axs[2].text(xi + w / 2, b + 1.5, f"{b:.0f}%", ha="center", fontsize=9)
axs[2].set_xticks(x, [m[1] for m in MAPS], fontsize=9); axs[2].set_ylabel("instances over budget (%)"); axs[2].set_ylim(0, 110)
axs[2].set_title("Budget 0.9 B$_0$, with road events", loc="left", fontsize=11)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=TEAL)], labels=["depart immediately (UAV hovers while it waits)", "just-in-time departure (UAV waits on the ground)"], loc="lower center", ncol=2, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f"{OUT}/fig_jit.png", dpi=200); plt.close(fig)
print("jit figure done")

# ---- 12. just-in-time departure across team sizes and with the sampling policy (scripts/jit_teams.py)
jt = read(os.path.join(IMPL, "results", "jit_teams.csv"))
def jv(pol, team, sc, j, key): return np.mean([r[key] for r in jt if r["policy"] == pol and r["team"] == team and r["scenario"] == sc and r["jit"] == j])
fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
tm = ["1U-1G", "2U-1G", "2U-2G", "4U-2G"]; x = np.arange(4); w = .36
off = [jv("P0 event + road layer", t, "events", 0, "E") for t in tm]; on = [jv("P0 event + road layer", t, "events", 1, "E") for t in tm]
axs[0].bar(x - w / 2, off, w, color=GRAY); axs[0].bar(x + w / 2, on, w, color=TEAL)
for xi, a, b in zip(x, off, on): axs[0].text(xi + w / 2, b + 120, f"{100 * (b / a - 1):.0f}%", ha="center", fontsize=9.5, color=NAVY)
axs[0].set_xticks(x, tm); axs[0].set_ylabel("UAV energy (kJ)"); axs[0].set_title("Base policy, with road events, by team size", loc="left", fontsize=11)
off = jv("P1 sampling + budget filter", "2U-1G", "static", 0, "E"); on = jv("P1 sampling + budget filter", "2U-1G", "static", 1, "E")
axs[1].bar([0 - w / 2], [off], w, color=GRAY); axs[1].bar([0 + w / 2], [on], w, color=TEAL); axs[1].text(w / 2, on + 60, f"{100 * (on / off - 1):.0f}%", ha="center", fontsize=9.5, color=NAVY)
axs[1].set_xticks([0], ["sampling + budget filter (2U-1G)"], fontsize=9); axs[1].set_xlim(-.7, .7); axs[1].set_title("Search policy", loc="left", fontsize=11)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=TEAL)], labels=["depart immediately", "just-in-time departure"], loc="lower center", ncol=2, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f"{OUT}/fig_jit_teams.png", dpi=200); plt.close(fig)
print("jit teams figure done")
