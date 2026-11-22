"""Build the second presentation from the preserved aggregate results."""
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/uav-ugv-presentation-v2-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "assets"
SIZES = ["U15G5", "U45G15", "U75G25"]
FLEETS = ["2U1G", "2U2G", "4U2G"]
KEYS = [f"{size}_{fleet}" for size in SIZES for fleet in FLEETS]

NAVY = "#17324D"
CORAL = "#E76F51"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
SLATE = "#5D6D7E"
LIGHT = "#D9E2E8"
RED = "#C44536"


def read_csv(name):
    with (ROOT / "results" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def label(key):
    size, fleet = key.split("_")
    return f"{size}  |  {fleet}"


def save(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(ASSETS / f"{name}.{suffix}", dpi=220,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def paired_plot(name, xlabel, keys, baseline, proposed, xlim=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    y = np.arange(len(keys))
    for yi, b_value, p_value in zip(y, baseline, proposed):
        ax.plot([b_value, p_value], [yi, yi], color=LIGHT, linewidth=2.4,
                linestyle=(0, (1.5, 1.5)), zorder=1)
    ax.scatter(baseline, y, s=75, color=SLATE, label="Nearest-UGV baseline",
               zorder=3, edgecolor="white", linewidth=.7)
    ax.scatter(proposed, y, s=85, color=CORAL, label="Virtual-slot policy",
               zorder=4, edgecolor="white", linewidth=.7)
    ax.set_yticks(y, [label(key) for key in keys], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    # The slide already carries the result title, so the figure reserves its
    # top band for the legend instead of repeating a competing heading.
    ax.grid(axis="x", color=LIGHT, linewidth=.8, alpha=.8)
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(.5, 1.02), fontsize=10)
    if xlim:
        ax.set_xlim(*xlim)
    save(fig, name)


def configuration_from_run(run_id):
    match = re.search(r"(U\d+G\d+_\d+U\d+G)", run_id)
    if not match:
        raise ValueError(f"Cannot identify configuration from {run_id}")
    return match.group(1)


def main():
    ASSETS.mkdir(exist_ok=True)
    summary = read_csv("full_matrix_summary.csv")
    metrics = {(row["configuration"], row["method"]): row for row in summary}
    if len(metrics) != 18:
        raise ValueError("Expected one baseline and one proposed row in nine cells.")

    for key in KEYS:
        for method in ("baseline", "proposed"):
            row = metrics[key, method]
            if int(row["test_instances"]) != 256:
                raise ValueError(f"Unexpected test count for {key} and {method}")
            expected = (float(row["R_makespan_min"])
                        + 800 * (256 - int(row["completed"])) / 256)
            if abs(expected - float(row["penalized_cost_min"])) > 1e-7:
                raise ValueError(f"Inconsistent penalized cost for {key} and {method}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "pdf.fonttype": 42,
    })

    scenario = json.loads(
        (ROOT / "presentation/end_term/assets/scenario_seed1234.json").read_text()
    )
    nodes = {node["id"]: node["xy"] for node in scenario["road_nodes"]}
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    for edge in scenario["road_edges"]:
        first, second = (nodes[index] for index in edge["nodes"])
        ax.plot([first[0], second[0]], [first[1], second[1]], color=NAVY,
                linewidth=3.0, solid_capstyle="round", alpha=.86)
    aerial = np.array([task["xy"] for task in scenario["tasks"]
                       if task["kind"] == "aerial"])
    ground = np.array([task["xy"] for task in scenario["tasks"]
                       if task["kind"] == "ground"])
    depot = nodes[scenario["depot_node"]]
    ax.scatter(aerial[:, 0], aerial[:, 1], marker="^", s=85, color=CORAL,
               edgecolor="white", linewidth=.8, label="Aerial task", zorder=4)
    ax.scatter(ground[:, 0], ground[:, 1], marker="s", s=65, color=TEAL,
               edgecolor="white", linewidth=.8, label="Road task", zorder=5)
    ax.scatter(*depot, marker="*", s=250, color=GOLD, edgecolor=NAVY,
               linewidth=1.0, label="Depot", zorder=6)
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="x position (km)",
           ylabel="y position (km)")
    ax.set_aspect("equal")
    ax.set_title("Generated U15G5 instance", color=NAVY, fontweight="bold")
    ax.grid(color=LIGHT, linewidth=.7, alpha=.7)
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(.5, 1.0), fontsize=9)
    save(fig, "scenario_v2")

    baseline_cost = [float(metrics[key, "baseline"]["penalized_cost_min"])
                     for key in KEYS]
    proposed_cost = [float(metrics[key, "proposed"]["penalized_cost_min"])
                     for key in KEYS]
    paired_plot("cost_pairs", "Mean cost over 256 shared scenarios (min)", KEYS,
                baseline_cost, proposed_cost, (150, 1080))

    gains = [100 * (1 - proposed / baseline)
             for baseline, proposed in zip(baseline_cost, proposed_cost)]
    order = np.argsort(gains)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    ordered_gains = np.array(gains)[order]
    ordered_keys = np.array(KEYS)[order]
    colors = [TEAL if value >= 0 else RED for value in ordered_gains]
    y = np.arange(9)
    ax.barh(y, ordered_gains, color=colors, height=.62)
    ax.axvline(0, color=NAVY, linewidth=1.2)
    ax.set_yticks(y, [label(key) for key in ordered_keys], fontsize=10)
    ax.set_xlabel("Reduction in penalized cost (%)")
    ax.set_title("Configurations ranked by relative cost change",
                 color=NAVY, fontweight="bold", pad=10)
    ax.grid(axis="x", color=LIGHT, linewidth=.8)
    for yi, value in zip(y, ordered_gains):
        side = 1.0 if value >= 0 else -1.0
        ax.text(value + side * .8, yi, f"{value:+.1f}%",
                ha="left" if value >= 0 else "right", va="center",
                fontsize=10, fontweight="bold", color=NAVY)
    ax.set_xlim(-11, 44)
    save(fig, "gain_ranked")

    baseline_wait = [float(metrics[key, "baseline"]["T_wait_min_per_event"])
                     for key in KEYS]
    proposed_wait = [float(metrics[key, "proposed"]["T_wait_min_per_event"])
                     for key in KEYS]
    paired_plot("wait_pairs", "Mean wait per observed recharge event (min)", KEYS,
                baseline_wait, proposed_wait, (0, 40))

    load_keys = [key for key in KEYS if key.endswith(("2U2G", "4U2G"))]
    baseline_load = [float(metrics[key, "baseline"]["sigma_UGV_busy_min"])
                     for key in load_keys]
    proposed_load = [float(metrics[key, "proposed"]["sigma_UGV_busy_min"])
                     for key in load_keys]
    paired_plot("load_pairs", "Within-mission busy-time spread (min)", load_keys,
                baseline_load, proposed_load, (0, 145))

    completion_delta = [int(metrics[key, "proposed"]["completed"])
                        - int(metrics[key, "baseline"]["completed"])
                        for key in KEYS]
    fig, ax = plt.subplots(figsize=(8.5, 4.1), layout="constrained")
    y = np.arange(9)
    colors = [TEAL if value >= 0 else RED for value in completion_delta]
    ax.barh(y, completion_delta, color=colors, height=.58)
    ax.axvline(0, color=NAVY, linewidth=1.2)
    ax.set_yticks(y, [label(key) for key in KEYS], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Change in completed missions out of 256")
    ax.set_title("Completion change under the virtual-slot policy",
                 color=NAVY, fontweight="bold", pad=10)
    ax.grid(axis="x", color=LIGHT, linewidth=.8)
    for yi, value in zip(y, completion_delta):
        ax.text(value + (.25 if value >= 0 else -.25), yi, f"{value:+d}",
                ha="left" if value >= 0 else "right", va="center",
                fontweight="bold", color=NAVY)
    ax.set_xlim(-7.5, 9.5)
    save(fig, "completion_delta")

    stability = read_csv("hard_argmax_stability.csv")
    collision = {}
    for row in stability:
        key = row["configuration"] or configuration_from_run(row["run_id"])
        if row["method"] == "hard_argmax":
            collision.setdefault(key, []).append(
                100 * float(row["mean_argmax_collision_rate"])
            )
    collision_means = [np.mean(collision[key]) for key in KEYS]
    fig, ax = plt.subplots(figsize=(8.5, 3.9), layout="constrained")
    y = np.arange(9)
    ax.barh(y, collision_means, color=GOLD, edgecolor=NAVY, linewidth=.8,
            height=.58)
    ax.set_yticks(y, [label(key) for key in KEYS], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("States with duplicate row-ArgMax slots (%)")
    ax.set_title("Independent hard choices often violate slot capacity",
                 color=NAVY, fontweight="bold", pad=10)
    ax.grid(axis="x", color=LIGHT, linewidth=.8)
    save(fig, "collision_rates")

    shutil.copy2(ROOT / "presentation/end_term/assets/midterm_logo-000.png",
                 ASSETS / "iiita_logo.png")

    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "Presentation_V2.tex"], cwd=HERE, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if result.returncode:
            raise RuntimeError(result.stdout[-10000:])

    pdf = HERE / "Presentation_V2.pdf"
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    if pages != 15:
        raise RuntimeError(f"Expected 15 slides, found {pages}.")
    pdf_text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
    for name in ("Aditya Pachauri", "Jalendu Pandey", "Kostubh Birla"):
        if name not in pdf_text:
            raise RuntimeError(f"Missing presenter name {name}")
    if re.search(r"queue[- ]aware", pdf_text, re.I):
        raise RuntimeError("The removed term is still present in the presentation.")
    log = (HERE / "Presentation_V2.log").read_text()
    if re.search(r"Overfull|undefined|multiply defined", log):
        raise RuntimeError("The presentation log contains a layout or reference issue.")

    inputs = [
        ROOT / "results/full_matrix_summary.csv",
        ROOT / "results/hard_argmax_stability.csv",
        ROOT / "presentation/end_term/assets/scenario_seed1234.json",
        HERE / "Presentation_V2.tex",
        HERE / "build.py",
    ]
    artifacts = sorted(ASSETS.glob("*.pdf")) + [pdf]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "status": "built_from_preserved_aggregate_results",
        "slides": pages,
        "title": "Learning Virtual-Slot Rendezvous Scheduling for Cooperative UAV-UGV Missions",
        "numerical_results_changed": False,
        "visual_encoding_changed": True,
        "source_presentation_preserved": True,
        "inputs": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "artifacts": {str(path.relative_to(ROOT)): digest(path) for path in artifacts},
    }
    (HERE / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for suffix in ("aux", "log", "nav", "out", "snm", "toc"):
        (HERE / f"Presentation_V2.{suffix}").unlink(missing_ok=True)
    print(f"Built {pdf} with {pages} slides.")


if __name__ == "__main__":
    main()
