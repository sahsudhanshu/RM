"""Build source-linked tables and figures for the nine-cell policy matrix."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIGURES = HERE / "figures"
SIZES = [(15, 5), (45, 15), (75, 25)]
FLEETS = [(2, 1), (2, 2), (4, 2)]


def tag(aerial: int, ground: int, uavs: int, ugvs: int) -> str:
    return f"U{aerial}G{ground}_{uavs}U{ugvs}G"


def short_label(key: str) -> str:
    size, fleet = key.split("_")
    return f"{size}\n{fleet}"


def primary_run(key: str) -> Path:
    old = {
        "U15G5_2U2G": "20261010_U15G5_2U2G_pair_seed11_proposed",
        "U15G5_4U2G": "20261010_U15G5_4U2G_pair_seed11",
    }
    return RUNS / old.get(key, f"20261011_{key}_pair_seed11")


def zero_run(key: str) -> Path:
    return RUNS / f"20261011_{key}_gamma0_seed11"


def ablation_run(key: str) -> Path:
    date = "20261010" if key in {"U15G5_2U2G", "U15G5_4U2G"} else "20261011"
    label = "trained_assignment_training_ablation" if key.startswith("U75G25_") else "assignment_training_ablation"
    return RUNS / f"{date}_{key}_{label}"


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _event_stats(run: Path, ugvs: int) -> tuple[dict[tuple[int, str], float], dict[str, list[float]]]:
    result = {}
    waits: dict[str, list[float]] = defaultdict(list)
    with (run / "events.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            loads = [0.0] * ugvs
            for service in item["charging_services"]:
                waits[item["method"]].append(service["waiting_time"])
                loads[service["ugv_id"]] += (
                    service["ugv_arrival_time"] - service["departure_time"]
                    + service["charging_end_time"] - service["charging_start_time"]
                )
            result[(item["instance_seed"], item["method"])] = statistics.pstdev(loads)
    return result, waits


def _paired_difference(before: list[dict], after: list[dict]) -> tuple[float, list[float]]:
    first = {int(row["instance_seed"]): row for row in before}
    second = {int(row["instance_seed"]): row for row in after}
    if first.keys() != second.keys() or len(first) != 256:
        raise ValueError("paired comparison lacks 256 shared seeds")
    seeds = sorted(first)
    if any(first[s]["scenario_sha256"] != second[s]["scenario_sha256"] for s in seeds):
        raise ValueError("paired scenarios differ")
    differences = np.array([
        float(second[s]["penalized_cost_min"])
        - float(first[s]["penalized_cost_min"])
        for s in seeds
    ])
    rng = np.random.default_rng(37)
    indices = rng.integers(0, len(differences), (2000, len(differences)))
    return float(differences.mean()), np.quantile(
        differences[indices].mean(axis=1), [.025, .975]
    ).tolist()


def gather() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    metrics, primary_effects, gamma_effects, frozen_effects, stability = [], [], [], [], []
    for aerial, ground in SIZES:
        for uavs, ugvs in FLEETS:
            key = tag(aerial, ground, uavs, ugvs)
            primary = primary_run(key)
            zero = zero_run(key)
            expected_status = "evaluated"
            if any(json.loads((run / "manifest.json").read_text())["status"] != expected_status
                   for run in (primary, zero)):
                raise RuntimeError(f"unfinished policy comparison: {key}")
            latency = json.loads((primary / "latency_summary.json").read_text())
            primary_rows = csv_rows(primary / "evaluation.csv")
            zero_rows = csv_rows(zero / "evaluation.csv")
            busy, event_waits = _event_stats(primary, ugvs)
            grouped = defaultdict(list)
            for row in primary_rows:
                grouped[row["method"]].append(row)
            for method in ("baseline", "proposed"):
                rows = grouped[method]
                if len(rows) != 256:
                    raise ValueError(f"expected 256 {method} rows: {key}")
                metrics.append({
                    "configuration": key,
                    "size": f"U{aerial}G{ground}",
                    "fleet": f"{uavs}U{ugvs}G",
                    "method": method,
                    "run_id": primary.name,
                    "test_instances": 256,
                    "completed": sum(row["status"] == "completed" for row in rows),
                    "R_makespan_min": statistics.fmean(
                        float(row["makespan_min"]) for row in rows
                    ),
                    "T_wait_min_per_event": statistics.fmean(event_waits[method]),
                    "recharge_event_count": len(event_waits[method]),
                    "T_wait_instance_macro_min": statistics.fmean(
                        float(row["mean_wait_per_event_min"]) for row in rows
                    ),
                    "mean_total_wait_per_uav_min": statistics.fmean(
                        float(row["mean_total_wait_per_uav_min"]) for row in rows
                    ),
                    "sigma_UGV_busy_min": statistics.fmean(
                        busy[(int(row["instance_seed"]), method)] for row in rows
                    ),
                    "t_plan_ms_per_round": latency["methods"][method][
                        "mean_tplan_ms_per_replanning_round"
                    ],
                    "t_plan_p95_ms_per_round": latency["methods"][method][
                        "p95_tplan_ms_per_replanning_round"
                    ],
                    "penalized_cost_min": statistics.fmean(
                        float(row["penalized_cost_min"]) for row in rows
                    ),
                })
            proposed = grouped["proposed"]
            primary_difference, primary_ci = _paired_difference(
                grouped["baseline"], proposed
            )
            primary_effects.append({
                "configuration": key,
                "run_id": primary.name,
                "proposed_minus_baseline_cost_min": primary_difference,
                "bootstrap_ci_low_min": primary_ci[0],
                "bootstrap_ci_high_min": primary_ci[1],
            })
            gamma_zero = [row for row in zero_rows if row["method"] == "proposed"]
            difference, ci = _paired_difference(gamma_zero, proposed)
            gamma_effects.append({
                "configuration": key,
                "learned_gamma_run": primary.name,
                "gamma_zero_run": zero.name,
                "learned_minus_zero_cost_min": difference,
                "bootstrap_ci_low_min": ci[0],
                "bootstrap_ci_high_min": ci[1],
                "gamma_zero_cost_min": statistics.fmean(
                    float(row["penalized_cost_min"]) for row in gamma_zero
                ),
                "learned_gamma_cost_min": statistics.fmean(
                    float(row["penalized_cost_min"]) for row in proposed
                ),
            })
            frozen = json.loads((primary / "frozen_gamma_toggle_summary.json").read_text())
            frozen_effects.append({
                "configuration": key,
                "run_id": primary.name,
                "learned_gamma_value": frozen["learned_gamma_value"],
                "learned_minus_zero_cost_min": frozen["learned_minus_zero_cost_min"],
                "bootstrap_ci_low_min": frozen["paired_bootstrap_95_ci_min"][0],
                "bootstrap_ci_high_min": frozen["paired_bootstrap_95_ci_min"][1],
            })
            stability.extend(csv_rows(ablation_run(key) / "summary.csv"))
    return metrics, primary_effects, gamma_effects, frozen_effects, stability


def _finish(figure, stem: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_metrics(records: list[dict]) -> None:
    config_order = [tag(a, g, u, v) for a, g in SIZES for u, v in FLEETS]
    paired = {(row["configuration"], row["method"]): row for row in records}
    specs = [
        ("R_makespan_min", r"$R_{makespan}$ (min)"),
        ("T_wait_min_per_event", r"$\bar{T}_{wait}$ (min/event)"),
        ("sigma_UGV_busy_min", r"$\sigma_{UGV}$ busy time (min)"),
        ("t_plan_ms_per_round", r"$t_{plan}$ (ms/round)"),
        ("penalized_cost_min", "Penalized mission cost (min)"),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(17, 12), constrained_layout=True)
    x = np.arange(len(config_order))
    for axis, (field, ylabel) in zip(axes.flat, specs):
        baseline = [paired[(key, "baseline")][field] for key in config_order]
        proposed = [paired[(key, "proposed")][field] for key in config_order]
        axis.bar(x - .19, baseline, width=.38, label="Base-style route + nearest UGV", color="#677a9e")
        axis.bar(x + .19, proposed, width=.38, label="Queue-aware proposal", color="#158a7b")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [short_label(key) for key in config_order], fontsize=10)
        axis.grid(axis="y", alpha=.2)
    axis = axes.flat[-1]
    axis.axis("off")
    axis.text(.02, .88,
        "One training seed per configuration. 256 paired test scenarios each.\n"
        "Failures remain in averages; penalized cost adds 800 min per failure.\n"
        "Planning time uses 64 separate CPU inference scenarios per configuration.\n"
        "All nine size-fleet cells have separately trained policies.",
        va="top", fontsize=12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.03))
    _finish(figure, "full_matrix_metrics")

    for stem, chosen in [
        ("mission_time_and_wait", specs[:2]),
        ("load_and_planning_latency", specs[2:4]),
    ]:
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for axis, (field, ylabel) in zip(axes, chosen):
            baseline = [paired[(key, "baseline")][field] for key in config_order]
            proposed = [paired[(key, "proposed")][field] for key in config_order]
            axis.bar(x - .19, baseline, width=.38, color="#677a9e", label="Baseline")
            axis.bar(x + .19, proposed, width=.38, color="#158a7b", label="Proposed")
            axis.set_ylabel(ylabel)
            axis.set_xticks(x, [short_label(key) for key in config_order], fontsize=10)
            axis.grid(axis="y", alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=2,
                      bbox_to_anchor=(.5, 1.05))
        _finish(figure, stem)


def plot_gamma(records: list[dict]) -> None:
    x = np.arange(len(records))
    means = np.array([row["learned_minus_zero_cost_min"] for row in records])
    low = np.array([row["bootstrap_ci_low_min"] for row in records])
    high = np.array([row["bootstrap_ci_high_min"] for row in records])
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    axis.bar(x, means, color=np.where(means < 0, "#158a7b", "#b85c4a"))
    axis.errorbar(x, means, yerr=[means-low, high-means], fmt="none", color="black", capsize=3)
    axis.axhline(0, color="black", lw=1)
    axis.set_xticks(x, [short_label(row["configuration"]) for row in records], fontsize=10)
    axis.set_ylabel(r"Learned $\gamma_{queue}$ minus fixed zero, penalized cost (min)")
    axis.set_title("Queue-penalty ablation: same baseline start, separate joint fine-tunes")
    axis.grid(axis="y", alpha=.2)
    _finish(figure, "gamma_zero_ablation")


def plot_frozen_gamma(records: list[dict]) -> None:
    x = np.arange(len(records))
    means = np.array([row["learned_minus_zero_cost_min"] for row in records])
    low = np.array([row["bootstrap_ci_low_min"] for row in records])
    high = np.array([row["bootstrap_ci_high_min"] for row in records])
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    axis.bar(x, means, color=np.where(means < 0, "#158a7b", "#b85c4a"))
    axis.errorbar(x, means, yerr=[means-low, high-means], fmt="none", color="black", capsize=3)
    axis.axhline(0, color="black", lw=1)
    axis.set_xticks(x, [short_label(row["configuration"]) for row in records], fontsize=10)
    axis.set_ylabel(r"Learned $\gamma_{queue}$ minus fixed zero, penalized cost (min)")
    axis.set_title("Direct queue term effect with route and head weights frozen")
    axis.grid(axis="y", alpha=.2)
    _finish(figure, "frozen_gamma_toggle")


def plot_primary_effects(records: list[dict]) -> None:
    x = np.arange(len(records))
    means = np.array([row["proposed_minus_baseline_cost_min"] for row in records])
    low = np.array([row["bootstrap_ci_low_min"] for row in records])
    high = np.array([row["bootstrap_ci_high_min"] for row in records])
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    axis.bar(x, means, color=np.where(means < 0, "#158a7b", "#b85c4a"))
    axis.errorbar(x, means, yerr=[means-low, high-means], fmt="none", color="black", capsize=3)
    axis.axhline(0, color="black", lw=1)
    axis.set_xticks(x, [short_label(row["configuration"]) for row in records], fontsize=10)
    axis.set_ylabel("Proposed minus baseline penalized mission cost (min)")
    axis.set_title("Paired test effect, 256 scenarios per configuration")
    axis.grid(axis="y", alpha=.2)
    _finish(figure, "primary_policy_effects")


def plot_stability(records: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in records:
        name = row["run_id"].split("_pair_seed11")[0].split("_assignment_training_ablation")[0]
        key = row.get("configuration") or name.replace("20261010_", "").replace("20261011_", "")
        grouped[(key, row["method"])].append(row)
    keys = [tag(a, g, u, v) for a, g in SIZES for u, v in FLEETS]
    x = np.arange(len(keys))
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    for method, shift, color in [("sinkhorn_soft", -.18, "#158a7b"),
                                 ("hard_argmax", .18, "#b85c4a")]:
        tail = [statistics.fmean(float(r["tail_loss_std"]) for r in grouped[(key, method)])
                for key in keys]
        collision = [statistics.fmean(float(r["mean_argmax_collision_rate"])
                    for r in grouped[(key, method)]) for key in keys]
        axes[0].bar(x+shift, tail, width=.36, color=color, label=method)
        axes[1].bar(x+shift, collision, width=.36, color=color)
    axes[0].set_ylabel("Last-20-update training-loss SD")
    axes[1].set_ylabel("Duplicate-slot fraction in training batches")
    for axis in axes:
        axis.set_xticks(x, [short_label(key) for key in keys], fontsize=10)
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(ncol=2)
    figure.suptitle("Assignment-head surrogate training, three seeds per configuration")
    _finish(figure, "soft_vs_hard_training_stability")

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    for method, shift, color in [("sinkhorn_soft", -.18, "#158a7b"),
                                 ("hard_argmax", .18, "#b85c4a")]:
        values = [statistics.fmean(float(r["validation_hungarian_cost"])
                  for r in grouped[(key, method)]) for key in keys]
        axis.bar(x+shift, values, width=.36, color=color, label=method)
    axis.set_xticks(x, [short_label(key) for key in keys], fontsize=10)
    axis.set_ylabel("Validation Hungarian slot-cost surrogate")
    axis.set_title("Both heads decoded with the same feasible Hungarian assignment")
    axis.grid(axis="y", alpha=.2)
    axis.legend(ncol=2)
    _finish(figure, "soft_vs_hard_validation_cost")


def plot_training_curves() -> None:
    figure, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    for axis, (aerial, ground), (uavs, ugvs) in zip(
        axes.flat, [size for size in SIZES for _ in FLEETS], FLEETS * 3
    ):
        key = tag(aerial, ground, uavs, ugvs)
        for run, label, color in [
            (primary_run(key), "learned queue penalty", "#158a7b"),
            (zero_run(key), r"fixed $\gamma=0$", "#b85c4a"),
        ]:
            with (run / "training.jsonl").open() as stream:
                points = [json.loads(line) for line in stream]
            points = [row for row in points if row["stage"] == "proposed"]
            xs = [row["update"] for row in points]
            ys = [row["mean_cost_min"] for row in points]
            width = 25
            smoothed = np.convolve(ys, np.ones(width)/width, mode="valid")
            axis.plot(xs[width-1:], smoothed, label=label, color=color, lw=1.5)
        axis.set_title(key, fontsize=10)
        axis.set_xlabel("Policy-gradient update")
        axis.set_ylabel("Training cost, 25-update mean (min)")
        axis.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.02))
    _finish(figure, "joint_training_curves")


def plot_soft_hard_loss_curves() -> None:
    figure, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
    for axis, (aerial, ground), (uavs, ugvs) in zip(
        axes.flat, [size for size in SIZES for _ in FLEETS], FLEETS * 3
    ):
        key = tag(aerial, ground, uavs, ugvs)
        records = csv_rows(ablation_run(key) / "training.csv")
        for method, label, color in [
            ("sinkhorn_soft", "Soft Sinkhorn", "#158a7b"),
            ("hard_argmax", "Hard ArgMax", "#b85c4a"),
        ]:
            by_seed = defaultdict(dict)
            for row in records:
                if row["method"] == method:
                    by_seed[int(row["seed"])][int(row["update"])] = float(
                        row["train_surrogate_loss"]
                    )
            if len(by_seed) != 3 or any(len(points) != 100 for points in by_seed.values()):
                raise ValueError(f"missing soft/hard loss trace for {key}: {method}")
            values = np.array([[points[update] for update in range(1, 101)]
                               for points in by_seed.values()])
            means = values.mean(axis=0)
            deviation = values.std(axis=0)
            smooth = np.ones(5) / 5
            x = np.arange(5, 101)
            y = np.convolve(means, smooth, mode="valid")
            spread = np.convolve(deviation, smooth, mode="valid")
            axis.plot(x, y, color=color, lw=1.6, label=label)
            axis.fill_between(x, y - spread, y + spread, color=color, alpha=.12)
        axis.set_title(key, fontsize=10)
        axis.set_xlabel("Head update")
        axis.set_ylabel("Surrogate loss")
        axis.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2,
                  bbox_to_anchor=(.5, 1.07))
    _finish(figure, "soft_vs_hard_loss_curves")


def main() -> None:
    metrics, primary, gamma, frozen, stability = gather()
    save_csv(HERE / "full_matrix_summary.csv", metrics)
    save_csv(HERE / "primary_policy_effects.csv", primary)
    save_csv(HERE / "gamma_zero_effects.csv", gamma)
    save_csv(HERE / "frozen_gamma_effects.csv", frozen)
    save_csv(HERE / "hard_argmax_stability.csv", stability)
    plot_metrics(metrics)
    plot_primary_effects(primary)
    plot_gamma(gamma)
    plot_frozen_gamma(frozen)
    plot_stability(stability)
    plot_training_curves()
    plot_soft_hard_loss_curves()
    print(f"Wrote {len(metrics)} policy rows, {len(gamma)} gamma contrasts, and ten figures")


if __name__ == "__main__":
    main()
