"""Write a compact, data-derived report from completed matrix CSVs."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent


def rows(name: str) -> list[dict]:
    with (HERE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def pair(record: dict, field: str) -> str:
    return f"{float(record[field]):.2f}"


def main() -> None:
    metrics = rows("full_matrix_summary.csv")
    primary = rows("primary_policy_effects.csv")
    gamma = rows("gamma_zero_effects.csv")
    frozen = rows("frozen_gamma_effects.csv")
    stability = rows("hard_argmax_stability.csv")
    by_metric = {(row["configuration"], row["method"]): row for row in metrics}
    by_frozen = {row["configuration"]: row for row in frozen}
    by_gamma = {row["configuration"]: row for row in gamma}
    by_stability = defaultdict(list)
    for row in stability:
        key = row["run_id"].split("_pair_seed11")[0]
        key = row.get("configuration") or key.replace("20261010_", "").replace("20261011_", "")
        by_stability[(key, row["method"])].append(row)

    lines = [
        "# Nine-cell policy results",
        "",
        "Baseline and proposed policies were trained separately for each of",
        "the nine size-fleet configurations, including U75G25.",
        "Every row has one training seed, 256 paired test scenarios, and an",
        "800-minute penalty for a failed mission in penalized cost.",
        "",
        "## Requested evaluation metrics",
        "",
        "Values are **baseline / proposed**. Makespan is the slowest-agent",
        "mission time; wait is pooled across observed landing events; UGV load",
        "spread is the within-mission SD of road travel plus charging busy time.",
        "Recharge event counts and total wait per UAV are retained in the CSV",
        "to explain cases where wait per event and mission cost move differently.",
        "Planning latency is the mean CPU policy time per recharge round on a",
        "separate 64-scenario benchmark after training stopped; p95 is retained",
        "in `full_matrix_summary.csv`.",
        "",
        "| Configuration | Completed / 256 | R makespan (min) | Mean wait/event (min) | UGV busy SD (min) | t plan (ms/round) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for effect in primary:
        key = effect["configuration"]
        baseline = by_metric[(key, "baseline")]
        proposed = by_metric[(key, "proposed")]
        lines.append(
            f"| {key} | {baseline['completed']} / {proposed['completed']} | "
            f"{pair(baseline, 'R_makespan_min')} / {pair(proposed, 'R_makespan_min')} | "
            f"{pair(baseline, 'T_wait_min_per_event')} / {pair(proposed, 'T_wait_min_per_event')} | "
            f"{pair(baseline, 'sigma_UGV_busy_min')} / {pair(proposed, 'sigma_UGV_busy_min')} | "
            f"{pair(baseline, 't_plan_ms_per_round')} / {pair(proposed, 't_plan_ms_per_round')} |"
        )
    lines += [
        "",
        "A one-UGV fleet has structurally zero UGV-load SD; it cannot show",
        "balancing across vehicles. Failed missions can have a short raw",
        "makespan, so completion and penalized cost should be read alongside it.",
        "",
        "## Paired policy effect",
        "",
        "Difference is proposed minus baseline mean penalized mission cost.",
        "Intervals are 95% paired-scenario bootstrap intervals with 2,000",
        "resamples; they do not include variability across training seeds.",
        "",
        "| Configuration | Baseline cost (min) | Proposed cost (min) | Difference (min) | 95% interval (min) |",
        "|---|---:|---:|---:|---:|",
    ]
    for effect in primary:
        key = effect["configuration"]
        baseline = by_metric[(key, "baseline")]
        proposed = by_metric[(key, "proposed")]
        lines.append(
            f"| {key} | {pair(baseline, 'penalized_cost_min')} | "
            f"{pair(proposed, 'penalized_cost_min')} | "
            f"{pair(effect, 'proposed_minus_baseline_cost_min')} | "
            f"[{pair(effect, 'bootstrap_ci_low_min')}, {pair(effect, 'bootstrap_ci_high_min')}] |"
        )
    improved = sum(float(effect["proposed_minus_baseline_cost_min"]) < 0 for effect in primary)
    lines += ["", f"The proposed policy has lower penalized cost in {improved} of {len(primary)} cells."]
    for effect in primary:
        if float(effect["proposed_minus_baseline_cost_min"]) <= 0:
            continue
        key = effect["configuration"]
        baseline = by_metric[(key, "baseline")]
        proposed = by_metric[(key, "proposed")]
        lines += [
            f"The {key.replace('_', ' ')} cell is the exception: its proposed policy has",
            f"higher mean cost and completes {proposed['completed']} missions versus",
            f"{baseline['completed']} for the baseline. Its paired interval includes zero,",
            "and one training seed is insufficient to judge repeatability.",
        ]
    lines.append("Lower penalized cost does not imply lower wait per landing event in every cell.")
    lines += [
        "",
        "## Queue-penalty ablation",
        "",
        "Both columns are learned-γ minus γ=0 penalized-cost differences.",
        "Retrained compares two models fine-tuned from the same baseline;",
        "frozen toggle keeps the selected proposed weights fixed and changes",
        "only the explicit coefficient on a separate 256-scenario split.",
        "",
        "| Configuration | Retrained difference (min) [95% interval] | Frozen toggle (min) [95% interval] |",
        "|---|---:|---:|",
    ]
    for effect in primary:
        key = effect["configuration"]
        a = by_gamma[key]
        b = by_frozen[key]
        lines.append(
            f"| {key} | {pair(a, 'learned_minus_zero_cost_min')} "
            f"[{pair(a, 'bootstrap_ci_low_min')}, {pair(a, 'bootstrap_ci_high_min')}] | "
            f"{pair(b, 'learned_minus_zero_cost_min')} "
            f"[{pair(b, 'bootstrap_ci_low_min')}, {pair(b, 'bootstrap_ci_high_min')}] |"
        )
    lines += [
        "",
        "## Soft Sinkhorn versus hard ArgMax head training",
        "",
        "Each method used three training seeds and the same baseline-generated",
        "recharge states. This is a head-only surrogate stability ablation.",
        "Hard row-wise ArgMax may select the same virtual slot for multiple",
        "UAVs. Validation cost uses the same feasible Hungarian decoder for",
        "both heads. Surrogate cost is normalized by 500 minutes.",
        "",
        "| Configuration | Tail loss SD: soft / hard | Hard duplicate-slot fraction | Validation cost: soft / hard | Nonfinite updates: soft / hard |",
        "|---|---:|---:|---:|---:|",
    ]
    for effect in primary:
        key = effect["configuration"]
        soft = by_stability[(key, "sinkhorn_soft")]
        hard = by_stability[(key, "hard_argmax")]
        avg = lambda group, field: statistics.fmean(float(row[field]) for row in group)
        lines.append(
            f"| {key} | {avg(soft, 'tail_loss_std'):.3f} / {avg(hard, 'tail_loss_std'):.3f} | "
            f"{avg(hard, 'mean_argmax_collision_rate'):.3f} | "
            f"{avg(soft, 'validation_hungarian_cost'):.3f} / "
            f"{avg(hard, 'validation_hungarian_cost'):.3f} | "
            f"{sum(int(row['nonfinite_updates']) for row in soft)} / "
            f"{sum(int(row['nonfinite_updates']) for row in hard)} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "The baseline is a local base-style reconstruction, and the local",
        "budget is far below the published 5.12 million instances per model.",
        "One training seed per size and fleet limits generalization claims.",
        "The head-only soft/hard ablation does not establish full-mission",
        "policy-gradient stability.",
        "Raw scenario records, checkpoints, selected validation scores, and",
        "event traces remain in `results/runs/`. Figures are in",
        "`results/figures/`; aggregate CSVs sit beside this report.",
        "The two U15G5 pilot cells dated 2026-10-10 retain their own archived source",
        "snapshots. Their source hashes differ from the current training source",
        "because later code added baseline-only and exact-zero-penalty run options.",
        "",
    ]
    output = HERE / "full_matrix_report.md"
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
