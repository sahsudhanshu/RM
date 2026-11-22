"""Metric extraction for paired policy comparisons."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable

from ..domain import RoundResult


def result_metrics(result: RoundResult) -> dict[str, float]:
    service_counts = Counter(item.ugv_id for item in result.services)
    all_ugv_ids = [state.ugv_id for state in result.final_ugv_states]
    loads = [float(service_counts.get(ugv_id, 0)) for ugv_id in all_ugv_ids]
    return {
        "makespan": result.makespan,
        "mean_wait": result.mean_waiting_time,
        "max_wait": result.max_waiting_time,
        "total_road_distance": result.total_road_distance,
        "ugv_service_load_std": statistics.pstdev(loads) if loads else 0.0,
    }


def aggregate_results(results: Iterable[RoundResult]) -> dict[str, dict[str, float]]:
    rows = [result_metrics(result) for result in results]
    if not rows:
        return {}
    aggregate: dict[str, dict[str, float]] = {}
    for metric in rows[0]:
        values = [row[metric] for row in rows]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        half_width = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        aggregate[metric] = {
            "mean": mean,
            "std": std,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
        }
    return aggregate
