"""Time neural policy decisions without charging simulator work to planning."""

from __future__ import annotations

from time import perf_counter_ns


class TimedRoutePolicy:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.elapsed_ns = 0
        self.call_count = 0

    def start_episode(self, scenario) -> None:
        self.policy.start_episode(scenario)

    def choose_next(self, *args, **kwargs):
        started = perf_counter_ns()
        try:
            return self.policy.choose_next(*args, **kwargs)
        finally:
            self.elapsed_ns += perf_counter_ns() - started
            self.call_count += 1


class TimedAssignmentPolicy:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.name = policy.name
        self.elapsed_ns = 0
        self.call_count = 0

    def assign(self, *args, **kwargs):
        started = perf_counter_ns()
        try:
            return self.policy.assign(*args, **kwargs)
        finally:
            self.elapsed_ns += perf_counter_ns() - started
            self.call_count += 1


def planning_metrics(route: TimedRoutePolicy, assignment: TimedAssignmentPolicy,
                     charging_rounds: int) -> dict[str, float | int]:
    total_ms = (route.elapsed_ns + assignment.elapsed_ns) / 1_000_000
    return {
        "tplan_ms_per_replanning_round": total_ms / max(1, charging_rounds),
        "route_decision_ms_mean": route.elapsed_ns / 1_000_000 / max(1, route.call_count),
        "assignment_decision_ms_mean": (
            assignment.elapsed_ns / 1_000_000 / max(1, assignment.call_count)
        ),
        "route_decision_count": route.call_count,
        "assignment_decision_count": assignment.call_count,
    }
