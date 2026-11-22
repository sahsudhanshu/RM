"""Static 2D map and mission-result plots."""

from __future__ import annotations

from pathlib import Path

from .mission import MissionResult, MissionScenario


def plot_scenario(
    scenario: MissionScenario,
    output_path: str | Path,
    result: MissionResult | None = None,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plotting requires the optional dependency: pip install -e '.[viz]'"
        ) from error

    figure, axis = plt.subplots(figsize=(8, 8))
    for left, right in scenario.road_network.graph.edges:
        x1, y1 = scenario.road_network.coordinates(left)
        x2, y2 = scenario.road_network.coordinates(right)
        axis.plot([x1, x2], [y1, y2], color="0.65", linewidth=2, zorder=1)

    if result is not None:
        colors = plt.get_cmap("tab10")
        for leg in result.flight_legs:
            axis.plot(
                [leg.start_x, leg.end_x],
                [leg.start_y, leg.end_y],
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                color=colors(leg.uav_id % 10),
                zorder=2,
            )
        current_ugv_node = {
            state.ugv_id: scenario.depot_node
            for state in result.charging_rounds[0].final_ugv_states
        } if result.charging_rounds else {}
        ugv_label_added = False
        for charging_round in result.charging_rounds:
            for service in sorted(
                charging_round.services,
                key=lambda item: (item.ugv_id, item.slot_position),
            ):
                path = scenario.road_network.shortest_path(
                    current_ugv_node[service.ugv_id], service.landing_node
                )
                path_coordinates = [
                    scenario.road_network.coordinates(node) for node in path
                ]
                axis.plot(
                    [coordinate[0] for coordinate in path_coordinates],
                    [coordinate[1] for coordinate in path_coordinates],
                    color="tab:green",
                    linewidth=3.0,
                    alpha=0.75,
                    label="Executed UGV route" if not ugv_label_added else None,
                    zorder=3,
                )
                ugv_label_added = True
                current_ugv_node[service.ugv_id] = service.landing_node

    axis.scatter(
        [task.x for task in scenario.aerial_tasks],
        [task.y for task in scenario.aerial_tasks],
        facecolors="none",
        edgecolors="tab:red",
        marker="o",
        s=65,
        label="Aerial tasks",
        zorder=4,
    )
    axis.scatter(
        [task.x for task in scenario.ground_tasks],
        [task.y for task in scenario.ground_tasks],
        color="tab:blue",
        marker="s",
        s=55,
        label="Ground / rendezvous tasks",
        zorder=5,
    )
    depot_x, depot_y = scenario.depot_coordinates
    axis.scatter(
        [depot_x],
        [depot_y],
        color="black",
        marker="*",
        s=180,
        label="Depot",
        zorder=6,
    )
    axis.set_xlim(0, scenario.width_km)
    axis.set_ylim(0, scenario.height_km)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (km)")
    axis.set_ylabel("y (km)")
    title = f"2D UAV-UGV Scenario — U{len(scenario.aerial_tasks)}G{len(scenario.ground_tasks)}"
    if result is not None:
        title += f" — makespan {result.makespan:.1f} min"
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
