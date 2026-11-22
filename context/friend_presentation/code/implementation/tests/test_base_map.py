import math

from uav_ugv.base_map import BasePaperScenarioGenerator, base_paper_road_network


def test_base_map_is_connected_and_inside_20_km_workspace() -> None:
    road = base_paper_road_network()
    assert road.graph.number_of_nodes() == 28
    assert road.graph.number_of_edges() == 27
    for node in road.graph.nodes:
        x, y = road.coordinates(node)
        assert 0.0 <= x <= 20.0
        assert 0.0 <= y <= 20.0


def test_u15g5_generation_is_reproducible_and_bounded() -> None:
    generator = BasePaperScenarioGenerator()
    left = generator.generate(15, 5, seed=19)
    right = generator.generate(15, 5, seed=19)
    assert left.aerial_tasks == right.aerial_tasks
    assert left.ground_tasks == right.ground_tasks
    assert len(left.aerial_tasks) == 15
    assert len(left.ground_tasks) == 5
    ground_coordinates = {(task.x, task.y) for task in left.ground_tasks}
    assert {(3.0, 8.0), (10.0, 12.0), (10.0, 5.0)} <= ground_coordinates
    for aerial in left.aerial_tasks:
        assert 0.0 <= aerial.x <= 20.0
        assert 0.0 <= aerial.y <= 20.0
        assert min(
            math.hypot(aerial.x - ground.x, aerial.y - ground.y)
            for ground in left.ground_tasks
        ) <= 7.0 + 1e-4
