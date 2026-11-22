from uav_ugv.base_map import BasePaperScenarioGenerator
from uav_ugv.mission_simulator import TwoDMissionSimulator
from uav_ugv.policies import QueueAwareHungarianPolicy


def test_complete_2d_mission_visits_every_task() -> None:
    scenario = BasePaperScenarioGenerator().generate(
        aerial_task_count=5,
        ground_task_count=5,
        seed=4,
    )
    result = TwoDMissionSimulator(
        scenario=scenario,
        number_of_uavs=2,
        number_of_ugvs=2,
        assignment_policy=QueueAwareHungarianPolicy(),
    ).run()
    assert result.completed, result.failure_reason
    assert result.makespan > 0.0
    assert result.charging_rounds
    assert len(result.visited_task_ids) == len(scenario.tasks)
