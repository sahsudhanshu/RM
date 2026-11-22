import torch

from uav_ugv.base_map import BasePaperScenarioGenerator
from uav_ugv.domain import RechargeRequest, UGVState
from uav_ugv.mission import UAVMissionState
from uav_ugv.model import QueueAwareCrossAttention
from uav_ugv.mission_simulator import TwoDMissionSimulator
from uav_ugv.policies import NearestEuclideanPolicy
from uav_ugv.simulator import ChargingRoundSimulator
from uav_ugv.training.assignment_policy import (
    NearestWithLearnedOrderPolicy,
    SampledAssignmentPolicy,
)
from uav_ugv.training.route_policy import LearnedRoutePolicy, RoutingTransformer


def test_route_decoder_only_selects_feasible_actions() -> None:
    scenario = BasePaperScenarioGenerator().generate(5, 5, seed=8)
    model = RoutingTransformer(embedding_dim=16, heads=2, layers=1)
    policy = LearnedRoutePolicy(model, "cpu", mode="sample")
    policy.start_episode(scenario)
    x, y = scenario.depot_coordinates
    uav = UAVMissionState(0, x, y, 0.0, 287_700.0)
    allowed = [scenario.aerial_tasks[0], scenario.aerial_tasks[1]]
    landing = [scenario.ground_tasks[0]]
    selected = policy.choose_next(
        uav, set(), allowed, landing, allowed[0]
    )
    assert selected.task_id in {task.task_id for task in allowed + landing}
    (-policy.episode_log_probability()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_both_assignment_training_policies_produce_valid_service_orders() -> None:
    scenario = BasePaperScenarioGenerator().generate(5, 5, seed=9)
    nodes = [task.road_node for task in scenario.ground_tasks[:2]]
    requests = [
        RechargeRequest(uav_id=i, landing_node=int(node), touchdown_time=10.0 + i)
        for i, node in enumerate(nodes)
    ]
    ugvs = [UGVState(ugv_id=i, current_node=scenario.depot_node) for i in range(2)]
    simulator = ChargingRoundSimulator(scenario.road_network, 0.27, 15.0)
    route_model = RoutingTransformer(embedding_dim=16, heads=2, layers=1)
    baseline = NearestWithLearnedOrderPolicy(route_model, "cpu", mode="sample")
    head = QueueAwareCrossAttention(5, 4, embedding_dim=16, attention_dim=8)
    proposed = SampledAssignmentPolicy(head, "cpu", mode="sample")
    for policy in (baseline, proposed):
        policy.start_episode()
        result = simulator.run(requests, ugvs, policy)
        assert len(result.services) == len(requests)
        assert {service.uav_id for service in result.services} == {0, 1}
        (-policy.episode_log_probability()).backward()
    assert any(parameter.grad is not None for parameter in route_model.parameters())
    assert any(parameter.grad is not None for parameter in head.parameters())
    assert NearestEuclideanPolicy().assign(
        requests, ugvs, scenario.road_network, 0.27, 15.0
    )


def test_learned_route_must_make_task_progress_before_recharging() -> None:
    class EarlyLandingPolicy:
        def start_episode(self, scenario):
            self.scenario = scenario

        def choose_next(self, uav, visited, feasible_tasks, reachable_grounds, teacher_choice):
            del uav, visited, teacher_choice
            return reachable_grounds[0] if reachable_grounds else feasible_tasks[0]

    scenario = BasePaperScenarioGenerator().generate(5, 5, seed=4)
    result = TwoDMissionSimulator(
        scenario, 2, 2, NearestEuclideanPolicy(),
        route_policy=EarlyLandingPolicy(),
    ).run()
    assert result.completed, result.failure_reason


def test_fixed_zero_queue_penalty_removes_latency_term() -> None:
    model = QueueAwareCrossAttention(
        5, 4, embedding_dim=16, attention_dim=8,
        zero_queue_penalty=True,
    )
    uav = torch.randn(1, 2, 5)
    ugv = torch.randn(1, 2, 4)
    zero = torch.zeros(1, 2, 2, 2)
    high = torch.full((1, 2, 2, 2), 1000.0)
    assert torch.equal(model(uav, ugv, zero), model(uav, ugv, high))
    assert float(model.queue_weight) == 0.0
    assert not model.raw_queue_weight.requires_grad
