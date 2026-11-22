from uav_ugv.domain import RechargeRequest, UGVState
from uav_ugv.policies import NearestEuclideanPolicy, QueueAwareHungarianPolicy
from uav_ugv.road_network import RoadNetwork
from uav_ugv.simulator import ChargingRoundSimulator


def line_network() -> RoadNetwork:
    return RoadNetwork.from_edges(
        coordinates={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)},
        edges=[(0, 1), (1, 2)],
    )


def test_one_ugv_recharges_two_uavs_sequentially() -> None:
    road = line_network()
    simulator = ChargingRoundSimulator(road, ugv_speed=1.0, recharge_time=5.0)
    requests = [
        RechargeRequest(0, landing_node=0, touchdown_time=0.0),
        RechargeRequest(1, landing_node=0, touchdown_time=0.0),
    ]
    result = simulator.run(requests, [UGVState(0, current_node=0)], NearestEuclideanPolicy())
    services = sorted(result.services, key=lambda item: item.slot_position)
    assert [item.charging_end_time for item in services] == [5.0, 10.0]
    assert [item.waiting_time for item in services] == [0.0, 5.0]
    assert result.makespan == 10.0


def test_queue_aware_hungarian_uses_multiple_physical_ugvs() -> None:
    road = line_network()
    simulator = ChargingRoundSimulator(road, ugv_speed=1.0, recharge_time=10.0)
    requests = [
        RechargeRequest(0, landing_node=1, touchdown_time=0.0),
        RechargeRequest(1, landing_node=1, touchdown_time=0.0),
    ]
    result = simulator.run(
        requests,
        [UGVState(0, current_node=0), UGVState(1, current_node=2)],
        QueueAwareHungarianPolicy(),
    )
    assert {assignment.ugv_id for assignment in result.assignments} == {0, 1}
    assert result.mean_waiting_time == 1.0
    assert result.makespan == 11.0
