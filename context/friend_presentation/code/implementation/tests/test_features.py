from uav_ugv.domain import RechargeRequest, UGVState
from uav_ugv.features import build_model_inputs
from uav_ugv.road_network import RoadNetwork
from uav_ugv.virtual_slots import build_slot_matrices


def test_physical_state_converts_to_expected_model_shapes() -> None:
    road = RoadNetwork.from_edges(
        {0: (0.0, 0.0), 1: (1.0, 0.0)},
        [(0, 1)],
    )
    requests = [
        RechargeRequest(0, 0, 1.0),
        RechargeRequest(1, 1, 2.0),
    ]
    ugvs = [UGVState(0, 0), UGVState(1, 1)]
    matrices = build_slot_matrices(requests, ugvs, road, 1.0, 5.0)
    inputs = build_model_inputs(requests, ugvs, road, matrices, 5.0)
    assert inputs.uav_features.shape == (1, 2, 5)
    assert inputs.ugv_features.shape == (1, 2, 4)
    assert inputs.normalized_queue_latency.shape == (1, 2, 2, 2)
