import torch

from uav_ugv.model import QueueAwareCrossAttention


def test_model_scores_every_virtual_slot() -> None:
    model = QueueAwareCrossAttention(
        uav_feature_dim=5,
        ugv_feature_dim=4,
        embedding_dim=16,
        attention_dim=8,
        max_slots_per_ugv=4,
    )
    uav = torch.randn(2, 4, 5)
    ugv = torch.randn(2, 2, 4)
    queue = torch.rand(2, 4, 2, 4)
    scores = model(uav, ugv, queue)
    assert scores.shape == (2, 4, 8)

    soft = model.soft_assignment(
        uav,
        ugv,
        queue,
        add_gumbel_noise=False,
        sinkhorn_iterations=100,
    )
    assert soft.shape == (2, 8, 8)
    torch.testing.assert_close(
        soft.sum(dim=-1),
        torch.ones(2, 8),
        atol=1e-4,
        rtol=1e-4,
    )
