import torch

from uav_ugv.model import QueueAwareCrossAttention
from uav_ugv.training import AssignmentBatch, AssignmentTrainer, TrainerSettings


def test_assignment_head_training_step_updates_without_shape_errors() -> None:
    torch.manual_seed(3)
    model = QueueAwareCrossAttention(
        uav_feature_dim=5,
        ugv_feature_dim=4,
        embedding_dim=16,
        attention_dim=8,
        max_slots_per_ugv=2,
    )
    batch = AssignmentBatch(
        uav_features=torch.randn(3, 2, 5),
        ugv_features=torch.randn(3, 2, 4),
        normalized_queue_latency=torch.rand(3, 2, 2, 2),
        normalized_assignment_cost=torch.rand(3, 2, 4),
    )
    trainer = AssignmentTrainer(
        model,
        TrainerSettings(sinkhorn_iterations=10),
        device="cpu",
    )
    metrics = trainer.train_step(batch)
    assert metrics["loss"] >= 0.0
    assert metrics["gradient_norm"] >= 0.0
