import torch

from uav_ugv.sinkhorn import gumbel_sinkhorn, pad_with_dummy_rows


def test_dummy_padding_makes_rectangular_matrix_square() -> None:
    logits = torch.zeros(2, 4, 8)
    padded = pad_with_dummy_rows(logits)
    assert padded.shape == (2, 8, 8)


def test_sinkhorn_has_unit_row_and_column_sums() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8)
    matrix = gumbel_sinkhorn(
        logits,
        temperature=0.5,
        iterations=100,
        add_noise=False,
    )
    torch.testing.assert_close(
        matrix.sum(dim=-1),
        torch.ones(2, 8),
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        matrix.sum(dim=-2),
        torch.ones(2, 8),
        atol=1e-4,
        rtol=1e-4,
    )
