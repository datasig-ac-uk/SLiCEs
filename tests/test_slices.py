import pytest
import torch

from slices.slices import SLiCE, SLiCEBlock, StackedSLiCE


def _rand_x(batch: int, seq: int, dim: int, *, seed: int = 0, dtype=torch.float32):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn(batch, seq, dim, generator=g, dtype=dtype)


def _assert_no_nan(t: torch.Tensor):
    assert torch.isfinite(t).all(), "Tensor contains NaN/Inf"


def _assert_grads_exist(module: torch.nn.Module):
    # At least one parameter should get a grad (and usually all of them)
    grads = [p.grad for p in module.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "No parameter gradients found"


# -----------------------
# SLiCE: constructor paths
# -----------------------


def test_slice_raises_if_hidden_dim_not_divisible_by_block_size():
    with pytest.raises(ValueError, match="hidden_dim must be divisible"):
        SLiCE(input_dim=5, hidden_dim=6, block_size=4)


def test_slice_raises_if_block_size_less_than_one():
    with pytest.raises(ValueError, match="block_size must be at least 1"):
        SLiCE(input_dim=3, hidden_dim=3, block_size=0)


# -----------------------
# SLiCE: forward paths
# -----------------------


def test_slice_forward_block_diagonal_block_size_gt1_bias_true_with_grads():
    # diagonal_dense=False, block_size>1 hits the 4D block-matmul path
    x = _rand_x(batch=3, seq=5, dim=4, seed=2).requires_grad_(True)

    m = SLiCE(
        input_dim=4,
        hidden_dim=4,
        block_size=2,
        diagonal_dense=False,
        bias=True,
    )

    y = m(x)
    assert y.shape == (3, 5, 4)
    _assert_no_nan(y)

    # Gradient check
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    _assert_grads_exist(m)


def test_slice_forward_block_size_one_bias_false():
    # diagonal_dense=False, block_size==1 hits elementwise state_transition path
    x = _rand_x(batch=2, seq=4, dim=3, seed=3)

    m = SLiCE(
        input_dim=3,
        hidden_dim=3,
        block_size=1,
        diagonal_dense=False,
        bias=False,  # covers bias-off path
    )
    y = m(x)
    assert y.shape == (2, 4, 3)
    _assert_no_nan(y)


def test_slice_forward_diagonal_dense_bias_true_with_grads():
    # diagonal_dense=True hits diag + dense block path
    x = _rand_x(batch=2, seq=6, dim=5, seed=4).requires_grad_(True)

    m = SLiCE(
        input_dim=5,
        hidden_dim=8,
        block_size=4,
        diagonal_dense=True,
        bias=True,
    )

    y = m(x)
    assert y.shape == (2, 6, 8)
    _assert_no_nan(y)

    # Gradient check: ensure both vf_A_diag and vf_A_dense participate
    y.mean().backward()
    assert x.grad is not None
    assert m.vf_A_diag.weight.grad is not None
    assert m.vf_A_dense.weight.grad is not None
    _assert_grads_exist(m)


def test_slice_diagonal_dense_edge_case_hidden_dim_equals_block_size():
    # Edge case: hidden_dim - block_size == 0, so "diagonal" part is empty
    x = _rand_x(batch=2, seq=3, dim=2, seed=5)

    m = SLiCE(
        input_dim=2,
        hidden_dim=2,  # equals block_size
        block_size=2,
        diagonal_dense=True,
        bias=False,
    )

    y = m(x)
    assert y.shape == (2, 3, 2)
    _assert_no_nan(y)


def test_slice_diagonal_dense_block_size_one_runs():
    x = _rand_x(batch=2, seq=4, dim=3, seed=9)

    m = SLiCE(
        input_dim=3,
        hidden_dim=3,
        block_size=1,
        diagonal_dense=True,
        bias=True,
    )
    y = m(x)
    assert y.shape == (2, 4, 3)
    _assert_no_nan(y)


# -----------------------
# SLiCEBlock: both GLU and tanh paths
# -----------------------


@pytest.mark.parametrize("use_glu", [False, True])
@pytest.mark.parametrize("diagonal_dense", [False, True])
def test_slice_block_forward_covers_glu_and_tanh(use_glu: bool, diagonal_dense: bool):
    x = _rand_x(batch=2, seq=4, dim=6, seed=7)

    block = SLiCEBlock(
        input_dim=6,
        block_size=2 if not diagonal_dense else 4,
        diagonal_dense=diagonal_dense,
        dropout_rate=0.05,
        use_glu=use_glu,
    )

    y = block(x)
    assert y.shape == x.shape
    _assert_no_nan(y)


# -----------------------
# StackedSLiCE: tokens vs continuous values
# -----------------------


def test_stacked_slice_tokens_path():
    # tokens=True uses nn.Embedding
    batch, seq = 2, 5
    vocab = 11
    hidden = 8
    label = 3

    x = torch.randint(low=0, high=vocab, size=(batch, seq))

    m = StackedSLiCE(
        num_blocks=2,
        data_dim=vocab,
        hidden_dim=hidden,
        label_dim=label,
        tokens=True,
        block_size=4,
        diagonal_dense=False,
        dropout_rate=0.0,
        use_glu=True,
    )
    m.eval()

    y = m(x)
    assert y.shape == (batch, seq, label)
    _assert_no_nan(y)


def test_stacked_slice_continuous_path():
    # tokens=False uses nn.Linear embedding
    batch, seq = 2, 4
    data_dim = 6
    hidden = 8
    label = 5

    x = _rand_x(batch=batch, seq=seq, dim=data_dim, seed=8)

    m = StackedSLiCE(
        num_blocks=3,
        data_dim=data_dim,
        hidden_dim=hidden,
        label_dim=label,
        tokens=False,
        block_size=4,
        diagonal_dense=True,
        dropout_rate=0.0,
        use_glu=False,
    )
    m.eval()

    y = m(x)
    assert y.shape == (batch, seq, label)
    _assert_no_nan(y)


def test_slice_trainable_init_input_dim2_four_basis_matrices_example():
    """
    Explicit calculation test with input_dim=2 so augmented inp has 4 channels:
        inp = [inc_ts, ts, x1, x2]
    """

    m = SLiCE(
        input_dim=2,
        hidden_dim=4,
        block_size=2,
        diagonal_dense=False,
        bias=False,
    )
    m.eval()

    # Trainable init state (state BEFORE step-0 update)
    init_vec = torch.tensor([1.0, -1.0, 0.0, 2.0], dtype=torch.float32)

    assert hasattr(m, "init"), "Model must define a trainable initial state: self.init"
    with torch.no_grad():
        m.init.copy_(init_vec.reshape_as(m.init))

        # Basis matrices (each is 4x4 block-diagonal with block size 2).
        # Flatten order is row-major per block:
        # [b1_00, b1_01, b1_10, b1_11,  b2_00, b2_01, b2_10, b2_11]
        A1_flat = torch.tensor(
            [0.40, 0.00, 0.00, -0.40, 0.2, 0.08, 0.00, 0.12], dtype=m.vf_A.weight.dtype
        )  # inc
        A2_flat = torch.tensor(
            [0.00, 0.2, 0.00, 0.00, -0.08, 0.00, 0.16, 0.04], dtype=m.vf_A.weight.dtype
        )  # ts
        A3_flat = torch.tensor(
            [0.20, -0.10, 0.10, 0.00, 0.00, 0.03, -0.01, 0.05],
            dtype=m.vf_A.weight.dtype,
        )  # x1
        A4_flat = torch.tensor(
            [-0.10, 0.00, 0.00, 0.05, 0.02, -0.02, 0.03, 0.00],
            dtype=m.vf_A.weight.dtype,
        )  # x2

        # vf_A.weight shape is (8, 4) because inp has 4 channels [inc_ts, ts, x1, x2]
        m.vf_A.weight.copy_(torch.stack([A1_flat, A2_flat, A3_flat, A4_flat], dim=1))

    # Fixed input sequence
    X = torch.tensor(
        [
            [
                [4.0, -4.0],
                [2.0, 8.0],
                [-4.0, 6.0],
                [8.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    # Expected outputs AFTER each update i=0..3 (hand-computed):
    expected = torch.tensor(
        [
            [1.45, -0.75, 0.14, 2.18],
            [1.4125, -0.6775, 0.1361, 2.3624],
            [0.89, -0.8018125, 0.044326, 2.4098415],
            [1.335, -0.54363125, 0.23578354, 2.8257202],
        ],
        dtype=torch.float32,
    )

    Y = m(X)[0]  # (seq_len=4, hidden_dim=4)

    assert Y.shape == expected.shape
    torch.testing.assert_close(Y, expected, rtol=1e-6, atol=1e-6)
