import pytest
import torch

import slices.slices as slices_module
from slices import SLiCE, SLiCELayer, StackedSLiCE


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
        use_parallel=False,
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
        use_parallel=False,
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
        use_parallel=False,
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


def test_slice_parallel_falls_back_when_associative_scan_is_unavailable(monkeypatch):
    x = _rand_x(batch=2, seq=4, dim=3, seed=10)
    monkeypatch.setattr(
        slices_module.torch, "_higher_order_ops", object(), raising=False
    )

    m = SLiCE(
        input_dim=3,
        hidden_dim=3,
        block_size=1,
        diagonal_dense=False,
        bias=True,
        use_parallel=True,
    )

    assert not m.use_parallel
    expected = m._forward_recurrent(x)
    y = m(x)
    assert y.shape == expected.shape
    assert torch.allclose(y, expected)


def test_slice_diagonal_dense_edge_case_hidden_dim_equals_block_size():
    # Edge case: hidden_dim - block_size == 0, so "diagonal" part is empty
    x = _rand_x(batch=2, seq=3, dim=2, seed=5)

    m = SLiCE(
        input_dim=2,
        hidden_dim=2,  # equals block_size
        block_size=2,
        diagonal_dense=True,
        bias=False,
        use_parallel=False,
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
        use_parallel=False,
    )
    y = m(x)
    assert y.shape == (2, 4, 3)
    _assert_no_nan(y)


# -----------------------
# SLiCELayer: both GLU and tanh paths
# -----------------------


@pytest.mark.parametrize("use_glu", [False, True])
@pytest.mark.parametrize("diagonal_dense", [False, True])
def test_slice_block_forward_covers_glu_and_tanh(use_glu: bool, diagonal_dense: bool):
    x = _rand_x(batch=2, seq=4, dim=6, seed=7)

    block = SLiCELayer(
        input_dim=6,
        block_size=2 if not diagonal_dense else 4,
        diagonal_dense=diagonal_dense,
        use_parallel=False,
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
        num_layers=2,
        data_dim=vocab,
        hidden_dim=hidden,
        label_dim=label,
        tokens=True,
        block_size=4,
        diagonal_dense=False,
        use_parallel=False,
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
        num_layers=3,
        data_dim=data_dim,
        hidden_dim=hidden,
        label_dim=label,
        tokens=False,
        block_size=4,
        diagonal_dense=True,
        use_parallel=False,
        dropout_rate=0.0,
        use_glu=False,
    )
    m.eval()

    y = m(x)
    assert y.shape == (batch, seq, label)
    _assert_no_nan(y)


def test_slice_input_dim2_golden_example():
    """
    Hand calculated example with input_dim=2 so augmented inp has 3 channels:
        inp = [inc_ts, x1, x2]
    """

    m = SLiCE(
        input_dim=2,
        hidden_dim=4,
        block_size=2,
        diagonal_dense=False,
        bias=False,
        scale=1.0,
        use_parallel=False,
    )
    m.eval()

    # Trainable init state (state BEFORE step-0 update)
    init_vec = torch.tensor([1.0, -1.0, 0.0, 2.0], dtype=torch.float32)

    assert hasattr(m, "init"), "Model must define a trainable initial state: self.init"
    with torch.no_grad():
        m.init.copy_(init_vec.reshape_as(m.init))

        # Basis matrices (each is 4x4 block-diagonal with block size 2)
        # Flatten order is row-major per block:
        # [b1_00, b1_01, b1_10, b1_11,  b2_00, b2_01, b2_10, b2_11]
        A1_flat = torch.tensor(
            [0.10, 0.00, 0.00, -0.10, 0.05, 0.02, 0.00, 0.03],
            dtype=m.vf_A.weight.dtype,
        )  # inc
        A2_flat = torch.tensor(
            [0.20, -0.10, 0.10, 0.00, 0.00, 0.03, -0.01, 0.05],
            dtype=m.vf_A.weight.dtype,
        )  # x1
        A3_flat = torch.tensor(
            [-0.10, 0.00, 0.00, 0.05, 0.02, -0.02, 0.03, 0.00],
            dtype=m.vf_A.weight.dtype,
        )  # x2

        # vf_A input channels are now [inc_ts, x1, x2]
        m.vf_A.weight.copy_(torch.stack([A1_flat, A2_flat, A3_flat], dim=1))

    # Fixed input sequence
    X = torch.tensor(
        [
            [
                [1.0, -1.0],
                [0.5, 2.0],
                [-1.0, 1.5],
                [2.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    # Expected outputs AFTER each update i=0..3 (hand-computed):
    expected = torch.tensor(
        [
            [1.5, -0.75, 0.14, 2.16],
            [1.5375, -0.675, 0.1418, 2.2865],
            [1.085625, -0.811875, 0.061684, 2.248569],
            [1.7908125, -0.5135625, 0.24465372, 2.53964929],
        ],
        dtype=torch.float32,
    )

    Y = m(X)[0]  # (seq_len=4, hidden_dim=4)

    assert Y.shape == expected.shape
    torch.testing.assert_close(Y, expected, rtol=0.0, atol=1e-5)


# -----------------------
# SLiCE: parallel paths
# -----------------------


def test_slice_input_dim2_golden_example_parallel():
    """
    Same hand-calculated setup as test_slice_input_dim2_golden_example,
    but evaluated through the parallel/chunked path.
    """

    m = SLiCE(
        input_dim=2,
        hidden_dim=4,
        block_size=2,
        diagonal_dense=False,
        bias=False,
        scale=1.0,
        use_parallel=True,
        chunk_size=4,
    )
    m.eval()

    init_vec = torch.tensor([1.0, -1.0, 0.0, 2.0], dtype=torch.float32)

    with torch.no_grad():
        m.init.copy_(init_vec.reshape_as(m.init))

        A1_flat = torch.tensor(
            [0.10, 0.00, 0.00, -0.10, 0.05, 0.02, 0.00, 0.03],
            dtype=m.vf_A.weight.dtype,
        )  # inc
        A2_flat = torch.tensor(
            [0.20, -0.10, 0.10, 0.00, 0.00, 0.03, -0.01, 0.05],
            dtype=m.vf_A.weight.dtype,
        )  # x1
        A3_flat = torch.tensor(
            [-0.10, 0.00, 0.00, 0.05, 0.02, -0.02, 0.03, 0.00],
            dtype=m.vf_A.weight.dtype,
        )  # x2

        m.vf_A.weight.copy_(torch.stack([A1_flat, A2_flat, A3_flat], dim=1))

    X = torch.tensor(
        [
            [
                [1.0, -1.0],
                [0.5, 2.0],
                [-1.0, 1.5],
                [2.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    expected = torch.tensor(
        [
            [1.5, -0.75, 0.14, 2.16],
            [1.5375, -0.675, 0.1418, 2.2865],
            [1.085625, -0.811875, 0.061684, 2.248569],
            [1.7908125, -0.5135625, 0.24465372, 2.53964929],
        ],
        dtype=torch.float32,
    )

    Y = m(X)[0]

    assert Y.shape == expected.shape
    torch.testing.assert_close(Y, expected, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "cfg",
    [
        # diagonal / elementwise
        dict(input_dim=3, hidden_dim=3, block_size=1, diagonal_dense=False),
        # block-diagonal
        dict(input_dim=4, hidden_dim=4, block_size=2, diagonal_dense=False),
        # diagonal + one dense block
        dict(input_dim=5, hidden_dim=8, block_size=4, diagonal_dense=True),
        # dense (single full block)
        dict(input_dim=6, hidden_dim=6, block_size=6, diagonal_dense=False),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_slice_parallel_matches_recurrent(cfg, bias: bool, chunk_size: int):
    x = _rand_x(batch=2, seq=7, dim=cfg["input_dim"], seed=11)
    m_recurrent = SLiCE(**cfg, bias=bias, use_parallel=False)
    m_parallel = SLiCE(**cfg, bias=bias, use_parallel=True, chunk_size=chunk_size)
    m_parallel.load_state_dict(m_recurrent.state_dict())

    y_recurrent = m_recurrent(x)
    y_parallel = m_parallel(x)

    assert y_parallel.shape == y_recurrent.shape
    _assert_no_nan(y_parallel)
    torch.testing.assert_close(y_parallel, y_recurrent, rtol=1e-5, atol=1e-6)


def test_slice_parallel_with_input_dependent_init_matches_recurrent():
    x = _rand_x(batch=3, seq=6, dim=4, seed=12)
    m_recurrent = SLiCE(
        input_dim=4,
        hidden_dim=4,
        block_size=2,
        diagonal_dense=False,
        bias=True,
        input_dependent_init=True,
        use_parallel=False,
    )
    m_parallel = SLiCE(
        input_dim=4,
        hidden_dim=4,
        block_size=2,
        diagonal_dense=False,
        bias=True,
        input_dependent_init=True,
        use_parallel=True,
        chunk_size=3,
    )
    m_parallel.load_state_dict(m_recurrent.state_dict())

    y_recurrent = m_recurrent(x)
    y_parallel = m_parallel(x)

    assert y_parallel.shape == y_recurrent.shape
    _assert_no_nan(y_parallel)
    torch.testing.assert_close(y_parallel, y_recurrent, rtol=1e-5, atol=1e-6)


def test_slice_parallel_forward_with_grads():
    x = _rand_x(batch=2, seq=5, dim=4, seed=13).requires_grad_(True)
    m = SLiCE(
        input_dim=4,
        hidden_dim=4,
        block_size=2,
        diagonal_dense=False,
        bias=True,
        use_parallel=True,
    )

    y = m(x)
    assert y.shape == (2, 5, 4)
    _assert_no_nan(y)

    y.sum().backward()
    assert x.grad is not None
    _assert_grads_exist(m)
