# SLiCEs

`slices` is a small PyTorch package providing **Structured Linear CDE (SLiCE)** recurrences.

## Mathematical form

Given an input sequence $x_i \in \mathbb{R}^D$ for $i=1,\dots,T$, a SLiCE computes hidden states $y_i \in \mathbb{R}^H$ via

$$
y_i = y_{i-1} + A(X_i)y_{i-1} + B(X_i),
$$

where $A(\cdot): \mathbb{R}^D \rightarrow \mathbb{R}^{H \times H}$ and $B(\cdot): \mathbb{R}^D \rightarrow \mathbb{R}^H$ are *learned linear maps*, the initial state $y_0$ is either a function of $X_0$ or a learnt vector, and the input is augmented with an extra channel:

- `inc` = a constant “increment” channel (all ones)

such that

$$
X_i = [inc_i, x_i] \in \mathbb{R}^{D+1}.
$$

## Installation

```bash
pip install torch-slices
```

Or install from source:

```bash
pip install git+https://github.com/datasig-ac-uk/slices.git
```

## What's included

- **`SLiCE`**: the core structured recurrence
- **`SLiCELayer`**: a residual layer wrapping `SLiCE` with a post-activation stage (`GLU` or `tanh`)
- **`StackedSLiCE`**: stacks multiple `SLiCELayer`s with an embedding + output projection (supports tokens or continuous inputs)

`SLiCE` supports both:
- **Recurrent execution** (step-by-step update)
- **Parallel chunked scan execution** using `torch.associative_scan`

## Structured transition matrices

SLiCE supports different $A(X_i)$ structures:

### 1) Diagonal (elementwise update)
Set:
- `diagonal_dense=False`
- `block_size=1`

Then $A(X_i)$ is diagonal, which aligns with the approach used by Mamba (see [here](https://arxiv.org/abs/2505.17761) for more details).

### 2) Block-diagonal
Set:
- `diagonal_dense=False`
- `block_size > 1`

Then $A(X_i)$ is block-diagonal with blocks of shape `(block_size × block_size)`.

### 3) Diagonal + dense tail block
Set:
- `diagonal_dense=True`
- `block_size > 1`

Then the first `(hidden_dim - block_size)` dimensions are diagonal, and the final `block_size` dimensions are updated via a dense `(block_size × block_size)` matrix.

## Quickstart

### Use `SLiCE` directly

```python
import torch
from slices import SLiCE

x = torch.randn(8, 128, 32)  # (batch, seq, input_dim)

layer = SLiCE(
    input_dim=32,
    hidden_dim=64,
    block_size=4,
    diagonal_dense=False,
    bias=True,
    use_parallel=True,
    chunk_size=256,
)

y = layer(x)  # (8, 128, 64)
print(y.shape)
```

Execution mode is configured via constructor arguments (`use_parallel`, `chunk_size`).

### Use `SLiCELayer` as a residual sequence layer

```python
import torch
from slices import SLiCELayer

x = torch.randn(4, 256, 64)

layer = SLiCELayer(
    input_dim=64,
    block_size=4,
    diagonal_dense=True,
    dropout_rate=0.01,
    use_glu=True,
)

y = layer(x)  # (4, 256, 64)
```

### Stack layers for a full model

#### Token sequence mode (`tokens=True`)

Uses an `nn.Embedding(data_dim, hidden_dim)` front-end.

```python
import torch
from slices import StackedSLiCE

batch, seq_len = 2, 128
vocab_size = 5000

x = torch.randint(0, vocab_size, (batch, seq_len))

model = StackedSLiCE(
    num_layers=4,
    data_dim=vocab_size,
    hidden_dim=256,
    label_dim=vocab_size,
    tokens=True,
    block_size=4,
    diagonal_dense=False,
    use_glu=True,
)

logits = model(x)  # (batch, seq_len, vocab_size)
```

#### Continuous time-series mode (`tokens=False`)

Uses an `nn.Linear(data_dim, hidden_dim)` front-end.

```python
import torch
from slices import StackedSLiCE

x = torch.randn(16, 100, 12)  # (batch, seq, data_dim)

model = StackedSLiCE(
    num_layers=3,
    data_dim=12,
    hidden_dim=64,
    label_dim=10,
    tokens=False,
    block_size=4,
    diagonal_dense=True,
)

y = model(x)  # (16, 100, 10)
```

## Training Example

`examples/language_disambiguation.py` is a simple example of training a
`StackedSLiCE` model end-to-end on a real dataset.

This example:
- uses a real dataset (**wikimedia/wikipedia**, English/French subset) for
  **character-level language disambiguation**
- trains a compact token-mode `StackedSLiCE` end-to-end
- evaluates validation accuracy every `--eval-every` training steps
- prints sample predictions so you can inspect model behavior quickly

To run it, first install the example dependencies:

```bash
uv sync --group examples
```

and then execute the script:

```bash
uv run python examples/language_disambiguation.py
```

Useful flags:
- `--device cpu|cuda`
- `--train-steps 3000`
- `--eval-every 300`
- `--train-size 12000 --val-size 3000`
- `--max-seq-len 192`

## Benchmarking

To compare recurrent vs parallel throughput across sequence lengths and hidden dimensions:

```bash
uv run python examples/benchmark_parallel_vs_recurrent.py
```

This script:
- benchmarks all four SLiCE matrix modes (`diagonal`, `block_diagonal`, `diagonal_dense`, `dense`)
- prints timing/speedup tables
- saves a combined 3D plot to `examples/images/parallel_vs_recurrent_speedup_3d_all_modes.png`

For plotting in development, install development dependencies (includes `matplotlib`):

```bash
uv sync --dev
```

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.8.0
- NumPy ≥ 2.4.1

## License

MIT License. See [LICENSE](LICENSE).
