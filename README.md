# SLiCEs

`slices` is a small PyTorch package providing **Structured Linear CDE (SLiCE)** layers.

## Mathematical form

Given an input sequence $x_i \in \mathbb{R}^D$ for $i=1,\dots,T$, a SLiCE computes hidden states $y_i \in \mathbb{R}^H$ via

$$
y_i = y_{i-1} + A(X_i)y_{i-1} + B(X_i),
$$

where $A(\cdot): \mathbb{R}^D \rightarrow \mathbb{R}^{H \times H}$ and $B(\cdot): \mathbb{R}^D \rightarrow \mathbb{R}^H$ are *learned linear maps*, the initial state $y_0$ is learned, and the input is augmented with two extra channels:

- `inc_ts` = a constant “increment” channel (all ones)
- `ts` = a cumulative “sample time” channel (`cumsum(inc_ts)`)

such that

$$
X_i = [\text{inc\_ts}_i,\; \text{ts}_i,\; X_i] \in \mathbb{R}^{D+2}.
$$

## What’s included

- **`SLiCE`**: the core structured linear recurrence layer
- **`SLiCEBlock`**: a residual block wrapping `SLiCE` with a post-activation stage (`GLU` or `tanh`)
- **`StackedSLiCE`**: stacks multiple `SLiCEBlock`s with an embedding + output projection (supports tokens or continuous inputs)

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
from slices.slices import SLiCE

x = torch.randn(8, 128, 32)  # (batch, seq, input_dim)

layer = SLiCE(
    input_dim=32,
    hidden_dim=64,
    block_size=4,
    diagonal_dense=False,
    bias=True,
)

y = layer(x)  # (8, 128, 64)
print(y.shape)
````

### Use `SLiCEBlock` as a residual sequence block

```python
import torch
from slices.slices import SLiCEBlock

x = torch.randn(4, 256, 64)

block = SLiCEBlock(
    input_dim=64,
    block_size=4,
    diagonal_dense=True,
    dropout_rate=0.01,
    use_glu=True,
)

y = block(x)  # (4, 256, 64)
```

### Stack blocks for a full model

#### Token sequence mode (`tokens=True`)

Uses an `nn.Embedding(data_dim, hidden_dim)` front-end.

```python
import torch
from slices.slices import StackedSLiCE

batch, seq_len = 2, 128
vocab_size = 5000

x = torch.randint(0, vocab_size, (batch, seq_len))

model = StackedSLiCE(
    num_blocks=4,
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
from slices.slices import StackedSLiCE

x = torch.randn(16, 100, 12)  # (batch, seq, data_dim)

model = StackedSLiCE(
    num_blocks=3,
    data_dim=12,
    hidden_dim=64,
    label_dim=10,
    tokens=False,
    block_size=4,
    diagonal_dense=True,
)

y = model(x)  # (16, 100, 10)
```

## Installation (development)

Clone the repo and install dependencies with Poetry:

```bash
poetry install
```

Run the test suite:

```bash
poetry run pytest -q
```

## Linting and formatting

This repo uses **Ruff** for linting + formatting.

```bash
poetry run ruff check .
poetry run ruff format .
```

## Pre-commit hooks

Install git hooks:

```bash
poetry run pre-commit install
```

Run all hooks:

```bash
poetry run pre-commit run --all-files
```

## Package layout

```text
src/slices/        # Python package
tests/             # pytest tests
.github/workflows/ # CI configuration
```

## License

MIT License. See `LICENSE`.

