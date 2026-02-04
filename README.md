# SLiCEs Development Workspace

This is the **development mono-repo** for the `slices` PyTorch package — providing **Structured Linear CDE (SLiCE)** layers for sequence modeling.

## Repository Structure

```
SLiCEs/
├── pyproject.toml          # Dev workspace (uv-managed)
├── ruff.toml               # Linting configuration
├── README.md               # This file
├── slices/                # Main deployable package
│   ├── pyproject.toml      # Package metadata (PyPI-ready)
│   ├── README.md           # Package documentation
│   ├── LICENSE             # MIT License
│   ├── src/
│   │   └── slices/         # Python package
│   │       ├── __init__.py
│   │       └── slices.py
│   └── tests/
│       └── test_slices.py
└── .github/
    └── workflows/
        └── ci.yml          # GitHub Actions CI
```

## Quick Start (Development)

### Prerequisites

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

Clone and install in development mode:

```bash
git clone <repo-url>
cd SLiCEs

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
uv run pytest slices/tests -v
```

### Linting and Formatting

```bash
# Check linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Format code
uv run ruff format .
```

### Pre-commit Hooks

```bash
# Run all hooks on all files
uv run pre-commit run --all-files
```

## Package Development

The main package code lives in `slices/`. This structure allows:

1. **Local development**: Install as editable via `uv sync`
2. **Package publishing**: Publish `slices/` to PyPI independently
3. **Isolation**: Dev dependencies stay in the workspace, not in the published package

### Building the Package

```bash
cd slices
uv build
```

### Publishing to PyPI

```bash
cd slices
uv publish
```

## Mathematical Background

Given an input sequence $x_i \in \mathbb{R}^D$ for $i=1,\dots,T$, a SLiCE computes hidden states $y_i \in \mathbb{R}^H$ via:

$$
y_i = y_{i-1} + A(X_i)y_{i-1} + B(X_i)
$$

See [slices/README.md](slices/README.md) for full documentation.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `uv run pytest slices/tests -v`
5. Run linting: `uv run ruff check . && uv run ruff format .`
6. Commit: `git commit -m "Add my feature"`
7. Push and create a Pull Request

## License

MIT License. See [LICENSE](slices/LICENSE).
