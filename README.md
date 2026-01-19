# SLiCEs

`slices` is a small PyTorch package for Structured Linear CDE (SLiCE) layers.

This repository currently contains:
- A `src/`-based Python package layout (`src/slices/`)
- Poetry for dependency management
- Ruff linting + formatting (run locally via pre-commit)
- GitHub Actions CI (runs on push and pull requests)
- A minimal pytest import smoke test

## Installation (development)

Clone the repo and install dependencies with Poetry:

```bash
poetry install
````

Run the test suite:

```bash
poetry run pytest -q
```

## Linting and formatting

This repo uses Ruff for linting + formatting.

Run Ruff manually:

```bash
poetry run ruff check .
poetry run ruff format .
```

## Pre-commit hooks

Install the git hooks:

```bash
poetry run pre-commit install
```

Run all hooks across the repo:

```bash
poetry run pre-commit run --all-files
```

## CI

GitHub Actions runs:

* Ruff + formatting checks
* pytest

on every push and pull request.

## Package layout

```text
src/slices/        # Python package
tests/             # pytest tests
.github/workflows/ # CI configuration
```

## License

TBD.

