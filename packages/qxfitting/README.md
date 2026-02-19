# qxfitting

`qxfitting` is the fitting and curve-analysis layer extracted from the Qubex project. It provides fit models, result containers, and plotting helpers that higher-level packages like `qubex` build on.

## Relationship to qubex

- `qxfitting` is a standalone package with no dependency on `qubex`.
- `qubex` depends on `qxfitting` and provides the full experiment framework (backend, measurement, etc.).
- If you only need fitting and curve-analysis utilities, install `qxfitting` directly.

## Requirements

- Python 3.10 or higher

## Installation

```bash
# pip
pip install "qxfitting @ git+https://github.com/amachino/qubex.git@develop-next#subdirectory=packages/qxfitting"

# uv
uv pip install "qxfitting @ git+https://github.com/amachino/qubex.git@develop-next#subdirectory=packages/qxfitting"
```

## pyproject.toml

```toml
[project]
dependencies = [
  "qxfitting @ git+https://github.com/amachino/qubex.git@develop-next#subdirectory=packages/qxfitting",
]
```

## Development

```bash
git clone -b develop-next https://github.com/amachino/qubex.git

cd qubex/packages/qxfitting

uv sync
```
