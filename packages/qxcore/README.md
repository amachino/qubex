# qxcore

`qxcore` is the core data-model and utility layer extracted from the Qubex project. It contains shared primitives (models, units, quantities, expressions, serialization) that higher-level packages like `qubex` build on.

## Relationship to qubex

- `qxcore` is a standalone package with no dependency on `qubex`.
- `qubex` depends on `qxcore` and provides the full experiment framework (pulse, simulator, backend, etc.).
- If you only need the core models and utilities, install `qxcore` directly.

## Requirements

- Python 3.10 or higher

## Installation

```bash
# pip
pip install "qxcore @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxcore"

# uv
uv pip install "qxcore @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxcore"
```

## pyproject.toml

```toml
[project]
dependencies = [
  "qxcore @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxcore",
]
```

## Development

```bash
git clone -b develop https://github.com/amachino/qubex.git

cd qubex/packages/qxcore

uv sync
```
