# qxsimulator

`qxsimulator` is the quantum system simulation layer extracted from the Qubex project. It contains quantum system models, control signal definitions, and simulation utilities that higher-level packages like `qubex` build on.

## Relationship to qubex

- `qxsimulator` is a standalone package with no dependency on `qubex`.
- `qubex` depends on `qxsimulator` and provides the full experiment framework (backend, measurement, etc.).
- If you only need simulation utilities, install `qxsimulator` directly.

## Requirements

- Python 3.10 or higher

## Installation

```bash
# pip
pip install "qxsimulator @ git+https://github.com/amachino/qubex.git@develop-next#subdirectory=packages/qxsimulator"

# uv
uv pip install "qxsimulator @ git+https://github.com/amachino/qubex.git@develop-next#subdirectory=packages/qxsimulator"
```

## pyproject.toml

```toml
[project]
dependencies = [
  "qxsimulator @ git+https://github.com/amachino/qubex.git@develop-next#subdirectory=packages/qxsimulator",
]
```

## Development

```bash
git clone -b develop-next https://github.com/amachino/qubex.git

cd qubex/packages/qxsimulator

uv sync
```
