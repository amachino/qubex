# qxpulse

`qxpulse` is the pulse primitives and scheduling layer extracted from the Qubex project. It contains waveforms, pulse libraries, and scheduling utilities that higher-level packages like `qubex` build on.

## Relationship to qubex

- `qxpulse` is a standalone package with no dependency on `qubex`.
- `qubex` depends on `qxpulse` and provides the full experiment framework (backend, simulator, measurement, etc.).
- If you only need pulse primitives and scheduling utilities, install `qxpulse` directly.

## Requirements

- Python 3.10 or higher

## Installation

```bash
# pip
pip install "qxpulse @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxpulse"

# uv
uv pip install "qxpulse @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxpulse"
```

## pyproject.toml

```toml
[project]
dependencies = [
  "qxpulse @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxpulse",
]
```

## Development

```bash
git clone -b develop https://github.com/amachino/qubex.git

cd qubex/packages/qxpulse

uv sync
```
