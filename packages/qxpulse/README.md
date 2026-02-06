# qxpulse: Qubex Pulse

`qxpulse` is the pulse primitives and scheduling layer extracted from the Qubex project. It contains waveforms, pulse libraries, and scheduling utilities that higher-level packages like `qubex` build on.

## Relationship to qubex

- `qxpulse` is a standalone package with no dependency on `qubex`.
- `qubex` depends on `qxpulse` and provides the full experiment framework (backend, simulator, measurement, etc.).
- If you only need pulse primitives and scheduling utilities, install `qxpulse` directly.

## Requirements

- Python 3.10 or higher

## Installation

`qxpulse` is not on PyPI yet. Install from the Qubex GitHub repository using the subdirectory.

### Install latest

```bash
# pip
pip install "qxpulse @ git+https://github.com/amachino/qubex.git#subdirectory=packages/qxpulse"

# uv
uv pip install "qxpulse @ git+https://github.com/amachino/qubex.git#subdirectory=packages/qxpulse"
```

### Install specific version

```bash
# pip
pip install "qxpulse @ git+https://github.com/amachino/qubex.git@<version>#subdirectory=packages/qxpulse"

# uv
uv pip install "qxpulse @ git+https://github.com/amachino/qubex.git@<version>#subdirectory=packages/qxpulse"
```

Check available versions on the [Release Page](https://github.com/amachino/qubex/releases).

## pyproject.toml

Add the GitHub dependency with a subdirectory reference:

```toml
[project]
dependencies = [
  "qxpulse @ git+https://github.com/amachino/qubex.git#subdirectory=packages/qxpulse",
]
```

For a pinned version:

```toml
[project]
dependencies = [
  "qxpulse @ git+https://github.com/amachino/qubex.git@<version>#subdirectory=packages/qxpulse",
]
```

## Development

```bash
git clone https://github.com/amachino/qubex.git
cd qubex

# uv
uv sync --all-groups

# editable install for qxpulse
uv pip install -e packages/qxpulse
```
