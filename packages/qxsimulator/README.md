# qxsimulator: Qubex Simulator

`qxsimulator` is the quantum system simulation layer extracted from the Qubex project. It contains quantum system models, control signal definitions, and simulation utilities that higher-level packages like `qubex` build on.

## Relationship to qubex

- `qxsimulator` is a standalone package with no dependency on `qubex`.
- `qubex` depends on `qxsimulator` and provides the full experiment framework (backend, measurement, etc.).
- If you only need simulation utilities, install `qxsimulator` directly.

## Requirements

- Python 3.10 or higher

## Installation

`qxsimulator` is not on PyPI yet. Install from the Qubex GitHub repository using the subdirectory.

### Install latest

```bash
# pip
pip install "qxsimulator @ git+https://github.com/amachino/qubex.git#subdirectory=packages/qxsimulator"

# uv
uv pip install "qxsimulator @ git+https://github.com/amachino/qubex.git#subdirectory=packages/qxsimulator"
```

### Install specific version

```bash
# pip
pip install "qxsimulator @ git+https://github.com/amachino/qubex.git@<version>#subdirectory=packages/qxsimulator"

# uv
uv pip install "qxsimulator @ git+https://github.com/amachino/qubex.git@<version>#subdirectory=packages/qxsimulator"
```

Check available versions on the [Release Page](https://github.com/amachino/qubex/releases).

## pyproject.toml

Add the GitHub dependency with a subdirectory reference:

```toml
[project]
dependencies = [
  "qxsimulator @ git+https://github.com/amachino/qubex.git#subdirectory=packages/qxsimulator",
]
```

For a pinned version:

```toml
[project]
dependencies = [
  "qxsimulator @ git+https://github.com/amachino/qubex.git@<version>#subdirectory=packages/qxsimulator",
]
```

## Development

```bash
git clone https://github.com/amachino/qubex.git
cd qubex

# uv
uv sync --all-groups

# editable install for qxsimulator
uv pip install -e packages/qxsimulator
```
