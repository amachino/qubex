# qxcore: Qubex Core

`qxcore` is the core data-model and utility layer extracted from the Qubex project. It contains shared primitives (models, units, quantities, expressions, serialization) that higher-level packages like `qubex` build on.

## Relationship to qubex

- `qxcore` is a standalone package with no dependency on `qubex`.
- `qubex` depends on `qxcore` and provides the full experiment framework (pulse, simulator, backend, etc.).
- If you only need the core models and utilities, install `qxcore` directly.

## Requirements

- Python 3.10 or higher

## Installation

`qxcore` is not on PyPI yet. Install from the Qubex GitHub repository using the subdirectory.

### Install latest

```bash
# pip
pip install "qxcore @ git+https://github.com/amachino/qubex.git#subdirectory=packages/qxcore"

# uv
uv pip install "qxcore @ git+https://github.com/amachino/qubex.git#subdirectory=packages/qxcore"
```

### Install specific version

```bash
# pip
pip install "qxcore @ git+https://github.com/amachino/qubex.git@<version>#subdirectory=packages/qxcore"

# uv
uv pip install "qxcore @ git+https://github.com/amachino/qubex.git@<version>#subdirectory=packages/qxcore"
```

Check available versions on the [Release Page](https://github.com/amachino/qubex/releases).

## pyproject.toml

Add the GitHub dependency with a subdirectory reference:

```toml
[project]
dependencies = [
  "qxcore @ git+https://github.com/amachino/qubex.git#subdirectory=packages/qxcore",
]
```

For a pinned version:

```toml
[project]
dependencies = [
  "qxcore @ git+https://github.com/amachino/qubex.git@<version>#subdirectory=packages/qxcore",
]
```

## Development

```bash
git clone https://github.com/amachino/qubex.git
cd qubex

# uv
uv sync --all-groups

# editable install for qxcore
uv pip install -e packages/qxcore
```
