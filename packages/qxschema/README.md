# qxschema

`qxschema` is a collection of data models for quantum experiment configurations and results, built on top of the `qxcore` serialization framework to provide a shared experiment interface across different software.

## Requirements

- Python 3.10 or higher

## Installation

```bash
# pip
pip install "qxschema @ git+https://github.com/amachino/qubex.git@develop-next#subdirectory=packages/qxschema"

# uv
uv pip install "qxschema @ git+https://github.com/amachino/qubex.git@develop-next#subdirectory=packages/qxschema"
```

## pyproject.toml

```toml
[project]
dependencies = [
  "qxschema @ git+https://github.com/amachino/qubex.git@develop-next#subdirectory=packages/qxschema",
]
```

## Development

```bash
git clone -b develop-next https://github.com/amachino/qubex.git

cd qubex/packages/qxschema

uv sync
```
