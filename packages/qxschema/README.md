# qubex-schema

PyPI distribution: `qubex-schema`. Python import: `qxschema` (unchanged).

```bash
pip install "qubex-schema==1.5.0rc4"
```

`qxschema` is a collection of data models for quantum experiment configurations and results, built on top of the `qxcore` serialization framework to provide a shared experiment interface across different software.

## Requirements

- Python 3.10 or higher

## Installation

```bash
# pip
pip install "qubex-schema @ git+https://github.com/amachino/qubex.git@v1.5.0rc4#subdirectory=packages/qxschema"

# uv
uv pip install "qubex-schema @ git+https://github.com/amachino/qubex.git@v1.5.0rc4#subdirectory=packages/qxschema"
```

## pyproject.toml

```toml
[project]
dependencies = [
  "qubex-schema @ git+https://github.com/amachino/qubex.git@v1.5.0rc4#subdirectory=packages/qxschema",
]
```

## Development

```bash
git clone -b develop https://github.com/amachino/qubex.git

cd qubex/packages/qxschema

uv sync
```
