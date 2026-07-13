# qxfitting

`qxfitting` is currently a placeholder package reserved for future fitting and curve-analysis APIs.

## Migration policy

- Legacy fitting APIs remain in `qubex.analysis.fitting` during transition.
- New fitting APIs are added to `qxfitting` incrementally.
- Domain code in `qubex` will migrate to `qxfitting` step by step.
- After migration completes, `qubex.analysis.fitting` will be deprecated and removed.

## Relationship to qubex

- `qxfitting` is a standalone package with no dependency on `qubex`.
- `qubex` depends on `qxfitting` and provides the full experiment framework (backend, measurement, etc.).
- If you only need fitting and curve-analysis utilities, install `qxfitting` directly.

## Requirements

- Python 3.10 or higher

## Installation

```bash
# pip
pip install "qxfitting @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxfitting"

# uv
uv pip install "qxfitting @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxfitting"
```

## pyproject.toml

```toml
[project]
dependencies = [
  "qxfitting @ git+https://github.com/amachino/qubex.git@develop#subdirectory=packages/qxfitting",
]
```

## Development

```bash
git clone -b develop https://github.com/amachino/qubex.git

cd qubex/packages/qxfitting

uv sync
```
