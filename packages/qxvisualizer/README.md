# qubex-visualizer

PyPI distribution: `qubex-visualizer`. Python import: `qxvisualizer` (unchanged).

```bash
pip install "qubex-visualizer==1.5.0rc4"
```

`qxvisualizer` is the shared visualization layer extracted from the Qubex project. It provides common Plotly style, figure factory helpers, and generic plotting helpers that higher-level packages like `qxpulse`, `qxsimulator`, and `qubex` build on.

## Relationship to qubex

- `qxvisualizer` is a standalone package with no dependency on `qubex`.
- `qubex`, `qxpulse`, and `qxsimulator` depend on `qxvisualizer` for shared Plotly-based visualization helpers.
- If you only need shared Plotly style and figure helpers, install `qubex-visualizer` directly.

## Requirements

- Python 3.10 or higher

## Installation

```bash
# pip
pip install "qubex-visualizer @ git+https://github.com/amachino/qubex.git@v1.5.0rc4#subdirectory=packages/qxvisualizer"

# uv
uv pip install "qubex-visualizer @ git+https://github.com/amachino/qubex.git@v1.5.0rc4#subdirectory=packages/qxvisualizer"
```

## pyproject.toml

```toml
[project]
dependencies = [
  "qubex-visualizer @ git+https://github.com/amachino/qubex.git@v1.5.0rc4#subdirectory=packages/qxvisualizer",
]
```

## Development

```bash
git clone -b develop https://github.com/amachino/qubex.git

cd qubex/packages/qxvisualizer

uv sync
```
