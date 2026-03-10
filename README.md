# Qubex: Qubit experiment framework

[![GitHub license](https://img.shields.io/github/license/amachino/qubex)](https://github.com/amachino/qubex/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/amachino/qubex)](https://github.com/amachino/qubex/stargazers)

**Qubex** is a Python library designed to simplify and accelerate quantum experiments by providing a flexible and extensible framework for pulse-level control, readout, and calibration on quantum hardware. It streamlines experiment workflows such as:

- **Qubit and resonator spectroscopy**
- **T1 and T2 measurement**
- **Single- and two-qubit gate calibration**
- **Randomized Benchmarking**
- **Pulse-level experiments and simulations**

## Requirements

- Python 3.10 or higher

## Installation

### Install latest release

```bash
# pip
pip install git+https://github.com/amachino/qubex.git

# uv
uv pip install git+https://github.com/amachino/qubex.git
```

### Install with backend support (Linux only)

```bash
# pip
pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"

# uv
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"
```

This installs packaged backend dependencies such as `qxdriver-quel1`.
QuEL-3 support also requires compatible `quelware-client` sources or packages.
This repository fetches `packages/quelware-client` as a submodule for
development and CI, and prefers `lib/quelware-client-internal` when present
in a local development workspace.

Available backend extras:

- `backend`: install packaged QuEL-1 and QuEL-3 dependencies.
- `quel1`: install `qxdriver-quel1` only.
- `quel3`: install `quelware-client` only.
- `qubecalib`: install `qubecalib` only.

### Install specific version

```bash
# pip
pip install "git+https://github.com/amachino/qubex.git@<version>"

# uv
uv pip install "git+https://github.com/amachino/qubex.git@<version>"
```

Check available versions on the [Release Page](https://github.com/amachino/qubex/releases).

### Install for development

Development in this repository assumes a `uv`-managed environment.

```bash
git clone -b develop https://github.com/amachino/qubex.git
cd qubex
make sync
```

## Documentation

The documentation is available at [https://amachino.github.io/qubex/](https://amachino.github.io/qubex/).
The source files live in [docs](https://github.com/amachino/qubex/tree/main/docs).
