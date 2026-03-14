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

During the `v1.5.0 beta` period and until the official release, installing Qubex requires `uv`.
We plan to publish Qubex to PyPI for the official release, which will make `pip install` available.
To install `uv`, see the official guide:
[https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install latest release

```bash
uv pip install git+https://github.com/amachino/qubex.git
```

### Install with backend support (Linux only)

```bash
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"
```

### Install specific version

```bash
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
