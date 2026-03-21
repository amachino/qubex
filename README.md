# Qubex: Qubit experiment framework

[![GitHub license](https://img.shields.io/github/license/amachino/qubex)](https://github.com/amachino/qubex/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/amachino/qubex)](https://github.com/amachino/qubex/stargazers)

**Qubex** is a Python library designed to simplify and accelerate quantum experiments by providing a flexible and extensible framework for pulse-level control, readout, and calibration on quantum hardware. It streamlines experiment workflows such as:

- **Qubit and resonator spectroscopy**
- **T1 and T2 measurement**
- **Single- and two-qubit gate calibration**
- **Randomized Benchmarking**
- **Pulse-level experiments and simulations**

## Recommended docs paths

- **Experiment**: The recommended user-facing workflow for most hardware-backed qubit experiments, from setup through analysis. See the [Experiment guide](https://amachino.github.io/qubex/user-guide/experiment/).
- **Simulator**: For researchers who want to study pulse-level Hamiltonian dynamics without using real hardware. See the [Simulator guide](https://amachino.github.io/qubex/user-guide/simulator/).
- **Pulse sequences**: Learn the shared `PulseSchedule` model used across hardware and simulation workflows. See the [PulseSchedule guide](https://amachino.github.io/qubex/user-guide/pulse-sequences/).
- **Low-level APIs**: Start here when you need measurement-side abstractions such as sessions, schedules, readout, sweeps, and backend integration. See the [overview](https://amachino.github.io/qubex/user-guide/low-level-apis/).

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

### Install with QuEL-1 backend support for real hardware use (Linux only)

```bash
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"
```

`backend` is the public extra for QuEL-1 support. `quel1` is available as the
equivalent explicit extra name.
If you only want to use the simulator, you do not need either extra.

### Install with QuEL-3 support

```bash
uv pip install "qubex[quel3] @ git+https://github.com/amachino/qubex.git"
```

`quel3` requires `quelware-client`. Depending on the release timing, you may
need an additional package index or an internal distribution channel for that
dependency.

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

- User guides: [Installation](https://amachino.github.io/qubex/user-guide/getting-started/installation/), [System configuration](https://amachino.github.io/qubex/user-guide/getting-started/system-configuration/), [Choose where to start](https://amachino.github.io/qubex/user-guide/getting-started/choose-where-to-start/), [Examples](https://amachino.github.io/qubex/examples/), and [API reference](https://amachino.github.io/qubex/api-reference/qubex/).
- Contributors: [Contributing](https://amachino.github.io/qubex/CONTRIBUTING/), [Developer guide](https://amachino.github.io/qubex/developer-guide/), and the source files in [docs](https://github.com/amachino/qubex/tree/main/docs).
