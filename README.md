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

- Python 3.9 or higher
- pip 24.0

## Installation

### Install latest release

```bash
pip install git+https://github.com/amachino/qubex.git
```

### Install with backend support (Linux only)

```bash
pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"
```

### Install for development

```bash
git clone https://github.com/amachino/qubex.git
cd qubex
pip install -e .
```

Check available versions on the [Release Page](https://github.com/amachino/qubex/releases).

## Usage

See the [examples](https://github.com/amachino/qubex/tree/main/docs/examples) folder for usage examples.
