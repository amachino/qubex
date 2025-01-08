# Qubex: Quantum Experimentation and Simulation Framework

Qubex is a Python library designed to facilitate quantum experiments by providing a flexible, modular, and extensible framework for pulse-level control, readout, and calibration on quantum hardware. It streamlines experiment workflows (T1/T2 experiments, 1Q/2Q gate calibration, Randomized Benchmarking and more) and supports advanced simulations, data analysis, and experiment orchestration.

## Requirements

- Python 3.9 or higher
- pip 24.0

## Installation

### basic installation

```bash
pip install git+https://github.com/amachino/qubex.git
```

### qubex with backend (only for Linux)

```bash
pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"
```

### editable installation (for development)

```bash
git clone git+https://github.com/amachino/qubex.git
cd qubex
pip install -e .
```

Check available versions on the [release page](https://github.com/amachino/qubex/releases).

## Usage

See the [examples](https://github.com/amachino/qubex/tree/main/docs/examples) folder for usage examples.
