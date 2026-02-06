# Installation

Qubex runs on Python 3.10+ and supports macOS and Linux. Hardware backends require additional dependencies and are typically Linux-only.

## Install latest version

```bash
# pip
python -m pip install git+https://github.com/amachino/qubex.git

# uv
uv pip install git+https://github.com/amachino/qubex.git
```

## Install with backend support (Linux)

```bash
# pip
python -m pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"

# uv
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"
```

## Development install

```bash
git clone https://github.com/amachino/qubex.git
cd qubex

# uv
uv sync --dev

# pip
python -m pip install -e .
```

## Configuration prerequisites

Qubex loads configuration and parameter files that describe your chip, wiring, and control settings.

- Default location: `/home/shared/qubex-config/<chip_id>/`.
- Custom locations can be passed via `config_dir` and `params_dir` when creating an `Experiment` object.

See [Configuration](../reference/configuration.md) for details.
