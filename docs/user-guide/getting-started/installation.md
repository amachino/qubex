# Installation

Qubex runs on Python 3.10+. When you are not using hardware backends, it is not tied to a specific operating system. Hardware backends require additional dependencies and typically assume a Linux host.
During the `v1.5.0 beta` period and until the official release, installing Qubex requires `uv`.
We plan to publish Qubex to PyPI for the official release, which will make `pip install` available.

## Prepare Python environment

We recommend using [uv](https://docs.astral.sh/uv/) to manage Python and virtual environments. `uv` is a unified tool for installing Python, creating virtual environments, and managing dependencies.

## Create virtual environment

It is best practice to install Qubex in a dedicated virtual environment to prevent conflicts with system packages or other projects.

Run one of the following commands in your project directory.

=== "uv"
    ```bash
    uv venv
    ```

=== "venv"
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

## Install Qubex

This section introduces installation options for standard use, backend-enabled setups, pinned versions, and development workflows.

### Install latest version

Use this option when you want the newest Qubex features from the repository.

```bash
uv pip install "qubex @ git+https://github.com/amachino/qubex.git"
```

### Install with QuEL-1 backend support for real hardware use (Linux)

Use this variant when you plan to use Qubex with real hardware on a Linux host.
If you only want to use `QuantumSimulator`, you do not need the `backend` or
`quel1` extra.

```bash
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"
```

This installs the public QuEL-1 backend libraries required for hardware-backed
execution. `qubex[quel1]` is available as the equivalent explicit extra.

### Install with QuEL-3 support

Use this variant when you plan to use Qubex with QuEL-3 runtime support.

```bash
uv pip install "qubex[quel3] @ git+https://github.com/amachino/qubex.git"
```

`quel3` depends on `quelware-client`. During the `v1.5.0 beta` period,
install with `uv`, which resolves the bundled source dependency declared by
Qubex.

### Install specific version

Use this option when you need a pinned version for reproducibility.

```bash
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git@<version>"
```

### Install for development

Use this setup to prepare a local environment for developing and testing Qubex.
Development commands in this repository assume a `uv`-managed environment.

```bash
git clone -b develop https://github.com/amachino/qubex.git
cd qubex
make sync
```

## Next steps

- Start with [Choose where to start](choose-where-to-start.md) to pick the entry point that matches your goal.
- If you are moving on to hardware-backed `Experiment` or `Low-level APIs`, continue with [System configuration](system-configuration.md).
