# Installation

Qubex runs on Python 3.10+ and supports macOS and Linux. Hardware backends require additional dependencies and are typically Linux-only.
During the `v1.5.0 beta` period and until the official release, installing Qubex requires `uv`.
We plan to publish Qubex to PyPI for the official release, which will make `pip install` available.

## Prepare Python environment

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python installation and virtual environments. `uv` is a fast, standalone tool that helps keep environments reproducible.

Install Python first if it is not already available on your system.

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
uv pip install "qubex @ git+https://github.com/amachino/qubex.git@main"
```

### Install with backend support for real hardware use (Linux)

Use this variant when you plan to use Qubex with real hardware on a Linux host.
If you only want to use the simulator, you do not need the `backend` extra.

```bash
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git@main"
```

This installs the backend libraries required for hardware-backed execution.
QuEL-3 support also requires compatible `quelware-client` sources or packages.
For repository development, `packages/quelware-client` is fetched as a
submodule, while a local `lib/quelware-client-internal` checkout takes
precedence when present.

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

- If you plan to use real hardware, continue with [System configuration](system-configuration.md).
- Choose the path that matches your goal: [Choose where to start](choose-where-to-start.md).
