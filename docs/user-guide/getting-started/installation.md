# Installation

Qubex runs on Python 3.10+ and supports macOS and Linux. Hardware backends require additional dependencies and are typically Linux-only.

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

=== "uv"
    ```bash
    uv pip install "qubex @ git+https://github.com/amachino/qubex.git@main"
    ```

=== "pip"
    ```bash
    pip install "qubex @ git+https://github.com/amachino/qubex.git@main"
    ```

### Install with backend support (Linux)

Use this variant when you need hardware backend dependencies on Linux hosts.

=== "uv"
    ```bash
    uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git@main"
    ```

=== "pip"
    ```bash
    pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git@main"
    ```

### Install specific version

Use this option when you need a pinned version for reproducibility.

=== "uv"
    ```bash
    uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git@<version>"
    ```

=== "pip"
    ```bash
    pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git@<version>"
    ```

### Install for development

Use this setup to prepare a local environment for developing and testing Qubex.
Development commands in this repository assume a `uv`-managed environment.

```bash
git clone https://github.com/amachino/qubex.git
cd qubex
make sync
```
