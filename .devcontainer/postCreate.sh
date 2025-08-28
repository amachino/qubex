#!/usr/bin/env bash
set -euo pipefail
echo "[postCreate] Installing system packages..."
apt-get update -y && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libgirepository1.0-dev libcairo2-dev build-essential git && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

echo "[postCreate] Installing qubex (editable) with extras..."
pip install -e ".[backend,dev]"

echo "[postCreate] Running ruff + pytest smoke (short) ..."
ruff check . || true
pytest -q || true

echo "[postCreate] Done."
