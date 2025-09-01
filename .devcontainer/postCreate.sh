#!/usr/bin/env bash
set -euo pipefail

echo "[postCreate] Creating virtual environment..."
python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip==24.0

echo "[postCreate] Installing qubex (editable) with extras..."
pip install -e ".[dev]"

echo "[postCreate] Running ruff + pytest smoke (short) ..."
ruff check . || true
pytest -q || true

echo "[postCreate] Done."
