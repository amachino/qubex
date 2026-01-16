#!/usr/bin/env bash
set -euo pipefail

echo "[postCreate] Syncing dependencies..."
uv sync --dev --all-extras

echo "[postCreate] Done."
