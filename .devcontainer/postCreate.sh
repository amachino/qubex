#!/usr/bin/env bash
set -euo pipefail

echo "[postCreate] Syncing dependencies..."
uv sync  --all-groups --all-extras

echo "[postCreate] Done."
