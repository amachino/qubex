"""Backward-compatible ConfigLoader export."""

from __future__ import annotations

# TODO(v1.6): Remove this compatibility shim after callers migrate to
# `qubex.configuration.config_loader`.
from qubex.configuration.config_loader import ConfigLoader

__all__ = ["ConfigLoader"]
