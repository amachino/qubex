"""Tests for ConfigLoader canonical imports."""

from __future__ import annotations

import qubex.backend as backend
from qubex.system import ConfigLoader
from qubex.system.config_loader import ConfigLoader as ModuleConfigLoader


def test_config_loader_is_exported_from_system_namespace() -> None:
    """Given public imports, when resolving ConfigLoader, then they point to one class."""
    assert ConfigLoader is ModuleConfigLoader
    assert not hasattr(backend, "ConfigLoader")
