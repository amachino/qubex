"""Tests for ConfigLoader import compatibility."""

from __future__ import annotations

from qubex.backend import ConfigLoader as BackendExportConfigLoader
from qubex.backend.config_loader import ConfigLoader as LegacyConfigLoader
from qubex.configuration import ConfigLoader


def test_config_loader_is_exported_from_configuration_and_backend() -> None:
    """Given public imports, when resolving ConfigLoader, then they point to one class."""
    assert ConfigLoader is LegacyConfigLoader
    assert ConfigLoader is BackendExportConfigLoader
