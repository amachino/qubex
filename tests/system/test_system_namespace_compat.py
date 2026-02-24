"""Tests for system namespace canonical exports."""

from __future__ import annotations

import qubex.backend as backend
from qubex.system import BackendSettings, SystemManager, SystemSynchronizer
from qubex.system.quel1 import Quel1SystemSynchronizer
from qubex.system.quel3 import Quel3SystemSynchronizer


def test_system_namespace_exposes_system_manager_components() -> None:
    """Given system namespace, when importing components, then canonical symbols are available."""
    assert SystemManager.__name__ == "SystemManager"
    assert BackendSettings.__name__ == "BackendSettings"
    assert SystemSynchronizer.__name__ == "SystemSynchronizer"
    assert Quel1SystemSynchronizer.__name__ == "Quel1SystemSynchronizer"
    assert Quel3SystemSynchronizer.__name__ == "Quel3SystemSynchronizer"


def test_backend_namespace_does_not_reexport_migrated_system_symbols() -> None:
    """Given backend namespace, when checking migrated symbols, then they are not re-exported."""
    assert not hasattr(backend, "SystemManager")
    assert not hasattr(backend, "ConfigLoader")
    assert not hasattr(backend, "ControlSystem")
    assert not hasattr(backend, "ExperimentSystem")
    assert not hasattr(backend, "TargetRegistry")
