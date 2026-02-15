"""Tests for QuEL driver protocol contracts used by qubex type checking."""

from __future__ import annotations

from qubex.backend.quel1 import driver_protocols


def test_driver_protocols_export_expected_contracts() -> None:
    """Given protocol module, contract types used by qubex are exported."""
    assert hasattr(driver_protocols, "QuelDriverModulesProtocol")
    assert hasattr(driver_protocols, "QubeCalibProtocol")
    assert hasattr(driver_protocols, "SequencerProtocol")
    assert hasattr(driver_protocols, "Quel1ConfigOptionProtocol")
