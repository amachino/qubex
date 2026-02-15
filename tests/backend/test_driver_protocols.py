"""Tests for QuEL driver protocol contracts used by qubex type checking."""

from __future__ import annotations

from qubex.backend.quel1 import quel1_driver_protocols


def test_quel1_driver_protocols_export_expected_contracts() -> None:
    """Given protocol module, contract types used by qubex are exported."""
    assert hasattr(quel1_driver_protocols, "QuelDriverClassesProtocol")
    assert hasattr(quel1_driver_protocols, "QubeCalibProtocol")
    assert hasattr(quel1_driver_protocols, "SequencerProtocol")
    assert hasattr(quel1_driver_protocols, "Quel1ConfigOptionProtocol")
