"""Tests for legacy qubecalib protocol contracts used by qubex type checking."""

from __future__ import annotations

from qubex.backend.quel1.compat import quel1_qubecalib_protocols


def test_quel1_qubecalib_protocols_export_expected_contracts() -> None:
    """Given legacy protocol module, contract types used by qubex are exported."""
    assert hasattr(quel1_qubecalib_protocols, "QuelDriverClassesProtocol")
    assert hasattr(quel1_qubecalib_protocols, "QubeCalibProtocol")
    assert hasattr(quel1_qubecalib_protocols, "SequencerProtocol")
    assert hasattr(quel1_qubecalib_protocols, "Quel1ConfigOptionProtocol")
    assert hasattr(quel1_qubecalib_protocols, "ActionProtocol")
    assert hasattr(quel1_qubecalib_protocols, "MultiActionProtocol")
    assert hasattr(quel1_qubecalib_protocols, "SingleActionProtocol")
    assert hasattr(quel1_qubecalib_protocols, "AwgIdProtocol")
    assert hasattr(quel1_qubecalib_protocols, "RunitIdProtocol")
    assert hasattr(quel1_qubecalib_protocols, "NamedBoxProtocol")
    assert hasattr(quel1_qubecalib_protocols, "SkewProtocol")


def test_sequencer_protocol_exports_parallel_execution_contracts() -> None:
    """Given sequencer protocol, methods used by parallel execution are defined."""
    assert hasattr(
        quel1_qubecalib_protocols.SequencerProtocol, "set_measurement_option"
    )
    assert hasattr(quel1_qubecalib_protocols.SequencerProtocol, "generate_e7_settings")
    assert hasattr(quel1_qubecalib_protocols.SequencerProtocol, "parse_capture_result")
