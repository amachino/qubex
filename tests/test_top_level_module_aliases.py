"""Tests for top-level qubex module aliases."""

from __future__ import annotations

import importlib

import pytest

import qubex as qx


@pytest.mark.parametrize(
    ("name", "module_name", "probe"),
    [
        ("contrib", "qubex.contrib", "measure_cr_crosstalk"),
        ("core", "qubex.core", "normalize_time_to_ns"),
        ("fit", "qubex.analysis.fitting", "fit_linear"),
        ("viz", "qubex.visualization", "scatter_iq_data"),
    ],
)
def test_qubex_exports_top_level_module_aliases(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    module_name: str,
    probe: str,
) -> None:
    """Given top-level aliases, when reading them from qubex, then the module is returned."""
    monkeypatch.delitem(qx.__dict__, name, raising=False)

    module = getattr(qx, name)
    expected_module = importlib.import_module(module_name)

    assert module is expected_module
    assert callable(getattr(module, probe))
