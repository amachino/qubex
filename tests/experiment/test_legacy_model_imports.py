"""Compatibility tests for legacy experiment model imports."""

from __future__ import annotations

import sys
import warnings
from importlib import import_module

import pytest

from qubex.compat.deprecated_imports import reset_deprecated_import_warning


def _reset_legacy_shim_state(module_name: str) -> None:
    """Reset cached shim import state for deterministic warning checks."""
    sys.modules.pop(module_name, None)
    reset_deprecated_import_warning(legacy_module=module_name)


def _import_symbol(module_name: str, symbol_name: str) -> object:
    """Resolve a symbol from the imported legacy module."""
    module = import_module(module_name)
    return getattr(module, symbol_name)


@pytest.mark.parametrize(
    ("legacy_module_name", "canonical_module_name", "exported_names"),
    [
        (
            "qubex.experiment.calibration_note",
            "qubex.experiment.models.calibration_note",
            [
                "CalibrationNote",
                "CrossResonanceParam",
                "DragParam",
                "FlatTopParam",
                "Parameter",
                "RabiParam",
                "StateParam",
            ],
        ),
        (
            "qubex.experiment.experiment_note",
            "qubex.experiment.models.experiment_note",
            ["ExperimentNote", "FILE_PATH"],
        ),
        (
            "qubex.experiment.experiment_record",
            "qubex.experiment.models.experiment_record",
            ["DEFAULT_DATA_DIR", "ExperimentRecord"],
        ),
        (
            "qubex.experiment.experiment_result",
            "qubex.experiment.models.experiment_result",
            [
                "AmplCalibData",
                "AmplRabiData",
                "ExperimentResult",
                "FreqRabiData",
                "RabiData",
                "RBData",
                "RamseyData",
                "SweepData",
                "T1Data",
                "T2Data",
                "TargetData",
            ],
        ),
        (
            "qubex.experiment.rabi_param",
            "qubex.experiment.models.rabi_param",
            ["RabiParam"],
        ),
        (
            "qubex.experiment.result",
            "qubex.experiment.models.result",
            ["Result"],
        ),
    ],
)
def test_legacy_experiment_model_module_exports_canonical_symbols(
    legacy_module_name: str,
    canonical_module_name: str,
    exported_names: list[str],
) -> None:
    """Legacy experiment model imports should resolve to canonical module symbols."""
    _reset_legacy_shim_state(legacy_module_name)
    canonical_module = import_module(canonical_module_name)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", FutureWarning)
        import_module(legacy_module_name)

    assert not captured

    for exported_name in exported_names:
        _reset_legacy_shim_state(legacy_module_name)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", FutureWarning)
            imported_symbol = _import_symbol(legacy_module_name, exported_name)

        assert imported_symbol is getattr(canonical_module, exported_name)
        assert len(captured) == 1
        assert legacy_module_name in str(captured[0].message)
        assert canonical_module_name in str(captured[0].message)
