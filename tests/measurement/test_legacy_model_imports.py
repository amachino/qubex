"""Compatibility tests for legacy measurement model imports."""

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
            "qubex.measurement.measurement_record",
            "qubex.measurement.models.measurement_record",
            ["MeasurementRecord"],
        ),
        (
            "qubex.measurement.measurement_result",
            "qubex.measurement.models.measure_result",
            [
                "MeasureData",
                "MeasureMode",
                "MeasureResult",
                "MultipleMeasureResult",
            ],
        ),
    ],
)
def test_legacy_measurement_model_module_exports_canonical_symbols(
    legacy_module_name: str,
    canonical_module_name: str,
    exported_names: list[str],
) -> None:
    """Legacy measurement model imports should resolve to canonical module symbols."""
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


def test_legacy_measurement_record_module_exports_default_data_dir_alias() -> None:
    """Legacy measurement record import should preserve DEFAULT_DATA_DIR."""
    _reset_legacy_shim_state("qubex.measurement.measurement_record")
    canonical_module = import_module("qubex.measurement.models.measurement_record")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", FutureWarning)
        imported_symbol = _import_symbol(
            "qubex.measurement.measurement_record",
            "DEFAULT_DATA_DIR",
        )

    assert imported_symbol == canonical_module.DEFAULT_RAWDATA_DIR
    assert len(captured) == 1
