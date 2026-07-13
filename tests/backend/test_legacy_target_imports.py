"""Compatibility tests for legacy backend target imports."""

from __future__ import annotations

import sys
import warnings
from importlib import import_module

import pytest

import qubex.backend as backend
from qubex.compat.deprecated_imports import reset_deprecated_import_warning
from qubex.system import Target as SystemTarget, TargetType as SystemTargetType


def _reset_legacy_shim_state(module_name: str) -> None:
    """Reset cached shim import state for deterministic warning checks."""
    sys.modules.pop(module_name, None)
    reset_deprecated_import_warning(legacy_module=module_name)


def test_legacy_backend_target_module_exports_canonical_symbols() -> None:
    """Legacy backend target module imports should resolve to canonical symbols."""
    legacy_module_name = "qubex.backend.target"
    _reset_legacy_shim_state(legacy_module_name)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", FutureWarning)
        import_module(legacy_module_name)

    assert not captured

    for exported_name, expected_symbol in (
        ("Target", SystemTarget),
        ("TargetType", SystemTargetType),
    ):
        _reset_legacy_shim_state(legacy_module_name)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", FutureWarning)
            imported_symbol = getattr(import_module(legacy_module_name), exported_name)

        assert imported_symbol is expected_symbol
        assert len(captured) == 1
        assert legacy_module_name in str(captured[0].message)
        assert "qubex.system" in str(captured[0].message)


def test_backend_package_exposes_legacy_target_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy backend Target aliases should resolve to system symbols."""
    reset_deprecated_import_warning(legacy_module="qubex.backend")
    monkeypatch.delitem(backend.__dict__, "Target", raising=False)
    monkeypatch.delitem(backend.__dict__, "TargetType", raising=False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", FutureWarning)
        imported_target = backend.Target
        imported_target_type = backend.TargetType

    assert imported_target is SystemTarget
    assert imported_target_type is SystemTargetType
    assert len(captured) == 1
    assert "qubex.backend" in str(captured[0].message)
    assert "qubex.system" in str(captured[0].message)
