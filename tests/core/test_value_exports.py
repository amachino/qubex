"""Tests for Value and ValueArray exports."""

from __future__ import annotations

import tunits


def test_qxcore_exports_value_types() -> None:
    """Given qxcore, when importing value types, then tunits classes are re-exported."""
    from qxcore import Value, ValueArray

    assert Value is tunits.Value
    assert ValueArray is tunits.ValueArray


def test_qubex_core_exports_value_types() -> None:
    """Given qubex.core, when importing value types, then tunits classes are re-exported."""
    from qubex.core import Value, ValueArray

    assert Value is tunits.Value
    assert ValueArray is tunits.ValueArray
