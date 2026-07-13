"""Tests for deprecated option normalization helpers."""

from __future__ import annotations

from typing import Any

import pytest

from qubex.compat.deprecated_options import (
    DeprecatedOptionSpec,
    normalize_deprecated_options,
    partition_deprecated_options,
    resolve_deprecated_option,
)
from qubex.core.sentinel import MISSING


def test_resolve_deprecated_option_returns_legacy_value_with_warning() -> None:
    """Given a legacy option, when resolving, then it warns and returns the legacy value."""
    deprecated_options: dict[str, Any] = {"shots": 123}

    with pytest.warns(DeprecationWarning, match="`shots` is deprecated"):
        resolved = resolve_deprecated_option(
            value=None,
            deprecated_options=deprecated_options,
            deprecated_name="shots",
            replacement_name="n_shots",
            default=64,
        )

    assert resolved == 123
    assert deprecated_options == {}


def test_normalize_deprecated_options_applies_defaults_and_aliases() -> None:
    """Given canonical and legacy options, when normalizing, then defaults and aliases are resolved."""
    with pytest.warns(DeprecationWarning, match="`shots` is deprecated"):
        normalized = normalize_deprecated_options(
            values={
                "n_shots": None,
                "shot_interval": None,
                "readout_amplification": None,
            },
            deprecated_options={
                "shots": 128,
                "interval": 240.0,
            },
            specs=(
                DeprecatedOptionSpec("shots", "n_shots", default=64),
                DeprecatedOptionSpec("interval", "shot_interval", default=120.0),
                DeprecatedOptionSpec(
                    "add_pump_pulses",
                    "readout_amplification",
                    default=False,
                ),
            ),
        )

    assert normalized == {
        "n_shots": 128,
        "shot_interval": 240.0,
        "readout_amplification": False,
    }


def test_normalize_deprecated_options_rejects_unknown_kwargs() -> None:
    """Given unknown kwargs, when normalizing, then it raises TypeError."""
    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        normalize_deprecated_options(
            values={"n_shots": None},
            deprecated_options={"n_shot": 1},
            specs=(DeprecatedOptionSpec("shots", "n_shots", default=MISSING),),
        )


def test_partition_deprecated_options_preserves_unresolved_kwargs() -> None:
    """Given partial alias specs, when partitioning, then unresolved kwargs are preserved."""
    with pytest.warns(DeprecationWarning, match="`shots` is deprecated"):
        normalized, remaining = partition_deprecated_options(
            values={"n_shots": None},
            deprecated_options={"shots": 32, "interval": 120.0},
            specs=(DeprecatedOptionSpec("shots", "n_shots", default=MISSING),),
        )

    assert normalized == {"n_shots": 32}
    assert remaining == {"interval": 120.0}
