"""Tests for public GEF population estimation exports."""

from __future__ import annotations

import qubex.contrib as contrib
import qubex.contrib.experiment as experiment
from qubex.contrib import measure_gef_populations
from qubex.contrib.experiment import (
    measure_gef_populations as experiment_measure_gef_populations,
)

_LOW_LEVEL_GEF_SYMBOLS = (
    "GefPopulationBootstrap",
    "IQMomentSummary",
    "bootstrap_gef_populations",
    "reconstruct_gef_state_features",
    "summarize_iq_shots",
)


def test_measure_gef_populations_is_exported_from_contrib() -> None:
    """The contrib namespaces should expose the GEF population workflow."""
    assert experiment_measure_gef_populations is measure_gef_populations


def test_low_level_gef_symbols_are_not_exported_from_contrib() -> None:
    """The contrib namespaces should not expose low-level GEF helpers."""
    for namespace in (contrib, experiment):
        for symbol in _LOW_LEVEL_GEF_SYMBOLS:
            assert not hasattr(namespace, symbol)
