"""Tests for public GEF population estimation exports."""

from __future__ import annotations

from qubex.contrib import bootstrap_gef_populations, measure_gef_populations
from qubex.contrib.experiment import (
    bootstrap_gef_populations as experiment_bootstrap_gef_populations,
    measure_gef_populations as experiment_measure_gef_populations,
)


def test_measure_gef_populations_is_exported_from_contrib() -> None:
    """The contrib namespaces should expose the GEF population workflow."""
    assert experiment_measure_gef_populations is measure_gef_populations
    assert experiment_bootstrap_gef_populations is bootstrap_gef_populations
