"""Tests for functional APIs in `qubex.contrib.experiment.cpmg_noise_spectroscopy`."""

from __future__ import annotations

from qubex.contrib import (
    cpmg_noise_spectroscopy,
    plot_cpmg_results,
)


def test_all_cpmg_noise_spectroscopy_functions_are_exported_from_contrib() -> (
    None
):
    """Given contrib package, when imported, then CPMG noise spectroscopy helpers are available."""
    assert callable(cpmg_noise_spectroscopy)
    assert callable(plot_cpmg_results)
