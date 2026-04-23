"""Tests for functional APIs in `qubex.contrib.experiment.ckp_characterization`."""

from __future__ import annotations

from qubex.contrib import (
    ckp_measurement_v2,
    filtered_ckp_experiment,
)


def test_all_ckp_characterization_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then CKP characterization helpers are available."""
    assert callable(ckp_measurement_v2)
    assert callable(filtered_ckp_experiment)
