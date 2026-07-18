"""Tests for functional APIs in `qubex.contrib.experiment.spin_lock_spectroscopy`."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from qubex.contrib import spin_lock_sequence, spin_lock_spectroscopy
from qubex.contrib.experiment import (
    spin_lock_sequence as experiment_spin_lock_sequence,
    spin_lock_spectroscopy as experiment_spin_lock_spectroscopy,
)

spin_lock_module = importlib.import_module(
    "qubex.contrib.experiment.spin_lock_spectroscopy"
)


def test_all_spin_lock_spectroscopy_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then spin-lock helpers are available."""
    assert callable(spin_lock_sequence)
    assert callable(spin_lock_spectroscopy)


def test_all_spin_lock_spectroscopy_functions_are_exported_from_experiment() -> None:
    """Given experiment package, when imported, then spin-lock helpers are available."""
    assert experiment_spin_lock_sequence is spin_lock_sequence
    assert experiment_spin_lock_spectroscopy is spin_lock_spectroscopy


def test_default_spin_lock_frequency_range_is_log_spaced_to_200_mhz() -> None:
    """Given defaults, when inspected, then the Rabi-frequency range reaches 200 MHz."""
    frequency_range = spin_lock_module.DEFAULT_SPIN_LOCK_FREQUENCY_RANGE

    assert frequency_range[0] == pytest.approx(0.001)
    assert frequency_range[-1] == pytest.approx(0.2)
    np.testing.assert_allclose(
        np.diff(np.log(frequency_range)),
        np.diff(np.log(frequency_range))[0],
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"duration": 0.0, "drive_amplitude": 0.1}, "duration"),
        ({"duration": 100.0, "drive_amplitude": np.nan}, "drive_amplitude"),
        ({"duration": 100.0, "drive_amplitude": 1.1}, "drive_amplitude"),
        (
            {
                "duration": 100.0,
                "drive_amplitude": 0.1,
                "drive_detuning": np.nan,
            },
            "drive_detuning",
        ),
    ],
)
def test_spin_lock_sequence_rejects_invalid_scalar_inputs(
    kwargs: dict[str, float],
    match: str,
) -> None:
    """Given invalid scalar inputs, when building a sequence, then validation fails."""
    with pytest.raises(ValueError, match=match):
        spin_lock_sequence(
            object(),  # type: ignore[arg-type]
            target="Q00",
            **kwargs,
        )
