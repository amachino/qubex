"""Tests for sampling-period usage in simultaneous coherence helper."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from qubex.contrib.experiment.simultaneous_coherence_measurement import (
    simultaneous_coherence_measurement,
)


def test_simultaneous_coherence_uses_measurement_sampling_period_for_discretization() -> (
    None
):
    """Given measurement dt, when discretizing time range, then helper uses backend-derived dt."""
    captured: dict[str, float] = {}

    def _discretize_time_range(
        *,
        time_range: np.ndarray,
        sampling_period: float,
    ) -> np.ndarray:
        captured["sampling_period"] = sampling_period
        return np.array([], dtype=int)

    exp = SimpleNamespace(
        ctx=SimpleNamespace(
            qubit_labels=["Q00"],
            qubits={},
            state_centers={},
            measurement=SimpleNamespace(sampling_period=0.4),
            util=SimpleNamespace(discretize_time_range=_discretize_time_range),
        ),
        pulse=SimpleNamespace(
            validate_rabi_params=lambda _targets: None,
            get_hpi_pulse=lambda _target: object(),
            rabi_params={},
        ),
        measurement_service=SimpleNamespace(measure=lambda **_kwargs: None),
    )

    result = simultaneous_coherence_measurement(
        exp,  # type: ignore[arg-type]
        targets=["Q00"],
        time_range=np.array([0.0, 10.0]),
        plot=False,
    )

    assert captured["sampling_period"] == 2 * 0.4
    assert set(result) == {"T1", "T2", "Ramsey"}
