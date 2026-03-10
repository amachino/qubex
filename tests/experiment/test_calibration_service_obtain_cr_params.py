"""Tests for CR parameter history returned by calibration service."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import qubex.visualization as viz
from qubex.experiment.services.calibration_service import CalibrationService


class _FigureStub:
    def add_trace(self, _trace: object) -> None:
        """Accept added traces."""

    def update_layout(self, **_kwargs: object) -> None:
        """Accept layout updates."""

    def show(self) -> None:
        """Accept show calls."""


def test_obtain_cr_params_returns_fig_history(monkeypatch) -> None:
    """Given iterative CR updates, when obtaining CR params, then per-iteration figures are returned."""
    monkeypatch.setattr(viz, "make_figure", lambda: _FigureStub())

    service = cast(Any, object.__new__(CalibrationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        measurement=SimpleNamespace(sampling_period=2.0),
        qubits={
            "Q00": SimpleNamespace(frequency=5.0),
            "Q01": SimpleNamespace(frequency=5.2),
        },
        calib_note=SimpleNamespace(get_cr_param=lambda _label: None),
    )
    service.__dict__["_pulse_service"] = SimpleNamespace(
        calc_control_amplitude=lambda _control_qubit, _max_cr_rabi: 0.25,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace()

    update_results = [
        {
            "zx90_duration": 16.0,
            "cr_param": {
                "cr_phase": 0.1,
                "cancel_amplitude": 0.2,
                "cancel_phase": 0.3,
            },
            "coeffs": {"IX": 1.0e-4, "IY": 2.0e-4},
            "fig_c": "fig-c-1",
            "fig_t": "fig-t-1",
        },
        {
            "zx90_duration": 16.0,
            "cr_param": {
                "cr_phase": 0.4,
                "cancel_amplitude": 0.5,
                "cancel_phase": 0.6,
            },
            "coeffs": {"IX": 0.5e-4, "IY": 1.0e-4},
            "fig_c": "fig-c-2",
            "fig_t": "fig-t-2",
        },
    ]
    service.__dict__["update_cr_params"] = lambda **_kwargs: update_results.pop(0)

    result = service.obtain_cr_params(
        control_qubit="Q00",
        target_qubit="Q01",
        n_iterations=2,
        n_cycles=1,
        n_points_per_cycle=4,
        ramptime=16.0,
        plot=False,
    )

    assert result["figs_history"] == [
        {"fig_c": "fig-c-1", "fig_t": "fig-t-1"},
        {"fig_c": "fig-c-2", "fig_t": "fig-t-2"},
    ]
