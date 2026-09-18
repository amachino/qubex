"""Tests for CR parameter history returned by calibration service."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

import qubex.visualization as viz
from qubex.experiment.services.calibration_service import CalibrationService


class _FigureStub:
    def add_trace(self, _trace: object) -> None:
        """Accept added traces."""

    def update_layout(self, **_kwargs: object) -> None:
        """Accept layout updates."""

    def show(self) -> None:
        """Accept show calls."""


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Provide a calibration service without hardware dependencies."""
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
    return service


@pytest.mark.parametrize("last_duration", [16.0, 10000.0])
def test_obtain_cr_params_returns_fig_history(
    service: Any, last_duration: float
) -> None:
    """Given iterative CR updates, when obtaining CR params, then per-iteration figures are returned."""
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
            "zx90_duration": last_duration,
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


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_short_explicit_range_is_input_error(service: Any, count: int) -> None:
    """Explicit time ranges with fewer than four points should fail before measuring."""
    service.update_cr_params = Mock()
    with pytest.raises(ValueError, match=r"time_range.*at least 4"):
        service.obtain_cr_params(
            "Q00", "Q01", time_range=list(range(count)), ramptime=16.0
        )
    service.update_cr_params.assert_not_called()


@pytest.mark.parametrize(
    ("duration", "count"), [(3000.0, 3), (5000.0, 2), (10000.0, 1)]
)
@pytest.mark.parametrize("stored", [False, True])
def test_short_generated_range_reports_failure(
    service: Any, duration: float, count: int, stored: bool
) -> None:
    """Short generated time ranges should report calibration failure before measuring."""
    params = dict(
        cr_amplitude=0.25, cr_phase=0.0, cancel_amplitude=0.0, cancel_phase=0.0
    )
    service.ctx.calib_note.get_cr_param = lambda _: dict(
        params, zx_rotation_rate=1 / duration
    )
    service.update_cr_params = Mock(
        return_value=dict(
            zx90_duration=duration,
            cr_param=params,
            coeffs={"IX": 0.0, "IY": 0.0},
            fig_c=None,
            fig_t=None,
        )
    )
    iteration = 1 if stored else 2
    with pytest.raises(
        RuntimeError,
        match=rf"CR calibration failed.*Q00-Q01.*iteration {iteration}.*{count} time points",
    ) as caught:
        service.obtain_cr_params(
            "Q00", "Q01", use_stored_params=stored, ramptime=16.0, plot=False
        )
    assert type(caught.value) is RuntimeError
    assert service.update_cr_params.call_count == iteration - 1
