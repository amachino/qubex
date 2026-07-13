"""Tests for sampling-period usage in calibration service."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from qubex.experiment.models.experiment_result import SweepData
from qubex.experiment.models.rabi_param import RabiParam
from qubex.experiment.services.calibration_service import CalibrationService


class _FlatTopStub:
    SAMPLING_PERIOD = 2.0

    def __init__(self, *, duration: float, amplitude: float, tau: float) -> None:
        self.duration = duration
        self.tau = tau
        self.real = np.array([1.0], dtype=float)

    def scaled(self, _: float) -> _FlatTopStub:
        return self


class _DragStub:
    SAMPLING_PERIOD = 2.0

    def __init__(self, *, duration: float, amplitude: float, beta: float) -> None:
        self.duration = duration
        self.beta = beta
        self.real = np.array([1.0], dtype=float)

    def scaled(self, _: float) -> _DragStub:
        return self

    def repeated(self, _: int) -> _DragStub:
        return self


def _make_sweep_data(target: str) -> SweepData:
    rabi_param = RabiParam(
        target=target,
        amplitude=1.0,
        frequency=0.0,
        phase=0.0,
        offset=0.0,
        noise=0.0,
        angle=0.0,
        distance=0.0,
        r2=1.0,
        reference_phase=0.0,
    )
    return SweepData(
        target=target,
        data=np.array([1.0 + 0.0j], dtype=np.complex128),
        sweep_range=np.array([0.5], dtype=float),
        rabi_param=rabi_param,
    )


def test_calibrate_default_pulse_uses_measurement_sampling_period(
    monkeypatch,
) -> None:
    """Given measurement dt, when calibrating default pulse, then area uses measurement sampling period."""
    captured: dict[str, float] = {}
    target = "Q00"
    sweep_data = _make_sweep_data(target)

    monkeypatch.setattr(
        "qubex.experiment.services.calibration_service.FlatTop",
        _FlatTopStub,
    )
    monkeypatch.setattr(
        "qubex.experiment.services.calibration_service.fitting.fit_ampl_calib_data",
        lambda **_: {"r2": 0.0, "amplitude": 0.5},
    )

    def _calc_control_amplitude(_target: str, rabi_rate: float) -> float:
        captured["rabi_rate"] = float(rabi_rate)
        return 0.5

    service = object.__new__(CalibrationService)
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=[target],
        measurement=SimpleNamespace(sampling_period=0.4),
        calib_note=SimpleNamespace(
            rabi_params={target: {}},
        ),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(
        sweep_parameter=lambda **_: SimpleNamespace(data={target: sweep_data}),
    )
    service.__dict__["_pulse_service"] = SimpleNamespace(
        rabi_params={target: object()},
        calc_control_amplitude=_calc_control_amplitude,
    )

    service.calibrate_default_pulse(
        targets=[target],
        pulse_type="hpi",
        plot=False,
        update_params=False,
    )

    assert np.isclose(captured["rabi_rate"], 0.25 / 0.4)


def test_calibrate_drag_amplitude_uses_measurement_sampling_period(monkeypatch) -> None:
    """Given measurement dt, when calibrating DRAG amplitude, then area uses measurement sampling period."""
    captured: dict[str, float] = {}
    target = "Q00"
    sweep_data = _make_sweep_data(target)

    monkeypatch.setattr(
        "qubex.experiment.services.calibration_service.Drag",
        _DragStub,
    )
    monkeypatch.setattr(
        "qubex.experiment.services.calibration_service.fitting.fit_ampl_calib_data",
        lambda **_: {"r2": 0.0, "amplitude": 0.5},
    )

    def _calc_control_amplitude(_target: str, rabi_rate: float) -> float:
        captured["rabi_rate"] = float(rabi_rate)
        return 0.5

    service = object.__new__(CalibrationService)
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubits={target: SimpleNamespace(alpha=1.0)},
        measurement=SimpleNamespace(sampling_period=0.4),
        calib_note=SimpleNamespace(
            get_drag_hpi_param=lambda _target: None,
            get_drag_pi_param=lambda _target: None,
            update_drag_hpi_param=lambda *_args, **_kwargs: None,
            update_drag_pi_param=lambda *_args, **_kwargs: None,
        ),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(
        sweep_parameter=lambda **_: SimpleNamespace(data={target: sweep_data}),
    )
    service.__dict__["_pulse_service"] = SimpleNamespace(
        rabi_params={target: object()},
        validate_rabi_params=lambda _params: None,
        calc_control_amplitude=_calc_control_amplitude,
        get_pulse_for_state=lambda **_: object(),
    )

    service.calibrate_drag_amplitude(
        targets=[target],
        pulse_type="hpi",
        plot=False,
    )

    assert np.isclose(captured["rabi_rate"], 0.25 / 0.4)
