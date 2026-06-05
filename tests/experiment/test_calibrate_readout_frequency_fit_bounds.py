"""Tests for readout-frequency calibration fit bounds."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest

from qubex.analysis import FitResult, FitStatus
from qubex.experiment.models.result import Result
from qubex.experiment.services.characterization_service import CharacterizationService


def test_calibrate_readout_frequency_constrains_lorentzian_amplitude_positive(
    monkeypatch,
) -> None:
    """Given readout calibration data, when fitting the Lorentzian, then A uses a non-negative bound and initial guess."""
    fit_calls: list[dict[str, Any]] = []

    @contextmanager
    def _no_output() -> Any:
        yield

    readout_amplitude = {"Q00": 0.2}
    control_amplitude = defaultdict(lambda: 0.1, {"Q00": 0.1})

    def _rabi_experiment(**kwargs: Any) -> SimpleNamespace:
        resonator_freq = kwargs["frequencies"]["RQ00"]
        detuning = resonator_freq - 5.0
        amplitude = 1.0 / (1.0 + (detuning / 0.002) ** 2)
        return SimpleNamespace(
            data={
                "Q00": SimpleNamespace(
                    rabi_param=SimpleNamespace(amplitude=float(amplitude))
                )
            }
        )

    def _fit_lorentzian(**kwargs: Any) -> FitResult:
        fit_calls.append(kwargs)
        return FitResult(status=FitStatus.SUCCESS, data={"f0": 5.0})

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q00"],
        resonators={"Q00": SimpleNamespace(label="RQ00", frequency=5.0)},
        resonator_labels=["RQ00"],
        params=SimpleNamespace(
            readout_amplitude=readout_amplitude,
            control_amplitude=control_amplitude,
        ),
        util=SimpleNamespace(no_output=_no_output),
        resolve_qubit_label=lambda label: label,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(
        rabi_experiment=_rabi_experiment
    )
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.fitting.fit_lorentzian",
        _fit_lorentzian,
    )

    service.calibrate_readout_frequency(
        targets=["Q00"],
        detuning_range=np.array([-0.002, 0.0, 0.002]),
        time_range=np.array([0.0, 4.0, 8.0]),
        plot=False,
        save_image=False,
        fit_func="lorentzian",
    )

    assert len(fit_calls) == 1
    fit_call = fit_calls[0]
    assert fit_call["p0"][0] > 0
    assert fit_call["bounds"][0][0] == 0


def test_calibrate_readout_frequency_double_lorentzian_starts_from_two_peaks(
    monkeypatch,
) -> None:
    """Given two-peaked readout data, when double fitting, then the initial guess uses both peaks."""
    fit_calls: list[dict[str, Any]] = []

    @contextmanager
    def _no_output() -> Any:
        yield

    readout_amplitude = {"Q00": 0.2}
    control_amplitude = defaultdict(lambda: 0.1, {"Q00": 0.1})

    def _rabi_experiment(**kwargs: Any) -> SimpleNamespace:
        resonator_freq = kwargs["frequencies"]["RQ00"]
        amplitude_by_frequency = {
            4.996: 0.05,
            4.998: 0.90,
            5.000: 0.20,
            5.002: 0.80,
            5.004: 0.05,
        }
        amplitude = amplitude_by_frequency[round(float(resonator_freq), 3)]
        return SimpleNamespace(
            data={
                "Q00": SimpleNamespace(
                    rabi_param=SimpleNamespace(amplitude=float(amplitude))
                )
            }
        )

    def _fit_lorentzian(**kwargs: Any) -> FitResult:
        fit_calls.append(kwargs)
        return FitResult(status=FitStatus.SUCCESS, data={"f0": 5.0})

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q00"],
        resonators={"Q00": SimpleNamespace(label="RQ00", frequency=5.0)},
        resonator_labels=["RQ00"],
        params=SimpleNamespace(
            readout_amplitude=readout_amplitude,
            control_amplitude=control_amplitude,
        ),
        util=SimpleNamespace(no_output=_no_output),
        resolve_qubit_label=lambda label: label,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(
        rabi_experiment=_rabi_experiment
    )
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.fitting.fit_double_lorentzian",
        _fit_lorentzian,
    )

    service.calibrate_readout_frequency(
        targets=["Q00"],
        detuning_range=np.array([-0.004, -0.002, 0.0, 0.002, 0.004]),
        time_range=np.array([0.0, 4.0, 8.0]),
        plot=False,
        save_image=False,
    )

    fit_call = fit_calls[0]
    assert fit_call["p0"][1] == pytest.approx(4.998, abs=1e-12)
    assert fit_call["p0"][4] == pytest.approx(5.002, abs=1e-12)
    assert fit_call["p0"][6] == pytest.approx(0.05, abs=1e-12)


def test_calibrate_readout_frequency_returns_peak_frequency_for_low_quality_fit(
    monkeypatch,
) -> None:
    """Given a low-r2 readout fit, when calibrating, then the maximum-response frequency is returned."""

    @contextmanager
    def _no_output() -> Any:
        yield

    readout_amplitude = {"Q00": 0.2}
    control_amplitude = defaultdict(lambda: 0.1, {"Q00": 0.1})

    def _rabi_experiment(**kwargs: Any) -> SimpleNamespace:
        resonator_freq = kwargs["frequencies"]["RQ00"]
        amplitude_by_frequency = {
            4.998: 0.1,
            5.000: 0.2,
            5.002: 0.9,
        }
        amplitude = amplitude_by_frequency[round(float(resonator_freq), 3)]
        return SimpleNamespace(
            data={
                "Q00": SimpleNamespace(
                    rabi_param=SimpleNamespace(amplitude=float(amplitude))
                )
            }
        )

    def _fit_lorentzian(**_kwargs: Any) -> FitResult:
        return FitResult(
            status=FitStatus.SUCCESS,
            message="Fitting returned low quality.",
            data={"f0": 4.998, "r2": 0.8},
        )

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q00"],
        resonators={"Q00": SimpleNamespace(label="RQ00", frequency=5.0)},
        resonator_labels=["RQ00"],
        params=SimpleNamespace(
            readout_amplitude=readout_amplitude,
            control_amplitude=control_amplitude,
        ),
        util=SimpleNamespace(no_output=_no_output),
        resolve_qubit_label=lambda label: label,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(
        rabi_experiment=_rabi_experiment
    )
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.fitting.fit_lorentzian",
        _fit_lorentzian,
    )

    result = service.calibrate_readout_frequency(
        targets=["Q00"],
        detuning_range=np.array([-0.002, 0.0, 0.002]),
        time_range=np.array([0.0, 4.0, 8.0]),
        plot=False,
        save_image=False,
        fit_func="lorentzian",
    )

    assert isinstance(result, Result)
    assert set(result.data) == {"data", "fig"}
    assert result["data"]["Q00"] == pytest.approx(5.002, abs=1e-12)


def test_calibrate_readout_frequency_returns_peak_frequency_when_fit_rejects_guess(
    monkeypatch,
) -> None:
    """Given a rejected readout fit initial guess, when calibrating, then the maximum-response frequency is returned."""

    @contextmanager
    def _no_output() -> Any:
        yield

    readout_amplitude = {"Q00": 0.2}
    control_amplitude = defaultdict(lambda: 0.1, {"Q00": 0.1})

    def _rabi_experiment(**kwargs: Any) -> SimpleNamespace:
        resonator_freq = kwargs["frequencies"]["RQ00"]
        amplitude_by_frequency = {
            4.998: 0.1,
            5.000: np.nan,
            5.002: 0.9,
        }
        amplitude = amplitude_by_frequency[round(float(resonator_freq), 3)]
        return SimpleNamespace(
            data={
                "Q00": SimpleNamespace(
                    rabi_param=SimpleNamespace(amplitude=float(amplitude))
                )
            }
        )

    def _fit_lorentzian(**kwargs: Any) -> FitResult:
        assert np.all(np.isfinite(kwargs["p0"]))
        raise ValueError("Initial guess is outside of provided bounds")

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q00"],
        resonators={"Q00": SimpleNamespace(label="RQ00", frequency=5.0)},
        resonator_labels=["RQ00"],
        params=SimpleNamespace(
            readout_amplitude=readout_amplitude,
            control_amplitude=control_amplitude,
        ),
        util=SimpleNamespace(no_output=_no_output),
        resolve_qubit_label=lambda label: label,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(
        rabi_experiment=_rabi_experiment
    )
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.fitting.fit_lorentzian",
        _fit_lorentzian,
    )

    result = service.calibrate_readout_frequency(
        targets=["Q00"],
        detuning_range=np.array([-0.002, 0.0, 0.002]),
        time_range=np.array([0.0, 4.0, 8.0]),
        plot=False,
        save_image=False,
        fit_func="lorentzian",
    )

    assert isinstance(result, Result)
    assert set(result.data) == {"data", "fig"}
    assert result["data"]["Q00"] == pytest.approx(5.002, abs=1e-12)


def test_calibrate_readout_frequency_shows_peak_figure_when_fit_has_no_figure(
    monkeypatch,
) -> None:
    """Given readout fit failure without a figure, when plotting, then a measured-data figure is shown."""
    show_calls: list[go.Figure] = []

    @contextmanager
    def _no_output() -> Any:
        yield

    def _show(self: go.Figure, *_args: Any, **_kwargs: Any) -> None:
        show_calls.append(self)

    readout_amplitude = {"Q00": 0.2}
    control_amplitude = defaultdict(lambda: 0.1, {"Q00": 0.1})

    def _rabi_experiment(**kwargs: Any) -> SimpleNamespace:
        resonator_freq = kwargs["frequencies"]["RQ00"]
        amplitude_by_frequency = {
            4.998: 0.1,
            5.000: 0.2,
            5.002: 0.9,
        }
        amplitude = amplitude_by_frequency[round(float(resonator_freq), 3)]
        return SimpleNamespace(
            data={
                "Q00": SimpleNamespace(
                    rabi_param=SimpleNamespace(amplitude=float(amplitude))
                )
            }
        )

    def _fit_lorentzian(**_kwargs: Any) -> FitResult:
        return FitResult(
            status=FitStatus.ERROR,
            message="Failed to fit the data.",
        )

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q00"],
        resonators={"Q00": SimpleNamespace(label="RQ00", frequency=5.0)},
        resonator_labels=["RQ00"],
        params=SimpleNamespace(
            readout_amplitude=readout_amplitude,
            control_amplitude=control_amplitude,
        ),
        util=SimpleNamespace(no_output=_no_output),
        resolve_qubit_label=lambda label: label,
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(
        rabi_experiment=_rabi_experiment
    )
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.fitting.fit_lorentzian",
        _fit_lorentzian,
    )
    monkeypatch.setattr(go.Figure, "show", _show)

    result = service.calibrate_readout_frequency(
        targets=["Q00"],
        detuning_range=np.array([-0.002, 0.0, 0.002]),
        time_range=np.array([0.0, 4.0, 8.0]),
        plot=True,
        save_image=False,
        fit_func="lorentzian",
    )

    assert isinstance(result, Result)
    assert set(result.data) == {"data", "fig"}
    assert result.figures is not None
    fig = result.figures["Q00"]
    assert fig is not None
    assert show_calls == [fig]
    assert fig.data[0].name == "Data"
    assert fig.data[1].name == "Maximum response"
    assert result["data"]["Q00"] == pytest.approx(5.002, abs=1e-12)
