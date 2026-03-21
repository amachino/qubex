"""Tests for readout-frequency calibration fit bounds."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

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

    def _fit_lorentzian(**kwargs: Any) -> Result:
        fit_calls.append(kwargs)
        return Result(data={"f0": 5.0}, figures={})

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
    )

    assert len(fit_calls) == 1
    fit_call = fit_calls[0]
    assert fit_call["p0"][0] > 0
    assert fit_call["bounds"][0][0] == 0
