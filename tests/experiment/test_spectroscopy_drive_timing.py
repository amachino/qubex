"""Regression tests for spectroscopy drive timing through Experiment."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from qxpulse import PulseSchedule

from qubex.experiment.experiment import Experiment
from qubex.experiment.services.characterization_service import CharacterizationService


@pytest.mark.parametrize(
    "method",
    [
        "scan_qubit_frequencies",
        "qubit_spectroscopy",
        "measure_qubit_resonance",
        "estimate_control_amplitude",
    ],
)
@pytest.mark.parametrize("drive_mode", ["omitted", None, True, False])
def test_experiment_spectroscopy_drive_timing(method, drive_mode, monkeypatch) -> None:
    """Spectroscopy should separate control and readout only when explicitly requested."""
    schedules: list[PulseSchedule] = []

    def execute(*, schedule: PulseSchedule, **kwargs):
        schedules.append(schedule)
        return SimpleNamespace(data={"Q00": [SimpleNamespace(kerneled=1 + 1j)]})

    ctx = SimpleNamespace(
        resolve_qubit_label=lambda target: "Q00",
        resolve_read_label=lambda target: "R00",
        qubits={"Q00": SimpleNamespace(frequency=5.0)},
        targets={"R00": SimpleNamespace(frequency=7.0)},
        params=SimpleNamespace(
            control_amplitude={"Q00": 0.1}, readout_amplitude={"Q00": 0.2}
        ),
        experiment_system=SimpleNamespace(
            get_control_box_for_qubit=lambda target: SimpleNamespace(
                traits=SimpleNamespace(ctrl_ssb="L")
            )
        ),
        system_manager=SimpleNamespace(
            modified_backend_settings=lambda **kwargs: nullcontext()
        ),
        modified_frequencies=lambda frequencies: nullcontext(),
        reset_awg_and_capunits=lambda **kwargs: None,
    )
    service = CharacterizationService(
        experiment_context=cast(Any, ctx),
        measurement_service=cast(Any, SimpleNamespace(execute=execute)),
        calibration_service=cast(Any, None),
        pulse_service=cast(Any, None),
    )
    exp = object.__new__(Experiment)
    exp.__dict__["_characterization_service"] = service
    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.fitting.fit_sqrt_lorentzian",
        Mock(return_value={}),
    )
    kwargs: dict[str, Any] = dict(
        frequency_range=[5.0, 5.01, 5.02], plot=False, save_image=False
    )
    if drive_mode != "omitted":
        kwargs["simultaneous_drive"] = drive_mode
    if method == "qubit_spectroscopy":
        kwargs["power_range"] = [-30, -20]

    if method == "estimate_control_amplitude":
        with pytest.warns(DeprecationWarning, match="measure_qubit_resonance"):
            getattr(exp, method)("Q00", **kwargs)
    else:
        getattr(exp, method)("Q00", **kwargs)

    assert len(schedules) == (6 if method == "qubit_spectroscopy" else 3)
    for schedule in schedules:
        control = schedule.get_pulse_ranges()["Q00"][0]
        readout = schedule.get_pulse_ranges()["R00"][0]
        if drive_mode is False:
            assert readout.start == control.stop
        else:
            assert readout.start == control.start
