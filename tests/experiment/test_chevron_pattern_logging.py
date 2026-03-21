"""Tests for chevron-pattern logging behavior."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from qubex.experiment.models.result import Result
from qubex.experiment.services.characterization_service import CharacterizationService


def test_chevron_pattern_suppresses_low_r2_fit_warning(monkeypatch) -> None:
    """Given chevron sweeps, when fitting each sweep point, then low-r2 warnings are suppressed."""
    fit_rabi_calls: list[dict[str, Any]] = []

    @contextmanager
    def _no_output() -> Any:
        yield

    def _fit_rabi(**kwargs: Any) -> dict[str, float]:
        fit_rabi_calls.append(kwargs)
        return {"frequency": 0.25}

    def _fit_detuned_rabi(**kwargs: Any) -> dict[str, float]:
        return {"f_resonance": kwargs["control_frequencies"][0]}

    service = cast(Any, object.__new__(CharacterizationService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        qubit_labels=["Q00"],
        targets={"Q00": SimpleNamespace(frequency=5.0)},
        params=SimpleNamespace(control_amplitude={"Q00": 0.1}),
        util=SimpleNamespace(
            create_qubit_subgroups=lambda targets: [list(targets)],
            no_output=_no_output,
        ),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(
        obtain_rabi_params=lambda **kwargs: SimpleNamespace(
            rabi_params={"Q00": SimpleNamespace()}
        ),
        sweep_parameter=lambda **kwargs: Result(
            data={
                "Q00": SimpleNamespace(
                    target="Q00",
                    sweep_range=np.array([0.0, 4.0, 8.0]),
                    data=np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j]),
                    rabi_param=None,
                    normalized=np.array([0.0, 1.0, 0.0]),
                )
            }
        ),
    )
    service.__dict__["_calibration_service"] = SimpleNamespace()
    service.__dict__["_pulse_service"] = SimpleNamespace()

    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.fitting.fit_rabi",
        _fit_rabi,
    )
    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.fitting.fit_detuned_rabi",
        _fit_detuned_rabi,
    )

    result = service.chevron_pattern(
        targets=["Q00"],
        detuning_range=np.array([0.0]),
        time_range=np.array([0.0, 4.0, 8.0]),
        plot=False,
        save_image=False,
    )

    assert fit_rabi_calls
    assert all(call["warn_low_r2"] is False for call in fit_rabi_calls)
    assert result.data["resonant_frequencies"] == {"Q00": 5.0}
