"""Tests for classifier validation in randomized benchmarking."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from qxpulse import PulseSchedule

from qubex.experiment.services.benchmarking_service import BenchmarkingService


@pytest.mark.parametrize("missing_classifier", ["Q17", "Q18"])
def test_randomized_benchmarking_2q_checks_classifiers_before_measurement(
    missing_classifier: str,
) -> None:
    """Given a missing 2Q classifier, when RB starts, then it fails before measurement."""
    measurement_calls: list[object] = []
    classifiers = {
        qubit: object() for qubit in ("Q17", "Q18") if qubit != missing_classifier
    }

    def measure(**_kwargs: object) -> None:
        measurement_calls.append(object())
        raise ValueError(f"Classifier not found for {missing_classifier}.")

    service = cast(Any, object.__new__(BenchmarkingService))
    service.__dict__["_experiment_context"] = SimpleNamespace(
        classifiers=classifiers,
        state_centers={"Q17": {0: 0j}, "Q18": {0: 0j}},
        experiment_system=SimpleNamespace(
            get_target=lambda _label: SimpleNamespace(is_cr=True)
        ),
        calib_note=SimpleNamespace(cr_params={"CR17-18": object()}),
        cr_pair=lambda _label: ("Q17", "Q18"),
    )
    service.__dict__["_measurement_service"] = SimpleNamespace(measure=measure)
    service.__dict__["_pulse_service"] = SimpleNamespace()
    service.__dict__["rb_sequence_2q"] = lambda **_kwargs: PulseSchedule(["Q17", "Q18"])

    with pytest.raises(
        ValueError,
        match=rf"^Classifier not found for {missing_classifier}\.$",
    ):
        service.randomized_benchmarking(
            targets="CR17-18",
            n_cliffords_range=[0],
            n_trials=1,
            seeds=[0],
            plot=False,
            save_image=False,
        )

    assert measurement_calls == []
