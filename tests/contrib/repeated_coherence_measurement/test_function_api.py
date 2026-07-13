"""Tests for `qubex.contrib.experiment.repeated_coherence_measurement`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.contrib import repeated_coherence_measurement
from qubex.experiment.models.experiment_result import ExperimentResult


class _FakeExperiment:
    def __init__(self) -> None:
        self.ctx = SimpleNamespace(qubit_labels=["Q00", "Q01"])
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.t1_values = [
            {"Q00": 10.0, "Q01": 20.0},
            {"Q00": 12.0, "Q01": 22.0},
            {"Q00": 14.0, "Q01": 24.0},
        ]
        self.t2_values = [
            {"Q00": 30.0, "Q01": 40.0},
            {"Q00": 32.0, "Q01": 42.0},
            {"Q00": 34.0, "Q01": 44.0},
        ]
        self.ramsey_values = [
            {"Q00": 50.0, "Q01": 60.0},
            {"Q00": 52.0, "Q01": 62.0},
            {"Q00": 54.0, "Q01": 64.0},
        ]

    def t1_experiment(self, **kwargs: Any) -> ExperimentResult[Any]:
        self.calls.append(("t1", kwargs))
        return _result(self.t1_values[len(self.calls_for("t1")) - 1], "t1")

    def t2_experiment(self, **kwargs: Any) -> ExperimentResult[Any]:
        self.calls.append(("t2_echo", kwargs))
        return _result(self.t2_values[len(self.calls_for("t2_echo")) - 1], "t2")

    def ramsey_experiment(self, **kwargs: Any) -> ExperimentResult[Any]:
        self.calls.append(("ramsey", kwargs))
        return _result(self.ramsey_values[len(self.calls_for("ramsey")) - 1], "t2")

    def calls_for(self, mode: str) -> list[tuple[str, dict[str, Any]]]:
        return [call for call in self.calls if call[0] == mode]


def _result(values: dict[str, float], attribute: str) -> ExperimentResult[Any]:
    return ExperimentResult(
        data={
            target: SimpleNamespace(**{attribute: value})
            for target, value in values.items()
        }
    )


def test_repeated_coherence_measurement_is_exported_from_contrib() -> None:
    """Given contrib package, when imported, then repeated coherence helper is available."""
    assert callable(repeated_coherence_measurement)


def test_repeated_coherence_measurement_returns_raw_values_and_statistics() -> None:
    """Given repeated measurements, when all fits succeed, then all data and stats are returned."""
    exp = _FakeExperiment()

    result = repeated_coherence_measurement(
        cast(Any, exp),
        targets=["Q00", "Q01"],
        n_runs=3,
        modes=["T1", "t2", "t2_star"],
        n_shots=100,
        shot_interval=1.5,
        t1_options={"n_shots": 200, "time_range": [1, 2, 3]},
        ramsey_options={"detuning": 0.001},
    )

    assert result.data["modes"] == ["t1", "t2_echo", "ramsey"]
    assert result.data["metrics"] == {
        "t1": "t1",
        "t2_echo": "t2_echo",
        "ramsey": "t2_star",
    }
    assert result.data["values"]["t1"]["Q00"] == [10.0, 12.0, 14.0]
    assert result.data["values"]["t2_echo"]["Q01"] == [40.0, 42.0, 44.0]
    assert result.data["values"]["ramsey"]["Q00"] == [50.0, 52.0, 54.0]
    assert result.data["statistics"]["t1"]["Q00"] == {
        "metric": "t1",
        "mean": 12.0,
        "std": 2.0,
        "count": 3,
        "n_runs": 3,
    }
    assert result.data["statistics"]["ramsey"]["Q01"] == {
        "metric": "t2_star",
        "mean": 62.0,
        "std": 2.0,
        "count": 3,
        "n_runs": 3,
    }
    assert len(result.data["raw_results"]["t1"]) == 3
    assert result.data["failed_runs"]["t1"]["Q00"] == []

    assert exp.calls_for("t1")[0][1]["n_shots"] == 200
    assert exp.calls_for("t1")[0][1]["time_range"] == [1, 2, 3]
    assert exp.calls_for("t2_echo")[0][1]["n_shots"] == 100
    assert exp.calls_for("t2_echo")[0][1]["shot_interval"] == 1.5
    assert exp.calls_for("ramsey")[0][1]["detuning"] == 0.001


def test_repeated_coherence_measurement_summarizes_successful_values_only() -> None:
    """Given missing target results, when summarized, then failed runs are recorded."""
    exp = _FakeExperiment()
    exp.t1_values = [
        {"Q00": 10.0, "Q01": 20.0},
        {"Q00": 12.0},
        {"Q00": 14.0, "Q01": 24.0},
    ]

    result = repeated_coherence_measurement(
        cast(Any, exp),
        targets=["Q00", "Q01"],
        n_runs=3,
        modes=["t1"],
    )

    q01_values = result.data["values"]["t1"]["Q01"]
    assert q01_values[0] == 20.0
    assert np.isnan(q01_values[1])
    assert q01_values[2] == 24.0
    assert result.data["failed_runs"]["t1"]["Q01"] == [1]
    assert result.data["statistics"]["t1"]["Q01"]["mean"] == 22.0
    assert result.data["statistics"]["t1"]["Q01"]["std"] == pytest.approx(np.sqrt(8.0))
    assert result.data["statistics"]["t1"]["Q01"]["count"] == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_runs": 0}, "n_runs must be at least 1"),
        ({"n_runs": 1, "modes": ["unknown"]}, "Unknown coherence measurement mode"),
        ({"n_runs": 1, "modes": ["t1", "T1"]}, "Duplicate coherence measurement mode"),
        ({"n_runs": 1, "modes": []}, "At least one coherence measurement mode"),
    ],
)
def test_repeated_coherence_measurement_validates_inputs(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """Given invalid options, when called, then a clear ValueError is raised."""
    with pytest.raises(ValueError, match=match):
        repeated_coherence_measurement(cast(Any, _FakeExperiment()), **kwargs)
