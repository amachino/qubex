"""Tests for `MeasurementService.build_classifier` result figures."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go

import qubex.visualization as viz
from qubex.experiment.models.result import Result
from qubex.experiment.services.measurement_service import MeasurementService
from qubex.measurement.classifiers import StateClassifierGMM


class _PredictOnlyClassifier:
    """Minimal classifier stub that supports predict but rejects classify."""

    def __init__(self) -> None:
        self.n_states = 2
        self.centers = {
            0: 0.0 + 0.0j,
            1: 1.0 + 0.0j,
        }
        self.predict_calls: list[np.ndarray] = []
        self.classify_calls = 0

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return deterministic labels for the provided prepared-state shots."""
        self.predict_calls.append(np.asarray(data))
        real = np.real(data)
        return (real >= 0.5).astype(int)

    def classify(
        self, target: str, data: np.ndarray, plot: bool = True
    ) -> dict[int, int]:
        """Fail when legacy classify-path is used."""
        _ = target, data, plot
        self.classify_calls += 1
        raise AssertionError("build_classifier should not call classify().")

    def save(self, path: object) -> None:
        """Ignore classifier persistence in tests."""
        _ = path


def test_build_classifier_aggregates_figure_fields_across_targets() -> None:
    """Given per-target classifier results, when building classifiers, then primary and named figures are preserved."""
    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = SimpleNamespace(qubit_labels=["Q00", "Q01"])
    service.__dict__["_pulse_service"] = SimpleNamespace()
    figures = {
        "Q00": go.Figure(),
        "Q01": go.Figure(),
    }
    classifiers = {
        "Q00": object(),
        "Q01": object(),
    }

    def fake_build_classifier(
        self: MeasurementService,
        targets: str,
        **_: object,
    ) -> Result:
        figure = figures[targets]
        return Result(
            data={
                "readout_fidelities": {targets: [1.0, 0.5]},
                "average_readout_fidelity": {targets: 0.75},
                "data": {0: np.array([0.0 + 0.0j])},
                "classifiers": {targets: classifiers[targets]},
            },
            figure=figure,
            figures={targets: figure},
        )

    service.__dict__["_build_classifier"] = MethodType(fake_build_classifier, service)

    result = service.build_classifier(
        targets=["Q00", "Q01"],
        save_classifier=False,
        plot=False,
    )

    assert result.figure is figures["Q00"]
    assert result.figures == figures
    assert result["classifiers"] == classifiers
    assert np.array_equal(result["data"]["Q00"][0], np.array([0.0 + 0.0j]))
    assert np.array_equal(result["data"]["Q01"][0], np.array([0.0 + 0.0j]))


def test__build_classifier_uses_predict_counts_and_returns_figures(
    monkeypatch,
) -> None:
    """Given classifier training data, when building classifiers, then figures are returned without calling classify."""
    service = cast(Any, object.__new__(MeasurementService))
    updated_classifiers: dict[str, object] = {}
    service.__dict__["_ctx"] = SimpleNamespace(
        qubit_labels=["Q00"],
        classifier_type="gmm",
        reference_phases={"Q00": 0.0},
        measurement=SimpleNamespace(
            update_classifiers=lambda classifiers: updated_classifiers.update(
                classifiers
            )
        ),
        chip_id="TEST",
        calib_note=SimpleNamespace(
            reference_phases={"Q00": 0.0},
            state_params={},
        ),
    )
    service.__dict__["_pulse_service"] = SimpleNamespace()

    prepared_state_0 = np.array([0.0 + 0.0j, 0.1 + 0.0j, 1.0 + 0.0j])
    prepared_state_1 = np.array([1.0 + 0.0j, 0.9 + 0.0j, 0.0 + 0.0j])
    classifier = _PredictOnlyClassifier()
    figure_calls: list[go.Figure] = []

    def fake_obtain_reference_points(
        self: MeasurementService,
        targets: list[str],
        **_: object,
    ) -> None:
        assert targets == ["Q00"]

    def fake_measure_state_distribution(
        self: MeasurementService,
        **_: object,
    ) -> list[object]:
        return [
            SimpleNamespace(
                data={"Q00": SimpleNamespace(kerneled=prepared_state_0)},
            ),
            SimpleNamespace(
                data={"Q00": SimpleNamespace(kerneled=prepared_state_1)},
            ),
        ]

    def fake_make_classification_figure(
        *,
        target: str,
        data: np.ndarray,
        labels: np.ndarray,
        centers: dict[int, complex],
        stddevs: dict[int, float] | None = None,
        n_samples: int = 1000,
    ) -> go.Figure:
        _ = stddevs, n_samples
        figure = go.Figure()
        figure.update_layout(title=f"{target}:{len(figure_calls)}")
        figure_calls.append(figure)
        assert centers == classifier.centers
        assert len(data) == len(labels)
        return figure

    monkeypatch.setattr(
        service,
        "obtain_reference_points",
        MethodType(fake_obtain_reference_points, service),
    )
    monkeypatch.setattr(
        service,
        "measure_state_distribution",
        MethodType(fake_measure_state_distribution, service),
    )
    monkeypatch.setattr(
        StateClassifierGMM,
        "fit",
        classmethod(lambda cls, data, phase=0.0: classifier),
    )
    monkeypatch.setattr(
        viz, "make_classification_figure", fake_make_classification_figure
    )

    result = service._build_classifier(  # noqa: SLF001
        targets="Q00",
        n_states=2,
        save_classifier=False,
        plot=False,
    )

    assert classifier.classify_calls == 0
    assert len(classifier.predict_calls) == 2
    assert updated_classifiers == {"Q00": classifier}
    assert result.figure is figure_calls[0]
    assert list((result.figures or {}).values()) == figure_calls
    assert result["readout_fidelities"]["Q00"] == [2 / 3, 2 / 3]
    assert result["average_readout_fidelity"]["Q00"] == 2 / 3
