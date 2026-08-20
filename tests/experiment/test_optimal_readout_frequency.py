"""Tests for optimal readout-frequency selection."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.experiment.services.characterization_service import CharacterizationService


class _FakeFigure:
    """Plotly-like figure stub for optimal-frequency tests."""

    def add_scatter(self, **_kwargs: Any) -> None:
        """Accept scatter traces."""
        return

    def add_vline(self, **_kwargs: Any) -> None:
        """Accept vertical markers."""
        return

    def add_annotation(self, **_kwargs: Any) -> None:
        """Accept annotations."""
        return

    def update_layout(self, **_kwargs: Any) -> None:
        """Accept layout updates."""
        return


class _FakeContext:
    """Experiment-context stub for optimal-frequency tests."""

    def __init__(self) -> None:
        self.reference_phases = {"Q00": 0.0}
        self.targets = {"RQ00": SimpleNamespace(frequency=5.105)}
        self.current_frequency: float | None = None
        self.frequency_stack: list[float | None] = []

    @staticmethod
    def resolve_read_label(_target: str) -> str:
        """Resolve the readout target label."""
        return "RQ00"

    @contextmanager
    def modified_frequencies(
        self,
        frequencies: dict[str, float] | None,
    ) -> Iterator[None]:
        """Record the active readout frequency."""
        self.frequency_stack.append(self.current_frequency)
        self.current_frequency = None if frequencies is None else frequencies["RQ00"]
        try:
            yield
        finally:
            self.current_frequency = self.frequency_stack.pop()


class _FakeMeasurementService:
    """Measurement-service stub for optimal-frequency tests."""

    def __init__(self, ctx: _FakeContext) -> None:
        self.ctx = ctx
        self.measure_state_calls: list[dict[str, Any]] = []
        self.build_classifier_calls: list[dict[str, Any]] = []

    def measure_state(self, states: dict[str, str], **kwargs: Any) -> Any:
        """Return IQ data tagged by the active frequency and state."""
        state = next(iter(states.values()))
        frequency = cast(float, self.ctx.current_frequency)
        self.measure_state_calls.append(
            {"states": states, "frequency": frequency, **kwargs}
        )
        value = frequency + (0.0 if state == "0" else 0.001)
        iq = np.full(4, value + 0j, dtype=np.complex128)
        return SimpleNamespace(data={"Q00": SimpleNamespace(kerneled=iq)})

    def build_classifier(self, **kwargs: Any) -> Any:
        """Record classifier build calls."""
        self.build_classifier_calls.append(
            {"frequency": self.ctx.current_frequency, **kwargs}
        )
        return SimpleNamespace(figures={"classifier": object()})


def test_find_optimal_readout_frequency_uses_fidelity_plateau_threshold(
    monkeypatch,
) -> None:
    """Given fidelity objective, first frequency within peak ratio is selected."""
    service = cast(Any, object.__new__(CharacterizationService))
    ctx = _FakeContext()
    measurement_service = _FakeMeasurementService(ctx)
    service.__dict__["_experiment_context"] = ctx
    service.__dict__["_measurement_service"] = measurement_service

    frequency_range = np.array([5.0, 5.1, 5.2])

    def fake_measure_reflection_coefficient(
        *_args: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        return {"frequency_range": frequency_range}

    class _Classifier:
        def __init__(self, predictions: dict[int, np.ndarray]) -> None:
            self._predictions = predictions

        def predict(self, iq: np.ndarray) -> np.ndarray:
            return self._predictions[id(iq)]

    def fake_fit(data: dict[int, np.ndarray], *, phase: float) -> _Classifier:
        _ = phase
        frequency = round(float(np.real(data[0][0])), 1)
        if frequency == 5.0:
            pred_0 = np.array([0, 1, 1, 1])
            pred_1 = np.array([1, 1, 1, 0])
        elif frequency == 5.1:
            pred_0 = np.array([0, 0, 0, 0])
            pred_1 = np.array([1, 1, 1, 1])
        else:
            pred_0 = np.array([0, 0, 0, 0])
            pred_1 = np.array([1, 1, 1, 1])
        return _Classifier({id(data[0]): pred_0, id(data[1]): pred_1})

    monkeypatch.setattr(
        service,
        "measure_reflection_coefficient",
        fake_measure_reflection_coefficient,
    )
    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.viz.make_figure",
        lambda **_kwargs: _FakeFigure(),
    )
    monkeypatch.setattr(
        "qubex.measurement.classifiers.state_classifier_gmm.StateClassifierGMM.fit",
        fake_fit,
    )

    result = service.find_optimal_readout_frequency(
        "Q00",
        df=0.1,
        frequency_width=0.21,
        objective="fidelity",
        fidelity_ratio=0.99,
        shots=4,
        interval=10.0,
        plot=False,
        save_image=False,
    )

    np.testing.assert_allclose(
        result.data["optimal_frequency"], 5.1, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(result.data["readout_fidelity"], [0.5, 1.0, 1.0])
    assert result.data["signals_0"].shape == (3, 4)
    assert result.data["signals_1"].shape == (3, 4)
    with pytest.warns(DeprecationWarning, match="legacy figure payload key"):
        assert result.data["fig"] is result.figure
    np.testing.assert_allclose(
        [call["frequency"] for call in measurement_service.measure_state_calls],
        [5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        rtol=0,
        atol=1e-12,
    )
    assert measurement_service.build_classifier_calls == [
        {
            "frequency": 5.1,
            "targets": "Q00",
            "readout_amplitudes": None,
            "n_shots": 4,
            "shot_interval": 10.0,
            "plot": False,
        }
    ]
