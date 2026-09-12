"""Tests for single-result measurement analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qubex.measurement._analysis import (
    get_mitigated_probabilities,
    get_probabilities,
)
from qubex.measurement.models import (
    CaptureData,
    CapturePayload,
    ClassifierRef,
    MeasurementConfig,
    MeasurementResult,
    ReturnItem,
)


def _capture(
    target: str,
    values: list[float],
    *,
    waveform: bool = False,
    averaged: bool = False,
    states: bool = False,
) -> CaptureData:
    config = MeasurementConfig(
        n_shots=len(values),
        shot_interval=100.0,
        shot_averaging=averaged,
        time_integration=not waveform,
        state_classification=states,
    )
    data = np.asarray(np.mean(values) if averaged else values, dtype=complex)
    if waveform:
        data = np.stack([data / 2, data / 2], axis=-1)
    payload: dict[str, Any] = {config.primary_return_item.value: data}
    if states:
        payload[ReturnItem.STATE_SERIES.value] = np.asarray(values)
    return CaptureData(
        target=target,
        config=config,
        payload=CapturePayload(**payload),
        sampling_period=0.8,
    )


def _result(*captures: CaptureData) -> MeasurementResult:
    data: dict[str, list[CaptureData]] = {}
    for capture in captures:
        data.setdefault(capture.target, []).append(capture)
    return MeasurementResult(data=data, measurement_config=captures[0].config)


def _classifier(matrix: np.ndarray | None = None) -> Any:
    class Classifier:
        def __init__(self) -> None:
            self.confusion_matrix = np.eye(2) if matrix is None else matrix
            self.n_states = len(self.confusion_matrix)
            self.calls = 0

        def predict(self, data: np.ndarray) -> np.ndarray:
            self.calls += 1
            return np.asarray(data.real, dtype=int)

    return Classifier()


@pytest.mark.parametrize("waveform", [False, True])
@pytest.mark.parametrize("averaged", [False, True])
@pytest.mark.parametrize("values", [[1.0], [1.0, 3.0]])
def test_kerneled_preserves_shots_and_integrates_only_time(
    waveform: bool, averaged: bool, values: list[float]
) -> None:
    """Kerneled IQ should preserve shots or the averaged scalar without scaling."""
    capture = _capture("Q0", values, waveform=waveform, averaged=averaged)
    before = capture.data.copy()
    writeable = capture.data.flags.writeable
    expected = np.asarray(np.mean(values) if averaged else values)

    np.testing.assert_allclose(capture.kerneled, expected, rtol=0, atol=1e-12)
    assert capture.kerneled.shape == expected.shape
    np.testing.assert_array_equal(capture.data, before)
    assert capture.data.flags.writeable == writeable


@pytest.mark.parametrize("averaged", [False, True])
def test_kerneled_prefers_returned_iq(averaged: bool) -> None:
    """Returned IQ should take precedence over software waveform integration."""
    capture = _capture("Q0", [1, 3], waveform=True, averaged=averaged)
    item = ReturnItem.AVERAGED_IQ if averaged else ReturnItem.IQ_SERIES
    iq = np.asarray(7 if averaged else [7, 9], dtype=complex)
    capture = CaptureData(
        target="Q0",
        config=capture.config.model_copy(
            update={"return_items": (*capture.config.return_items, item)}
        ),
        payload=capture.payload.model_copy(update={item.value: iq}),
        sampling_period=0.8,
    )
    np.testing.assert_array_equal(capture.kerneled, iq)


def test_analysis_preserves_serialized_data(tmp_path: Path) -> None:
    """Derived IQ and probabilities should not change JSON or NetCDF payloads."""
    result = _result(_capture("Q0", [0, 1], waveform=True, states=True))
    before = result.to_json()
    _ = result.data["Q0"][0].kerneled
    assert get_probabilities(result) == {"0": 0.5, "1": 0.5}
    assert result.to_json() == before
    assert "kerneled" not in before
    restored = MeasurementResult.load(result.save(tmp_path / "result.nc"))
    assert get_probabilities(restored) == {"0": 0.5, "1": 0.5}
    np.testing.assert_array_equal(restored.data["Q0"][0].kerneled, [0, 1])


@pytest.mark.parametrize("waveform", [False, True])
def test_probabilities_preserve_joint_shot_correlations(waveform: bool) -> None:
    """Joint probabilities should count aligned shots instead of multiplying marginals."""
    result = _result(
        _capture("Q0", [0, 0, 1, 1], waveform=waveform),
        _capture("Q1", [1, 1, 0, 0], waveform=waveform),
    )
    assert get_probabilities(
        result, classifiers={"Q0": _classifier(), "Q1": _classifier()}
    ) == {"01": 0.5, "10": 0.5}


def test_stored_states_bypass_prediction_and_file_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stored per-shot states should be usable without prediction or reference loading."""
    result = _result(_capture("Q0", [0, 1, 1], states=True, averaged=True))
    result = result.model_copy(
        update={"classifier_refs": {"Q0": ClassifierRef(path="missing.pkl")}}
    )

    def fail_load(_self: ClassifierRef) -> Any:
        raise AssertionError("unexpected classifier load")

    monkeypatch.setattr(ClassifierRef, "load", fail_load)
    classifier = _classifier()
    assert get_probabilities(result) == {"0": 1 / 3, "1": 2 / 3}
    assert get_probabilities(result, classifiers={"Q0": classifier}) == {
        "0": 1 / 3,
        "1": 2 / 3,
    }
    assert classifier.calls == 0


def test_probabilities_select_capture_and_target_order() -> None:
    """Capture selection and target order should determine joint state labels."""
    result = _result(
        _capture("Q0", [0], states=True),
        _capture("Q0", [1], states=True),
        _capture("Q1", [2], states=True),
    )
    assert get_probabilities(result, ["Q1", "Q0"]) == {"21": 1.0}
    assert get_probabilities(result, ["Q1", "Q0"], capture_index=0) == {"20": 1.0}


def test_mitigation_respects_target_order_and_mixed_dimensions() -> None:
    """Correction should invert the ordered normalized response without clipping."""
    result = _result(
        _capture("Q0", [0, 0, 1, 1], states=True),
        _capture("Q1", [0, 0, 1, 2], states=True),
    )
    cm0 = np.array([[8, 2], [1, 4]], dtype=float)
    cm1 = np.array([[9, 1, 0], [1, 7, 2], [0, 1, 4]], dtype=float)
    corrected = get_mitigated_probabilities(
        result,
        ["Q1", "Q0"],
        classifiers={"Q0": _classifier(cm0), "Q1": _classifier(cm1)},
    )
    response = np.kron(
        cm1 / cm1.sum(axis=1, keepdims=True), cm0 / cm0.sum(axis=1, keepdims=True)
    )
    values = np.array(
        [corrected[label] for label in ["00", "01", "10", "11", "20", "21"]]
    )
    np.testing.assert_allclose(
        values @ response, [0.5, 0, 0, 0.25, 0, 0.25], rtol=0, atol=1e-12
    )
    assert np.any(values < 0)


@pytest.mark.parametrize("targets", [["missing"], ["Q0", "Q0"], []])
def test_probabilities_reject_invalid_targets(targets: list[str]) -> None:
    """Analysis should reject missing, duplicate, and empty targets."""
    with pytest.raises(ValueError, match=r"No measurement capture|nonempty, unique"):
        get_probabilities(_result(_capture("Q0", [0], states=True)), targets)


def test_probabilities_reject_invalid_capture_index() -> None:
    """An out-of-range capture index should raise IndexError."""
    with pytest.raises(IndexError):
        get_probabilities(_result(_capture("Q0", [0], states=True)), capture_index=1)


@pytest.mark.parametrize("averaged", [False, True])
def test_probabilities_require_usable_iq_and_explicit_classifier(
    averaged: bool,
) -> None:
    """IQ analysis should require per-shot data and an explicitly provided classifier."""
    result = _result(_capture("Q0", [0, 1], averaged=averaged))
    result = result.model_copy(
        update={"classifier_refs": {"Q0": ClassifierRef(path="missing.pkl")}}
    )
    with pytest.raises(ValueError, match=r"Per-shot IQ|Classifier for target"):
        get_probabilities(
            result, classifiers={"Q0": _classifier()} if averaged else None
        )


@pytest.mark.parametrize("values", [[-1], [0.5], [float("nan")]])
def test_probabilities_reject_invalid_states(values: list[float]) -> None:
    """State labels should be finite nonnegative integers."""
    with pytest.raises(ValueError, match="nonnegative integer state"):
        get_probabilities(_result(_capture("Q0", values, states=True)))


def test_probabilities_reject_mismatched_shots() -> None:
    """Joint state aggregation should reject unequal shot counts."""
    with pytest.raises(ValueError, match="Shot counts must match"):
        get_probabilities(
            _result(
                _capture("Q0", [0], states=True), _capture("Q1", [0, 1], states=True)
            )
        )


@pytest.mark.parametrize("matrix", [np.ones((2, 2)), np.array([[0, 0], [1, 1]])])
def test_mitigation_rejects_unusable_confusion_matrix(matrix: np.ndarray) -> None:
    """Singular responses and empty prepared-state rows should fail explicitly."""
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        get_mitigated_probabilities(
            _result(_capture("Q0", [0], states=True)),
            classifiers={"Q0": _classifier(matrix)},
        )
