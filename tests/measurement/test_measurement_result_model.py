"""Tests for the high-level measurement result model."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from netCDF4 import Dataset
from pydantic import ValidationError
from sklearn import __version__ as SKLEARN_VERSION

from qubex.measurement.classifiers.state_classifier import StateClassifier
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.models import (
    CaptureData,
    CapturePayload,
    ClassifierRef,
    MeasurementConfig,
    MeasurementResult,
    ReturnItem,
)
from qubex.measurement.models.measure_result import (
    MeasureData,
    MeasureMode,
    MeasureResult,
    MultipleMeasureResult,
)
from qubex.typing import MeasurementMode


def _make_multiple_measure_result() -> MultipleMeasureResult:
    data0 = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.array([1.0 + 0.0j]),
        classifier=None,
    )
    data1 = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.array([2.0 + 0.0j]),
        classifier=None,
    )
    return MultipleMeasureResult(
        mode=MeasureMode.AVG,
        data={"Q00": [data0, data1]},
        config={"shots": 2},
    )


def test_legacy_multiple_measure_result_does_not_warn_on_init() -> None:
    """MultipleMeasureResult instantiation should not emit deprecation warnings."""
    data = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.array([1.0 + 0.0j]),
        classifier=None,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        MultipleMeasureResult(
            mode=MeasureMode.AVG,
            data={"Q00": [data]},
            config={"shots": 1},
        )

    assert not any(
        issubclass(warning.category, DeprecationWarning) for warning in captured
    )


def _make_config(
    *,
    mode: MeasurementMode = "avg",
    shots: int = 2,
    time_integration: bool = False,
) -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=shots,
        shot_interval=100.0,
        shot_averaging=(mode == "avg"),
        time_integration=time_integration,
        state_classification=False,
    )


def _make_capture(
    *,
    target: str,
    raw: np.ndarray,
    measurement_config: MeasurementConfig,
    sampling_period: float,
    classifier_ref: ClassifierRef | None = None,
) -> CaptureData:
    return CaptureData.from_primary_data(
        target=target,
        data=raw,
        config=measurement_config,
        sampling_period=sampling_period,
        classifier_ref=classifier_ref,
    )


def test_to_multiple_measure_result_returns_wrapped_result() -> None:
    """Given legacy multiple result, when converting round-trip, then mode and config are preserved."""
    multiple = _make_multiple_measure_result()
    config = _make_config()
    result = MeasurementResultConverter.from_multiple(
        multiple,
        measurement_config=config,
    )
    restored = MeasurementResultConverter.to_multiple_measure_result(
        result,
        config=multiple.config,
    )

    assert restored.mode == multiple.mode
    assert restored.config == multiple.config
    assert result.device_config == multiple.config
    assert result.measurement_config == config
    assert result.data["Q00"][0].data.flags.writeable is False
    assert np.array_equal(
        restored.data["Q00"][0].raw,
        multiple.data["Q00"][0].raw,
    )
    assert result.measurement_config.shot_averaging is True


def test_to_measure_result_selects_requested_index() -> None:
    """Given wrapped multiple result, when converting with an index, then selected capture is returned."""
    multiple = _make_multiple_measure_result()
    result = MeasurementResultConverter.from_multiple(
        multiple,
        measurement_config=_make_config(),
    )

    single: MeasureResult = MeasurementResultConverter.to_measure_result(
        result,
        index=1,
        config=multiple.config,
    )

    assert single.mode == MeasureMode.AVG
    assert np.array_equal(single.data["Q00"].raw, multiple.data["Q00"][1].raw)
    assert single.config == multiple.config


def test_to_measure_result_raises_for_invalid_index() -> None:
    """Given wrapped multiple result, when index is out of range, then IndexError is raised."""
    result = MeasurementResultConverter.from_multiple(
        _make_multiple_measure_result(),
        measurement_config=_make_config(),
    )

    with pytest.raises(IndexError):
        MeasurementResultConverter.to_measure_result(result, index=10)


def test_measurement_result_repr_includes_targets_counts_and_config() -> None:
    """MeasurementResult repr should summarize targets, captures, and config flags."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=False,
        state_classification=True,
    )
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([[1.0 + 0.0j], [2.0 + 0.0j]]),
                    measurement_config=config,
                    sampling_period=0.4,
                )
            ],
            "Q01": [
                _make_capture(
                    target="Q01",
                    raw=np.array([[3.0 + 0.0j], [4.0 + 0.0j]]),
                    measurement_config=config,
                    sampling_period=0.4,
                )
            ],
        },
        measurement_config=config,
    )

    text = repr(result)

    assert "targets=[Q00, Q01]" in text
    assert "captures=2" in text
    assert "shot_averaging=False" in text
    assert "time_integration=False" in text
    assert "state_classification=True" in text


def test_to_measure_result_propagates_sampling_period() -> None:
    """Given canonical capture data with sampling period, conversion keeps per-capture period."""
    config = _make_config(mode="avg", shots=2)
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([1.0 + 0.0j]),
                    measurement_config=config,
                    sampling_period=0.4,
                ),
                _make_capture(
                    target="Q00",
                    raw=np.array([2.0 + 0.0j]),
                    measurement_config=config,
                    sampling_period=0.8,
                ),
            ]
        },
        device_config={"shots": 2},
        measurement_config=config,
    )

    single = MeasurementResultConverter.to_measure_result(result, index=1)

    assert single.data["Q00"].sampling_period == 0.8
    assert np.array_equal(single.data["Q00"].times, np.array([0.0]))


def test_json_roundtrip_preserves_capture_data() -> None:
    """Given serialized measurement result, when deserializing, then capture payload is preserved."""
    config = _make_config(mode="avg", shots=2)
    classifier_ref = ClassifierRef(path="classifier-Q00.pkl", version=SKLEARN_VERSION)
    original = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([1.0 + 0.0j]),
                    measurement_config=config,
                    sampling_period=2.0,
                    classifier_ref=classifier_ref,
                ),
                _make_capture(
                    target="Q00",
                    raw=np.array([2.0 + 0.0j]),
                    measurement_config=config,
                    sampling_period=2.0,
                    classifier_ref=classifier_ref,
                ),
            ]
        },
        device_config={"shots": 2},
        measurement_config=config,
    )
    serialized = original.to_dict()

    restored = MeasurementResult.from_json(original.to_json())

    assert serialized["__meta__"]["format"] == "qxdata"
    assert serialized["__meta__"]["version"] == 1
    assert np.array_equal(
        restored.data["Q00"][0].data,
        original.data["Q00"][0].data,
    )
    assert restored.data["Q00"][0].classifier_ref == classifier_ref
    assert restored.device_config == original.device_config
    assert restored.measurement_config == original.measurement_config


def test_netcdf_roundtrip_preserves_capture_data(tmp_path) -> None:
    """Given NetCDF save/load, when round-tripping, then capture data and metadata are preserved."""
    config = _make_config(mode="single", shots=2)
    original = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([[1.0 + 2.0j], [3.0 + 4.0j]]),
                    measurement_config=config,
                    sampling_period=2.0,
                )
            ],
            "Q01": [
                _make_capture(
                    target="Q01",
                    raw=np.array([[5.0 + 6.0j], [9.0 + 10.0j]]),
                    measurement_config=config,
                    sampling_period=1.0,
                ),
                _make_capture(
                    target="Q01",
                    raw=np.array([[7.0 + 8.0j], [11.0 + 12.0j]]),
                    measurement_config=config,
                    sampling_period=1.0,
                ),
            ],
        },
        device_config={"shots": 2},
        measurement_config=config,
    )
    path = tmp_path / "measurement_result.nc"

    saved = original.save_netcdf(path)
    restored = MeasurementResult.load_netcdf(saved)

    assert restored.device_config == original.device_config
    assert restored.measurement_config == original.measurement_config
    assert np.array_equal(
        restored.data["Q00"][0].data,
        original.data["Q00"][0].data,
    )
    assert np.array_equal(
        restored.data["Q01"][0].data,
        original.data["Q01"][0].data,
    )
    assert np.array_equal(
        restored.data["Q01"][1].data,
        original.data["Q01"][1].data,
    )
    assert restored.data["Q00"][0].sampling_period == pytest.approx(2.0)
    assert restored.data["Q01"][0].sampling_period == pytest.approx(1.0)


def test_save_writes_netcdf_file(tmp_path) -> None:
    """Given save(), when called, then it writes a readable NetCDF file."""
    config = _make_config(mode="avg", shots=2)
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([1.0 + 0.0j]),
                    measurement_config=config,
                    sampling_period=2.0,
                )
            ]
        },
        measurement_config=config,
    )

    path = result.save(tmp_path / "result.nc")
    restored = MeasurementResult.load_netcdf(path)

    assert path.name == "result.nc"
    assert np.array_equal(restored.data["Q00"][0].data, np.array([1.0 + 0.0j]))


def test_load_alias_reads_netcdf_file(tmp_path) -> None:
    """Given netcdf file, load alias should delegate to load_netcdf."""
    config = _make_config(mode="avg", shots=1)
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([1.0 + 0.0j]),
                    measurement_config=config,
                    sampling_period=2.0,
                )
            ]
        },
        measurement_config=config,
    )
    path = result.save(tmp_path / "result.nc")
    restored = MeasurementResult.load(path)

    assert np.array_equal(restored.data["Q00"][0].data, np.array([1.0 + 0.0j]))


def test_capture_data_save_writes_netcdf_file(tmp_path) -> None:
    """Given capture data, save alias should write a readable NetCDF file."""
    config = _make_config(mode="avg", shots=1)
    capture = _make_capture(
        target="Q00",
        raw=np.array([1.0 + 0.0j]),
        measurement_config=config,
        sampling_period=2.0,
    )

    path = capture.save(tmp_path / "capture.nc")
    restored = CaptureData.load_netcdf(path)

    assert path.name == "capture.nc"
    assert restored.target == "Q00"
    assert np.array_equal(restored.data, np.array([1.0 + 0.0j]))
    assert restored.config == config


def test_capture_data_load_alias_reads_netcdf_file(tmp_path) -> None:
    """Given capture NetCDF file, load alias should delegate to load_netcdf."""
    config = _make_config(mode="avg", shots=1)
    capture = _make_capture(
        target="Q00",
        raw=np.array([1.0 + 0.0j]),
        measurement_config=config,
        sampling_period=2.0,
    )
    path = capture.save_netcdf(tmp_path / "capture.nc")

    restored = CaptureData.load(path)

    assert restored.target == "Q00"
    assert np.array_equal(restored.data, np.array([1.0 + 0.0j]))
    assert restored.config == config


def test_measurement_result_requires_measurement_config() -> None:
    """Given missing measurement config, result construction raises validation error."""
    with pytest.raises(ValidationError):
        _ = MeasurementResult.model_validate(
            {"data": {"Q00": [np.array([1.0 + 0.0j])]}}
        )


def test_measurement_result_rejects_legacy_array_data() -> None:
    """Given legacy array payload, result validation should fail."""
    config = _make_config(mode="avg", shots=1)
    with pytest.raises(ValidationError):
        _ = MeasurementResult.model_validate(
            {
                "data": {"Q00": [np.array([1.0 + 0.0j])]},
                "measurement_config": config,
            }
        )


def test_capture_data_rejects_non_averaged_scalar_raw() -> None:
    """Given non-averaged capture, scalar raw data should fail validation."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=False,
        state_classification=False,
    )

    with pytest.raises(ValidationError):
        _ = CaptureData.from_primary_data(
            target="Q00",
            data=np.array(1.0 + 0.0j),
            config=config,
            sampling_period=0.4,
        )


def test_capture_data_rejects_non_averaged_shot_count_mismatch() -> None:
    """Given non-averaged capture, first-axis length should match measurement shots."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=False,
    )

    with pytest.raises(ValidationError):
        _ = CaptureData.from_primary_data(
            target="Q00",
            data=np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
            config=config,
            sampling_period=0.4,
        )


def test_capture_data_accepts_structured_waveform_payload() -> None:
    """Given waveform_series payload, raw compatibility property should expose the same data."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=False,
        state_classification=False,
    )
    waveform = np.array(
        [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
        dtype=np.complex128,
    )

    capture = CaptureData(
        target="Q00",
        config=config,
        payload=CapturePayload(waveform_series=waveform),
        sampling_period=0.4,
    )

    assert capture.payload.waveform_series is not None
    assert np.array_equal(capture.payload.waveform_series, waveform)
    assert np.array_equal(capture.data, waveform)


def test_capture_data_from_primary_data_populates_mode_field() -> None:
    """Given averaged waveform mode, from_primary_data should populate averaged_waveform."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=True,
        time_integration=False,
        state_classification=False,
    )
    raw = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)

    capture = CaptureData.from_primary_data(
        target="Q00",
        data=raw,
        config=config,
        sampling_period=0.4,
    )

    assert capture.payload.averaged_waveform is not None
    assert np.array_equal(capture.payload.averaged_waveform, raw)
    assert np.array_equal(capture.data, raw)


def test_capture_data_payload_properties_allow_none() -> None:
    """Given waveform-only capture, non-matching payload properties return None."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=True,
        time_integration=False,
        state_classification=False,
    )
    raw = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    capture = CaptureData.from_primary_data(
        target="Q00",
        data=raw,
        config=config,
        sampling_period=0.4,
    )

    assert capture.waveform_series is None
    assert capture.iq_series is None
    assert capture.state_series is None
    assert capture.averaged_waveform is not None
    assert np.array_equal(capture.averaged_waveform, raw)
    assert capture.averaged_iq is None


def test_capture_data_data_returns_waveform_payload() -> None:
    """Given waveform capture, data and waveform_series should expose waveform payload."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=False,
        state_classification=False,
    )
    waveform = np.array(
        [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
        dtype=np.complex128,
    )
    capture = CaptureData.from_primary_data(
        target="Q00",
        data=waveform,
        config=config,
        sampling_period=0.4,
    )

    assert np.array_equal(capture.data, waveform)
    assert capture.waveform_series is not None
    assert np.array_equal(capture.waveform_series, waveform)


def test_capture_data_waveform_series_is_none_for_iq_primary_payload() -> None:
    """Given IQ primary payload, waveform_series should be None and data should expose IQ data."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=False,
    )
    iq_series = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    capture = CaptureData.from_primary_data(
        target="Q00",
        data=iq_series,
        config=config,
        sampling_period=0.4,
    )

    assert capture.waveform_series is None
    assert np.array_equal(capture.data, iq_series)


def test_capture_data_normalizes_iq_series_singleton_axis() -> None:
    """Given singleton IQ axis, capture data should normalize to one value per shot."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=False,
    )
    capture = CaptureData.from_primary_data(
        target="Q00",
        data=np.array([[1.0 + 0.0j], [2.0 + 0.0j]], dtype=np.complex128),
        config=config,
        sampling_period=0.4,
    )

    assert capture.data.shape == (2,)
    assert np.array_equal(
        capture.data,
        np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128),
    )


def test_capture_data_normalizes_one_shot_waveform_to_2d_series() -> None:
    """Given one-shot waveform vector, capture data should keep explicit shot and time axes."""
    config = MeasurementConfig(
        n_shots=1,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=False,
        state_classification=False,
    )
    capture = CaptureData.from_primary_data(
        target="Q00",
        data=np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128),
        config=config,
        sampling_period=0.4,
    )

    assert capture.data.shape == (1, 2)
    assert np.array_equal(
        capture.data,
        np.array([[1.0 + 0.0j, 2.0 + 0.0j]], dtype=np.complex128),
    )


def test_capture_data_normalizes_averaged_iq_to_scalar() -> None:
    """Given singleton averaged IQ axis, capture data should normalize to a scalar."""
    config = MeasurementConfig(
        n_shots=4,
        shot_interval=100.0,
        shot_averaging=True,
        time_integration=True,
        state_classification=False,
    )
    capture = CaptureData.from_primary_data(
        target="Q00",
        data=np.array([[1.0 + 0.0j]], dtype=np.complex128),
        config=config,
        sampling_period=0.4,
    )

    assert np.asarray(capture.data).shape == ()
    assert np.array_equal(
        np.asarray(capture.data),
        np.array(1.0 + 0.0j, dtype=np.complex128),
    )


def test_capture_data_from_primary_data_uses_mode_not_return_item_order() -> None:
    """Given reordered return_items, from_primary_data should still follow mode flags."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=True,
        return_items=(ReturnItem.STATE_SERIES, ReturnItem.IQ_SERIES),
    )
    raw = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)

    capture = CaptureData.from_primary_data(
        target="Q00",
        data=raw,
        config=config,
        sampling_period=0.4,
    )

    assert capture.payload.iq_series is not None
    assert np.array_equal(capture.payload.iq_series, raw)
    assert capture.payload.state_series is None
    assert np.array_equal(capture.data, raw)


def test_capture_data_state_series_returns_payload_when_available() -> None:
    """Given state payload, state_series should return it as-is."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=True,
        return_items=(ReturnItem.IQ_SERIES, ReturnItem.STATE_SERIES),
    )
    state_series = np.array([0, 1], dtype=np.int64)
    capture = CaptureData(
        target="Q00",
        config=config,
        payload=CapturePayload(
            iq_series=np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128),
            state_series=state_series,
        ),
        sampling_period=0.4,
    )

    assert capture.state_series is state_series


def test_capture_data_state_series_returns_none_when_not_provided() -> None:
    """Given no state payload, state_series should return None."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=True,
        return_items=(ReturnItem.IQ_SERIES, ReturnItem.STATE_SERIES),
    )
    capture = CaptureData.from_primary_data(
        target="Q00",
        data=np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128),
        config=config,
        sampling_period=0.4,
    )

    assert capture.state_series is None


def test_capture_data_rejects_payload_not_requested_by_return_items() -> None:
    """Given payload outside return_items, capture validation should fail."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=False,
        return_items=(ReturnItem.IQ_SERIES,),
    )

    with pytest.raises(ValidationError):
        _ = CaptureData(
            target="Q00",
            config=config,
            payload=CapturePayload(
                waveform_series=np.array(
                    [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
                    dtype=np.complex128,
                ),
            ),
            sampling_period=0.4,
        )


def test_capture_data_rejects_multiple_iq_values_per_shot() -> None:
    """Given non-singleton integrated IQ payload, capture validation should fail."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=False,
    )

    with pytest.raises(ValidationError, match="iq_series must contain exactly one"):
        _ = CaptureData.from_primary_data(
            target="Q00",
            data=np.array(
                [
                    [1.0 + 0.0j, 2.0 + 0.0j],
                    [3.0 + 0.0j, 4.0 + 0.0j],
                ],
                dtype=np.complex128,
            ),
            config=config,
            sampling_period=0.4,
        )


def test_capture_data_repr_summarizes_raw_array() -> None:
    """CaptureData repr should summarize long raw arrays."""
    config = _make_config(mode="avg", shots=1)
    capture = _make_capture(
        target="Q00",
        raw=np.arange(1024, dtype=np.float64),
        measurement_config=config,
        sampling_period=0.4,
    )

    text = repr(capture)

    assert "data=array([0., ...], shape=(1024,))" in text


def test_capture_data_classifier_uses_shared_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dummy_classifier,
) -> None:
    """Given same classifier ref across captures, classifier loading should happen once."""
    config = _make_config(mode="avg", shots=1)
    classifier_path = tmp_path / "classifier-Q00.pkl"
    classifier_path.write_bytes(b"classifier-v1")
    ref = ClassifierRef(path=str(classifier_path), version=SKLEARN_VERSION)
    capture0 = _make_capture(
        target="Q00",
        raw=np.array([1.0 + 0.0j]),
        measurement_config=config,
        sampling_period=0.4,
        classifier_ref=ref,
    )
    capture1 = _make_capture(
        target="Q01",
        raw=np.array([2.0 + 0.0j]),
        measurement_config=config,
        sampling_period=0.4,
        classifier_ref=ref,
    )
    calls = {"count": 0}

    def _load(path: str) -> object:
        calls["count"] += 1
        assert Path(path) == classifier_path.resolve()
        return dummy_classifier

    ClassifierRef.clear_cache()
    monkeypatch.setattr(
        "qubex.measurement.models.classifier_ref.StateClassifier.load",
        _load,
    )

    assert capture0.classifier_ref is not None
    assert capture1.classifier_ref is not None

    first = capture0.classifier_ref.load()
    second = capture1.classifier_ref.load()

    assert first is dummy_classifier
    assert second is dummy_classifier
    assert calls["count"] == 1


def test_classifier_ref_reloads_classifier_after_file_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dummy_classifier,
) -> None:
    """Given same path with updated file metadata, classifier should be reloaded."""
    classifier_path = tmp_path / "classifier-Q00.pkl"
    classifier_path.write_bytes(b"v1")
    ref = ClassifierRef(path=str(classifier_path), version=SKLEARN_VERSION)
    calls = {"count": 0}

    def _load(path: str) -> object:
        calls["count"] += 1
        assert Path(path) == classifier_path.resolve()
        return dummy_classifier

    ClassifierRef.clear_cache()
    monkeypatch.setattr(
        "qubex.measurement.models.classifier_ref.StateClassifier.load",
        _load,
    )

    first = ref.load()
    classifier_path.write_bytes(b"v2-updated")
    second = ref.load()

    assert first is dummy_classifier
    assert second is dummy_classifier
    assert calls["count"] == 2


def test_capture_data_classifier_ref_rejects_unsupported_version() -> None:
    """Given unsupported classifier ref version, loading through classifier_ref should fail."""
    config = _make_config(mode="avg", shots=1)
    capture = _make_capture(
        target="Q00",
        raw=np.array([1.0 + 0.0j]),
        measurement_config=config,
        sampling_period=0.4,
        classifier_ref=ClassifierRef(
            path="classifier-Q00.pkl",
            version=f"{SKLEARN_VERSION}-incompatible",
        ),
    )

    assert capture.classifier_ref is not None
    with pytest.raises(ValueError, match="scikit-learn version mismatch"):
        _ = capture.classifier_ref.load()


def test_capture_data_does_not_force_raw_read_only() -> None:
    """Given writeable raw input, capture data should keep caller mutability as-is."""
    config = _make_config(mode="avg", shots=1)
    raw = np.array([1.0 + 0.0j])
    capture = _make_capture(
        target="Q00",
        raw=raw,
        measurement_config=config,
        sampling_period=0.4,
    )

    capture.data[0] = 2.0 + 0.0j
    assert capture.data[0] == 2.0 + 0.0j


def test_measurement_result_ignores_legacy_top_level_sampling_period() -> None:
    """Given legacy top-level sampling_period, capture data should remain unchanged."""
    config = _make_config(mode="avg", shots=1)
    result = MeasurementResult.model_validate(
        {
            "data": {
                "Q00": [
                    _make_capture(
                        target="Q00",
                        raw=np.array([1.0 + 0.0j]),
                        measurement_config=config,
                        sampling_period=0.4,
                    )
                ]
            },
            "measurement_config": config,
            "sampling_period": 0.8,
        }
    )

    assert result.data["Q00"][0].sampling_period == pytest.approx(0.4)


def test_converter_falls_back_to_empty_config_when_missing() -> None:
    """Given canonical result without device config, converter returns legacy results with empty config."""
    config = _make_config(mode="avg", shots=2)
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([1.0 + 0.0j]),
                    measurement_config=config,
                    sampling_period=2.0,
                ),
                _make_capture(
                    target="Q00",
                    raw=np.array([2.0 + 0.0j]),
                    measurement_config=config,
                    sampling_period=2.0,
                ),
            ]
        },
        measurement_config=config,
    )

    multiple = MeasurementResultConverter.to_multiple_measure_result(result)
    single = MeasurementResultConverter.to_measure_result(result)

    assert multiple.config == {}
    assert single.config == {}


def test_plot_calls_waveform_plot_for_avg_mode(monkeypatch) -> None:
    """Given AVG capture data, plot should call waveform plotting per capture."""
    config = _make_config(mode="avg", shots=1)
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([1.0 + 0.0j, 2.0 + 0.0j]),
                    measurement_config=config,
                    sampling_period=0.8,
                )
            ]
        },
        measurement_config=config,
    )
    called: dict[str, object] = {}

    def _plot_waveform(
        *,
        data: np.ndarray,
        sampling_period: float,
        title: str,
        xlabel: str,
        ylabel: str,
        save_image: bool,
    ) -> None:
        called["data"] = data
        called["sampling_period"] = sampling_period
        called["title"] = title
        called["xlabel"] = xlabel
        called["ylabel"] = ylabel
        called["save_image"] = save_image

    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.plot_waveform",
        _plot_waveform,
    )
    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.scatter_iq_data",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    result.plot()

    waveform = called["data"]
    assert isinstance(waveform, np.ndarray)
    assert np.array_equal(waveform, np.array([1.0 + 0.0j, 2.0 + 0.0j]))
    assert called["sampling_period"] == 0.8
    assert called["title"] == "Readout waveform : Q00"
    assert called["xlabel"] == "Capture time (ns)"
    assert called["ylabel"] == "Signal (arb. units)"
    assert called["save_image"] is False


def test_measurement_result_figure_returns_figure_list(monkeypatch) -> None:
    """Given waveform capture data, figure should return one waveform figure object."""
    config = _make_config(mode="avg", shots=1)
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([1.0 + 0.0j, 2.0 + 0.0j]),
                    measurement_config=config,
                    sampling_period=0.8,
                )
            ]
        },
        measurement_config=config,
    )
    sentinel = object()

    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.make_waveform_figure",
        lambda **kwargs: sentinel,
    )

    assert result.figure() == [sentinel]


def test_measurement_result_plot_return_figure_warns_deprecated(
    monkeypatch,
) -> None:
    """Given return_figure usage, result plot warns and still returns figure list."""
    config = _make_config(mode="avg", shots=1)
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([1.0 + 0.0j, 2.0 + 0.0j]),
                    measurement_config=config,
                    sampling_period=0.8,
                )
            ]
        },
        measurement_config=config,
    )
    sentinel = object()

    def _make_waveform_figure(**kwargs: object) -> object:
        assert kwargs["title"] == "Readout waveform : Q00"
        return sentinel

    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.make_waveform_figure",
        _make_waveform_figure,
    )

    with pytest.warns(DeprecationWarning, match="figure\\("):
        figures = result.plot(return_figure=True)

    assert figures == [sentinel]


def test_measurement_result_figure_uses_time_average_scatter_by_default(
    monkeypatch,
) -> None:
    """Given non-averaged waveform shots, figure should use time-average scatter by default."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=False,
        state_classification=False,
    )
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array(
                        [
                            [1.0 + 0.0j, 3.0 + 0.0j, 5.0 + 0.0j],
                            [3.0 + 0.0j, 5.0 + 0.0j, 7.0 + 0.0j],
                        ]
                    ),
                    measurement_config=config,
                    sampling_period=2.0,
                )
            ]
        },
        measurement_config=config,
    )
    called: dict[str, object] = {}

    def _make_iq_scatter_figure(**kwargs: object) -> object:
        called.update(kwargs)
        return object()

    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.make_iq_scatter_figure",
        _make_iq_scatter_figure,
    )
    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.make_waveform_figure",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    result.figure()

    plotted = called["data"]
    assert isinstance(plotted, dict)
    assert np.array_equal(plotted["Q00"], np.array([3.0 + 0.0j, 5.0 + 0.0j]))


def test_plot_calls_iq_scatter_for_single_mode(monkeypatch) -> None:
    """Given canonical IQ series data, plot should call IQ scatter with one value per shot."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=False,
    )
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([4.0 + 6.0j, 12.0 + 14.0j]),
                    measurement_config=config,
                    sampling_period=2.0,
                )
            ]
        },
        measurement_config=config,
    )
    called: dict[str, object] = {}

    def _scatter_iq_data(
        *,
        data: dict[str, np.ndarray],
        title: str,
        save_image: bool,
    ) -> None:
        called["data"] = data
        called["title"] = title
        called["save_image"] = save_image

    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.scatter_iq_data",
        _scatter_iq_data,
    )
    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.plot_waveform",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    result.plot()

    assert called["title"] == "Q00 : data[0]"
    assert called["save_image"] is False
    plotted = called["data"]
    assert isinstance(plotted, dict)
    assert np.array_equal(plotted["Q00"], np.array([4.0 + 6.0j, 12.0 + 14.0j]))


def test_plot_calls_scatter_for_single_mode_loopback_shape(monkeypatch) -> None:
    """Given SINGLE one-shot waveform-like data, plot should call IQ scatter by default."""
    config = _make_config(mode="single", shots=1)
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]]),
                    measurement_config=config,
                    sampling_period=2.0,
                )
            ]
        },
        measurement_config=config,
    )
    called: dict[str, object] = {}

    def _scatter_iq_data(
        *,
        data: dict[str, np.ndarray],
        title: str,
        save_image: bool,
    ) -> None:
        called["data"] = data
        called["title"] = title
        called["save_image"] = save_image

    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.scatter_iq_data",
        _scatter_iq_data,
    )
    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.plot_waveform",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    result.plot()

    plotted = called["data"]
    assert isinstance(plotted, dict)
    assert np.array_equal(plotted["Q00"], np.array([2.0 + 0.0j]))
    assert called["title"] == "Readout IQ data : Q00"
    assert called["save_image"] is False


def test_plot_uses_time_average_scatter_when_not_integrated_and_not_averaged(
    monkeypatch,
) -> None:
    """Given non-averaged non-integrated data, plot should use time-average scatter by default."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=False,
        state_classification=False,
    )
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array(
                        [
                            [1.0 + 0.0j, 3.0 + 0.0j, 5.0 + 0.0j],
                            [3.0 + 0.0j, 5.0 + 0.0j, 7.0 + 0.0j],
                        ]
                    ),
                    measurement_config=config,
                    sampling_period=2.0,
                )
            ]
        },
        measurement_config=config,
    )
    called: dict[str, object] = {}

    def _scatter_iq_data(
        *,
        data: dict[str, np.ndarray],
        title: str,
        save_image: bool,
    ) -> None:
        called["data"] = data
        called["title"] = title
        called["save_image"] = save_image

    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.scatter_iq_data",
        _scatter_iq_data,
    )
    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.plot_waveform",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    result.plot()

    plotted = called["data"]
    assert isinstance(plotted, dict)
    assert np.array_equal(plotted["Q00"], np.array([3.0 + 0.0j, 5.0 + 0.0j]))
    assert called["title"] == "Readout IQ data : Q00"
    assert called["save_image"] is False


def test_plot_calls_iq_scatter_for_averaged_integrated_mode(monkeypatch) -> None:
    """Given averaged integrated data, plot should call IQ scatter."""
    config = MeasurementConfig(
        n_shots=2,
        shot_interval=100.0,
        shot_averaging=True,
        time_integration=True,
        state_classification=False,
    )
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([1.0 + 2.0j]),
                    measurement_config=config,
                    sampling_period=2.0,
                )
            ]
        },
        measurement_config=config,
    )
    called: dict[str, object] = {}

    def _scatter_iq_data(
        *,
        data: dict[str, np.ndarray],
        title: str,
        save_image: bool,
    ) -> None:
        called["data"] = data
        called["title"] = title
        called["save_image"] = save_image

    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.scatter_iq_data",
        _scatter_iq_data,
    )
    monkeypatch.setattr(
        "qubex.measurement.models.measurement_result.viz.plot_waveform",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    result.plot()

    assert called["title"] == "Q00 : data[0]"
    assert called["save_image"] is False
    plotted = called["data"]
    assert isinstance(plotted, dict)
    assert np.array_equal(plotted["Q00"], np.array([1.0 + 2.0j]))


def test_netcdf_writes_codec_metadata_attributes(tmp_path) -> None:
    """Given NetCDF save, when opening the file, then codec metadata attributes are present."""
    config = _make_config(mode="single", shots=1)
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([[1.0 + 2.0j]]),
                    measurement_config=config,
                    sampling_period=2.0,
                )
            ]
        },
        device_config={"backend": "quel"},
        measurement_config=config,
    )
    path = result.save_netcdf(tmp_path / "metadata.nc")

    with Dataset(path, mode="r") as ds:
        format_name = ds.getncattr("format")
        format_version = ds.getncattr("format_version")
        model_class = ds.getncattr("model_class")
        payload_json = ds.getncattr("payload_json")

        assert format_name == "qxdata"
        assert int(format_version) == 1
        assert model_class.endswith(".MeasurementResult")

        payload = json.loads(payload_json)
        assert "Q00" in payload["data"]


def test_state_series_conversion_ignores_legacy_classifier_map() -> None:
    """Given canonical state-series data, legacy conversion should not re-attach software classifiers."""
    config = MeasurementConfig(
        n_shots=4,
        shot_interval=100.0,
        shot_averaging=False,
        time_integration=True,
        state_classification=True,
        classification_source="gmm_linear",
    )
    result = MeasurementResult(
        data={
            "Q00": [
                _make_capture(
                    target="Q00",
                    raw=np.array([0, 3, 3, 0], dtype=np.uint8),
                    measurement_config=config,
                    sampling_period=2.0,
                )
            ]
        },
        measurement_config=config,
    )

    class _ExplodingClassifier:
        n_states = 2

        def predict(self, data: np.ndarray) -> np.ndarray:
            raise AssertionError(data)

    classifiers = {"Q00": cast(StateClassifier, _ExplodingClassifier())}
    restored = MeasurementResultConverter.to_multiple_measure_result(
        result,
        classifiers=classifiers,
    )
    single = MeasurementResultConverter.to_measure_result(
        result,
        classifiers=classifiers,
    )

    assert restored.data["Q00"][0].classifier is None
    assert single.data["Q00"].classifier is None
    assert restored.data["Q00"][0].classified.tolist() == [0, 1, 1, 0]
    assert single.data["Q00"].classified.tolist() == [0, 1, 1, 0]
