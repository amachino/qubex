"""Tests for measurement result helpers."""

from __future__ import annotations

import numpy as np
import pytest

from qubex.measurement import MeasureData, MeasureMode, MeasureResult

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Use `CaptureData` in `measurement_result.py` instead\\.:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Use `MeasurementResult` in `measurement_result.py` instead\\.:DeprecationWarning"
    ),
]


def test_measure_data_counts_and_probabilities(dummy_classifier):
    """MeasureData should compute counts and probabilities."""
    raw = dummy_classifier.dataset[0][:5]
    data = MeasureData(
        target="Q00",
        mode=MeasureMode.SINGLE,
        raw=raw[:, None],
        classifier=dummy_classifier,
    )
    counts = data.counts
    probs = data.probabilities
    assert set(counts.keys()) <= {"0", "1"}
    assert np.isclose(sum(probs), 1.0)
    assert len(data.standard_deviations) == len(probs)


def test_measure_data_threshold_classification(dummy_classifier):
    """MeasureData should drop shots below threshold."""
    raw = np.array([0.0 + 0.0j, 2.0 + 0.0j])[:, None]
    data = MeasureData(
        target="Q00",
        mode=MeasureMode.SINGLE,
        raw=raw,
        classifier=dummy_classifier,
    )
    labels = data.get_classified_data(threshold=0.8)
    assert labels.tolist() == [-1, 1]


def test_measure_result_memory_and_counts(measure_result: MeasureResult):
    """MeasureResult should expose memory and counts."""
    memory = measure_result.get_memory()
    assert len(memory) == measure_result.data["Q00"].length
    assert all(bit in {"0", "1"} for bit in memory)
    counts = measure_result.get_counts()
    assert sum(counts.values()) == measure_result.data["Q00"].length


def test_measure_result_mitigated_counts(measure_result: MeasureResult):
    """MeasureResult should compute mitigated counts per basis label."""
    mitigated = measure_result.get_mitigated_counts()
    assert set(mitigated.keys()) == {"0", "1"}


def test_measure_data_times_use_runtime_sampling_period_in_single_mode() -> None:
    """MeasureData should use runtime sampling period directly in single mode."""
    data = MeasureData(
        target="Q00",
        mode=MeasureMode.SINGLE,
        raw=np.array([[1.0 + 0.0j], [2.0 + 0.0j], [3.0 + 0.0j]]),
        sampling_period=0.4,
    )

    assert np.array_equal(data.times, np.array([0.0, 0.4, 0.8]))


def test_measure_data_times_use_runtime_sampling_period_in_avg_mode() -> None:
    """MeasureData should use runtime sampling period directly in avg mode."""
    data = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
        sampling_period=0.8,
    )

    assert np.array_equal(data.times, np.array([0.0, 0.8, 1.6]))


def test_measure_data_plot_uses_scatter_for_scalar_avg_data(monkeypatch) -> None:
    """MeasureData.plot should use IQ scatter when AVG data is a single I/Q point."""
    data = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.array(1.0 + 0.0j),
        sampling_period=0.8,
    )
    called: dict[str, object] = {}

    def _scatter_iq_data(**kwargs: object) -> None:
        called.update(kwargs)

    monkeypatch.setattr(
        "qubex.measurement.models.measure_result.viz.scatter_iq_data",
        _scatter_iq_data,
    )
    monkeypatch.setattr(
        "qubex.measurement.models.measure_result.viz.plot_waveform",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    data.plot()

    plotted = called["data"]
    assert isinstance(plotted, dict)
    assert np.array_equal(plotted["Q00"], np.array([1.0 + 0.0j]))
    assert called["title"] == "Readout IQ data : Q00"
    assert called["save_image"] is False


def test_measure_data_plot_fft_skips_for_scalar_avg_data(
    monkeypatch,
    caplog,
) -> None:
    """MeasureData.plot_fft should log and skip when AVG data is a single I/Q point."""
    data = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.array(1.0 + 0.0j),
        sampling_period=0.8,
    )

    monkeypatch.setattr(
        "qubex.measurement.models.measure_result.viz.plot_fft",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    with caplog.at_level("INFO"):
        result = data.plot_fft()

    assert result is None
    assert "not waveform data" in caplog.text


def test_measure_data_repr_summarizes_raw_array() -> None:
    """MeasureData repr should summarize long raw arrays."""
    data = MeasureData(
        target="Q00",
        mode=MeasureMode.AVG,
        raw=np.arange(1024, dtype=np.float64),
        sampling_period=0.8,
    )

    text = repr(data)

    assert "raw=array([0., ...], shape=(1024,))" in text
