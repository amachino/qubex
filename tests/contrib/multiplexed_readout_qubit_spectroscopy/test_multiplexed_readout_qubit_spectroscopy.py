"""Tests for multiplexed-readout qubit spectroscopy contrib helper."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex import contrib
from qubex.contrib.experiment.multiplexed_readout_qubit_spectroscopy import (
    multiplexed_readout_qubit_spectroscopy,
)
from qubex.experiment.models.result import Result

QUBITS = ["Q00", "Q01", "Q02", "Q03", "Q04"]
READOUT_FREQUENCIES = {
    "RQ00": 7.30,
    "RQ01": 7.10,
    "RQ02": 7.40,
    "RQ03": 7.20,
}


class _Capture:
    def __init__(self, value: complex) -> None:
        self.kerneled = value


class _MeasurementResult:
    def __init__(self, data: dict[str, complex]) -> None:
        self.data = {key: [_Capture(value)] for key, value in data.items()}


class _SystemManager:
    """Plan sweeps in fixed-size, non-contiguous segments and track contexts."""

    def __init__(self, segment_indices: list[tuple[int, ...]] | None) -> None:
        self.segment_indices = segment_indices
        self.plan_calls: list[dict[str, Any]] = []
        self.entered: list[str] = []
        self.active = 0

    def plan_frequency_sweep(
        self,
        target: str,
        frequencies: Any,
        *,
        subrange_width: float | None = None,
    ) -> tuple[SimpleNamespace, ...]:
        values = np.asarray(frequencies, dtype=float)
        self.plan_calls.append(
            {"target": target, "frequencies": values, "subrange_width": subrange_width}
        )
        groups = self.segment_indices or [tuple(range(len(values)))]
        return tuple(
            SimpleNamespace(
                indices=indices,
                values=tuple(float(values[i]) for i in indices),
                configuration=f"segment-{n}",
            )
            for n, indices in enumerate(groups)
        )

    @contextmanager
    def modified_sweep_configuration(self, configuration: str):
        self.entered.append(configuration)
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1


class _Context:
    def __init__(self, segment_indices: list[tuple[int, ...]] | None) -> None:
        self.qubit_labels = list(QUBITS)
        self.params = SimpleNamespace(
            readout_amplitude={q: 0.01 * (i + 1) for i, q in enumerate(QUBITS)},
        )
        self.reset_calls: list[set[str]] = []
        self.system_manager = _SystemManager(segment_indices)
        self.experiment_system = SimpleNamespace(
            get_control_box_for_qubit=lambda _qubit: SimpleNamespace(
                traits=SimpleNamespace(
                    default_control_frequency_range=(5.00, 5.025, 0.01),
                )
            )
        )

    def resolve_qubit_label(self, label: str) -> str:
        if label in self.qubit_labels:
            return label
        if label.startswith("R") and label[1:] in self.qubit_labels:
            return label[1:]
        raise ValueError(f"Qubit label could not be resolved from `{label}`.")

    def resolve_read_label(self, label: str) -> str:
        return f"R{self.resolve_qubit_label(label)}"

    def reset_awg_and_capunits(self, *, qubits: set[str]) -> None:
        self.reset_calls.append(set(qubits))


class _MeasurementService:
    """Encode control point and readout identity in each slot's signal."""

    def __init__(self, ctx: _Context, fail_on_call: int | None) -> None:
        self.ctx = ctx
        self.fail_on_call = fail_on_call
        self.calls: list[dict[str, Any]] = []

    def execute(self, **kwargs: Any) -> _MeasurementResult:
        self.calls.append(kwargs)
        assert self.ctx.system_manager.active == 1
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("hardware failure")
        schedule = kwargs["schedule"]
        frequencies = kwargs["frequencies"]
        drives = _drive_labels(schedule)
        assert len(drives) == 1
        drive = drives[0]
        drive_amplitude = float(np.max(np.abs(schedule.get_sequence(drive).values)))
        data: dict[str, complex] = {}
        for label in schedule.labels:
            if label.startswith("R"):
                # Real part: control point; imaginary part: this slot's frequency.
                data[label[1:]] = complex(
                    frequencies[drive] + 100 * drive_amplitude,
                    frequencies[label],
                )
        return _MeasurementResult(data)


class _Experiment:
    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        segment_indices: list[tuple[int, ...]] | None = None,
    ) -> None:
        self.ctx = _Context(segment_indices)
        self.measurement_service = _MeasurementService(self.ctx, fail_on_call)


def _drive_labels(schedule: Any) -> list[str]:
    return [
        label
        for label in schedule.labels
        if not label.startswith("R")
        and np.any(np.abs(schedule.get_sequence(label).values) > 0)
    ]


def _readout_labels(schedule: Any) -> list[str]:
    return [label for label in schedule.labels if label.startswith("R")]


def _first_active_index(schedule: Any, label: str) -> int:
    values = np.abs(schedule.get_sequence(label).values)
    return int(np.flatnonzero(values > 1e-6)[0])


def _run(exp: _Experiment, **kwargs: Any) -> Result:
    options: dict[str, Any] = {
        "target": "Q04",
        "readout_frequencies": dict(READOUT_FREQUENCIES),
        "frequency_range": [5.00, 5.01, 5.02],
        "power_range": [-20.0, -10.0],
        "plot": False,
        "save_image": False,
    }
    options.update(kwargs)
    target = options.pop("target")
    return multiplexed_readout_qubit_spectroscopy(cast(Any, exp), target, **options)


def test_drives_only_the_requested_target_with_all_readouts() -> None:
    """Every schedule drives exactly one control target and reads every candidate slot."""
    exp = _Experiment()

    _run(exp)

    calls = exp.measurement_service.calls
    assert len(calls) == 6
    for call in calls:
        schedule = call["schedule"]
        assert _drive_labels(schedule) == ["Q04"]
        assert _readout_labels(schedule) == ["RQ00", "RQ01", "RQ02", "RQ03"]
        assert call["reset_awg_and_capunits"] is False


def test_drive_target_is_independent_of_readout_slots() -> None:
    """The driven qubit's own resonator is neither assumed nor added to the readouts."""
    exp = _Experiment()

    result = _run(exp)

    schedule = exp.measurement_service.calls[0]["schedule"]
    assert "RQ04" not in schedule.labels
    assert result.data["target"] == "Q04"
    assert list(result.data["candidates"]) == ["RQ00", "RQ01", "RQ02", "RQ03"]


def test_applies_candidate_frequencies_per_point() -> None:
    """Each point overrides the drive frequency and every candidate readout frequency."""
    exp = _Experiment()

    _run(exp)

    frequencies = [call["frequencies"] for call in exp.measurement_service.calls]
    assert frequencies[0] == {"Q04": 5.00, **READOUT_FREQUENCIES}
    assert [f["Q04"] for f in frequencies] == [5.00, 5.01, 5.02] * 2


def test_permuted_candidates_keep_independent_iq_identity() -> None:
    """Each candidate keeps its own complex IQ even when frequencies are permuted."""
    exp = _Experiment()

    result = _run(exp)

    for label, frequency in READOUT_FREQUENCIES.items():
        signals = result.data["candidates"][label]["signals"]
        assert signals.dtype == np.complex128
        assert_allclose(signals.imag, np.full((2, 3), frequency), atol=1e-12)


def test_payload_shapes_axes_and_metadata() -> None:
    """Payload exposes shared axes, candidate metadata, and spectroscopy-compatible phases."""
    exp = _Experiment()

    result = _run(exp, readout_amplitudes={"RQ02": 0.25}, simultaneous_drive=False)

    data = result.data
    assert data["target"] == "Q04"
    assert data["readout_labels"] == ["RQ00", "RQ01", "RQ02", "RQ03"]
    assert data["simultaneous_drive"] is False
    assert_allclose(data["frequency_range"], [5.00, 5.01, 5.02])
    assert_allclose(data["power_range"], [-20.0, -10.0])
    entry = data["candidates"]["RQ02"]
    assert entry["readout_label"] == "RQ02"
    assert entry["candidate_index"] == 2
    assert entry["readout_frequency"] == pytest.approx(7.40)
    assert entry["readout_amplitude"] == pytest.approx(0.25)
    assert data["candidates"]["RQ00"]["readout_amplitude"] == pytest.approx(0.01)
    assert entry["signals"].shape == (2, 3)
    assert entry["phases"].shape == (2, 3)
    assert entry["data"] is entry["phases"]
    assert np.all(np.abs(entry["phases"]) <= np.pi)


def test_signals_follow_frequency_and_power_axes() -> None:
    """Signals are placed by power row and sweep index even for non-contiguous segments."""
    exp = _Experiment(segment_indices=[(0, 2), (1,)])

    result = _run(exp)

    signals = result.data["candidates"]["RQ01"]["signals"]
    amplitudes = np.sqrt(10 ** (np.array([-20.0, -10.0]) / 10))
    expected = np.array([5.00, 5.01, 5.02])[None, :] + 100 * amplitudes[:, None]
    assert_allclose(signals.real, expected, rtol=1e-4)


def test_retunes_only_the_drive_target_once_per_segment() -> None:
    """Sweep planning targets the drive line and each segment is applied once."""
    exp = _Experiment(segment_indices=[(0, 1), (2,)])

    _run(exp, subrange_width=0.2)

    manager = exp.ctx.system_manager
    assert [c["target"] for c in manager.plan_calls] == ["Q04"]
    assert manager.plan_calls[0]["subrange_width"] == pytest.approx(0.2)
    assert manager.entered == ["segment-0", "segment-1"]


def test_resets_drive_and_all_readout_resources_per_segment() -> None:
    """AWG/capture reset covers the drive qubit and every candidate slot."""
    exp = _Experiment(segment_indices=[(0, 1), (2,)])

    _run(exp)

    assert exp.ctx.reset_calls == [{"Q00", "Q01", "Q02", "Q03", "Q04"}] * 2


def test_readout_amplitudes_are_applied_per_candidate() -> None:
    """Explicit and default readout amplitudes drive each candidate slot's pulse."""
    exp = _Experiment()

    _run(exp, readout_amplitudes={"RQ01": 0.3, "Q03": 0.4})

    schedule = exp.measurement_service.calls[0]["schedule"]
    peaks = {
        label: float(np.max(np.abs(schedule.get_sequence(label).values)))
        for label in _readout_labels(schedule)
    }
    assert peaks == pytest.approx(
        {"RQ00": 0.01, "RQ01": 0.3, "RQ02": 0.03, "RQ03": 0.4}
    )


@pytest.mark.parametrize(
    ("simultaneous_drive", "starts_after_drive"),
    [(None, False), (True, False), (False, True)],
)
def test_scheduling_timing_modes(
    simultaneous_drive: bool | None, starts_after_drive: bool
) -> None:
    """Readout starts with the drive by default and after it when not simultaneous."""
    exp = _Experiment()

    _run(exp, simultaneous_drive=simultaneous_drive)

    schedule = exp.measurement_service.calls[0]["schedule"]
    drive_samples = len(schedule.get_sequence("Q04").values)
    for label in _readout_labels(schedule):
        start = _first_active_index(schedule, label)
        if starts_after_drive:
            assert start >= drive_samples // 2
        else:
            assert start < 16


def test_figures_are_keyed_by_candidate() -> None:
    """One phase heatmap per candidate is returned with the first as primary figure."""
    exp = _Experiment()

    result = _run(exp)

    assert result.figures is not None
    assert list(result.figures) == ["RQ00", "RQ01", "RQ02", "RQ03"]
    assert result.figure is result.figures["RQ00"]
    heatmap: Any = result.figures["RQ03"].data[0]
    assert heatmap.type == "heatmap"
    assert_allclose(heatmap.z, result.data["candidates"]["RQ03"]["phases"])
    assert "Q04" in result.figures["RQ03"].layout.title.text
    assert "RQ03" in result.figures["RQ03"].layout.title.text


def test_supports_qubit_label_keys_and_other_candidate_counts() -> None:
    """Candidates may be keyed by qubit or readout labels and are not fixed to four."""
    exp = _Experiment()

    result = _run(
        exp,
        target="Q00",
        readout_frequencies={"Q02": 7.2, "RQ00": 7.0, "Q04": 7.4},
    )

    assert result.data["target"] == "Q00"
    assert result.data["readout_labels"] == ["RQ02", "RQ00", "RQ04"]
    schedule = exp.measurement_service.calls[0]["schedule"]
    assert _readout_labels(schedule) == ["RQ02", "RQ00", "RQ04"]


def test_uses_default_drive_frequency_range() -> None:
    """Omitted frequency range uses the drive target's control-box default range."""
    exp = _Experiment()

    result = _run(exp, frequency_range=None)

    assert_allclose(result.data["frequency_range"], [5.00, 5.01, 5.02])


def test_restores_sweep_configuration_after_failure() -> None:
    """Sweep configuration contexts are exited when a measurement fails."""
    exp = _Experiment(fail_on_call=2, segment_indices=[(0, 1), (2,)])

    with pytest.raises(RuntimeError, match="hardware failure"):
        _run(exp)

    assert exp.ctx.system_manager.active == 0


def test_does_not_mutate_persistent_parameters() -> None:
    """Running the helper leaves configured readout amplitudes untouched."""
    exp = _Experiment()
    before = dict(exp.ctx.params.readout_amplitude)

    _run(exp, readout_amplitudes={"RQ00": 0.9})

    assert exp.ctx.params.readout_amplitude == before


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"target": ["Q00", "Q01"]}, TypeError, "target"),
        ({"target": ""}, ValueError, "target"),
        ({"readout_frequencies": [7.0]}, TypeError, "readout_frequencies"),
        ({"readout_frequencies": {}}, ValueError, "readout_frequencies"),
        (
            {"readout_frequencies": {"Q00": 7.0, "RQ00": 7.1}},
            ValueError,
            "duplicate readout",
        ),
        ({"readout_frequencies": {"RQ00": float("nan")}}, ValueError, "finite"),
        ({"readout_frequencies": {"RQ00": -1.0}}, ValueError, "positive"),
        ({"readout_amplitudes": 0.5}, TypeError, "readout_amplitudes"),
        ({"readout_amplitudes": {"RQ04": 0.5}}, ValueError, "not a readout candidate"),
        ({"readout_amplitudes": {"RQ00": -0.1}}, ValueError, "readout_amplitudes"),
        ({"readout_amplitudes": {"RQ00": float("inf")}}, ValueError, "finite"),
        ({"frequency_range": {"Q04": [5.0]}}, TypeError, "frequency_range"),
        ({"frequency_range": []}, ValueError, "frequency_range"),
        ({"frequency_range": [5.0, float("nan")]}, ValueError, "frequency_range"),
        ({"power_range": [[-20.0]]}, ValueError, "power_range"),
        ({"power_range": []}, ValueError, "power_range"),
        ({"shots": 0}, ValueError, "shots"),
    ],
)
def test_rejects_invalid_inputs(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    """Invalid targets, readout candidates, or sweep axes are rejected before measuring."""
    exp = _Experiment()

    with pytest.raises(error, match=match):
        _run(exp, **kwargs)

    assert exp.measurement_service.calls == []
    assert exp.ctx.system_manager.entered == []


def test_rejects_candidate_without_default_amplitude() -> None:
    """A candidate without explicit or configured readout amplitude is rejected."""
    exp = _Experiment()
    del exp.ctx.params.readout_amplitude["Q01"]

    with pytest.raises(ValueError, match="RQ01"):
        _run(exp)

    assert exp.measurement_service.calls == []


def test_is_exported_from_contrib() -> None:
    """The helper is exported through the contrib namespace."""
    assert contrib.multiplexed_readout_qubit_spectroscopy is (
        multiplexed_readout_qubit_spectroscopy
    )
