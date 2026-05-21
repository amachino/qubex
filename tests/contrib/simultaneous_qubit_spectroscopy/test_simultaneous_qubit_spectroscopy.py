"""Tests for simultaneous qubit spectroscopy contrib helper."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qubex.contrib.experiment.simultaneous_qubit_spectroscopy import (
    simultaneous_qubit_spectroscopy,
)


class _Capture:
    def __init__(self, value: complex) -> None:
        self.kerneled = value


class _MeasurementResult:
    def __init__(self, data: dict[str, complex]) -> None:
        self.data = {target: [_Capture(value)] for target, value in data.items()}


class _MeasurementService:
    def __init__(self, qubits: list[str]) -> None:
        self.qubits = qubits
        self.calls: list[dict[str, Any]] = []

    def execute(self, **kwargs: Any) -> _MeasurementResult:
        self.calls.append(kwargs)
        point_index = len(self.calls)
        return _MeasurementResult(
            {
                qubit: complex(point_index, target_index + 1)
                for target_index, qubit in enumerate(self.qubits)
            }
        )


class _Context:
    def __init__(self, read_labels: dict[str, str] | None = None) -> None:
        self.qubit_labels = ["Q00", "Q01"]
        self.read_labels = read_labels or {"Q00": "RQ00", "Q01": "RQ01"}
        self.params = SimpleNamespace(
            control_amplitude={"Q00": 0.10, "Q01": 0.20},
            readout_amplitude={"Q00": 0.30, "Q01": 0.40},
        )
        self.targets = {
            "RQ00": SimpleNamespace(frequency=7.00),
            "RQ01": SimpleNamespace(frequency=7.10),
        }
        self.frequency_contexts: list[dict[str, float]] = []
        self.reset_calls: list[set[str]] = []
        self.backend_settings: list[dict[str, float | int | str | None]] = []
        self.system_manager = SimpleNamespace(
            modified_backend_settings=self._modified_backend_settings
        )
        self.experiment_system = SimpleNamespace(
            get_control_box_for_qubit=lambda _qubit: SimpleNamespace(
                traits=SimpleNamespace(
                    default_control_frequency_range=(5.00, 5.02, 0.01),
                    ctrl_ssb="L",
                )
            )
        )

    def resolve_qubit_label(self, target: str) -> str:
        return target

    def resolve_read_label(self, target: str) -> str:
        return self.read_labels[target]

    def reset_awg_and_capunits(self, *, qubits: set[str]) -> None:
        self.reset_calls.append(qubits)

    @contextmanager
    def modified_frequencies(self, frequencies: dict[str, float]):
        self.frequency_contexts.append(frequencies)
        yield

    @contextmanager
    def _modified_backend_settings(self, **settings: float | int | str | None):
        self.backend_settings.append(dict(settings))
        yield


class _Experiment:
    def __init__(self, read_labels: dict[str, str] | None = None) -> None:
        self.ctx = _Context(read_labels=read_labels)
        self.measurement_service = _MeasurementService(self.ctx.qubit_labels)


def test_simultaneous_qubit_spectroscopy_runs_one_combined_schedule_per_point() -> None:
    """Given shared sweep points, when helper runs, then each point measures all targets."""
    exp = _Experiment()

    result = simultaneous_qubit_spectroscopy(
        cast(Any, exp),
        targets=["Q00", "Q01"],
        power_range=[-20, -10],
        frequency_range=[5.00, 5.01],
        readout_frequencies={
            "Q00": 7.00,
            "Q01": 7.10,
        },
        shots=128,
        interval=2048,
        plot=False,
        save_image=False,
    )

    assert len(exp.measurement_service.calls) == 4
    assert exp.ctx.reset_calls == [{"Q00", "Q01"}] * 2
    assert exp.ctx.frequency_contexts == [
        {"Q00": 5.00, "Q01": 5.00, "RQ00": 7.00, "RQ01": 7.10},
        {"Q00": 5.01, "Q01": 5.01, "RQ00": 7.00, "RQ01": 7.10},
        {"Q00": 5.00, "Q01": 5.00, "RQ00": 7.00, "RQ01": 7.10},
        {"Q00": 5.01, "Q01": 5.01, "RQ00": 7.00, "RQ01": 7.10},
    ]
    first_schedule = exp.measurement_service.calls[0]["schedule"]
    assert first_schedule.labels == ["Q00", "Q01", "RQ00", "RQ01"]
    assert exp.measurement_service.calls[0]["shots"] == 128
    assert exp.measurement_service.calls[0]["interval"] == 2048
    assert exp.measurement_service.calls[0]["reset_awg_and_capunits"] is False
    q00 = result.data["Q00"]
    q01 = result.data["Q01"]
    assert isinstance(q00, dict)
    assert isinstance(q01, dict)
    assert result.figures is not None
    assert result.figure is result.figures["Q00"]
    assert set(result.figures) == {"Q00", "Q01"}
    assert_allclose(q00["frequency_range"], [5.00, 5.01])
    assert_allclose(q01["frequency_range"], [5.00, 5.01])
    assert_allclose(q00["power_range"], [-20, -10])
    assert_allclose(q01["power_range"], [-20, -10])
    assert_allclose(q00["signals"], [[1 + 1j, 2 + 1j], [3 + 1j, 4 + 1j]])
    assert_allclose(q01["signals"], [[1 + 2j, 2 + 2j], [3 + 2j, 4 + 2j]])
    assert q00["data"].shape == (2, 2)
    assert q01["data"].shape == (2, 2)
    assert exp.ctx.backend_settings[0]["label"] == "Q00"
    assert exp.ctx.backend_settings[1]["label"] == "Q01"
    assert exp.ctx.backend_settings[0]["fnco_freq"] == 0
    assert exp.ctx.backend_settings[1]["fnco_freq"] == 0
    assert len(exp.ctx.backend_settings) == 4


def test_simultaneous_qubit_spectroscopy_normalizes_phase_per_power_row() -> None:
    """Given complex signals, spectroscopy data stores phases normalized per power row."""
    exp = _Experiment()

    result = simultaneous_qubit_spectroscopy(
        cast(Any, exp),
        targets=["Q00"],
        frequency_range=[5.00, 5.01],
        power_range=[-20, -10],
        plot=False,
        save_image=False,
    )

    expected_signals = np.asarray([[1 + 1j, 2 + 1j], [3 + 1j, 4 + 1j]])
    expected_phases = np.angle(expected_signals)
    expected_phases -= np.median(expected_phases, axis=1, keepdims=True) - np.pi
    expected_phases %= 2 * np.pi
    expected_phases -= np.pi
    assert result.figures is not None
    assert result.figure is not None
    assert result.figure is result.figures["Q00"]
    assert_allclose(result.data["Q00"]["data"], expected_phases)
    assert_allclose(result.data["Q00"]["signals"], expected_signals)


def test_simultaneous_qubit_spectroscopy_rejects_per_target_frequency_range() -> None:
    """Given per-target frequency range, when helper runs, then it rejects it."""
    exp = _Experiment()

    with pytest.raises(TypeError, match="frequency_range must be shared"):
        simultaneous_qubit_spectroscopy(
            cast(Any, exp),
            targets=["Q00", "Q01"],
            frequency_range=cast(
                Any,
                {
                    "Q00": [5.00, 5.01],
                    "Q01": [5.20],
                },
            ),
            plot=False,
            save_image=False,
        )


def test_simultaneous_qubit_spectroscopy_uses_default_frequency_range() -> None:
    """Given no frequency range, when helper runs, then it uses the first target default."""
    exp = _Experiment()

    result = simultaneous_qubit_spectroscopy(
        cast(Any, exp),
        targets=["Q00", "Q01"],
        power_range=[-20],
        plot=False,
        save_image=False,
    )

    assert len(exp.measurement_service.calls) == 2
    assert_allclose(result.data["Q00"]["frequency_range"], [5.00, 5.01])
    assert_allclose(result.data["Q01"]["frequency_range"], [5.00, 5.01])
    assert exp.ctx.frequency_contexts == [
        {"Q00": 5.00, "Q01": 5.00, "RQ00": 7.00, "RQ01": 7.10},
        {"Q00": 5.01, "Q01": 5.01, "RQ00": 7.00, "RQ01": 7.10},
    ]


def test_simultaneous_qubit_spectroscopy_accepts_shared_frequency_range() -> None:
    """Given one shared frequency range, when helper runs, then it applies to every target."""
    exp = _Experiment()

    result = simultaneous_qubit_spectroscopy(
        cast(Any, exp),
        targets=["Q00", "Q01"],
        frequency_range=[5.10, 5.11],
        power_range=[-20],
        readout_amplitudes={"Q00": 0.5, "Q01": 0.6},
        readout_frequencies={"Q00": 7.2, "Q01": 7.3},
        plot=False,
        save_image=False,
    )

    assert len(exp.measurement_service.calls) == 2
    assert exp.ctx.frequency_contexts == [
        {"Q00": 5.10, "Q01": 5.10, "RQ00": 7.2, "RQ01": 7.3},
        {"Q00": 5.11, "Q01": 5.11, "RQ00": 7.2, "RQ01": 7.3},
    ]
    assert_allclose(result.data["Q00"]["frequency_range"], [5.10, 5.11])
    assert_allclose(result.data["Q01"]["frequency_range"], [5.10, 5.11])


def test_simultaneous_qubit_spectroscopy_rejects_scalar_readout_parameters() -> None:
    """Given scalar readout overrides, when helper runs, then it rejects them."""
    exp = _Experiment()

    with pytest.raises(TypeError, match="readout_amplitudes must be a mapping"):
        simultaneous_qubit_spectroscopy(
            cast(Any, exp),
            targets=["Q00", "Q01"],
            frequency_range=[5.00],
            readout_amplitudes=0.5,  # type: ignore[arg-type]
            plot=False,
            save_image=False,
        )

    with pytest.raises(TypeError, match="readout_frequencies must be a mapping"):
        simultaneous_qubit_spectroscopy(
            cast(Any, exp),
            targets=["Q00", "Q01"],
            frequency_range=[5.00],
            readout_frequencies=7.2,  # type: ignore[arg-type]
            plot=False,
            save_image=False,
        )


def test_simultaneous_qubit_spectroscopy_rejects_duplicate_targets() -> None:
    """Given duplicate targets, when validation is enabled, then it rejects the request."""
    exp = _Experiment()

    with pytest.raises(ValueError, match="duplicate target"):
        simultaneous_qubit_spectroscopy(
            cast(Any, exp),
            targets=["Q00", "Q00"],
            frequency_range=[5.00, 5.01],
            plot=False,
            save_image=False,
        )


def test_simultaneous_qubit_spectroscopy_rejects_duplicate_readout_labels() -> None:
    """Given shared readout labels, when validation is enabled, then it rejects the request."""
    exp = _Experiment(read_labels={"Q00": "RQ00", "Q01": "RQ00"})

    with pytest.raises(ValueError, match="readout label"):
        simultaneous_qubit_spectroscopy(
            cast(Any, exp),
            targets=["Q00", "Q01"],
            frequency_range=[5.00, 5.01],
            plot=False,
            save_image=False,
        )


def test_simultaneous_qubit_spectroscopy_can_skip_resource_validation() -> None:
    """Given intentional shared resources, when validation is disabled, then it runs."""
    exp = _Experiment(read_labels={"Q00": "RQ00", "Q01": "RQ00"})

    result = simultaneous_qubit_spectroscopy(
        cast(Any, exp),
        targets=["Q00", "Q01"],
        frequency_range=[5.00],
        power_range=[-20],
        validate_resources=False,
        plot=False,
        save_image=False,
    )

    assert len(exp.measurement_service.calls) == 1
    assert list(result.data) == ["Q00", "Q01"]


def test_simultaneous_qubit_spectroscopy_rejects_control_readout_label_collision() -> (
    None
):
    """Given a readout label matching a target label, validation rejects the ambiguous schedule."""
    exp = _Experiment(read_labels={"Q00": "Q01", "Q01": "RQ01"})

    with pytest.raises(ValueError, match="control and readout labels"):
        simultaneous_qubit_spectroscopy(
            cast(Any, exp),
            targets=["Q00", "Q01"],
            frequency_range=[5.00, 5.01],
            plot=False,
            save_image=False,
        )
