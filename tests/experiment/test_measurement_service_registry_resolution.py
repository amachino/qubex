"""Tests for target-registry resolution in measurement service."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import Blank, PulseSchedule

from qubex.experiment.services.measurement_service import MeasurementService


class _DummyResult:
    def plot(self) -> None:
        return None


def _make_service() -> tuple[MeasurementService, dict[str, object]]:
    reset_calls: list[set[str]] = []
    execute_calls: list[dict[str, object]] = []
    measure_calls: list[dict[str, object]] = []

    class _TargetRegistry:
        @staticmethod
        def resolve_qubit_label(label: str) -> str:
            return "Q17" if label == "custom-target" else label

        @staticmethod
        def resolve_read_label(label: str) -> str:
            return "RQ17" if label == "custom-target" else label

    @contextmanager
    def _modified_frequencies(_: object) -> Any:
        yield

    experiment_system = SimpleNamespace(
        target_registry=_TargetRegistry(),
        resolve_qubit_label=lambda label: _TargetRegistry.resolve_qubit_label(label),
        resolve_read_label=lambda label: _TargetRegistry.resolve_read_label(label),
        ordered_qubit_labels=lambda labels: list(
            dict.fromkeys(
                _TargetRegistry.resolve_qubit_label(label) for label in labels
            )
        ),
    )

    ctx = SimpleNamespace(
        experiment_system=experiment_system,
        resolve_qubit_label=lambda label: experiment_system.resolve_qubit_label(label),
        resolve_read_label=lambda label: experiment_system.resolve_read_label(label),
        ordered_qubit_labels=lambda labels: experiment_system.ordered_qubit_labels(
            labels
        ),
        measurement=SimpleNamespace(
            execute=lambda **kwargs: execute_calls.append(kwargs) or _DummyResult(),
            measure=lambda **kwargs: measure_calls.append(kwargs) or _DummyResult(),
        ),
        reset_awg_and_capunits=lambda *, qubits: reset_calls.append(set(qubits)),
        modified_frequencies=_modified_frequencies,
        qubit_labels=["Q17"],
    )
    pulse_service = SimpleNamespace(
        readout_duration=1024.0,
        readout_pre_margin=8.0,
        readout_post_margin=16.0,
    )

    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = ctx
    service.__dict__["_pulse_service"] = pulse_service

    captured = {
        "reset_calls": reset_calls,
        "execute_calls": execute_calls,
        "measure_calls": measure_calls,
    }
    return cast(MeasurementService, service), captured


def test_execute_resolves_qubits_via_target_registry_for_reset() -> None:
    """Given custom labels, when executing with reset, then qubits are resolved via target registry."""
    service, captured = _make_service()

    with PulseSchedule(["custom-target"]) as schedule:
        pass

    service.execute(schedule, reset_awg_and_capunits=True, plot=False)

    assert captured["reset_calls"] == [{"Q17"}]


def test_measure_resolves_qubits_via_target_registry_for_reset() -> None:
    """Given custom waveform labels, when measuring with reset, then qubits are resolved via target registry."""
    service, captured = _make_service()

    service.measure(
        sequence={"custom-target": np.array([0.0 + 0.0j], dtype=np.complex128)},
        reset_awg_and_capunits=True,
        plot=False,
    )

    assert captured["reset_calls"] == [{"Q17"}]


def test_check_waveform_resolves_read_labels_via_target_registry() -> None:
    """Given custom targets, when building readout amplitudes, then read labels use target registry mapping."""
    service, _ = _make_service()
    captured: dict[str, object] = {}

    def _measure(
        self: MeasurementService,
        sequence: object,
        **kwargs: object,
    ) -> _DummyResult:
        captured["readout_amplitudes"] = kwargs["readout_amplitudes"]
        captured["enable_dsp_sum"] = kwargs["enable_dsp_sum"]
        return _DummyResult()

    service.__dict__["measure"] = MethodType(_measure, service)

    with pytest.warns(DeprecationWarning, match="method=\\.\\.\\."):
        service.check_waveform(
            targets=["custom-target"],
            method="measure",
            readout_amplitude=0.5,
            plot=False,
        )

    assert captured["readout_amplitudes"] == {"RQ17": 0.5}
    assert captured["enable_dsp_sum"] is False


def test_check_waveform_for_execute_forces_dsp_sum_disabled() -> None:
    """Given execute method, check_waveform explicitly disables backend DSP sum."""
    service, _ = _make_service()
    captured: dict[str, object] = {}

    def _execute(
        self: MeasurementService,
        sequence: object,
        **kwargs: object,
    ) -> _DummyResult:
        captured["enable_dsp_sum"] = kwargs["enable_dsp_sum"]
        return _DummyResult()

    service.__dict__["execute"] = MethodType(_execute, service)

    with pytest.warns(DeprecationWarning, match="method=\\.\\.\\."):
        service.check_waveform(
            targets=["custom-target"],
            method="execute",
            plot=False,
        )

    assert captured["enable_dsp_sum"] is False


def test_measure_state_resolves_ef_labels_via_target_registry() -> None:
    """Given custom labels, when preparing f-state, then EF target label is resolved via target registry."""

    class _TargetRegistry:
        @staticmethod
        def resolve_ef_label(label: str) -> str:
            return "Q17-ef" if label == "custom-target" else label

    experiment_system = SimpleNamespace(
        target_registry=_TargetRegistry(),
        resolve_ef_label=lambda label: _TargetRegistry.resolve_ef_label(label),
    )

    captured: dict[str, object] = {}

    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = SimpleNamespace(
        experiment_system=experiment_system,
        resolve_ef_label=lambda label: experiment_system.resolve_ef_label(label),
    )
    service.__dict__["_pulse_service"] = SimpleNamespace(
        get_pulse_for_state=lambda _target, _state: Blank(0),
        get_hpi_pulse=lambda _target: Blank(0),
    )

    def _measure(
        self: MeasurementService,
        sequence: PulseSchedule,
        **_: object,
    ) -> _DummyResult:
        captured["labels"] = list(sequence.labels)
        return _DummyResult()

    service.__dict__["measure"] = MethodType(_measure, service)

    service.measure_state(
        states={"custom-target": "f"},
        plot=False,
    )

    assert captured["labels"] == ["custom-target", "Q17-ef"]
