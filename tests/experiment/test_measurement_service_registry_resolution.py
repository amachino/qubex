"""Tests for target-registry resolution in measurement service."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import Blank, PulseSchedule

from qubex.experiment.services.measurement_service import MeasurementService
from qubex.measurement.models import (
    CaptureData,
    MeasurementConfig,
    MeasurementResult,
    MeasureResult,
)


class _DummyResult:
    def plot(self) -> None:
        return None


def _make_measurement_result(target: str = "custom-target") -> MeasurementResult:
    measurement_config = MeasurementConfig(
        n_shots=1,
        shot_interval=100.0,
        shot_averaging=True,
        time_integration=False,
        state_classification=False,
    )
    return MeasurementResult(
        data={
            target: [
                CaptureData.from_primary_data(
                    target=target,
                    data=np.array([0.0 + 0.0j], dtype=np.complex128),
                    config=measurement_config,
                    sampling_period=2.0,
                )
            ]
        },
        measurement_config=measurement_config,
    )


def _make_service() -> tuple[MeasurementService, dict[str, object]]:
    reset_calls: list[set[str]] = []
    execute_calls: list[dict[str, object]] = []
    measure_calls: list[dict[str, object]] = []
    measure_noise_calls: list[dict[str, object]] = []

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

    async def _measure_noise(
        targets: Any,
        duration: float,
        **kwargs: object,
    ) -> MeasurementResult:
        measure_noise_calls.append(
            {
                "targets": list(targets),
                "duration": duration,
                **kwargs,
            }
        )
        return _make_measurement_result()

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
            measure_noise=_measure_noise,
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
        "measure_noise_calls": measure_noise_calls,
    }
    return cast(MeasurementService, service), captured


def test_resolve_deprecated_option_uses_legacy_value_with_warning() -> None:
    """Given a legacy option, when resolving, then it warns and returns the legacy value."""
    deprecated_options: dict[str, Any] = {"shots": 123}

    with pytest.warns(DeprecationWarning, match="`shots` is deprecated"):
        resolved = MeasurementService.resolve_deprecated_option(
            value=None,
            deprecated_options=deprecated_options,
            deprecated_name="shots",
            replacement_name="n_shots",
            default=100,
        )

    assert resolved == 123
    assert "shots" not in deprecated_options


def test_resolve_deprecated_option_raises_on_conflict() -> None:
    """Given conflicting legacy and canonical options, when resolving, then it raises a conflict error."""
    deprecated_options: dict[str, Any] = {"interval": 120.0}

    with (
        pytest.warns(
            DeprecationWarning,
            match="`interval` is deprecated",
        ),
        pytest.raises(ValueError, match="conflicts with `shot_interval`"),
    ):
        MeasurementService.resolve_deprecated_option(
            value=100.0,
            deprecated_options=deprecated_options,
            deprecated_name="interval",
            replacement_name="shot_interval",
            default=200.0,
        )


def test_resolve_deprecated_option_uses_default_without_legacy() -> None:
    """Given no canonical or legacy value, when resolving, then it returns the provided default."""
    deprecated_options: dict[str, Any] = {}

    resolved = MeasurementService.resolve_deprecated_option(
        value=None,
        deprecated_options=deprecated_options,
        deprecated_name="shots",
        replacement_name="n_shots",
        default=64,
    )

    assert resolved == 64


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


def test_execute_does_not_reset_awg_by_default() -> None:
    """Given execute call without reset flag, when executed, then AWG reset is skipped by default."""
    service, captured = _make_service()

    with PulseSchedule(["custom-target"]) as schedule:
        pass

    service.execute(schedule, plot=False)

    assert captured["reset_calls"] == []


def test_measure_does_not_reset_awg_by_default() -> None:
    """Given measure call without reset flag, when measured, then AWG reset is skipped by default."""
    service, captured = _make_service()

    service.measure(
        sequence={"custom-target": np.array([0.0 + 0.0j], dtype=np.complex128)},
        plot=False,
    )

    assert captured["reset_calls"] == []


def test_check_waveform_resolves_read_labels_via_target_registry() -> None:
    """Given custom targets, when building readout amplitudes, then read labels use target registry mapping."""
    service, _ = _make_service()
    captured: dict[str, object] = {}

    async def _run_measurement(
        self: MeasurementService,
        schedule: object,
        **kwargs: object,
    ) -> MeasurementResult:
        captured["labels"] = list(cast(PulseSchedule, schedule).labels)
        captured["readout_amplitudes"] = kwargs["readout_amplitudes"]
        captured["time_integration"] = kwargs["time_integration"]
        return _make_measurement_result()

    service.__dict__["run_measurement"] = MethodType(_run_measurement, service)

    with pytest.warns(DeprecationWarning, match="method=\\.\\.\\."):
        service.check_waveform(
            targets=["custom-target"],
            method="measure",
            readout_amplitude=0.5,
            plot=False,
        )

    assert captured["labels"] == ["custom-target"]
    assert captured["readout_amplitudes"] == {"RQ17": 0.5}
    assert captured["time_integration"] is False


def test_check_waveform_for_execute_forces_dsp_sum_disabled() -> None:
    """Given execute-based waveform inspection, when check_waveform is called, then backend DSP summation is disabled."""
    service, _ = _make_service()
    captured: dict[str, object] = {}

    async def _run_measurement(
        self: MeasurementService,
        schedule: object,
        **kwargs: object,
    ) -> MeasurementResult:
        captured["labels"] = list(cast(PulseSchedule, schedule).labels)
        captured["time_integration"] = kwargs["time_integration"]
        return _make_measurement_result()

    service.__dict__["run_measurement"] = MethodType(_run_measurement, service)

    with pytest.warns(DeprecationWarning, match="method=\\.\\.\\."):
        service.check_waveform(
            targets=["custom-target"],
            method="execute",
            plot=False,
        )

    assert captured["labels"] == ["custom-target"]
    assert captured["time_integration"] is False


def test_check_waveform_returns_legacy_measure_result() -> None:
    """Given waveform inspection, when check_waveform is called, then it returns a legacy MeasureResult."""
    service, _ = _make_service()
    expected = _make_measurement_result()

    async def _run_measurement(
        self: MeasurementService,
        schedule: object,
        **kwargs: object,
    ) -> MeasurementResult:
        return expected

    service.__dict__["run_measurement"] = MethodType(_run_measurement, service)

    result = service.check_waveform(targets=["custom-target"], plot=False)

    assert isinstance(result, MeasureResult)
    assert result.data["custom-target"].raw.shape == (1,)


def test_check_waveform_rejects_unknown_keyword_arguments() -> None:
    """Given unknown kwargs, when check_waveform is called, then it raises TypeError."""
    service, _ = _make_service()

    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        service.check_waveform(targets=["custom-target"], n_shot=1, plot=False)


def test_check_noise_delegates_without_optional_noise_flags() -> None:
    """Given waveform-noise inspection, when check_noise is called, then it delegates with required args only."""
    service, captured = _make_service()

    service.check_noise(targets=["custom-target"], duration=512, plot=False)

    assert captured["measure_noise_calls"] == [
        {
            "targets": ["custom-target"],
            "duration": 512,
        }
    ]


def test_check_noise_returns_legacy_measure_result_and_plots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given plot enabled, when check_noise is called, then it returns and plots a legacy MeasureResult."""
    service, _ = _make_service()
    plot_calls = {"count": 0}
    expected = _make_measurement_result()

    def _plot(self: MeasureResult, *args: object, **kwargs: object) -> None:
        plot_calls["count"] += 1

    monkeypatch.setattr(MeasureResult, "plot", _plot)

    async def _measure_noise(*_args: object, **_kwargs: object) -> MeasurementResult:
        return expected

    service.ctx.measurement.measure_noise = _measure_noise

    result = service.check_noise(targets=["custom-target"], duration=512, plot=True)

    assert isinstance(result, MeasureResult)
    assert result.data["custom-target"].raw.shape == (1,)
    assert plot_calls["count"] == 1


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
