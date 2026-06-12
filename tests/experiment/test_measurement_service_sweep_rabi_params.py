"""Tests for sweep-parameter measurement execution paths."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import Blank, PulseSchedule

from qubex.experiment.models.rabi_param import RabiParam
from qubex.experiment.services.measurement_service import MeasurementService
from qubex.measurement import MeasurementResultConverter


def _stub_measurement_result_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MeasurementResultConverter,
        "to_measure_result",
        staticmethod(
            lambda _result: SimpleNamespace(
                data={"Q33": SimpleNamespace(kerneled=1.0 + 0.0j)}
            )
        ),
    )


def test_sweep_parameter_uses_legacy_measurement_by_default() -> None:
    """Given no batch opt-in, when sweeping, then the legacy measure loop is used."""
    accessed_targets: list[str] = []
    measured: list[dict[str, Any]] = []
    frequency_overrides: list[dict[str, float] | None] = []
    rabi_param = RabiParam.nan(target="Q33")

    @contextmanager
    def _modified_frequencies(frequencies: dict[str, float] | None) -> Any:
        frequency_overrides.append(frequencies)
        yield

    def _get_rabi_param(target: str) -> RabiParam | None:
        accessed_targets.append(target)
        return rabi_param if target == "Q33" else RabiParam.nan(target=target)

    ctx = SimpleNamespace(
        targets={
            "Q33": SimpleNamespace(is_ge=True, is_ef=False),
            "Q18": SimpleNamespace(is_ge=True, is_ef=False),
        },
        qubit_labels=["Q33", "Q18"],
        state_centers={},
        ordered_qubit_labels=lambda labels: list(labels),
        reset_awg_and_capunits=lambda *, qubits: None,
        modified_frequencies=_modified_frequencies,
        get_rabi_param=_get_rabi_param,
        resolve_ef_label=lambda label: f"{label}/ef",
    )
    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = ctx
    service.__dict__["_pulse_service"] = SimpleNamespace(
        readout_duration=512.0,
        readout_pre_margin=16.0,
        readout_post_margin=96.0,
        get_pulse_for_state=lambda _target, _state: Blank(0),
    )

    def _measure(self: MeasurementService, seq: object, **kwargs: object) -> Any:
        _ = self
        measured.append({"sequence": seq, "kwargs": kwargs})
        return SimpleNamespace(data={"Q33": SimpleNamespace(kerneled=1.0 + 0.0j)})

    async def _run_sweep_measurement(
        self: MeasurementService,
        _schedule: Any,
        **_kwargs: Any,
    ) -> Any:
        _ = self
        raise AssertionError("sweep_parameter must use measure by default")

    service.measure = MethodType(_measure, service)
    service.run_sweep_measurement = MethodType(_run_sweep_measurement, service)

    def _sequence(_sweep_value: float) -> PulseSchedule:
        with PulseSchedule(["Q33"]) as schedule:
            schedule.add("Q33", Blank(8))
        return schedule

    result = service.sweep_parameter(
        sequence=_sequence,
        sweep_range=np.array([0.0]),
        frequencies={"Q33": 5.0},
        readout_ramptime=12.0,
        time_integration=False,
        plot=False,
    )

    assert len(measured) == 1
    assert set(cast(dict[str, object], measured[0]["sequence"])) == {"Q33"}
    assert measured[0]["kwargs"] == {
        "initial_states": None,
        "mode": "avg",
        "n_shots": None,
        "shot_interval": None,
        "readout_amplitudes": None,
        "readout_duration": None,
        "readout_pre_margin": None,
        "readout_post_margin": None,
        "reset_awg_and_capunits": False,
        "readout_ramptime": 12.0,
        "time_integration": False,
    }
    assert frequency_overrides == [{"Q33": 5.0}]
    assert accessed_targets == ["Q33"]
    assert result.rabi_params == {"Q33": rabi_param}
    assert result.data["Q33"].rabi_param is rabi_param


def test_sweep_parameter_batch_execution_reads_rabi_params_only_for_swept_qubits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given batch opt-in, when sweeping one qubit, then async sweep execution is used."""
    accessed_targets: list[str] = []
    rabi_param = RabiParam.nan(target="Q33")

    def _get_rabi_param(target: str) -> RabiParam | None:
        accessed_targets.append(target)
        return rabi_param if target == "Q33" else RabiParam.nan(target=target)

    _stub_measurement_result_converter(monkeypatch)
    calls: dict[str, Any] = {}

    async def _run_sweep_measurement(
        self: MeasurementService,
        schedule: Any,
        *,
        sweep_values: Any,
        **kwargs: Any,
    ) -> Any:
        _ = self
        calls["sweep_values"] = sweep_values
        calls["kwargs"] = kwargs
        _ = schedule(np.asarray(sweep_values)[0])
        return SimpleNamespace(results=[object()])

    ctx = SimpleNamespace(
        targets={
            "Q33": SimpleNamespace(is_ge=True, is_ef=False),
            "Q18": SimpleNamespace(is_ge=True, is_ef=False),
        },
        qubit_labels=["Q33", "Q18"],
        state_centers={},
        ordered_qubit_labels=lambda labels: list(labels),
        reset_awg_and_capunits=lambda *, qubits: None,
        get_rabi_param=_get_rabi_param,
        resolve_ef_label=lambda label: f"{label}/ef",
    )
    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = ctx
    service.__dict__["_pulse_service"] = SimpleNamespace(
        readout_duration=512.0,
        readout_pre_margin=16.0,
        readout_post_margin=96.0,
        get_pulse_for_state=lambda _target, _state: Blank(0),
    )

    def _measure(self: MeasurementService, _seq: object, **_kwargs: object) -> Any:
        _ = self
        raise AssertionError("sweep_execution='batch' must use run_sweep_measurement")

    service.measure = MethodType(_measure, service)
    service.run_sweep_measurement = MethodType(_run_sweep_measurement, service)

    def _sequence(_sweep_value: float) -> dict[str, Blank]:
        return {"Q33": Blank(8)}

    result = service.sweep_parameter(
        sequence=_sequence,
        sweep_range=np.array([0.0]),
        sweep_execution="batch",
        plot=False,
    )

    assert accessed_targets == ["Q33"]
    assert result.rabi_params == {"Q33": rabi_param}
    assert result.data["Q33"].rabi_param is rabi_param
    assert np.array_equal(calls["sweep_values"], np.array([0.0]))
    assert calls["kwargs"]["shot_averaging"] is True
    assert calls["kwargs"]["time_integration"] is True


def test_sweep_parameter_uses_configured_experiment_batch_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given experiment batch config, omitted sweep_execution uses async sweep."""
    _stub_measurement_result_converter(monkeypatch)
    called: dict[str, bool] = {"batch": False}

    async def _run_sweep_measurement(
        self: MeasurementService,
        schedule: Any,
        *,
        sweep_values: Any,
        **_kwargs: Any,
    ) -> Any:
        _ = self
        called["batch"] = True
        _ = schedule(np.asarray(sweep_values)[0])
        return SimpleNamespace(results=[object()])

    ctx = SimpleNamespace(
        targets={"Q33": SimpleNamespace(is_ge=True, is_ef=False)},
        qubit_labels=["Q33"],
        config_loader=SimpleNamespace(
            backend_kind="quel1",
            experiment_config={"sweep_execution": "batch"},
        ),
        state_centers={},
        ordered_qubit_labels=lambda labels: list(labels),
        reset_awg_and_capunits=lambda *, qubits: None,
        get_rabi_param=lambda target: RabiParam.nan(target=target),
        resolve_ef_label=lambda label: f"{label}/ef",
    )
    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = ctx
    service.__dict__["_pulse_service"] = SimpleNamespace(
        readout_duration=512.0,
        readout_pre_margin=16.0,
        readout_post_margin=96.0,
        get_pulse_for_state=lambda _target, _state: Blank(0),
    )

    def _measure(self: MeasurementService, _seq: object, **_kwargs: object) -> Any:
        _ = self
        raise AssertionError("configured batch default must use run_sweep_measurement")

    service.measure = MethodType(_measure, service)
    service.run_sweep_measurement = MethodType(_run_sweep_measurement, service)

    def _sequence(_sweep_value: float) -> PulseSchedule:
        with PulseSchedule(["Q33"]) as schedule:
            schedule.add("Q33", Blank(8))
        return schedule

    result = service.sweep_parameter(
        sequence=_sequence,
        sweep_range=np.array([0.0]),
        plot=False,
    )

    assert called["batch"] is True
    assert result.data["Q33"].target == "Q33"


def test_sweep_parameter_prepends_initial_states_in_async_sweep_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given initial states, when sweeping, then state preparation is inside the async schedule callback."""
    built_schedules: list[PulseSchedule] = []
    _stub_measurement_result_converter(monkeypatch)

    async def _run_sweep_measurement(
        self: MeasurementService,
        schedule: Any,
        *,
        sweep_values: Any,
        **_kwargs: Any,
    ) -> Any:
        _ = self
        built_schedules.extend(schedule(value) for value in np.asarray(sweep_values))
        return SimpleNamespace(results=[object()])

    ctx = SimpleNamespace(
        qubit_labels=["Q33"],
        state_centers={},
        ordered_qubit_labels=lambda labels: list(labels),
        reset_awg_and_capunits=lambda *, qubits: None,
        get_rabi_param=lambda target: RabiParam.nan(target=target),
        resolve_ef_label=lambda label: f"{label}/ef",
    )
    service = cast(Any, object.__new__(MeasurementService))
    service.__dict__["_ctx"] = ctx
    service.__dict__["_pulse_service"] = SimpleNamespace(
        readout_duration=512.0,
        readout_pre_margin=16.0,
        readout_post_margin=96.0,
        get_pulse_for_state=lambda _target, _state: Blank(4),
    )

    def _measure(self: MeasurementService, _seq: object, **_kwargs: object) -> Any:
        _ = self
        raise AssertionError("sweep_execution='batch' must use run_sweep_measurement")

    service.measure = MethodType(_measure, service)
    service.run_sweep_measurement = MethodType(_run_sweep_measurement, service)

    def _sequence(_sweep_value: float) -> PulseSchedule:
        with PulseSchedule(["Q33"]) as schedule:
            schedule.add("Q33", Blank(8))
        return schedule

    _ = service.sweep_parameter(
        sequence=_sequence,
        sweep_range=np.array([0.0]),
        initial_states={"Q33": "1"},
        sweep_execution="batch",
        plot=False,
    )

    assert len(built_schedules) == 1
    assert built_schedules[0].labels == ["Q33"]
    assert built_schedules[0].duration == 12
