"""Tests for SessionService behavior."""

from __future__ import annotations

from typing import Any, cast

from qubex.experiment.experiment_context import ExperimentContext
from qubex.experiment.services.session_service import (
    SessionService,
)
from qubex.typing import ConfigurationMode


class _SystemManagerStub:
    def __init__(self, calls: list[tuple[str, dict[str, object]]]) -> None:
        self._calls = calls
        self.backend_settings: dict[str, object] = {}

    def load(self, **kwargs: object) -> None:
        self._calls.append(("system_manager.load", dict(kwargs)))

    def push(self, **kwargs: object) -> None:
        self._calls.append(("system_manager.push", dict(kwargs)))

    def is_synced(self, *, box_ids: list[str]) -> bool:
        return len(box_ids) > 0


class _ExperimentSystemStub:
    def get_boxes_for_qubits(self, qubits: object) -> list[Any]:
        return []


class _ContextStub:
    def __init__(
        self,
        *,
        system_manager: _SystemManagerStub,
        measurement: _MeasurementStub,
        config_path: str,
        params_path: str,
        calls: list[tuple[str, dict[str, object]]],
    ) -> None:
        self.configuration_mode: ConfigurationMode = "ge-cr-cr"
        self.box_ids = ["Q2A", "Q2B"]
        self.targets = {"Q00": object(), "RQ00": object()}
        self.backend_controller = object()
        self.system_manager = system_manager
        self.measurement = measurement
        self.chip_id = "64Qv2"
        self.config_path = config_path
        self.params_path = params_path
        self.experiment_system = _ExperimentSystemStub()


class _MeasurementStub:
    def __init__(self, calls: list[tuple[str, dict[str, object]]]) -> None:
        self._calls = calls
        self.sampling_period = 0.4

    def is_connected(self) -> bool:
        return True

    def disconnect(self) -> None:
        self._calls.append(("measurement.disconnect", {}))

    def connect(
        self,
        *,
        sync_clocks: bool | None = None,
        parallel: bool | None = None,
    ) -> None:
        self._calls.append(
            (
                "measurement.connect",
                {
                    "sync_clocks": sync_clocks,
                    "parallel": parallel,
                },
            )
        )

    def reload(
        self,
        *,
        configuration_mode: str | None = None,
    ) -> None:
        self._calls.append(
            (
                "measurement.reload",
                {"configuration_mode": configuration_mode},
            )
        )

    def check_link_status(self, box_list: list[str]) -> dict:
        return {"status": True, "links": {box: {} for box in box_list}}

    def check_clock_status(self, box_list: list[str]) -> dict:
        return {"status": True, "clocks": dict.fromkeys(box_list, ())}

    def linkup(
        self,
        box_list: list[str],
        noise_threshold: int | None = None,
    ) -> None:
        self._calls.append(
            (
                "measurement.linkup",
                {
                    "box_list": box_list,
                    "noise_threshold": noise_threshold,
                },
            )
        )


def test_connect_uses_measurement_and_sync_hook(monkeypatch) -> None:
    """Given connect args, when connect runs, then measurement connect and sync hook are called."""
    calls: list[tuple[str, dict[str, object]]] = []

    measurement = _MeasurementStub(calls)
    context = _ContextStub(
        system_manager=_SystemManagerStub(calls),
        measurement=measurement,
        config_path="config-dir",
        params_path="params-dir",
        calls=calls,
    )
    sync_calls: list[str] = []

    monkeypatch.setattr(
        SessionService,
        "_sync_pulse_sampling_period",
        lambda self: sync_calls.append("sync") or 0.4,
    )

    service = SessionService(
        experiment_context=cast(ExperimentContext, context),
    )

    service.connect(sync_clocks=False, parallel=True)

    assert calls == [
        (
            "measurement.connect",
            {
                "sync_clocks": False,
                "parallel": True,
            },
        ),
    ]
    assert sync_calls == ["sync", "sync"]


def test_configure_uses_system_manager_and_sync_hook(monkeypatch) -> None:
    """Given configure args, when configure runs, then system manager load/push and sync hook are called."""
    calls: list[tuple[str, dict[str, object]]] = []

    context = _ContextStub(
        system_manager=_SystemManagerStub(calls),
        measurement=_MeasurementStub(calls),
        config_path="config-dir",
        params_path="params-dir",
        calls=calls,
    )
    sync_calls: list[str] = []

    monkeypatch.setattr(
        SessionService,
        "_sync_pulse_sampling_period",
        lambda self: sync_calls.append("sync") or 0.4,
    )

    service = SessionService(
        experiment_context=cast(ExperimentContext, context),
    )

    service.configure(box_ids="Q2A", exclude="Q00", mode=None)

    assert calls == [
        (
            "system_manager.load",
            {
                "chip_id": "64Qv2",
                "config_dir": "config-dir",
                "params_dir": "params-dir",
                "targets_to_exclude": ["Q00"],
                "configuration_mode": "ge-cr-cr",
            },
        ),
        (
            "system_manager.push",
            {
                "box_ids": ["Q2A"],
                "target_labels": ["Q00", "RQ00"],
            },
        ),
    ]
    assert sync_calls == ["sync", "sync"]
