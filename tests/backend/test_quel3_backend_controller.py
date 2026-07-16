# ruff: noqa: SLF001

"""Tests for QuEL-3 backend controller behavior."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from enum import Enum
from io import StringIO
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from rich.console import Console

from qubex.backend import BackendExecutionRequest
from qubex.backend.backend_controller import BackendController
from qubex.backend.quel1 import Quel1BackendController
from qubex.backend.quel3 import (
    Quel3BackendController,
    Quel3BackendExecutionResult,
    Quel3CaptureMode,
    Quel3CaptureWindow,
    Quel3ConfigurationManager,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3HardwareState,
    Quel3RuntimeConfig,
    Quel3Waveform,
    Quel3WaveformEvent,
)
from qubex.backend.quel3.managers import execution_manager as execution_manager_module
from qubex.backend.quel3.managers.execution_manager import Quel3ExecutionManager
from qubex.backend.quel3.managers.session_workarounds import QuelwareSessionError


class _FakeCaptureMode(Enum):
    UNSPECIFIED = 1
    RAW_WAVEFORMS = 2
    AVERAGED_WAVEFORM = 3
    AVERAGED_VALUE = 4
    VALUES_PER_ITER = 5


@dataclass(frozen=True)
class _FakeInstrumentDefinition:
    role: str
    alias: str = ""


@dataclass(frozen=True)
class _FakeInstrumentInfo:
    port_id: str
    definition: _FakeInstrumentDefinition
    id: str = ""
    alias: str | None = None


class _FakeInstrumentResolver:
    def __init__(
        self,
        *,
        alias_to_info: dict[str, _FakeInstrumentInfo],
    ) -> None:
        self._alias_to_info = {
            alias: self._with_required_fields(alias=alias, instrument_info=info)
            for alias, info in alias_to_info.items()
        }

    @staticmethod
    def _with_required_fields(
        *,
        alias: str,
        instrument_info: _FakeInstrumentInfo,
    ) -> _FakeInstrumentInfo:
        runtime_alias = (
            instrument_info.alias or instrument_info.definition.alias or alias
        )
        return replace(
            instrument_info,
            id=instrument_info.id or alias,
            definition=replace(instrument_info.definition, alias=runtime_alias),
            alias=runtime_alias,
        )

    async def refresh(self, client: object) -> None:
        del client

    def resolve(self, aliases: list[str]) -> list[str]:
        return aliases

    def find_inst_info_by_alias(self, alias: str) -> _FakeInstrumentInfo:
        if alias not in self._alias_to_info:
            raise ValueError(alias)
        return self._alias_to_info[alias]


class _CountingInstrumentResolver(_FakeInstrumentResolver):
    def __init__(
        self,
        *,
        alias_to_info: dict[str, _FakeInstrumentInfo],
    ) -> None:
        super().__init__(alias_to_info=alias_to_info)
        self.refresh_calls = 0

    async def refresh(self, client: object) -> None:
        del client
        self.refresh_calls += 1


class _FakeHardwareStateReader:
    def __init__(self, state: Quel3HardwareState) -> None:
        self.state = state
        self.last_collect_kwargs: dict[str, object] = {}

    def collect_state(self, **kwargs: object) -> Quel3HardwareState:
        self.last_collect_kwargs = dict(kwargs)
        return self.state


def _make_payload(
    *,
    mode: str = "avg",
    n_iterations: int = 2,
    frequency_hz: float | None = None,
) -> Quel3ExecutionPayload:
    waveform_name = "wf0"
    timeline = Quel3FixedTimeline(
        events=(
            Quel3WaveformEvent(
                waveform_name=waveform_name,
                start_offset_ns=0.0,
            ),
        ),
        capture_windows=(
            Quel3CaptureWindow(name="capture_0", start_offset_ns=0.4, length_ns=0.4),
        ),
        length_ns=0.8,
        frequency_hz=frequency_hz,
    )
    return Quel3ExecutionPayload(
        waveform_library={
            waveform_name: Quel3Waveform(
                iq_array=np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
                sampling_period_ns=0.4,
            )
        },
        fixed_timelines={"alias-rq00": timeline},
        n_iterations=n_iterations,
        shot_interval_ns=100.0,
        capture_mode=(
            Quel3CaptureMode.AVERAGED_VALUE
            if mode == "avg"
            else Quel3CaptureMode.VALUES_PER_ITER
        ),
    )


def test_quel_controllers_implement_backend_controller_contract() -> None:
    """Given QuEL controllers, both satisfy BackendController protocol."""
    assert isinstance(Quel1BackendController(), BackendController)
    assert isinstance(Quel3BackendController(), BackendController)


def test_quel3_controller_is_not_quel1_subclass() -> None:
    """Given QuEL-3 controller, it is not a QuEL-1 subclass."""
    assert not isinstance(Quel3BackendController(), Quel1BackendController)


def test_quel3_constructor_rejects_config_path_argument() -> None:
    """Given legacy config_path kwarg, constructor raises TypeError."""
    with pytest.raises(TypeError, match="config_path"):
        cast(Any, Quel3BackendController)(config_path="dummy")


def test_quel3_constructor_rejects_sampling_period_override_argument() -> None:
    """Given legacy sampling-period override kwarg, constructor raises TypeError."""
    with pytest.raises(TypeError, match="sampling_period_ns"):
        cast(Any, Quel3BackendController)(sampling_period_ns=0.8)


def test_quel3_constructor_rejects_alias_map_argument() -> None:
    """Given legacy alias-map kwarg, constructor raises TypeError."""
    with pytest.raises(TypeError, match="alias_map"):
        cast(Any, Quel3BackendController)(alias_map={"RQ00": "inst-00"})


def test_get_hardware_state_delegates_to_hardware_state_reader() -> None:
    """Given hardware state reader, controller should delegate state collection."""
    state = Quel3HardwareState(
        generated_at="2026-07-07T00:00:00+00:00",
        endpoint="localhost",
        port=50051,
        selected_unit_labels=("unit-a",),
        units=(),
        ports=(),
        instruments=(),
        diagnostics=(),
        issues=(),
    )
    hardware_state_reader = _FakeHardwareStateReader(state)
    controller = Quel3BackendController(
        hardware_state_reader=cast(Any, hardware_state_reader)
    )

    result = controller.get_hardware_state(
        unit_labels=("unit-a",),
        port_ids=("tx_p01",),
        instrument_aliases=("Q00",),
        include_diagnostics=True,
        parallel=False,
        timeout_seconds=1.5,
    )

    assert result is state
    assert hardware_state_reader.last_collect_kwargs["unit_labels"] == ("unit-a",)
    assert hardware_state_reader.last_collect_kwargs["port_ids"] == ("tx_p01",)
    assert hardware_state_reader.last_collect_kwargs["instrument_aliases"] == ("Q00",)
    assert hardware_state_reader.last_collect_kwargs["include_diagnostics"] is True
    assert hardware_state_reader.last_collect_kwargs["parallel"] is False
    assert hardware_state_reader.last_collect_kwargs["timeout_seconds"] == 1.5


def test_get_hardware_state_rejects_old_filter_kwargs() -> None:
    """Given removed hardware-state filter kwargs, controller raises TypeError."""
    controller = Quel3BackendController(
        hardware_state_reader=cast(
            Any,
            _FakeHardwareStateReader(
                Quel3HardwareState(
                    generated_at="2026-07-07T00:00:00+00:00",
                    endpoint="localhost",
                    port=50051,
                    selected_unit_labels=(),
                    units=(),
                    ports=(),
                    instruments=(),
                    diagnostics=(),
                    issues=(),
                )
            ),
        )
    )

    with pytest.raises(TypeError, match="instrument_port_ids"):
        cast(Any, controller).get_hardware_state(instrument_port_ids=("unit-a:tx_p01",))
    with pytest.raises(TypeError, match="diagnostic_port_ids"):
        cast(Any, controller).get_hardware_state(diagnostic_port_ids=("unit-a:tx_p01",))


def test_print_hardware_state_collects_view_and_delegates_to_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given hardware state, controller should print a Rich hardware-state view."""
    state = Quel3HardwareState(
        generated_at="2026-07-07T00:00:00+00:00",
        endpoint="localhost",
        port=50051,
        selected_unit_labels=(),
        units=(),
        ports=(),
        instruments=(),
        diagnostics=(),
        issues=(),
    )
    hardware_state_reader = _FakeHardwareStateReader(state)
    controller = Quel3BackendController(
        hardware_state_reader=cast(Any, hardware_state_reader)
    )
    printed_views: list[str] = []
    monkeypatch.setattr(
        Quel3HardwareState,
        "print",
        lambda self, *, view: printed_views.append(view),
    )

    controller.print_hardware_state(view="summary")

    assert hardware_state_reader.last_collect_kwargs["view"] == "summary"
    assert printed_views == ["summary"]


def test_print_hardware_state_rejects_console_kwarg() -> None:
    """Given removed console kwarg, controller raises TypeError."""
    controller = Quel3BackendController(
        hardware_state_reader=cast(
            Any,
            _FakeHardwareStateReader(
                Quel3HardwareState(
                    generated_at="2026-07-07T00:00:00+00:00",
                    endpoint="localhost",
                    port=50051,
                    selected_unit_labels=(),
                    units=(),
                    ports=(),
                    instruments=(),
                    diagnostics=(),
                    issues=(),
                )
            ),
        )
    )
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)

    with pytest.raises(TypeError, match="console"):
        cast(Any, controller).print_hardware_state(console=console)


def test_execute_rejects_non_quel3_payload() -> None:
    """Given non-QuEL-3 payload, execute raises TypeError."""
    controller = Quel3BackendController()

    with pytest.raises(TypeError, match="Quel3ExecutionPayload"):
        asyncio.run(
            controller.execute_async(request=BackendExecutionRequest(payload=object()))
        )


def test_execute_surfaces_missing_quelware_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given missing quelware dependency, execute raises RuntimeError."""
    controller = Quel3BackendController()
    payload = _make_payload()

    monkeypatch.setattr(
        Quel3ExecutionManager,
        "_load_quelware_api",
        staticmethod(
            lambda: (_ for _ in ()).throw(ModuleNotFoundError("quelware_client"))
        ),
    )

    with pytest.raises(RuntimeError, match="quelware-client is not available"):
        asyncio.run(
            controller.execute_async(request=BackendExecutionRequest(payload=payload))
        )


def test_build_measurement_result_averages_shot_samples() -> None:
    """Given avg mode shots, result samples are averaged."""
    payload = _make_payload(mode="avg", n_iterations=2)
    shot_samples = {
        "alias-rq00": {
            "capture_0": [
                np.array([1.0 + 1.0j, 3.0 + 3.0j], dtype=np.complex128),
                np.array([3.0 + 3.0j, 5.0 + 5.0j], dtype=np.complex128),
            ]
        }
    }

    result = Quel3ExecutionManager._build_measurement_result(
        payload=payload,
        shot_samples=shot_samples,
        capture_sampling_period_ns=0.8,
        backend_sampling_period_ns=0.4,
        capture_decimation_factor=1,
    )

    assert isinstance(result, Quel3BackendExecutionResult)
    assert isinstance(result.status, dict)
    assert "alias-rq00" in result.data
    assert np.array_equal(
        result.data["alias-rq00"][0],
        np.array([2.0 + 2.0j, 4.0 + 4.0j], dtype=np.complex128),
    )
    assert result.config["sampling_period_ns"] == pytest.approx(0.8)


def test_build_measurement_result_keeps_backend_alias_labels() -> None:
    """Given backend flow result, measurement labels remain instrument aliases."""
    payload = _make_payload(mode="single", n_iterations=1)
    timeline = payload.fixed_timelines["alias-rq00"]
    payload = replace(
        payload,
        fixed_timelines={"alias-raw": timeline},
    )
    shot_samples = {
        "alias-raw": {
            "capture_0": [
                np.array([7.0 + 0.0j], dtype=np.complex128),
            ]
        }
    }

    result = Quel3ExecutionManager._build_measurement_result(
        payload=payload,
        shot_samples=shot_samples,
        capture_sampling_period_ns=0.4,
        backend_sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )

    assert isinstance(result, Quel3BackendExecutionResult)
    assert "alias-raw" in result.data


def test_extract_capture_samples_from_waveform_result_container() -> None:
    """Given waveform result container, extraction returns latest waveform samples."""

    class _Waveform:
        def __init__(self, values: np.ndarray) -> None:
            self.iq_array = values

    class _Result:
        def __init__(self) -> None:
            self.iq_waveform_result = {
                "RQ00:0": [
                    _Waveform(np.array([1.0 + 0.0j], dtype=np.complex128)),
                    _Waveform(np.array([2.0 + 0.0j], dtype=np.complex128)),
                ]
            }
            self.iq_point_result = {}

    values = Quel3ExecutionManager._extract_capture_samples(
        _Result(),
        "RQ00:0",
        capture_mode=Quel3CaptureMode.AVERAGED_WAVEFORM,
    )

    assert values is not None
    assert np.array_equal(values, np.array([2.0 + 0.0j], dtype=np.complex128))


def test_extract_capture_samples_from_raw_waveform_result_container() -> None:
    """Given raw waveform result container, extraction returns one waveform per shot."""

    class _Waveform:
        def __init__(self, values: np.ndarray) -> None:
            self.iq_array = values

    class _Result:
        def __init__(self) -> None:
            self.iq_waveform_result = {
                "RQ00:0": [
                    _Waveform(np.array([1.0 + 0.0j, 2.0 + 0.0j])),
                    _Waveform(np.array([3.0 + 0.0j, 4.0 + 0.0j])),
                ]
            }
            self.iq_point_result = {}

    values = Quel3ExecutionManager._extract_capture_samples(
        _Result(),
        "RQ00:0",
        capture_mode=Quel3CaptureMode.RAW_WAVEFORMS,
    )

    assert values is not None
    assert np.array_equal(
        values,
        np.array(
            [
                [1.0 + 0.0j, 2.0 + 0.0j],
                [3.0 + 0.0j, 4.0 + 0.0j],
            ],
            dtype=np.complex128,
        ),
    )


def test_extract_capture_samples_from_point_result_container() -> None:
    """Given point result container, extraction returns complex-point array."""

    class _Result:
        def __init__(self) -> None:
            self.iq_waveform_result = {}
            self.iq_point_result = {
                "RQ00:0": [1.0 + 2.0j, 3.0 + 4.0j],
            }

    values = Quel3ExecutionManager._extract_capture_samples(
        _Result(),
        "RQ00:0",
        capture_mode=Quel3CaptureMode.VALUES_PER_ITER,
    )

    assert values is not None
    assert np.array_equal(
        values,
        np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex128),
    )


def test_constructor_uses_builtin_quelware_defaults_ignoring_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given quelware env vars, constructor still uses builtin defaults."""
    monkeypatch.setenv("QUBEX_QUELWARE_ENDPOINT", "env-host")
    monkeypatch.setenv("QUBEX_QUELWARE_PORT", "12345")

    controller = Quel3BackendController()

    assert pytest.approx(0.4) == controller.sampling_period_ns
    assert controller.connection_manager.quelware_endpoint == "localhost"
    assert controller.connection_manager.quelware_port == 50051
    assert controller.session_manager.quelware_endpoint == "localhost"
    assert controller.session_manager.quelware_port == 50051


def test_constructor_rejects_standalone_runtime_mode() -> None:
    """Given standalone runtime mode, controller construction should fail fast."""
    with pytest.raises(ValueError, match="Unsupported QuEL-3 client mode"):
        Quel3BackendController(client_mode="standalone")


def test_constructor_accepts_quelware_pat_path_runtime_option() -> None:
    """Given PAT path runtime option, controller should propagate only the path."""
    pat_path = "/run/secrets/quelware-pat"
    controller = Quel3BackendController(quelware_pat_path=pat_path)

    assert controller.quelware_pat_path == pat_path
    assert controller.connection_manager.quelware_pat_path == pat_path
    assert controller.session_manager.quelware_pat_path == pat_path
    assert controller.configuration_manager.quelware_pat_path == pat_path
    assert controller.execution_manager.quelware_pat_path == pat_path
    assert controller.connection_manager.runtime_config is controller.runtime_config
    assert controller.session_manager.runtime_config is controller.runtime_config
    assert controller.configuration_manager.runtime_config is controller.runtime_config
    assert controller.execution_manager.runtime_config is controller.runtime_config


def test_constructor_accepts_injected_managers() -> None:
    """Given injected managers, controller should use those manager instances."""
    connection_manager = SimpleNamespace(
        hash=7,
        is_connected=True,
        quelware_endpoint="injected-host",
        quelware_port=61000,
        client_mode="server",
        quelware_pat_path="/run/secrets/quelware-pat",
        connect=lambda box_names=None, parallel=None: None,
        disconnect=lambda: None,
    )
    session_manager = SimpleNamespace(
        hash=11,
        quelware_endpoint="injected-host",
        quelware_port=61000,
        client_mode="server",
        quelware_pat_path="/run/secrets/quelware-pat",
        open=lambda box_names=None, parallel=None: None,
        close=lambda: None,
    )
    configuration_manager = SimpleNamespace(
        quelware_endpoint="injected-host",
        quelware_port=61000,
        client_mode="server",
        quelware_pat_path="/run/secrets/quelware-pat",
        target_alias_map={("BOX1", "Q00"): "Q00"},
        last_deployed_instrument_infos={"Q00": (object(),)},
        deploy_instruments=lambda *, requests: {"Q00": tuple(requests)},
    )
    execution_manager = SimpleNamespace(
        quelware_endpoint="injected-host",
        quelware_port=61000,
        sampling_period_ns=0.8,
        client_mode="server",
        quelware_pat_path="/run/secrets/quelware-pat",
        execute_sync=lambda *, request: request,
        execute_async=lambda *, request: request,
    )

    controller = Quel3BackendController(
        connection_manager=cast(Any, connection_manager),
        session_manager=cast(Any, session_manager),
        configuration_manager=cast(Any, configuration_manager),
        execution_manager=cast(Any, execution_manager),
    )

    assert controller.connection_manager is connection_manager
    assert controller.session_manager is session_manager
    assert controller.configuration_manager is configuration_manager
    assert controller.execution_manager is execution_manager
    assert controller.quelware_endpoint == "localhost"
    assert controller.quelware_port == 50051
    assert controller.sampling_period_ns == pytest.approx(0.8)
    assert controller.client_mode == "server"
    assert controller.quelware_pat_path is None


def test_constructor_accepts_injected_session_manager() -> None:
    """Given session manager injection, controller should expose it."""
    session_manager = SimpleNamespace(
        hash=11,
        quelware_endpoint="injected-host",
        quelware_port=61000,
        client_mode="server",
        quelware_pat_path=None,
        open=lambda box_names=None, parallel=None: None,
        close=lambda: None,
    )

    controller = Quel3BackendController(
        session_manager=cast(Any, session_manager),
    )

    assert controller.session_manager is session_manager


def test_connect_clears_existing_instrument_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QuEL-3 connect should clear instrument mappings without refreshing them."""
    calls: list[str] = []
    connection_manager = SimpleNamespace(
        connect=lambda box_names=None, parallel=None: calls.append("connect"),
        disconnect=lambda: None,
    )
    configuration_manager = Quel3ConfigurationManager()
    configuration_manager._last_deployed_instrument_infos = {"Q00": ()}
    configuration_manager._target_alias_map = {("BOX1", "Q00"): "unit-a:Q00"}
    controller = Quel3BackendController(
        connection_manager=cast(Any, connection_manager),
        configuration_manager=configuration_manager,
    )
    monkeypatch.setattr(
        controller.execution_manager,
        "invalidate_instrument_resolver",
        lambda: calls.append("invalidate-resolver"),
    )

    controller.connect(["BOX1"])

    assert calls == ["connect", "invalidate-resolver"]
    assert controller.last_deployed_instrument_infos == {}
    assert controller.target_alias_map == {}


def test_deploy_instruments_forwards_parallel_flag_to_configuration_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given parallel override, controller deploy_instruments should forward it."""
    captured: dict[str, object] = {}

    def _deploy_instruments(*, requests: object, parallel: bool) -> object:
        captured["requests"] = requests
        captured["parallel"] = parallel
        return {"Q00": ()}

    controller = Quel3BackendController(
        configuration_manager=cast(
            Any,
            SimpleNamespace(
                quelware_endpoint="host-a",
                quelware_port=50051,
                client_mode="server",
                quelware_pat_path=None,
                target_alias_map={},
                last_deployed_instrument_infos={},
                deploy_instruments=_deploy_instruments,
            ),
        )
    )
    invalidation_calls: list[str] = []
    monkeypatch.setattr(
        controller.execution_manager,
        "invalidate_instrument_resolver",
        lambda: invalidation_calls.append("invalidate-resolver"),
    )
    requests = (
        SimpleNamespace(
            port_id="quel3-02-a01:tx_p02",
            role="TRANSMITTER",
            frequency_range_min_hz=4.1e9,
            frequency_range_max_hz=4.3e9,
            alias="Q00",
            target_labels=("Q00",),
        ),
    )

    result = controller.deploy_instruments(requests=cast(Any, requests), parallel=False)

    assert result == {"Q00": ()}
    assert captured == {"requests": requests, "parallel": False}
    assert invalidation_calls == ["invalidate-resolver"]


def test_constructor_does_not_infer_runtime_config_from_injected_managers() -> None:
    """Given injected managers, controller runtime config should stay explicit."""
    connection_manager = SimpleNamespace(
        hash=11,
        is_connected=False,
        quelware_endpoint="host-a",
        quelware_port=50051,
        client_mode="server",
        quelware_pat_path=None,
        connect=lambda box_names=None, parallel=None: None,
        disconnect=lambda: None,
    )
    configuration_manager = SimpleNamespace(
        quelware_endpoint="host-b",
        quelware_port=50051,
        client_mode="server",
        quelware_pat_path=None,
        target_alias_map={},
        last_deployed_instrument_infos={},
        deploy_instruments=lambda *, requests: {},
    )

    controller = Quel3BackendController(
        quelware_endpoint="explicit-host",
        quelware_port=61000,
        quelware_pat_path="/run/secrets/explicit-pat",
        connection_manager=cast(Any, connection_manager),
        configuration_manager=cast(Any, configuration_manager),
    )

    assert controller.connection_manager is connection_manager
    assert controller.configuration_manager is configuration_manager
    assert controller.quelware_endpoint == "explicit-host"
    assert controller.quelware_port == 61000
    assert controller.quelware_pat_path == "/run/secrets/explicit-pat"


def test_resolve_payload_merges_targets_mapped_to_one_alias() -> None:
    """Given shared alias bindings, resolved payload merges timelines per alias."""
    payload = _make_payload(frequency_hz=6.2e9)
    payload = replace(
        payload,
        fixed_timelines={
            "RQ00": payload.fixed_timelines["alias-rq00"],
            "RQ01": payload.fixed_timelines["alias-rq00"],
        },
        instrument_bindings={
            "RQ00": "alias:unit-a:alias-shared",
            "RQ01": "alias:unit-a:alias-shared",
        },
    )

    resolved = Quel3ExecutionManager._resolve_payload(
        payload=payload,
    )

    assert set(resolved.fixed_timelines.keys()) == {"unit-a:alias-shared"}
    timeline = resolved.fixed_timelines["unit-a:alias-shared"]
    assert [window.name for window in timeline.capture_windows] == [
        "unit-a:alias-shared:0",
        "unit-a:alias-shared:1",
    ]
    assert timeline.frequency_hz == pytest.approx(6.2e9)


def test_resolve_payload_rejects_conflicting_frequencies_for_shared_alias() -> None:
    """Given shared alias with different frequencies, resolving payload should fail."""
    payload = _make_payload()
    base_timeline = payload.fixed_timelines["alias-rq00"]
    payload = replace(
        payload,
        fixed_timelines={
            "RQ00": replace(base_timeline, frequency_hz=6.0e9),
            "RQ01": replace(base_timeline, frequency_hz=6.1e9),
        },
        instrument_bindings={
            "RQ00": "alias:unit-a:alias-shared",
            "RQ01": "alias:unit-a:alias-shared",
        },
    )

    with pytest.raises(ValueError, match="Conflicting frequency"):
        Quel3ExecutionManager._resolve_payload(
            payload=payload,
        )


def test_filter_runnable_payload_drops_empty_aliases() -> None:
    """Given empty and active timelines, filtering should keep only runnable aliases."""
    payload = _make_payload()
    payload = replace(
        payload,
        fixed_timelines={
            "alias-empty": Quel3FixedTimeline(
                events=(),
                capture_windows=(),
                length_ns=payload.fixed_timelines["alias-rq00"].length_ns,
            ),
            "alias-rq00": payload.fixed_timelines["alias-rq00"],
        },
    )

    filtered = Quel3ExecutionManager._filter_runnable_payload(payload)

    assert set(filtered.fixed_timelines.keys()) == {"alias-rq00"}


def test_filter_runnable_payload_rejects_all_empty_timelines() -> None:
    """Given only empty timelines, filtering should fail with a clear error."""
    payload = _make_payload()
    payload = replace(
        payload,
        fixed_timelines={
            "alias-empty": Quel3FixedTimeline(
                events=(),
                capture_windows=(),
                length_ns=payload.fixed_timelines["alias-rq00"].length_ns,
            )
        },
    )

    with pytest.raises(ValueError, match="no waveform events or capture windows"):
        Quel3ExecutionManager._filter_runnable_payload(payload)


def test_resolve_payload_rejects_port_binding() -> None:
    """Given legacy port binding, resolving payload fails fast."""
    payload = _make_payload()
    payload = replace(
        payload,
        fixed_timelines={"RQ00": payload.fixed_timelines["alias-rq00"]},
        instrument_bindings={"RQ00": "port:unit-a-0"},
        capture_port_bindings={"RQ00": "unit-a-0"},
    )

    with pytest.raises(ValueError, match="Unsupported instrument binding"):
        Quel3ExecutionManager._resolve_payload(
            payload=payload,
        )


def test_resolve_payload_rejects_unqualified_alias_binding() -> None:
    """Given unqualified alias binding, resolving payload should fail fast."""
    payload = _make_payload()
    payload = replace(
        payload,
        fixed_timelines={"Q00": payload.fixed_timelines["alias-rq00"]},
        instrument_bindings={"Q00": "alias:inst-q00"},
    )

    with pytest.raises(ValueError, match="unit label"):
        Quel3ExecutionManager._resolve_payload(payload=payload)


def test_execute_resolves_unit_prefixed_alias_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given unit-prefixed alias binding, execute should resolve with the unit label."""
    payload = _make_payload()
    payload = replace(
        payload,
        fixed_timelines={"Q00": payload.fixed_timelines["alias-rq00"]},
        instrument_bindings={"Q00": "alias:quel3-02-a01:Q00"},
    )
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )

    @dataclass(frozen=True)
    class _Definition:
        alias: str
        role: str

    @dataclass(frozen=True)
    class _InstrumentInfo:
        id: str
        port_id: str
        definition: _Definition

    class _UnitAwareResolver:
        def __init__(self) -> None:
            self.find_calls: list[tuple[str, str | None]] = []

        async def refresh(self, client: object) -> None:
            del client

        def resolve(self, aliases: list[str]) -> list[str]:
            return aliases

        def find_inst_info_by_alias(
            self,
            alias: str,
            *,
            unit: str | None = None,
        ) -> _InstrumentInfo:
            self.find_calls.append((alias, unit))
            if (alias, unit) != ("Q00", "quel3-02-a01"):
                raise ValueError(alias)
            return _InstrumentInfo(
                id="inst-q00",
                port_id="quel3-02-a01:tx_p04",
                definition=_Definition(
                    alias="Q00",
                    role="TRANSMITTER",
                ),
            )

    resolver = _UnitAwareResolver()
    driver = _FakeInstrumentDriver()
    session = _FakeSession()
    client = _FakeClient(session)

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=lambda endpoint, port: client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=lambda _session, _instrument_info: driver,
        ),
    )

    result = asyncio.run(
        manager.execute_async(request=BackendExecutionRequest(payload=payload))
    )

    assert resolver.find_calls == [("Q00", "quel3-02-a01")]
    assert driver.apply_calls == [
        [
            ("capture_mode", _FakeCaptureMode.AVERAGED_VALUE),
            ("timeline", "quel3-02-a01:Q00"),
        ]
    ]
    assert session.trigger_calls == [["inst-q00"]]
    assert "quel3-02-a01:Q00" in result.data


def test_execute_rejects_invalid_payload_before_opening_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given invalid payload bindings, execute should fail before session setup."""
    payload = _make_payload()
    base_timeline = payload.fixed_timelines["alias-rq00"]
    payload = replace(
        payload,
        fixed_timelines={
            "RQ00": replace(base_timeline, frequency_hz=6.0e9),
            "RQ01": replace(base_timeline, frequency_hz=6.1e9),
        },
        instrument_bindings={
            "RQ00": "alias:quel3-02-a01:Q00",
            "RQ01": "alias:quel3-02-a01:Q00",
        },
    )
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    create_session_calls: list[tuple[str, ...]] = []
    driver_factory_calls = 0

    class _UnitAwareResolver:
        async def refresh(self, client: object) -> None:
            del client

        def resolve(self, aliases: list[str]) -> list[str]:
            return aliases

        def find_inst_info_by_alias(
            self,
            alias: str,
            *,
            unit: str | None = None,
        ) -> _FakeInstrumentInfo:
            if (alias, unit) != ("Q00", "quel3-02-a01"):
                raise ValueError(alias)
            return _FakeInstrumentInfo(
                id="inst-q00",
                port_id="quel3-02-a01:tx_p04",
                definition=_FakeInstrumentDefinition(
                    alias="Q00",
                    role="TRANSMITTER",
                ),
            )

    class _OrderProbeClient(_FakeClient):
        def create_session(
            self,
            resource_ids: list[str],
            ttl_ms: int = 4_000,
            tentative_ttl_ms: int = 1_000,
        ) -> _FakeSession:
            create_session_calls.append(tuple(resource_ids))
            return super().create_session(
                resource_ids,
                ttl_ms=ttl_ms,
                tentative_ttl_ms=tentative_ttl_ms,
            )

    def _create_driver(
        session: object,
        instrument_info: object,
    ) -> _FakeInstrumentDriver:
        nonlocal driver_factory_calls
        del session, instrument_info
        driver_factory_calls += 1
        return _FakeInstrumentDriver()

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=lambda endpoint, port: _OrderProbeClient(_FakeSession()),
            instrument_resolver_factory=_UnitAwareResolver,
            fixed_timeline_driver_factory=_create_driver,
        ),
    )

    with pytest.raises(ValueError, match="Conflicting frequency"):
        asyncio.run(
            manager.execute_async(request=BackendExecutionRequest(payload=payload))
        )

    assert create_session_calls == []
    assert driver_factory_calls == 0


@dataclass(frozen=True)
class _FakeInstrumentConfig:
    sampling_period_fs: int
    timeline_step_samples: int


class _FakeWaveformResult:
    def __init__(self, values: np.ndarray) -> None:
        self.iq_array = values


class _FakeResultContainer:
    def __init__(self) -> None:
        self.iq_waveform_result = {}
        self.iq_point_result = {"alias-rq00:0": [1.0 + 0.0j]}


class _FakeInstrumentDriver:
    def __init__(self) -> None:
        self.instrument_config = _FakeInstrumentConfig(
            sampling_period_fs=400_000,
            timeline_step_samples=64,
        )
        self.apply_calls: list[object] = []
        self.initialized = False

    async def apply(self, directive: object) -> None:
        self.apply_calls.append(directive)

    async def initialize(self) -> None:
        self.initialized = True

    async def wait_for_result(self) -> object:
        return _FakeResultContainer()


class _FakeSequencer:
    def __init__(self, default_sampling_period_ns: float) -> None:
        self.default_sampling_period_ns = default_sampling_period_ns

    def bind(
        self,
        alias: str,
        sampling_period_fs: int,
        step_samples: int,
    ) -> None:
        del alias, sampling_period_fs, step_samples

    def register_waveform(
        self,
        name: str,
        waveform: object,
        sampling_period_ns: float | None = None,
    ) -> None:
        del name, waveform, sampling_period_ns

    def add_event(
        self,
        instrument_alias: str,
        waveform_name: str,
        start_offset_ns: float,
        gain: float = 1.0,
        phase_offset_deg: float = 0.0,
    ) -> None:
        del instrument_alias, waveform_name, start_offset_ns, gain, phase_offset_deg

    def add_capture_window(
        self,
        instrument_alias: str,
        window_name: str,
        start_offset_ns: float,
        length_ns: float,
    ) -> None:
        del instrument_alias, window_name, start_offset_ns, length_ns

    def set_iterations(self, iterations: int) -> None:
        del iterations

    def extend_length_ns(self, additional_ns: float) -> None:
        del additional_ns

    def export_set_fixed_timeline_directive(self, instrument_alias: str) -> object:
        return ("timeline", instrument_alias)


class _PhaseBarrier:
    def __init__(self, expected: int) -> None:
        self._expected = expected
        self._arrived = 0
        self._event = asyncio.Event()

    async def wait(self) -> None:
        self._arrived += 1
        if self._arrived >= self._expected:
            self._event.set()
        await asyncio.wait_for(self._event.wait(), timeout=0.2)


class _ParallelResultContainer:
    def __init__(self, alias: str, value: complex) -> None:
        self.iq_waveform_result = {}
        self.iq_point_result = {f"{alias}:0": [value]}


class _ParallelInstrumentDriver:
    def __init__(
        self,
        *,
        alias: str,
        value: complex,
        initialize_barrier: _PhaseBarrier,
        apply_barrier: _PhaseBarrier,
        fetch_barrier: _PhaseBarrier,
    ) -> None:
        self.instrument_config = _FakeInstrumentConfig(
            sampling_period_fs=400_000,
            timeline_step_samples=64,
        )
        self._alias = alias
        self._value = value
        self._initialize_barrier = initialize_barrier
        self._apply_barrier = apply_barrier
        self._fetch_barrier = fetch_barrier
        self.apply_calls: list[object] = []

    async def apply(self, directive: object) -> None:
        self.apply_calls.append(directive)
        await self._apply_barrier.wait()

    async def initialize(self) -> None:
        await self._initialize_barrier.wait()

    async def wait_for_result(self) -> object:
        await self._fetch_barrier.wait()
        return _ParallelResultContainer(self._alias, self._value)


class _ConcurrencyProbe:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def step(self) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1


class _SerialProbeInstrumentDriver:
    def __init__(
        self,
        *,
        alias: str,
        value: complex,
        initialize_probe: _ConcurrencyProbe,
        apply_probe: _ConcurrencyProbe,
        fetch_probe: _ConcurrencyProbe,
    ) -> None:
        self.instrument_config = _FakeInstrumentConfig(
            sampling_period_fs=400_000,
            timeline_step_samples=64,
        )
        self._alias = alias
        self._value = value
        self._initialize_probe = initialize_probe
        self._apply_probe = apply_probe
        self._fetch_probe = fetch_probe
        self.apply_calls: list[object] = []

    async def apply(self, directive: object) -> None:
        self.apply_calls.append(directive)
        await self._apply_probe.step()

    async def initialize(self) -> None:
        await self._initialize_probe.step()

    async def wait_for_result(self) -> object:
        await self._fetch_probe.step()
        return _ParallelResultContainer(self._alias, self._value)


class _FakeSession:
    def __init__(self, *, session_id: str = "session-id") -> None:
        self.token = session_id
        self.trigger_calls: list[list[str]] = []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        del exc_type, exc, tb

    async def trigger(
        self,
        instrument_ids: list[str],
        wait_ms: int | None = None,
    ) -> int:
        del wait_ms
        self.trigger_calls.append(list(instrument_ids))
        return 0


class _FakeClient:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session
        self.exit_calls = 0

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        del exc_type, exc, tb
        self.exit_calls += 1

    def create_session(
        self,
        resource_ids: list[str],
        ttl_ms: int = 4_000,
        tentative_ttl_ms: int = 1_000,
    ) -> _FakeSession:
        del resource_ids, ttl_ms, tentative_ttl_ms
        return self._session


def _make_fake_execution_api(
    *,
    client_factory: Any,
    instrument_resolver_factory: Any,
    fixed_timeline_driver_factory: Any,
    capture_mode_namespace: Any = _FakeCaptureMode,
    sequencer_factory: Any = _FakeSequencer,
) -> Any:
    """Create one fake quelware API boundary for execution-manager tests."""
    return execution_manager_module._QuelwareExecutionApi(
        client_factory=client_factory,
        instrument_resolver_factory=instrument_resolver_factory,
        sequencer_factory=sequencer_factory,
        fixed_timeline_driver_factory=fixed_timeline_driver_factory,
        capture_mode_namespace=capture_mode_namespace,
        set_frequency_directive_factory=lambda *, hz: ("frequency", hz),
        set_capture_mode_directive_factory=lambda *, mode: ("capture_mode", mode),
    )


def test_execution_api_resolves_only_requested_capture_mode() -> None:
    """Given one capture mode, execution API resolves only the requested mode."""
    api = _make_fake_execution_api(
        client_factory=lambda endpoint, port: _FakeClient(_FakeSession()),
        instrument_resolver_factory=lambda: _FakeInstrumentResolver(alias_to_info={}),
        fixed_timeline_driver_factory=lambda _session, _instrument_info: (
            _FakeInstrumentDriver()
        ),
        capture_mode_namespace=SimpleNamespace(
            AVERAGED_VALUE=_FakeCaptureMode.AVERAGED_VALUE,
        ),
    )

    assert api.build_capture_mode_directive(Quel3CaptureMode.AVERAGED_VALUE) == (
        "capture_mode",
        _FakeCaptureMode.AVERAGED_VALUE,
    )


class _FlakyTriggerSession(_FakeSession):
    def __init__(
        self,
        *,
        fail_once: bool,
        session_id: str = "flaky-session-id",
        failed_session_id: str | None = None,
    ) -> None:
        super().__init__(session_id=session_id)
        self._fail_once = fail_once
        self._failed_session_id = failed_session_id
        self.exit_calls = 0

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        del exc_type, exc, tb
        self.exit_calls += 1

    async def trigger(
        self,
        instrument_ids: list[str],
        wait_ms: int | None = None,
    ) -> int:
        trigger_id = await super().trigger(instrument_ids, wait_ms=wait_ms)
        if self._fail_once:
            self._fail_once = False
            if self._failed_session_id is not None:
                self.token = self._failed_session_id
            raise RuntimeError("quelware request failed")
        return trigger_id


class _CloseFailingSession(_FakeSession):
    def __init__(
        self,
        *,
        fail_trigger: bool = False,
        session_id: str = "close-failing-session-id",
    ) -> None:
        super().__init__(session_id=session_id)
        self._fail_trigger = fail_trigger
        self.exit_calls = 0

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        del exc_type, exc, tb
        self.exit_calls += 1
        raise RuntimeError("quelware close failed")

    async def trigger(
        self,
        instrument_ids: list[str],
        wait_ms: int | None = None,
    ) -> int:
        trigger_id = await super().trigger(instrument_ids, wait_ms=wait_ms)
        if self._fail_trigger:
            raise RuntimeError("quelware request failed")
        return trigger_id


def test_execute_recreates_session_after_transient_request_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given transient quelware request failure, execute should retry with a new session."""
    caplog.set_level(
        logging.WARNING,
        logger="qubex.backend.quel3.managers.execution_manager",
    )
    payload = _make_payload()
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    resolver = _CountingInstrumentResolver(
        alias_to_info={
            "alias-rq00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
            )
        }
    )
    sessions = [
        _FlakyTriggerSession(
            fail_once=True,
            session_id="failed-trigger-session",
            failed_session_id="mutated-trigger-session",
        ),
        _FlakyTriggerSession(fail_once=False, session_id="retry-trigger-session"),
    ]
    clients: list[_FakeClient] = []
    drivers: list[_FakeInstrumentDriver] = []

    def _create_client(endpoint: str, port: int) -> _FakeClient:
        del endpoint, port
        client = _FakeClient(sessions[len(clients)])
        clients.append(client)
        return client

    def _create_driver(
        session: object, instrument_info: object
    ) -> _FakeInstrumentDriver:
        del session, instrument_info
        driver = _FakeInstrumentDriver()
        drivers.append(driver)
        return driver

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=_create_client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=_create_driver,
        ),
    )

    result = asyncio.run(
        manager.execute_async(request=BackendExecutionRequest(payload=payload))
    )

    assert len(clients) == 2
    assert resolver.refresh_calls == 2
    assert [client.exit_calls for client in clients] == [1, 1]
    assert [session.exit_calls for session in sessions] == [1, 1]
    assert sessions[0].trigger_calls == [["alias-rq00"]]
    assert sessions[1].trigger_calls == [["alias-rq00"]]
    assert "QuEL-3 quelware session request failed" in caplog.text
    assert "failed-trigger-session" in caplog.text
    assert "mutated-trigger-session" not in caplog.text
    assert "retry-trigger-session" not in caplog.text
    assert "attempt=1/4" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert len(drivers) == 2
    assert np.array_equal(
        result.data["alias-rq00"][0],
        np.array([1.0 + 0.0j], dtype=np.complex128),
    )


def test_execute_ignores_session_close_failure_after_success(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given request succeeds but close fails, execute should preserve the result."""
    caplog.set_level(
        logging.WARNING,
        logger="qubex.backend.quel3.managers.execution_manager",
    )
    payload = _make_payload()
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    resolver = _FakeInstrumentResolver(
        alias_to_info={
            "alias-rq00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
            )
        }
    )
    session = _CloseFailingSession(session_id="cleanup-failed-session")
    client = _FakeClient(session)

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=lambda endpoint, port: client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=lambda session, instrument_info: (
                _FakeInstrumentDriver()
            ),
        ),
    )

    result = asyncio.run(
        manager.execute_async(request=BackendExecutionRequest(payload=payload))
    )

    assert client.exit_calls == 1
    assert session.exit_calls == 1
    assert session.trigger_calls == [["alias-rq00"]]
    assert "QuEL-3 quelware session cleanup failed" in caplog.text
    assert "cleanup-failed-session" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert np.array_equal(
        result.data["alias-rq00"][0],
        np.array([1.0 + 0.0j], dtype=np.complex128),
    )


def test_execute_preserves_request_failure_when_session_close_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given request and close both fail, execute should preserve the request cause."""
    payload = _make_payload()
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    resolver = _FakeInstrumentResolver(
        alias_to_info={
            "alias-rq00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
            )
        }
    )
    expected_session_id = "close-failing-session-id"
    session = _CloseFailingSession(
        fail_trigger=True,
        session_id=expected_session_id,
    )
    client = _FakeClient(session)

    monkeypatch.setattr(
        execution_manager_module,
        "QUEL3_SESSION_REQUEST_MAX_ATTEMPTS",
        1,
    )
    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=lambda endpoint, port: client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=lambda session, instrument_info: (
                _FakeInstrumentDriver()
            ),
        ),
    )

    with pytest.raises(
        QuelwareSessionError,
        match=f"session_token={expected_session_id}",
    ) as exc_info:
        asyncio.run(
            manager.execute_async(request=BackendExecutionRequest(payload=payload))
        )

    assert exc_info.value.session_token == expected_session_id
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "quelware request failed"
    assert client.exit_calls == 1
    assert session.exit_calls == 1
    assert session.trigger_calls == [["alias-rq00"]]


def test_execute_uses_configured_session_request_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given configured retry limit, execute should stop after that many attempts."""
    payload = _make_payload()
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    resolver = _FakeInstrumentResolver(
        alias_to_info={
            "alias-rq00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
            )
        }
    )
    failed_session_ids = ("failed-trigger-session-1", "failed-trigger-session-2")
    sessions = [
        _FlakyTriggerSession(fail_once=True, session_id=failed_session_ids[0]),
        _FlakyTriggerSession(fail_once=True, session_id=failed_session_ids[1]),
    ]
    clients: list[_FakeClient] = []

    def _create_client(endpoint: str, port: int) -> _FakeClient:
        del endpoint, port
        client = _FakeClient(sessions[len(clients)])
        clients.append(client)
        return client

    monkeypatch.setattr(
        execution_manager_module,
        "QUEL3_SESSION_REQUEST_MAX_ATTEMPTS",
        2,
    )
    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=_create_client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=lambda session, instrument_info: (
                _FakeInstrumentDriver()
            ),
        ),
    )

    with pytest.raises(
        QuelwareSessionError,
        match=f"session_token={failed_session_ids[1]}",
    ) as exc_info:
        asyncio.run(
            manager.execute_async(request=BackendExecutionRequest(payload=payload))
        )

    assert exc_info.value.session_token == failed_session_ids[1]
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "quelware request failed"
    assert len(clients) == 2
    assert [client.exit_calls for client in clients] == [1, 1]
    assert [session.exit_calls for session in sessions] == [1, 1]
    assert [session.trigger_calls for session in sessions] == [
        [["alias-rq00"]],
        [["alias-rq00"]],
    ]


def test_execute_batch_retries_only_failed_payload_after_transient_request_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given one batch payload fails transiently, retry should resume from that payload."""
    payload = _make_payload()
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    resolver = _FakeInstrumentResolver(
        alias_to_info={
            "alias-rq00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
            )
        }
    )
    sessions = [
        _FlakyTriggerSession(fail_once=False, session_id="first-payload-session"),
        _FlakyTriggerSession(
            fail_once=True, session_id="failed-second-payload-session"
        ),
        _FlakyTriggerSession(
            fail_once=False, session_id="retry-second-payload-session"
        ),
    ]
    create_session_index = 0

    class _SequencedSessionClient(_FakeClient):
        def create_session(
            self,
            resource_ids: list[str],
            ttl_ms: int = 4_000,
            tentative_ttl_ms: int = 1_000,
        ) -> _FakeSession:
            nonlocal create_session_index
            del resource_ids, ttl_ms, tentative_ttl_ms
            session = sessions[create_session_index]
            create_session_index += 1
            return session

    clients: list[_SequencedSessionClient] = []

    def _create_client(endpoint: str, port: int) -> _FakeClient:
        del endpoint, port
        client = _SequencedSessionClient(_FakeSession())
        clients.append(client)
        return client

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=_create_client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=lambda session, instrument_info: (
                _FakeInstrumentDriver()
            ),
        ),
    )

    results = asyncio.run(
        manager.execute_batch_async(
            requests=(
                BackendExecutionRequest(payload=payload),
                BackendExecutionRequest(payload=payload),
            )
        )
    )

    assert len(results) == 2
    assert len(clients) == 2
    assert [client.exit_calls for client in clients] == [1, 1]
    assert [session.exit_calls for session in sessions] == [1, 1, 1]
    assert sessions[0].trigger_calls == [["alias-rq00"]]
    assert sessions[1].trigger_calls == [["alias-rq00"]]
    assert sessions[2].trigger_calls == [["alias-rq00"]]


def test_execute_batches_capture_mode_with_timeline_directive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given fixed-timeline execution, execute batches directives per instrument."""
    payload = _make_payload()
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    resolver = _FakeInstrumentResolver(
        alias_to_info={
            "alias-rq00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
            )
        }
    )
    driver = _FakeInstrumentDriver()
    session = _FakeSession()
    client = _FakeClient(session)

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=lambda endpoint, port: client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=lambda _session, _instrument_info: driver,
        ),
    )

    result = asyncio.run(
        manager.execute_async(request=BackendExecutionRequest(payload=payload))
    )

    assert driver.initialized is True
    assert driver.apply_calls == [
        [("capture_mode", _FakeCaptureMode.AVERAGED_VALUE), ("timeline", "alias-rq00")]
    ]
    assert session.trigger_calls == [["alias-rq00"]]
    assert np.array_equal(
        result.data["alias-rq00"][0],
        np.array([1.0 + 0.0j], dtype=np.complex128),
    )


def test_execute_batches_frequency_capture_mode_with_timeline_directive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given payload frequency, execute should apply frequency before capture mode and timeline."""
    payload = _make_payload(frequency_hz=6.25e9)
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    resolver = _FakeInstrumentResolver(
        alias_to_info={
            "alias-rq00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
            )
        }
    )
    driver = _FakeInstrumentDriver()
    session = _FakeSession()
    client = _FakeClient(session)

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=lambda endpoint, port: client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=lambda _session, _instrument_info: driver,
        ),
    )

    asyncio.run(manager.execute_async(request=BackendExecutionRequest(payload=payload)))

    assert driver.apply_calls == [
        [
            ("frequency", 6.25e9),
            ("capture_mode", _FakeCaptureMode.AVERAGED_VALUE),
            ("timeline", "alias-rq00"),
        ]
    ]


def test_execute_rejects_runtime_without_required_capture_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given missing runtime capture mode, execute raises RuntimeError."""
    payload = _make_payload()
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    resolver = _FakeInstrumentResolver(
        alias_to_info={
            "alias-rq00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
            )
        }
    )
    driver = _FakeInstrumentDriver()
    session = _FakeSession()
    client = _FakeClient(session)

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=lambda endpoint, port: client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=lambda _session, _instrument_info: driver,
            capture_mode_namespace=SimpleNamespace(
                RAW_WAVEFORMS=_FakeCaptureMode.RAW_WAVEFORMS,
                AVERAGED_WAVEFORM=_FakeCaptureMode.AVERAGED_WAVEFORM,
                VALUES_PER_ITER=_FakeCaptureMode.VALUES_PER_ITER,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="AVERAGED_VALUE"):
        asyncio.run(
            manager.execute_async(request=BackendExecutionRequest(payload=payload))
        )


def test_execute_parallelizes_driver_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given multiple instruments, execute should parallelize driver phases."""
    payload = _make_payload()
    payload = replace(
        payload,
        fixed_timelines={
            "alias-rq00": payload.fixed_timelines["alias-rq00"],
            "alias-rq01": payload.fixed_timelines["alias-rq00"],
        },
    )
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    resolver = _FakeInstrumentResolver(
        alias_to_info={
            "alias-rq00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
                alias="alias-rq00",
            ),
            "alias-rq01": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p01",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
                alias="alias-rq01",
            ),
        }
    )
    initialize_barrier = _PhaseBarrier(expected=2)
    apply_barrier = _PhaseBarrier(expected=2)
    fetch_barrier = _PhaseBarrier(expected=2)
    drivers = {
        "alias-rq00": _ParallelInstrumentDriver(
            alias="alias-rq00",
            value=1.0 + 0.0j,
            initialize_barrier=initialize_barrier,
            apply_barrier=apply_barrier,
            fetch_barrier=fetch_barrier,
        ),
        "alias-rq01": _ParallelInstrumentDriver(
            alias="alias-rq01",
            value=2.0 + 0.0j,
            initialize_barrier=initialize_barrier,
            apply_barrier=apply_barrier,
            fetch_barrier=fetch_barrier,
        ),
    }
    session = _FakeSession()
    client = _FakeClient(session)

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=lambda endpoint, port: client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=lambda _session, instrument_info: drivers[
                instrument_info.alias
            ],
        ),
    )

    result = asyncio.run(
        manager.execute_async(request=BackendExecutionRequest(payload=payload))
    )

    assert drivers["alias-rq00"].apply_calls == [
        [("capture_mode", _FakeCaptureMode.AVERAGED_VALUE), ("timeline", "alias-rq00")]
    ]
    assert drivers["alias-rq01"].apply_calls == [
        [("capture_mode", _FakeCaptureMode.AVERAGED_VALUE), ("timeline", "alias-rq01")]
    ]
    assert session.trigger_calls == [["alias-rq00", "alias-rq01"]]
    assert np.array_equal(
        result.data["alias-rq00"][0],
        np.array([1.0 + 0.0j], dtype=np.complex128),
    )
    assert np.array_equal(
        result.data["alias-rq01"][0],
        np.array([2.0 + 0.0j], dtype=np.complex128),
    )


def test_execute_batch_async_reopens_session_per_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given multiple requests, execute_batch_async should reopen each payload session."""
    payload_a = _make_payload()
    payload_b = _make_payload()
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=1,
    )
    resolver = _CountingInstrumentResolver(
        alias_to_info={
            "alias-rq00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
            )
        }
    )
    driver = _FakeInstrumentDriver()
    session = _FakeSession()
    create_session_calls: list[tuple[str, ...]] = []

    class _CountingClient(_FakeClient):
        def create_session(
            self,
            resource_ids: list[str],
            ttl_ms: int = 4_000,
            tentative_ttl_ms: int = 1_000,
        ) -> _FakeSession:
            create_session_calls.append(tuple(resource_ids))
            return super().create_session(
                resource_ids,
                ttl_ms=ttl_ms,
                tentative_ttl_ms=tentative_ttl_ms,
            )

    client = _CountingClient(session)

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=lambda endpoint, port: client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=lambda _session, _instrument_info: driver,
        ),
    )

    results = asyncio.run(
        manager.execute_batch_async(
            requests=[
                BackendExecutionRequest(payload=payload_a),
                BackendExecutionRequest(payload=payload_b),
            ]
        )
    )

    assert resolver.refresh_calls == 1
    assert create_session_calls == [("alias-rq00",), ("alias-rq00",)]
    assert len(session.trigger_calls) == 2
    assert len(driver.apply_calls) == 2
    assert len(results) == 2


def test_execute_async_reuses_resolver_until_invalidated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executions should reuse resolution while keeping resources call-scoped."""
    payload = _make_payload()
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=1,
    )
    resolvers: list[_CountingInstrumentResolver] = []
    clients: list[_FakeClient] = []
    sessions: list[_FakeSession] = []

    def _create_resolver() -> _CountingInstrumentResolver:
        resolver = _CountingInstrumentResolver(
            alias_to_info={
                "alias-rq00": _FakeInstrumentInfo(
                    port_id="quel3-02-a01:trx_p00",
                    definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
                )
            }
        )
        resolvers.append(resolver)
        return resolver

    def _create_client(endpoint: str, port: int) -> _FakeClient:
        del endpoint, port
        session = _FakeSession()
        sessions.append(session)
        client = _FakeClient(session)
        clients.append(client)
        return client

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=_create_client,
            instrument_resolver_factory=_create_resolver,
            fixed_timeline_driver_factory=lambda _session, _instrument_info: (
                _FakeInstrumentDriver()
            ),
        ),
    )

    async def _execute_around_invalidation() -> None:
        request = BackendExecutionRequest(payload=payload)
        await manager.execute_async(request=request)
        await manager.execute_async(request=request)
        manager.invalidate_instrument_resolver()
        await manager.execute_async(request=request)

    asyncio.run(_execute_around_invalidation())

    assert len(resolvers) == 2
    assert [resolver.refresh_calls for resolver in resolvers] == [1, 1]
    assert len(clients) == 3
    assert [client.exit_calls for client in clients] == [1, 1, 1]
    assert [session.trigger_calls for session in sessions] == [
        [["alias-rq00"]],
        [["alias-rq00"]],
        [["alias-rq00"]],
    ]


def test_execute_serializes_driver_phases_when_parallel_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given parallel disabled, execute should keep driver phases sequential."""
    payload = _make_payload()
    payload = replace(
        payload,
        fixed_timelines={
            "alias-rq00": payload.fixed_timelines["alias-rq00"],
            "alias-rq01": payload.fixed_timelines["alias-rq00"],
        },
    )
    manager = Quel3ExecutionManager(
        runtime_config=Quel3RuntimeConfig(),
        sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )
    resolver = _FakeInstrumentResolver(
        alias_to_info={
            "alias-rq00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
                alias="alias-rq00",
            ),
            "alias-rq01": _FakeInstrumentInfo(
                port_id="quel3-02-a01:trx_p01",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
                alias="alias-rq01",
            ),
        }
    )
    initialize_probe = _ConcurrencyProbe()
    apply_probe = _ConcurrencyProbe()
    fetch_probe = _ConcurrencyProbe()
    drivers = {
        "alias-rq00": _SerialProbeInstrumentDriver(
            alias="alias-rq00",
            value=1.0 + 0.0j,
            initialize_probe=initialize_probe,
            apply_probe=apply_probe,
            fetch_probe=fetch_probe,
        ),
        "alias-rq01": _SerialProbeInstrumentDriver(
            alias="alias-rq01",
            value=2.0 + 0.0j,
            initialize_probe=initialize_probe,
            apply_probe=apply_probe,
            fetch_probe=fetch_probe,
        ),
    }
    session = _FakeSession()
    client = _FakeClient(session)

    monkeypatch.setattr(
        manager,
        "_load_quelware_api",
        lambda: _make_fake_execution_api(
            client_factory=lambda endpoint, port: client,
            instrument_resolver_factory=lambda: resolver,
            fixed_timeline_driver_factory=lambda _session, instrument_info: drivers[
                instrument_info.alias
            ],
        ),
    )

    result = asyncio.run(
        manager.execute_async(
            request=BackendExecutionRequest(payload=payload),
            parallel=False,
        )
    )

    assert initialize_probe.max_active == 1
    assert apply_probe.max_active == 1
    assert fetch_probe.max_active == 1
    assert session.trigger_calls == [["alias-rq00", "alias-rq01"]]
    assert np.array_equal(
        result.data["alias-rq00"][0],
        np.array([1.0 + 0.0j], dtype=np.complex128),
    )
    assert np.array_equal(
        result.data["alias-rq01"][0],
        np.array([2.0 + 0.0j], dtype=np.complex128),
    )


def test_execute_sync_forwards_parallel_flag_to_execution_manager() -> None:
    """Given parallel override, controller execute_sync should forward it."""
    captured: dict[str, object] = {}

    def _execute_sync(*, request: object, parallel: bool) -> object:
        captured["request"] = request
        captured["parallel"] = parallel
        return "ok"

    controller = Quel3BackendController(
        execution_manager=cast(
            Any,
            SimpleNamespace(
                quelware_endpoint="localhost",
                quelware_port=50051,
                sampling_period_ns=0.4,
                client_mode="server",
                execute_sync=_execute_sync,
                execute_async=None,
            ),
        )
    )
    request = BackendExecutionRequest(payload=object())

    result = controller.execute_sync(request=request, parallel=False)

    assert result == "ok"
    assert captured == {"request": request, "parallel": False}
