# ruff: noqa: SLF001

"""Tests for QuEL-3 backend controller behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.backend import BackendExecutionRequest
from qubex.backend.backend_controller import BackendController
from qubex.backend.quel1 import Quel1BackendController
from qubex.backend.quel3 import (
    Quel3BackendController,
    Quel3BackendExecutionResult,
    Quel3CaptureMode,
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3Waveform,
    Quel3WaveformEvent,
)
from qubex.backend.quel3.managers.execution_manager import Quel3ExecutionManager


@dataclass(frozen=True)
class _FakeInstrumentDefinition:
    role: str


@dataclass(frozen=True)
class _FakeInstrumentInfo:
    port_id: str
    definition: _FakeInstrumentDefinition
    alias: str | None = None


class _FakeInstrumentResolver:
    def __init__(
        self,
        *,
        alias_to_info: dict[str, _FakeInstrumentInfo],
    ) -> None:
        self._alias_to_info = dict(alias_to_info)
        self._alias_to_id = {alias: alias for alias in self._alias_to_info}

    async def refresh(self, client: object) -> None:
        del client

    def resolve(self, aliases: list[str]) -> list[str]:
        return aliases

    def find_inst_info_by_alias(self, alias: str) -> _FakeInstrumentInfo:
        if alias not in self._alias_to_info:
            raise ValueError(alias)
        return self._alias_to_info[alias]


def _make_payload(*, mode: str = "avg", repeats: int = 2) -> Quel3ExecutionPayload:
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
    )
    return Quel3ExecutionPayload(
        waveform_library={
            waveform_name: Quel3Waveform(
                iq_array=np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
                sampling_period_ns=0.4,
            )
        },
        fixed_timelines={"alias-rq00": timeline},
        interval_ns=100.0,
        repeats=repeats,
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
    payload = _make_payload(mode="avg", repeats=2)
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
        sampling_period_ns=0.4,
        backend_sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )

    assert isinstance(result, Quel3BackendExecutionResult)
    assert isinstance(result.status, dict)
    assert "alias-rq00" in result.data
    assert np.array_equal(
        result.data["alias-rq00"][0],
        np.array([2.0 + 2.0j, 4.0 + 4.0j], dtype=np.complex128),
    )
    assert result.config["sampling_period_ns"] == pytest.approx(1.6)


def test_build_measurement_result_keeps_backend_alias_labels() -> None:
    """Given backend flow result, measurement labels remain instrument aliases."""
    payload = _make_payload(mode="single", repeats=1)
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
        sampling_period_ns=0.4,
        backend_sampling_period_ns=0.4,
        capture_decimation_factor=4,
    )

    assert isinstance(result, Quel3BackendExecutionResult)
    assert "alias-raw" in result.data


def test_resolve_timeline_iterations_uses_repeats_for_iterative_modes() -> None:
    """Given iterative capture mode, timeline iterations follow repeats."""
    iterations = Quel3ExecutionManager._resolve_timeline_iterations(
        capture_mode=Quel3CaptureMode.VALUES_PER_ITER,
        repeats=16,
    )
    assert iterations == 16


def test_resolve_timeline_iterations_uses_single_for_raw_waveforms() -> None:
    """Given raw waveform mode, timeline iterations stay at one."""
    iterations = Quel3ExecutionManager._resolve_timeline_iterations(
        capture_mode=Quel3CaptureMode.RAW_WAVEFORMS,
        repeats=16,
    )
    assert iterations == 1


def test_extract_capture_samples_from_waveform_result_container() -> None:
    """Given waveform-style iq_result, extraction returns latest waveform samples."""

    class _Waveform:
        def __init__(self, values: np.ndarray) -> None:
            self.iq_array = values

    class _Result:
        def __init__(self) -> None:
            self.iq_result = {
                "RQ00:0": [
                    _Waveform(np.array([1.0 + 0.0j], dtype=np.complex128)),
                    _Waveform(np.array([2.0 + 0.0j], dtype=np.complex128)),
                ]
            }

    values = Quel3ExecutionManager._extract_capture_samples(_Result(), "RQ00:0")

    assert values is not None
    assert np.array_equal(values, np.array([2.0 + 0.0j], dtype=np.complex128))


def test_extract_capture_samples_from_point_result_container() -> None:
    """Given point-style iq_result, extraction returns complex-point array."""

    class _Result:
        def __init__(self) -> None:
            self.iq_result = {
                "RQ00:0": [1.0 + 2.0j, 3.0 + 4.0j],
            }

    values = Quel3ExecutionManager._extract_capture_samples(_Result(), "RQ00:0")

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
    assert controller._connection_manager.quelware_endpoint == "localhost"
    assert controller._connection_manager.quelware_port == 50051


def test_constructor_accepts_standalone_runtime_options() -> None:
    """Given standalone runtime options, controller should propagate them to all managers."""
    controller = Quel3BackendController(
        client_mode="standalone",
        standalone_unit_label="quel3-02-a01",
    )

    assert controller.client_mode == "standalone"
    assert controller.standalone_unit_label == "quel3-02-a01"
    assert controller.connection_manager.client_mode == "standalone"
    assert controller.configuration_manager.client_mode == "standalone"
    assert controller.execution_manager.client_mode == "standalone"


def test_constructor_rejects_standalone_mode_without_unit_label() -> None:
    """Given standalone mode without unit label, controller construction should fail fast."""
    with pytest.raises(ValueError, match="standalone_unit_label"):
        Quel3BackendController(client_mode="standalone")


def test_constructor_accepts_injected_managers() -> None:
    """Given injected managers, controller should use those manager instances."""
    connection_manager = SimpleNamespace(
        hash=11,
        is_connected=True,
        quelware_endpoint="injected-host",
        quelware_port=61000,
        client_mode="standalone",
        standalone_unit_label="quel3-02-a01",
        connect=lambda box_names=None, parallel=None: None,
        disconnect=lambda: None,
    )
    configuration_manager = SimpleNamespace(
        quelware_endpoint="injected-host",
        quelware_port=61000,
        client_mode="standalone",
        standalone_unit_label="quel3-02-a01",
        target_alias_map={"Q00": "Q00"},
        last_deployed_instrument_infos={"Q00": (object(),)},
        deploy_instruments=lambda *, requests: {"Q00": tuple(requests)},
    )
    execution_manager = SimpleNamespace(
        quelware_endpoint="injected-host",
        quelware_port=61000,
        sampling_period_ns=0.8,
        client_mode="standalone",
        standalone_unit_label="quel3-02-a01",
        execute_sync=lambda *, request: request,
        execute_async=lambda *, request: request,
    )

    controller = Quel3BackendController(
        connection_manager=cast(Any, connection_manager),
        configuration_manager=cast(Any, configuration_manager),
        execution_manager=cast(Any, execution_manager),
    )

    assert controller.connection_manager is connection_manager
    assert controller.configuration_manager is configuration_manager
    assert controller.execution_manager is execution_manager
    assert controller.quelware_endpoint == "injected-host"
    assert controller.quelware_port == 61000
    assert controller.sampling_period_ns == pytest.approx(0.8)
    assert controller.client_mode == "standalone"
    assert controller.standalone_unit_label == "quel3-02-a01"


def test_connect_refreshes_existing_instrument_cache() -> None:
    """Given connect on QuEL-3, controller should refresh alias cache from existing instruments."""
    calls: list[str] = []
    connection_manager = SimpleNamespace(
        hash=11,
        is_connected=False,
        quelware_endpoint="host-a",
        quelware_port=50051,
        client_mode="server",
        standalone_unit_label=None,
        connect=lambda box_names=None, parallel=None: calls.append("connect"),
        disconnect=lambda: None,
    )
    configuration_manager = SimpleNamespace(
        quelware_endpoint="host-a",
        quelware_port=50051,
        client_mode="server",
        standalone_unit_label=None,
        target_alias_map={},
        last_deployed_instrument_infos={},
        refresh_instrument_cache=lambda: calls.append("refresh") or {},
        deploy_instruments=lambda *, requests: {},
    )

    controller = Quel3BackendController(
        connection_manager=cast(Any, connection_manager),
        configuration_manager=cast(Any, configuration_manager),
    )

    controller.connect(["A"])

    assert calls == ["connect", "refresh"]


def test_deploy_instruments_forwards_parallel_flag_to_configuration_manager() -> None:
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
                standalone_unit_label=None,
                target_alias_map={},
                last_deployed_instrument_infos={},
                deploy_instruments=_deploy_instruments,
            ),
        )
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


def test_constructor_rejects_mismatched_injected_manager_runtime_values() -> None:
    """Given mismatched injected manager runtime values, constructor should fail fast."""
    connection_manager = SimpleNamespace(
        hash=11,
        is_connected=False,
        quelware_endpoint="host-a",
        quelware_port=50051,
        connect=lambda box_names=None, parallel=None: None,
        disconnect=lambda: None,
    )
    configuration_manager = SimpleNamespace(
        quelware_endpoint="host-b",
        quelware_port=50051,
        client_mode="server",
        standalone_unit_label=None,
        target_alias_map={},
        last_deployed_instrument_infos={},
        deploy_instruments=lambda *, requests: {},
    )

    with pytest.raises(ValueError, match="quelware_endpoint"):
        Quel3BackendController(
            connection_manager=cast(Any, connection_manager),
            configuration_manager=cast(Any, configuration_manager),
        )


def test_constructor_rejects_mismatched_injected_client_runtime_values() -> None:
    """Given mismatched client runtime values, controller construction should fail fast."""
    connection_manager = SimpleNamespace(
        hash=11,
        is_connected=False,
        quelware_endpoint="host-a",
        quelware_port=50051,
        client_mode="server",
        standalone_unit_label=None,
        connect=lambda box_names=None, parallel=None: None,
        disconnect=lambda: None,
    )
    configuration_manager = SimpleNamespace(
        quelware_endpoint="host-a",
        quelware_port=50051,
        client_mode="standalone",
        standalone_unit_label="quel3-02-a01",
        target_alias_map={},
        last_deployed_instrument_infos={},
        deploy_instruments=lambda *, requests: {},
    )

    with pytest.raises(ValueError, match="client_mode"):
        Quel3BackendController(
            connection_manager=cast(Any, connection_manager),
            configuration_manager=cast(Any, configuration_manager),
        )


def test_resolve_payload_merges_targets_mapped_to_one_alias() -> None:
    """Given shared alias bindings, resolved payload merges timelines per alias."""
    payload = _make_payload()
    payload = replace(
        payload,
        fixed_timelines={
            "RQ00": payload.fixed_timelines["alias-rq00"],
            "RQ01": payload.fixed_timelines["alias-rq00"],
        },
        instrument_bindings={
            "RQ00": "alias:alias-shared",
            "RQ01": "alias:alias-shared",
        },
    )
    resolver = _FakeInstrumentResolver(
        alias_to_info={
            "alias-shared": _FakeInstrumentInfo(
                port_id="unit-a:trx_p00",
                definition=_FakeInstrumentDefinition(role="TRANSCEIVER"),
            )
        }
    )

    resolved = Quel3ExecutionManager._resolve_payload(
        payload=payload,
        resolver=cast(Any, resolver),
    )

    assert set(resolved.fixed_timelines.keys()) == {"alias-shared"}
    timeline = resolved.fixed_timelines["alias-shared"]
    assert [window.name for window in timeline.capture_windows] == [
        "alias-shared:0",
        "alias-shared:1",
    ]


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
    resolver = _FakeInstrumentResolver(alias_to_info={})

    with pytest.raises(ValueError, match="Unsupported instrument binding"):
        Quel3ExecutionManager._resolve_payload(
            payload=payload,
            resolver=cast(Any, resolver),
        )


def test_resolve_payload_accepts_alias_binding() -> None:
    """Given alias binding, resolving payload keeps the resolved alias."""
    payload = _make_payload()
    payload = replace(
        payload,
        fixed_timelines={"Q00": payload.fixed_timelines["alias-rq00"]},
        instrument_bindings={"Q00": "alias:inst-q00"},
    )
    resolver = _FakeInstrumentResolver(
        alias_to_info={
            "inst-q00": _FakeInstrumentInfo(
                port_id="quel3-02-a01:tx_p04",
                definition=_FakeInstrumentDefinition(role="TRANSMITTER"),
            )
        }
    )

    resolved = Quel3ExecutionManager._resolve_payload(
        payload=payload,
        resolver=cast(Any, resolver),
    )

    assert set(resolved.fixed_timelines.keys()) == {"inst-q00"}


@dataclass(frozen=True)
class _FakeInstrumentConfig:
    sampling_period_fs: int
    timeline_step_samples: int


class _FakeWaveformResult:
    def __init__(self, values: np.ndarray) -> None:
        self.iq_array = values


class _FakeResultContainer:
    def __init__(self) -> None:
        self.iq_result = {
            "alias-rq00:0": [
                _FakeWaveformResult(np.array([1.0 + 0.0j], dtype=np.complex128))
            ]
        }


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

    async def fetch_result(self) -> object:
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
        self.iq_result = {
            f"{alias}:0": [_FakeWaveformResult(np.array([value], dtype=np.complex128))]
        }


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

    async def fetch_result(self) -> object:
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

    async def fetch_result(self) -> object:
        await self._fetch_probe.step()
        return _ParallelResultContainer(self._alias, self._value)


class _FakeSession:
    def __init__(self) -> None:
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

    async def trigger(self, instrument_ids: list[str]) -> None:
        self.trigger_calls.append(list(instrument_ids))


class _FakeClient:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        del exc_type, exc, tb

    def create_session(self, resource_ids: list[str]) -> _FakeSession:
        del resource_ids
        return self._session


def test_execute_batches_capture_mode_with_timeline_directive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given fixed-timeline execution, execute batches directives per instrument."""
    payload = _make_payload()
    manager = Quel3ExecutionManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
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
        lambda: (
            lambda endpoint, port: client,
            lambda: resolver,
            _FakeSequencer,
            lambda _session, _instrument_info: driver,
            SimpleNamespace(AVERAGED_VALUE="avg"),
            lambda *, mode: ("capture_mode", mode),
        ),
    )

    result = asyncio.run(
        manager.execute_async(request=BackendExecutionRequest(payload=payload))
    )

    assert driver.initialized is True
    assert driver.apply_calls == [[("capture_mode", "avg"), ("timeline", "alias-rq00")]]
    assert session.trigger_calls == [["alias-rq00"]]
    assert np.array_equal(
        result.data["alias-rq00"][0],
        np.array([1.0 + 0.0j], dtype=np.complex128),
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
        quelware_endpoint="localhost",
        quelware_port=50051,
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
        lambda: (
            lambda endpoint, port: client,
            lambda: resolver,
            _FakeSequencer,
            lambda _session, instrument_info: drivers[instrument_info.alias],
            SimpleNamespace(AVERAGED_VALUE="avg"),
            lambda *, mode: ("capture_mode", mode),
        ),
    )

    result = asyncio.run(
        manager.execute_async(request=BackendExecutionRequest(payload=payload))
    )

    assert drivers["alias-rq00"].apply_calls == [
        [("capture_mode", "avg"), ("timeline", "alias-rq00")]
    ]
    assert drivers["alias-rq01"].apply_calls == [
        [("capture_mode", "avg"), ("timeline", "alias-rq01")]
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
        quelware_endpoint="localhost",
        quelware_port=50051,
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
        lambda: (
            lambda endpoint, port: client,
            lambda: resolver,
            _FakeSequencer,
            lambda _session, instrument_info: drivers[instrument_info.alias],
            SimpleNamespace(AVERAGED_VALUE="avg"),
            lambda *, mode: ("capture_mode", mode),
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
                standalone_unit_label=None,
                execute_sync=_execute_sync,
                execute_async=None,
            ),
        )
    )
    request = BackendExecutionRequest(payload=object())

    result = controller.execute_sync(request=request, parallel=False)

    assert result == "ok"
    assert captured == {"request": request, "parallel": False}
