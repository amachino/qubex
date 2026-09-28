# ruff: noqa: SLF001

"""Tests for QuEL-3 monitor configuration and IQ capture."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import Arbitrary, Blank, PhaseShift, PulseSchedule

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel3 import (
    Quel3BackendController,
    Quel3BackendExecutionResult,
    Quel3CaptureMode,
)
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol
from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
    Quel3HardwareStateReader,
)
from qubex.backend.quel3.models import InstrumentConfiguration, InstrumentSpec
from qubex.backend.quel3.services import Quel3MonitorService


class _MonitorClient:
    def __init__(self) -> None:
        self.controls = {"quel3.monitor.mode": "disabled"}
        self.allowed = ("disabled", "loopback")
        self.session_resources: tuple[str, ...] = ()
        self.configured: list[tuple[str, dict[str, str]]] = []

    async def __aenter__(self) -> _MonitorClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def list_unit_labels(self) -> list[str]:
        return ["unit-a", "unit-b"]

    async def list_resource_infos(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(id="unit-a:tx_p00", category="PORT"),
            SimpleNamespace(id="unit-a:mon", category="PORT"),
            SimpleNamespace(id="unit-b:tx_p00", category="PORT"),
        ]

    async def get_unit_configuration(self, unit_label: str) -> SimpleNamespace:
        assert unit_label == "unit-a"
        return SimpleNamespace(
            supported=(
                SimpleNamespace(
                    key="quel3.monitor.mode",
                    allowed_values=self.allowed,
                    current_value=self.controls["quel3.monitor.mode"],
                ),
            )
        )

    def create_session(self, resources: tuple[str, ...]) -> _MonitorClient:
        self.session_resources = tuple(resources)
        return self

    async def configure_unit(
        self, unit_label: str, controls: dict[str, str]
    ) -> dict[str, str]:
        self.configured.append((unit_label, controls))
        self.controls.update(controls)
        return dict(self.controls)


@pytest.fixture
def monitor_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Quel3BackendController, _MonitorClient]:
    """Provide a controller backed by a monitor-capable fake quelware client."""
    client = _MonitorClient()
    manager = Quel3ConfigurationManager()
    monkeypatch.setattr(
        manager, "_load_quelware_client_factory", lambda: lambda *args: client
    )
    return Quel3BackendController(configuration_manager=manager), client


def test_configure_monitor_mode_locks_every_unit_port(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """Monitor configuration should lock every port and return the applied mode."""
    controller, client = monitor_runtime

    value = controller.configuration_manager.configure_monitor_mode(
        unit_label="unit-a", mode="loopback"
    )

    assert value == "loopback"
    assert client.session_resources == ("unit-a:tx_p00", "unit-a:mon")
    assert client.configured == [("unit-a", {"quel3.monitor.mode": "loopback"})]


def test_configure_monitor_mode_rejects_unsupported_value(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """Unsupported monitor values should fail before changing hardware."""
    controller, client = monitor_runtime

    with pytest.raises(ValueError, match="allowed values"):
        controller.configuration_manager.configure_monitor_mode(
            unit_label="unit-a", mode="unknown"
        )

    assert client.configured == []


def test_configure_monitor_mode_requires_instruments_cleared(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """Cached instruments should prompt the caller to clear the unit first."""
    controller, client = monitor_runtime
    info = cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id="unit-a:instrument",
            port_id="unit-a:tx_p00",
            definition=SimpleNamespace(alias="output"),
        ),
    )
    controller._instrument_cache.replace_all(instrument_infos=(info,))

    with pytest.raises(RuntimeError, match="clear_instruments"):
        _service_for_controller(controller).configure_mode(
            unit_label="unit-a", mode="loopback"
        )

    assert client.configured == []


def test_get_monitor_mode_reads_live_unit_control(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """The current mode should be read from the unit before instrument changes."""
    controller, client = monitor_runtime
    client.controls["quel3.monitor.mode"] = "loopback"

    mode = controller.configuration_manager.get_monitor_mode(unit_label="unit-a")

    assert mode == "loopback"
    assert client.configured == []


@dataclass
class _MonitorExecutionManager:
    request: BackendExecutionRequest | None = None
    requests: list[BackendExecutionRequest] | None = None
    sampling_period_ns: float = 0.4

    def execute_sync(
        self, *, request: BackendExecutionRequest, **kwargs: object
    ) -> Quel3BackendExecutionResult:
        self.request = request
        if self.requests is None:
            self.requests = []
        self.requests.append(request)
        monitor_alias = next(
            alias
            for alias, timeline in request.payload.fixed_timelines.items()
            if timeline.capture_windows
        )
        return Quel3BackendExecutionResult(
            status={},
            data={monitor_alias: [np.array([[len(self.requests) + 2j, 3 + 4j]])]},
            config={"sampling_period_ns": 0.4},
        )


def _instrument_info(
    alias: str,
    port: str,
    *,
    frequency_range_min_hz: float = 4e9,
    frequency_range_max_hz: float = 6e9,
) -> InstrumentInfoProtocol:
    """Create one deployable hardware snapshot entry for monitor tests."""
    return cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id=f"unit-a:{alias}",
            port_id=f"unit-a:{port}",
            definition=SimpleNamespace(
                alias=alias,
                mode="FIXED_TIMELINE",
                role="TRANSMITTER",
                profile=SimpleNamespace(
                    frequency_range_min=frequency_range_min_hz,
                    frequency_range_max=frequency_range_max_hz,
                ),
            ),
        ),
    )


@pytest.fixture
def monitor_schedule_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]]:
    """Emulate one unit's instrument writes while retaining call order."""
    manager = _MonitorExecutionManager()
    controller = Quel3BackendController(execution_manager=cast(Any, manager))
    originals = (
        _instrument_info("output-a", "tx_p00"),
        _instrument_info("output-b", "tx_p01"),
        _instrument_info("idle", "tx_p02"),
    )
    actions: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        controller._hardware_state_reader,
        "read_instrument_infos",
        lambda **kwargs: originals,
    )
    monkeypatch.setattr(
        controller._configuration_manager,
        "get_monitor_mode",
        lambda **kwargs: "disabled",
        raising=False,
    )

    def clear(
        *, unit_label: str, instrument_cache: InstrumentCache, parallel: bool = True
    ) -> None:
        actions.append(("clear", unit_label))
        instrument_cache.replace_units(unit_labels=(unit_label,), instrument_infos=())

    def configure(*, unit_label: str, mode: str = "loopback") -> str:
        actions.append(("mode", mode))
        return mode

    def deploy(
        *,
        instrument: InstrumentSpec,
        instrument_cache: InstrumentCache,
        hardware_state_reader: object,
        append: bool = True,
        parallel: bool = True,
    ) -> InstrumentInfoProtocol:
        actions.append(
            (
                "deploy",
                instrument.alias,
                instrument.port_id,
                instrument.role,
                instrument.frequency_range_min_hz,
                instrument.frequency_range_max_hz,
            )
        )
        info = _instrument_info(instrument.alias, instrument.port_id.partition(":")[2])
        instrument_cache.replace_ports(
            port_ids=(instrument.port_id,), instrument_infos=(info,)
        )
        return info

    def restore(
        *,
        configuration: InstrumentConfiguration,
        instrument_cache: InstrumentCache,
        hardware_state_reader: object,
        parallel: bool = True,
    ) -> dict[str, InstrumentInfoProtocol]:
        actions.append(
            ("restore", tuple(spec.alias for spec in configuration.instruments))
        )
        instrument_cache.replace_units(
            unit_labels=("unit-a",), instrument_infos=originals
        )
        return {info.definition.alias: info for info in originals}

    monkeypatch.setattr(controller.configuration_manager, "clear_instruments", clear)
    monkeypatch.setattr(
        controller.configuration_manager, "configure_monitor_mode", configure
    )
    monkeypatch.setattr(controller.configuration_manager, "deploy_instrument", deploy)
    monkeypatch.setattr(controller.configuration_manager, "deploy_instruments", restore)
    return controller, manager, actions


def test_run_monitor_iq_returns_raw_capture_and_builds_two_timelines() -> None:
    """A monitor run should send IQ on the output and return raw monitor samples."""
    manager = _MonitorExecutionManager()
    controller = Quel3BackendController(execution_manager=cast(Any, manager))
    infos = tuple(
        cast(
            InstrumentInfoProtocol,
            SimpleNamespace(
                id=f"unit-a:{alias}",
                port_id=f"unit-a:{port}",
                definition=SimpleNamespace(alias=alias),
            ),
        )
        for alias, port in (("output", "tx_p00"), ("monitor", "mon"))
    )
    controller._instrument_cache.replace_all(instrument_infos=infos)

    iq = _service_for_controller(controller).run_iq(
        output_alias="output",
        monitor_alias="monitor",
        waveform=np.array([0.25 + 0.5j, 0.5 + 0.25j]),
    )

    assert np.array_equal(iq, np.array([[1 + 2j, 3 + 4j]]))
    assert manager.request is not None
    payload = manager.request.payload
    assert payload.capture_mode is Quel3CaptureMode.RAW_WAVEFORMS
    assert np.array_equal(
        payload.waveform_library["monitor_output"].iq_array, [0.25 + 0.5j, 0.5 + 0.25j]
    )
    assert len(payload.fixed_timelines["output"].events) == 1
    assert len(payload.fixed_timelines["monitor"].capture_windows) == 1
    assert payload.fixed_timelines["monitor"].capture_windows[
        0
    ].length_ns == pytest.approx(0.8)


def test_run_monitor_iq_requires_monitor_port_alias() -> None:
    """A non-monitor receiver alias should fail before execution."""
    manager = _MonitorExecutionManager()
    controller = Quel3BackendController(execution_manager=cast(Any, manager))
    infos = tuple(
        cast(
            InstrumentInfoProtocol,
            SimpleNamespace(
                id=f"unit-a:{alias}",
                port_id=f"unit-a:{port}",
                definition=SimpleNamespace(alias=alias),
            ),
        )
        for alias, port in (("output", "tx_p00"), ("receiver", "rx_p00"))
    )
    controller._instrument_cache.replace_all(instrument_infos=infos)

    with pytest.raises(ValueError, match="monitor port"):
        _service_for_controller(controller).run_iq(
            output_alias="output",
            monitor_alias="receiver",
            waveform=np.array([0.5 + 0j]),
        )

    assert manager.request is None


def test_run_monitor_iq_rejects_same_output_and_monitor_alias() -> None:
    """A receiver on the monitor port cannot also be the output instrument."""
    controller = Quel3BackendController()
    controller._instrument_cache.replace_all(
        instrument_infos=(
            cast(
                InstrumentInfoProtocol,
                SimpleNamespace(
                    id="unit-a:monitor",
                    port_id="unit-a:mon",
                    definition=SimpleNamespace(alias="monitor"),
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="distinct"):
        _service_for_controller(controller).run_iq(
            output_alias="monitor",
            monitor_alias="monitor",
            waveform=np.array([0.5 + 0j]),
        )


def test_run_monitor_iq_raises_when_capture_is_empty() -> None:
    """An empty capture should raise instead of masquerading as usable IQ."""
    manager = _MonitorExecutionManager()
    manager.execute_sync = cast(
        Any,
        lambda **kwargs: Quel3BackendExecutionResult(
            status={},
            data={"monitor": [np.array([], dtype=np.complex128)]},
            config={"sampling_period_ns": 0.4},
        ),
    )
    controller = Quel3BackendController(execution_manager=cast(Any, manager))
    controller._instrument_cache.replace_all(
        instrument_infos=tuple(
            cast(
                InstrumentInfoProtocol,
                SimpleNamespace(
                    id=f"unit-a:{alias}",
                    port_id=f"unit-a:{port}",
                    definition=SimpleNamespace(alias=alias),
                ),
            )
            for alias, port in (("output", "tx_p00"), ("monitor", "mon"))
        )
    )

    with pytest.raises(RuntimeError, match="no IQ data"):
        _service_for_controller(controller).run_iq(
            output_alias="output",
            monitor_alias="monitor",
            waveform=np.array([0.5 + 0j]),
        )


def test_run_monitor_schedule_builds_sparse_events_and_reuses_shape(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """A schedule should preserve pulse timing and modifiers without sampling blanks."""
    controller, manager, actions = monitor_schedule_runtime
    with PulseSchedule() as schedule:
        schedule.add(
            "drive",
            Arbitrary(
                [0.25 + 0.5j, 0.5 + 0.25j],
                sampling_period=0.4,
                scale=0.5,
                phase=np.deg2rad(30),
            ),
        )
        schedule.add("drive", Blank(0.8, sampling_period=0.4))
        schedule.add("drive", PhaseShift(np.pi / 2))
        schedule.add(
            "drive",
            Arbitrary([0.25 + 0.5j, 0.5 + 0.25j], sampling_period=0.4, scale=0.7),
        )
    schedule.set_frequency("drive", 5.0)

    iq = _service_for_controller(controller).run_schedule(
        unit_label="unit-a",
        pulse_schedule=schedule,
        output_alias="output-a",
        monitor_alias="monitor",
    )

    assert np.array_equal(iq["drive"], [[1 + 2j, 3 + 4j]])
    assert manager.request is not None
    payload = manager.request.payload
    assert len(payload.waveform_library) == 1
    assert np.array_equal(
        next(iter(payload.waveform_library.values())).iq_array,
        [0.25 + 0.5j, 0.5 + 0.25j],
    )
    events = payload.fixed_timelines["output-a"].events
    assert len(events) == 2
    assert events[0].waveform_name == events[1].waveform_name
    assert [event.start_offset_ns for event in events] == pytest.approx([0, 1.6])
    assert [event.gain for event in events] == pytest.approx([0.5, 0.7])
    assert [event.phase_offset_deg for event in events] == pytest.approx([30, 90])
    assert payload.fixed_timelines["output-a"].frequency_hz == pytest.approx(5e9)
    assert payload.fixed_timelines["monitor"].frequency_hz == pytest.approx(5e9)
    assert payload.fixed_timelines["monitor"].capture_windows[0].length_ns == (
        pytest.approx(2.4)
    )
    assert actions == [
        ("clear", "unit-a"),
        ("mode", "loopback"),
        ("deploy", "output-a", "unit-a:tx_p00", "TRANSMITTER", 4e9, 6e9),
        ("deploy", "monitor", "unit-a:mon", "RECEIVER", 4e9, 6e9),
        ("clear", "unit-a"),
        ("mode", "disabled"),
        ("restore", ("idle", "output-a", "output-b")),
    ]
    assert {
        spec.alias for spec in controller.get_instrument_configuration().instruments
    } == {"output-a", "output-b", "idle"}


def test_run_monitor_schedule_executes_multiple_output_channels(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """Each target should capture on the monitor port with its own frequency."""
    controller, manager, actions = monitor_schedule_runtime
    with PulseSchedule() as schedule:
        schedule.add("drive-a", Arbitrary([0.25 + 0j], sampling_period=0.4))
        schedule.add("drive-b", Arbitrary([0.5 + 0j], sampling_period=0.4))
    schedule.set_frequency("drive-b", 5.5)

    captured = _service_for_controller(controller).run_schedule(
        unit_label="unit-a",
        pulse_schedule=schedule,
        output_aliases={"drive-a": "output-a", "drive-b": "output-b"},
        monitor_alias="monitor",
    )

    assert set(captured) == {"drive-a", "drive-b"}
    assert captured["drive-a"][0, 0] == 1 + 2j
    assert captured["drive-b"][0, 0] == 2 + 2j
    assert manager.requests is not None
    assert len(manager.requests) == 2
    for request, output, frequency in zip(
        manager.requests, ("output-a", "output-b"), (5e9, 5.5e9), strict=True
    ):
        timelines = request.payload.fixed_timelines
        assert set(timelines) == {output, "monitor"}
        assert timelines[output].frequency_hz == pytest.approx(frequency)
        assert timelines["monitor"].frequency_hz == pytest.approx(frequency)
    assert [entry[1] for entry in actions if entry[0] == "deploy"] == [
        "output-a",
        "monitor",
        "output-b",
        "monitor",
    ]


def test_run_monitor_schedule_restores_instruments_after_execution_failure(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """Execution failure should still restore the original mode and instruments."""
    controller, manager, actions = monitor_schedule_runtime
    manager.execute_sync = cast(
        Any, lambda **kwargs: (_ for _ in ()).throw(RuntimeError("execute failed"))
    )
    with PulseSchedule() as schedule:
        schedule.add("drive", Arbitrary([1 + 0j], sampling_period=0.4))
    schedule.set_frequency("drive", 5.0)

    with pytest.raises(RuntimeError, match="execute failed"):
        _service_for_controller(controller).run_schedule(
            unit_label="unit-a",
            pulse_schedule=schedule,
            output_alias="output-a",
            monitor_alias="monitor",
        )

    assert actions[-3:] == [
        ("clear", "unit-a"),
        ("mode", "disabled"),
        ("restore", ("idle", "output-a", "output-b")),
    ]


def test_run_monitor_schedule_restores_instruments_after_deployment_failure(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deployment failure should still clear temporary resources and restore state."""
    controller, _, actions = monitor_schedule_runtime
    original_deploy = controller.configuration_manager.deploy_instrument

    def fail_monitor_deploy(
        *,
        instrument: InstrumentSpec,
        instrument_cache: InstrumentCache,
        hardware_state_reader: object,
        append: bool = True,
        parallel: bool = True,
    ) -> InstrumentInfoProtocol:
        if instrument.port_id == "unit-a:mon":
            raise RuntimeError("monitor deployment failed")
        return original_deploy(
            instrument=instrument,
            instrument_cache=instrument_cache,
            hardware_state_reader=cast(Any, hardware_state_reader),
            append=append,
            parallel=parallel,
        )

    monkeypatch.setattr(
        controller.configuration_manager, "deploy_instrument", fail_monitor_deploy
    )
    with PulseSchedule() as schedule:
        schedule.add("drive", Arbitrary([1 + 0j], sampling_period=0.4))
    schedule.set_frequency("drive", 5.0)

    with pytest.raises(RuntimeError, match="monitor deployment failed"):
        _service_for_controller(controller).run_schedule(
            unit_label="unit-a",
            pulse_schedule=schedule,
            output_alias="output-a",
            monitor_alias="monitor",
        )

    assert actions[-3:] == [
        ("clear", "unit-a"),
        ("mode", "disabled"),
        ("restore", ("idle", "output-a", "output-b")),
    ]


@pytest.mark.parametrize(
    ("label", "alias_options"),
    [
        ("output-a", {}),
        ("drive", {"output_alias": "output-a"}),
        ("drive", {"output_aliases": {"drive": "output-a"}}),
    ],
)
def test_run_monitor_schedule_defaults_to_live_instrument_center_frequency(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
    label: str,
    alias_options: dict[str, Any],
) -> None:
    """Missing frequency should use the mapped live instrument's range center."""
    controller, manager, _ = monitor_schedule_runtime
    controller._instrument_cache.replace_all(
        instrument_infos=(
            _instrument_info(
                "output-a",
                "tx_p00",
                frequency_range_min_hz=2e9,
                frequency_range_max_hz=3e9,
            ),
        )
    )
    with PulseSchedule() as schedule:
        schedule.add(label, Arbitrary([1 + 0j], sampling_period=0.4))

    captured = _service_for_controller(controller).run_schedule(
        unit_label="unit-a", pulse_schedule=schedule, **alias_options
    )

    assert np.array_equal(captured[label], [[1 + 2j, 3 + 4j]])
    assert manager.request is not None
    timelines = manager.request.payload.fixed_timelines
    assert timelines["output-a"].frequency_hz == pytest.approx(5e9)
    assert timelines["monitor"].frequency_hz == pytest.approx(5e9)
    assert schedule.get_frequency(label) is None


@pytest.mark.parametrize("frequency_ghz", [float("nan"), float("inf"), -float("inf")])
def test_run_monitor_schedule_rejects_nonfinite_frequency_before_deletion(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
    frequency_ghz: float,
) -> None:
    """An explicit nonfinite frequency should leave hardware untouched."""
    controller, _, actions = monitor_schedule_runtime
    with PulseSchedule() as schedule:
        schedule.add("drive", Arbitrary([1 + 0j], sampling_period=0.4))
    schedule.set_frequency("drive", frequency_ghz)

    with pytest.raises(ValueError, match="frequency"):
        _service_for_controller(controller).run_schedule(
            unit_label="unit-a",
            pulse_schedule=schedule,
            output_alias="output-a",
            monitor_alias="monitor",
        )

    assert actions == []


def test_run_monitor_schedule_rejects_missing_target_before_deletion(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """An unknown output alias should leave the unit's instruments untouched."""
    controller, _, actions = monitor_schedule_runtime
    with PulseSchedule() as schedule:
        schedule.add("drive", Arbitrary([1 + 0j], sampling_period=0.4))
    schedule.set_frequency("drive", 5.0)

    with pytest.raises(ValueError, match="has no instrument"):
        _service_for_controller(controller).run_schedule(
            unit_label="unit-a",
            pulse_schedule=schedule,
            output_alias="missing",
            monitor_alias="monitor",
        )

    assert actions == []


def test_run_monitor_schedule_rejects_invalid_capture_before_deletion(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """Invalid capture settings should not trigger instrument deletion."""
    controller, _, actions = monitor_schedule_runtime
    with PulseSchedule() as schedule:
        schedule.add("drive", Arbitrary([1 + 0j], sampling_period=0.4))
    schedule.set_frequency("drive", 5.0)

    with pytest.raises(ValueError, match="n_iterations"):
        _service_for_controller(controller).run_schedule(
            unit_label="unit-a",
            pulse_schedule=schedule,
            output_alias="output-a",
            n_iterations=0,
        )

    assert actions == []


def test_monitor_service_runs_iq_with_shared_cache() -> None:
    """The monitor service should execute IQ directly using its supplied cache."""
    cache = InstrumentCache()
    cache.replace_all(
        instrument_infos=(
            _instrument_info("output", "tx_p00"),
            _instrument_info("monitor", "mon"),
        )
    )
    manager = _MonitorExecutionManager()
    service = Quel3MonitorService(
        configuration_manager=Quel3ConfigurationManager(),
        execution_manager=cast(Any, manager),
        hardware_state_reader=Quel3HardwareStateReader(),
        instrument_cache=cache,
    )

    captured = service.run_iq(
        output_alias="output", monitor_alias="monitor", waveform=[0.5 + 0j]
    )

    assert np.array_equal(captured, [[1 + 2j, 3 + 4j]])
    assert manager.request is not None
    assert set(manager.request.payload.fixed_timelines) == {"output", "monitor"}


def test_monitor_service_runs_schedule_without_controller_dependency(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """A standalone monitor service should capture and restore the supplied unit."""
    controller, manager, actions = monitor_schedule_runtime
    service = Quel3MonitorService(
        configuration_manager=controller.configuration_manager,
        execution_manager=cast(Any, manager),
        hardware_state_reader=controller.hardware_state_reader,
        instrument_cache=controller._instrument_cache,
    )
    with PulseSchedule() as schedule:
        schedule.add("output-a", Arbitrary([1 + 0j], sampling_period=0.4))

    captured = service.run_schedule(unit_label="unit-a", pulse_schedule=schedule)

    assert np.array_equal(captured["output-a"], [[1 + 2j, 3 + 4j]])
    assert actions[-3:] == [
        ("clear", "unit-a"),
        ("mode", "disabled"),
        ("restore", ("idle", "output-a", "output-b")),
    ]


def _service_for_controller(controller: Quel3BackendController) -> Quel3MonitorService:
    """Bind a standalone service to the test controller's supplied components."""
    return Quel3MonitorService(
        configuration_manager=controller.configuration_manager,
        execution_manager=controller.execution_manager,
        hardware_state_reader=controller.hardware_state_reader,
        instrument_cache=controller._instrument_cache,
    )
