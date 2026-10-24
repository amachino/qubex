# ruff: noqa: SLF001

"""Tests for QuEL-3 monitor configuration and IQ capture."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, cast

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
)
from qubex.backend.quel3.models import (
    InstrumentConfiguration,
    InstrumentRoleName,
    InstrumentSpec,
)
from qubex.backend.quel3.tools import Quel3MonitorTool


class _MonitorClient:
    def __init__(self) -> None:
        self.controls = {"quel3.monitor.mode": "open"}
        self.allowed: tuple[str, ...] = ("open", "loopback")
        self.session_resources: tuple[str, ...] = ()
        self.session_count = 0
        self.configured: list[tuple[str, dict[str, str]]] = []
        self.live_instruments: dict[str, set[str]] = {}
        self.discarded: list[str] = []
        self.operations: list[tuple[str, str]] = []

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
        self.session_count += 1
        self.session_resources = tuple(resources)
        return self

    async def discard_instruments(self, port_id: str) -> None:
        self.discarded.append(port_id)
        self.operations.append(("discard", port_id))
        self.live_instruments.pop(port_id, None)

    async def configure_unit(
        self, unit_label: str, controls: dict[str, str]
    ) -> dict[str, str]:
        if any(
            aliases
            for port_id, aliases in self.live_instruments.items()
            if port_id.startswith(f"{unit_label}:")
        ):
            raise RuntimeError("Unit instruments remain deployed.")
        self.configured.append((unit_label, controls))
        self.operations.append(("mode", controls["quel3.monitor.mode"]))
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

    value = controller.configure_monitor_mode(unit_label="unit-a", mode="loopback")

    assert value == "loopback"
    assert client.session_resources == ("unit-a:tx_p00", "unit-a:mon")
    assert client.configured == [("unit-a", {"quel3.monitor.mode": "loopback"})]


def test_configure_monitor_mode_rejects_unsupported_value(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """Unsupported monitor values should fail before changing hardware."""
    controller, client = monitor_runtime

    with pytest.raises(ValueError, match="mode"):
        controller.configure_monitor_mode(
            unit_label="unit-a", mode=cast(Any, "unknown")
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
        controller.configure_monitor_mode(unit_label="unit-a", mode="loopback")

    assert client.configured == []


def test_configure_monitor_mode_defaults_to_open(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """Omitting the mode should restore normal external output without deletion."""
    controller, client = monitor_runtime
    client.controls["quel3.monitor.mode"] = "loopback"

    mode = controller.configure_monitor_mode(unit_label="unit-a")

    assert mode == "open"
    assert client.configured == [("unit-a", {"quel3.monitor.mode": "open"})]
    assert client.discarded == []


@pytest.mark.parametrize("mode", ["open", "loopback"])
def test_configure_monitor_mode_clears_live_instruments_in_one_session(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
    mode: Literal["open", "loopback"],
) -> None:
    """Explicit clearing should discard every selected port and preserve other units."""
    controller, client = monitor_runtime
    client.live_instruments = {
        "unit-a:tx_p00": {"cached-output", "uncached-output"},
        "unit-a:mon": {"uncached-monitor"},
        "unit-b:tx_p00": {"other-output"},
    }
    other = _instrument_info("other-output", "tx_p00", unit_label="unit-b")
    controller._instrument_cache.replace_all(
        instrument_infos=(_instrument_info("cached-output", "tx_p00"), other)
    )

    applied = controller.configure_monitor_mode(
        unit_label="unit-a", mode=mode, clear_instruments=True
    )

    assert applied == mode
    assert client.session_count == 1
    assert client.session_resources == ("unit-a:tx_p00", "unit-a:mon")
    assert set(client.discarded) == {"unit-a:tx_p00", "unit-a:mon"}
    assert client.operations[-1] == ("mode", mode)
    assert client.live_instruments == {"unit-b:tx_p00": {"other-output"}}
    assert controller._instrument_cache.snapshot() == {"other-output": other}


def test_configure_monitor_mode_preserves_uncached_instruments_by_default(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """Uncached instruments should prevent a mode change without implicit deletion."""
    controller, client = monitor_runtime
    client.live_instruments = {"unit-a:tx_p00": {"uncached-output"}}

    with pytest.raises(RuntimeError, match="instruments remain"):
        controller.configure_monitor_mode(unit_label="unit-a", mode="loopback")

    assert client.discarded == []
    assert client.controls["quel3.monitor.mode"] == "open"
    assert client.live_instruments == {"unit-a:tx_p00": {"uncached-output"}}


@pytest.mark.parametrize("mode", ["unknown", "loopback"])
def test_configure_monitor_mode_validates_before_clearing(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient], mode: str
) -> None:
    """Invalid or unsupported modes should leave live instruments and cache intact."""
    controller, client = monitor_runtime
    client.allowed = ("open",)
    client.live_instruments = {"unit-a:tx_p00": {"output"}}
    original = _instrument_info("output", "tx_p00")
    controller._instrument_cache.replace_all(instrument_infos=(original,))

    with pytest.raises(ValueError, match="mode"):
        controller.configure_monitor_mode(
            unit_label="unit-a", mode=cast(Any, mode), clear_instruments=True
        )

    assert client.session_count == 0
    assert client.discarded == []
    assert client.configured == []
    assert client.live_instruments == {"unit-a:tx_p00": {"output"}}
    assert controller._instrument_cache.snapshot() == {"output": original}


def test_configure_monitor_mode_stops_after_deletion_failure(
    monkeypatch: pytest.MonkeyPatch,
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """Deletion failures should prevent mode changes and invalidate the unit cache."""
    controller, client = monitor_runtime
    other = _instrument_info("other", "tx_p00", unit_label="unit-b")
    controller._instrument_cache.replace_all(
        instrument_infos=(_instrument_info("output", "tx_p00"), other)
    )
    discard = client.discard_instruments

    async def fail_monitor_discard(port_id: str) -> None:
        if port_id == "unit-a:mon":
            raise RuntimeError("discard failed")
        await discard(port_id)

    monkeypatch.setattr(client, "discard_instruments", fail_monitor_discard)

    with pytest.raises(RuntimeError, match="discard failed"):
        controller.configure_monitor_mode(
            unit_label="unit-a", mode="loopback", clear_instruments=True
        )

    assert client.configured == []
    assert client.controls["quel3.monitor.mode"] == "open"
    assert controller._instrument_cache.snapshot() == {"other": other}


def test_get_monitor_mode_reads_live_unit_control(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """The current mode should be read from the unit before instrument changes."""
    controller, client = monitor_runtime
    client.controls["quel3.monitor.mode"] = "loopback"

    mode = controller.configuration_manager.get_monitor_mode(unit_label="unit-a")

    assert mode == "loopback"
    assert client.configured == []


def test_get_monitor_mode_rejects_unknown_readback(
    monitor_runtime: tuple[Quel3BackendController, _MonitorClient],
) -> None:
    """Unexpected live modes should fail before any instrument changes."""
    controller, client = monitor_runtime
    client.controls["quel3.monitor.mode"] = "unknown"

    with pytest.raises(ValueError, match="readback"):
        controller.configuration_manager.get_monitor_mode(unit_label="unit-a")

    assert client.discarded == []
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
    unit_label: str = "unit-a",
    role: InstrumentRoleName = "TRANSMITTER",
    frequency_range_min_hz: float = 4e9,
    frequency_range_max_hz: float = 6e9,
) -> InstrumentInfoProtocol:
    """Create one deployable hardware snapshot entry for monitor tests."""
    return cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id=f"{unit_label}:{alias}",
            port_id=f"{unit_label}:{port}",
            definition=SimpleNamespace(
                alias=alias,
                mode="FIXED_TIMELINE",
                role=role,
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
    """Emulate instrument writes while retaining call order and unit boundaries."""
    manager = _MonitorExecutionManager()
    controller = Quel3BackendController(execution_manager=cast(Any, manager))
    originals = (
        _instrument_info("output-a", "tx_p00"),
        _instrument_info("output-b", "tx_p01"),
        _instrument_info("idle", "tx_p02"),
    )
    actions: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        controller._resource_reader,
        "read_instrument_infos",
        lambda **kwargs: originals,
    )
    monkeypatch.setattr(
        controller._configuration_manager,
        "get_monitor_mode",
        lambda **kwargs: "open",
        raising=False,
    )

    def clear(
        *, unit_label: str, instrument_cache: InstrumentCache, parallel: bool = True
    ) -> None:
        actions.append(("clear", unit_label))
        instrument_cache.replace_units(unit_labels=(unit_label,), instrument_infos=())

    def configure(
        *,
        unit_label: str,
        instrument_cache: InstrumentCache,
        mode: str = "open",
        clear_instruments: bool = False,
    ) -> str:
        actions.append(("mode", mode))
        return mode

    def deploy(
        *,
        instrument: InstrumentSpec,
        instrument_cache: InstrumentCache,
        resource_reader: object,
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
        info = _instrument_info(
            instrument.alias,
            instrument.port_id.partition(":")[2],
            unit_label=instrument.port_id.partition(":")[0],
            role=instrument.role,
        )
        instrument_cache.replace_ports(
            port_ids=(instrument.port_id,), instrument_infos=(info,)
        )
        return info

    def restore(
        *,
        configuration: InstrumentConfiguration,
        instrument_cache: InstrumentCache,
        resource_reader: object,
        parallel: bool = True,
    ) -> dict[str, InstrumentInfoProtocol]:
        actions.append(
            ("restore", tuple(spec.alias for spec in configuration.instruments))
        )
        restored = tuple(
            _instrument_info(
                spec.alias,
                spec.port_id.partition(":")[2],
                unit_label=spec.port_id.partition(":")[0],
                role=spec.role,
                frequency_range_min_hz=spec.frequency_range_min_hz,
                frequency_range_max_hz=spec.frequency_range_max_hz,
            )
            for spec in configuration.instruments
        )
        instrument_cache.replace_units(
            unit_labels=tuple(
                {spec.port_id.partition(":")[0] for spec in configuration.instruments}
            ),
            instrument_infos=restored,
        )
        return {info.definition.alias: info for info in restored}

    monkeypatch.setattr(controller.configuration_manager, "clear_instruments", clear)
    monkeypatch.setattr(
        controller.configuration_manager, "configure_monitor_mode", configure
    )
    monkeypatch.setattr(controller.configuration_manager, "deploy_instrument", deploy)
    monkeypatch.setattr(controller.configuration_manager, "deploy_instruments", restore)
    return controller, manager, actions


@pytest.mark.parametrize("capture_kind", ["missing", "no_windows", "empty"])
def test_run_monitor_schedule_restores_instruments_after_empty_capture(
    monkeypatch: pytest.MonkeyPatch,
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
    capture_kind: str,
) -> None:
    """Missing or empty IQ should raise and restore the unit's original state."""
    controller, manager, actions = monitor_schedule_runtime

    def execute(
        *, request: BackendExecutionRequest, **kwargs: object
    ) -> Quel3BackendExecutionResult:
        monitor_alias = next(
            alias
            for alias, timeline in request.payload.fixed_timelines.items()
            if timeline.capture_windows
        )
        data = (
            {}
            if capture_kind == "missing"
            else {monitor_alias: []}
            if capture_kind == "no_windows"
            else {monitor_alias: [np.array([], dtype=np.complex128)]}
        )
        return Quel3BackendExecutionResult(status={}, data=data, config={})

    monkeypatch.setattr(manager, "execute_sync", execute)
    with PulseSchedule() as schedule:
        schedule.add("output-a", Arbitrary([0.5 + 0j], sampling_period=0.4))

    with pytest.raises(RuntimeError, match="no IQ data"):
        controller.run_monitor_schedule(pulse_schedule=schedule)

    assert actions[-3:] == [
        ("clear", "unit-a"),
        ("mode", "open"),
        ("restore", ("idle", "output-a", "output-b")),
    ]
    assert set(controller._instrument_cache.snapshot()) == {
        "output-a",
        "output-b",
        "idle",
    }


def test_run_monitor_schedule_builds_sparse_events_and_reuses_shape(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """A schedule should preserve pulse timing and modifiers without sampling blanks."""
    controller, manager, actions = monitor_schedule_runtime
    with PulseSchedule() as schedule:
        schedule.add(
            "output-a",
            Arbitrary(
                [0.25 + 0.5j, 0.5 + 0.25j],
                sampling_period=0.4,
                scale=0.5,
                phase=np.deg2rad(30),
            ),
        )
        schedule.add("output-a", Blank(0.8, sampling_period=0.4))
        schedule.add("output-a", PhaseShift(np.pi / 2))
        schedule.add(
            "output-a",
            Arbitrary([0.25 + 0.5j, 0.5 + 0.25j], sampling_period=0.4, scale=0.7),
        )
    schedule.set_frequency("output-a", 5.0)

    iq = controller.run_monitor_schedule(
        pulse_schedule=schedule,
    )

    assert np.array_equal(iq["output-a"], [[1 + 2j, 3 + 4j]])
    assert manager.request is not None
    payload = manager.request.payload
    assert payload.capture_mode is Quel3CaptureMode.RAW_WAVEFORMS
    monitor_alias = next(
        alias
        for alias, timeline in payload.fixed_timelines.items()
        if timeline.capture_windows
    )
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
    assert payload.fixed_timelines[monitor_alias].frequency_hz == pytest.approx(5e9)
    assert payload.fixed_timelines[monitor_alias].capture_windows[0].length_ns == (
        pytest.approx(2.4)
    )
    assert actions == [
        ("clear", "unit-a"),
        ("mode", "loopback"),
        ("deploy", "output-a", "unit-a:tx_p00", "TRANSMITTER", 4e9, 6e9),
        ("deploy", monitor_alias, "unit-a:mon", "RECEIVER", 4e9, 6e9),
        ("clear", "unit-a"),
        ("mode", "open"),
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
        schedule.add("output-a", Arbitrary([0.25 + 0j], sampling_period=0.4))
        schedule.add("output-b", Arbitrary([0.5 + 0j], sampling_period=0.4))
    schedule.set_frequency("output-b", 5.5)

    captured = controller.run_monitor_schedule(
        pulse_schedule=schedule,
    )

    assert set(captured) == {"output-a", "output-b"}
    assert captured["output-a"][0, 0] == 1 + 2j
    assert captured["output-b"][0, 0] == 2 + 2j
    assert manager.requests is not None
    assert len(manager.requests) == 2
    monitor_alias = next(
        alias
        for alias, timeline in manager.requests[0].payload.fixed_timelines.items()
        if timeline.capture_windows
    )
    for request, output, frequency in zip(
        manager.requests, ("output-a", "output-b"), (5e9, 5.5e9), strict=True
    ):
        timelines = request.payload.fixed_timelines
        assert set(timelines) == {output, monitor_alias}
        assert timelines[output].frequency_hz == pytest.approx(frequency)
        assert timelines[monitor_alias].frequency_hz == pytest.approx(frequency)
    assert [entry[1] for entry in actions if entry[0] == "deploy"] == [
        "output-a",
        monitor_alias,
        "output-b",
        monitor_alias,
    ]


def test_run_monitor_schedule_infers_unit_from_live_target_and_restores_it(
    monkeypatch: pytest.MonkeyPatch,
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """A live target should select its unit and restore all of that unit's instruments."""
    controller, manager, actions = monitor_schedule_runtime
    other = _instrument_info("other", "tx_p00")
    originals = (
        _instrument_info("unit-b:output-a", "tx_p00", unit_label="unit-b"),
        _instrument_info("idle", "tx_p01", unit_label="unit-b"),
    )
    controller._instrument_cache.replace_all(instrument_infos=(other,))
    reads: list[bool] = []

    def read(*, parallel: bool = True) -> tuple[InstrumentInfoProtocol, ...]:
        reads.append(parallel)
        return (other, *originals)

    monkeypatch.setattr(controller.resource_reader, "read_instrument_infos", read)
    with PulseSchedule() as schedule:
        schedule.add("output-a", Arbitrary([0.5 + 0j], sampling_period=0.4))

    captured = controller.run_monitor_schedule(pulse_schedule=schedule, parallel=False)

    assert np.array_equal(captured["output-a"], [[1 + 2j, 3 + 4j]])
    assert manager.request is not None
    assert reads == [False]
    assert [entry for entry in actions if entry[0] == "clear"] == [
        ("clear", "unit-b"),
        ("clear", "unit-b"),
    ]
    assert ("deploy", "output-a", "unit-b:tx_p00", "TRANSMITTER", 4e9, 6e9) in actions
    assert any(entry[0] == "deploy" and entry[2] == "unit-b:mon" for entry in actions)
    assert actions[-2:] == [("mode", "open"), ("restore", ("idle", "output-a"))]
    snapshot = controller._instrument_cache.snapshot()
    assert set(snapshot) == {"other", "idle", "output-a"}
    assert snapshot["other"] is other
    assert snapshot["idle"].port_id == "unit-b:tx_p01"
    assert snapshot["output-a"].port_id == "unit-b:tx_p00"


@pytest.mark.parametrize("qualified_aliases", [False, True])
def test_run_monitor_schedule_rejects_ambiguous_target_before_deletion(
    monkeypatch: pytest.MonkeyPatch,
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
    qualified_aliases: bool,
) -> None:
    """A target alias on multiple units should fail even when one match is cached."""
    controller, manager, actions = monitor_schedule_runtime
    originals = tuple(
        _instrument_info(
            f"{unit}:output-a" if qualified_aliases else "output-a",
            "tx_p00",
            unit_label=unit,
        )
        for unit in ("unit-a", "unit-b")
    )
    controller._instrument_cache.replace_all(instrument_infos=originals[:1])
    original_snapshot = controller._instrument_cache.snapshot()
    monkeypatch.setattr(
        controller.resource_reader,
        "read_instrument_infos",
        lambda **kwargs: originals,
    )
    with PulseSchedule() as schedule:
        schedule.add("output-a", Arbitrary([0.5 + 0j], sampling_period=0.4))

    with pytest.raises(ValueError, match=r"ambiguous.*unit-a.*unit-b"):
        controller.run_monitor_schedule(pulse_schedule=schedule)

    assert actions == []
    assert manager.request is None
    assert controller._instrument_cache.snapshot() == original_snapshot


def test_run_monitor_schedule_rejects_multiple_units_before_deletion(
    monkeypatch: pytest.MonkeyPatch,
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """Targets spanning multiple units should fail before changing any instruments."""
    controller, manager, actions = monitor_schedule_runtime
    originals = (
        _instrument_info("output-a", "tx_p00"),
        _instrument_info("output-b", "tx_p00", unit_label="unit-b"),
    )
    controller._instrument_cache.replace_all(instrument_infos=originals)
    original_snapshot = controller._instrument_cache.snapshot()
    monkeypatch.setattr(
        controller.resource_reader,
        "read_instrument_infos",
        lambda **kwargs: originals,
    )
    with PulseSchedule() as schedule:
        schedule.add("output-a", Arbitrary([0.5 + 0j], sampling_period=0.4))
        schedule.add("output-b", Arbitrary([0.5 + 0j], sampling_period=0.4))

    with pytest.raises(ValueError, match=r"single unit.*unit-a.*unit-b"):
        controller.run_monitor_schedule(pulse_schedule=schedule)

    assert actions == []
    assert manager.request is None
    assert controller._instrument_cache.snapshot() == original_snapshot


def test_run_monitor_schedule_ignores_unrelated_instrument_configurations(
    monkeypatch: pytest.MonkeyPatch,
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """Other units' duplicate aliases and unsupported profiles should not block a run."""
    controller, _, actions = monitor_schedule_runtime
    unrelated = tuple(
        _instrument_info("unrelated", "tx_p00", unit_label=unit)
        for unit in ("unit-b", "unit-c")
    )
    cast(Any, unrelated[0]).definition.mode = "OTHER_MODE"
    cast(Any, unrelated[0]).definition.profile = None
    original = _instrument_info("output-a", "tx_p00")
    monkeypatch.setattr(
        controller.resource_reader,
        "read_instrument_infos",
        lambda **kwargs: (original, *unrelated),
    )
    with PulseSchedule() as schedule:
        schedule.add("output-a", Arbitrary([0.5 + 0j], sampling_period=0.4))

    captured = controller.run_monitor_schedule(pulse_schedule=schedule)

    assert np.array_equal(captured["output-a"], [[1 + 2j, 3 + 4j]])
    assert [entry for entry in actions if entry[0] == "clear"] == [
        ("clear", "unit-a"),
        ("clear", "unit-a"),
    ]
    assert actions[-1] == ("restore", ("output-a",))


def test_run_monitor_schedule_read_failure_leaves_instruments_untouched(
    monkeypatch: pytest.MonkeyPatch,
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """Failure to read live instruments should leave hardware and cached state intact."""
    controller, manager, actions = monitor_schedule_runtime
    controller._instrument_cache.replace_all(
        instrument_infos=(_instrument_info("output-a", "tx_p00"),)
    )
    original_snapshot = controller._instrument_cache.snapshot()

    def read(**kwargs: object) -> tuple[InstrumentInfoProtocol, ...]:
        raise RuntimeError("instrument snapshot incomplete")

    monkeypatch.setattr(controller.resource_reader, "read_instrument_infos", read)
    with PulseSchedule() as schedule:
        schedule.add("output-a", Arbitrary([0.5 + 0j], sampling_period=0.4))

    with pytest.raises(RuntimeError, match="snapshot incomplete"):
        controller.run_monitor_schedule(pulse_schedule=schedule)

    assert actions == []
    assert manager.request is None
    assert controller._instrument_cache.snapshot() == original_snapshot


@pytest.mark.parametrize("role", ["TRANSMITTER", "TRANSCEIVER", "TRANSCEIVER_LOOPBACK"])
def test_run_monitor_schedule_normalizes_trx_port_waveforms(
    monkeypatch: pytest.MonkeyPatch,
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
    role: InstrumentRoleName,
) -> None:
    """TRX ports should use 0.8 ns waveform samples regardless of instrument role."""
    controller, manager, actions = monitor_schedule_runtime
    original = _instrument_info("readout", "trx_p00p01", role=role)
    monkeypatch.setattr(
        controller.resource_reader,
        "read_instrument_infos",
        lambda **kwargs: (original,),
    )
    with PulseSchedule() as schedule:
        schedule.add("readout", Arbitrary([0.1, 0.3, 0.5], sampling_period=0.4))

    captured = controller.run_monitor_schedule(pulse_schedule=schedule)

    assert np.array_equal(captured["readout"], [[1 + 2j, 3 + 4j]])
    assert manager.request is not None
    timeline = manager.request.payload.fixed_timelines["readout"]
    waveform = manager.request.payload.waveform_library[
        timeline.events[0].waveform_name
    ]
    assert waveform.sampling_period_ns == pytest.approx(0.8)
    np.testing.assert_allclose(waveform.iq_array, [0.2, 0.5], rtol=1e-12, atol=1e-12)
    assert (
        next(action for action in actions if action[:2] == ("deploy", "readout"))[3]
        == "TRANSMITTER"
    )
    assert controller.get_instrument_configuration().instruments[0].role == role


@pytest.mark.parametrize("role", ["TRANSMITTER", "TRANSCEIVER", "TRANSCEIVER_LOOPBACK"])
def test_run_monitor_schedule_rejects_incompatible_readout_before_deletion(
    monkeypatch: pytest.MonkeyPatch,
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
    role: InstrumentRoleName,
) -> None:
    """Incompatible readout sampling should fail before modifying instruments."""
    controller, manager, actions = monitor_schedule_runtime
    monkeypatch.setattr(
        controller.resource_reader,
        "read_instrument_infos",
        lambda **kwargs: (_instrument_info("readout", "trx_p00p01", role=role),),
    )
    with PulseSchedule() as schedule:
        schedule.add("readout", Arbitrary([0.1, 0.3], sampling_period=0.3))

    with pytest.raises(ValueError, match="must divide"):
        controller.run_monitor_schedule(pulse_schedule=schedule)

    assert actions == []
    assert manager.request is None


@pytest.mark.parametrize("target_label", ["monitor", "_qubex_monitor"])
def test_run_monitor_schedule_avoids_temporary_alias_collisions(
    monkeypatch: pytest.MonkeyPatch,
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
    target_label: str,
) -> None:
    """Temporary monitor aliases should avoid live outputs and other units' cache."""
    controller, manager, actions = monitor_schedule_runtime
    originals = (
        _instrument_info(target_label, "tx_p00"),
        _instrument_info("_qubex_monitor_2", "tx_p02"),
    )
    other_unit_infos = tuple(
        _instrument_info(alias, f"tx_p0{index}", unit_label="unit-b")
        for index, alias in enumerate(("_qubex_monitor", "_qubex_monitor_1"))
        if alias != target_label
    )
    uncached = _instrument_info("_qubex_monitor_3", "tx_p02", unit_label="unit-b")
    controller._instrument_cache.replace_all(instrument_infos=other_unit_infos)
    monkeypatch.setattr(
        controller.resource_reader,
        "read_instrument_infos",
        lambda **kwargs: (*originals, uncached),
    )
    with PulseSchedule() as schedule:
        schedule.add(target_label, Arbitrary([0.5 + 0j], sampling_period=0.4))

    captured = controller.run_monitor_schedule(pulse_schedule=schedule)

    assert np.array_equal(captured[target_label], [[1 + 2j, 3 + 4j]])
    assert manager.request is not None
    monitor_alias = next(
        alias
        for alias, timeline in manager.request.payload.fixed_timelines.items()
        if timeline.capture_windows
    )
    assert monitor_alias not in {
        info.definition.alias for info in (*originals, *other_unit_infos, uncached)
    }
    assert ("deploy", monitor_alias, "unit-a:mon", "RECEIVER", 4e9, 6e9) in actions
    snapshot = controller._instrument_cache.snapshot()
    assert set(snapshot) == {
        info.definition.alias for info in (*originals, *other_unit_infos)
    }
    for info in other_unit_infos:
        assert snapshot[info.definition.alias] is info


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
        schedule.add("output-a", Arbitrary([1 + 0j], sampling_period=0.4))
    schedule.set_frequency("output-a", 5.0)

    with pytest.raises(RuntimeError, match="execute failed"):
        controller.run_monitor_schedule(
            pulse_schedule=schedule,
        )

    assert actions[-3:] == [
        ("clear", "unit-a"),
        ("mode", "open"),
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
        resource_reader: object,
        append: bool = True,
        parallel: bool = True,
    ) -> InstrumentInfoProtocol:
        if instrument.port_id == "unit-a:mon":
            raise RuntimeError("monitor deployment failed")
        return original_deploy(
            instrument=instrument,
            instrument_cache=instrument_cache,
            resource_reader=cast(Any, resource_reader),
            append=append,
            parallel=parallel,
        )

    monkeypatch.setattr(
        controller.configuration_manager, "deploy_instrument", fail_monitor_deploy
    )
    with PulseSchedule() as schedule:
        schedule.add("output-a", Arbitrary([1 + 0j], sampling_period=0.4))
    schedule.set_frequency("output-a", 5.0)

    with pytest.raises(RuntimeError, match="monitor deployment failed"):
        controller.run_monitor_schedule(
            pulse_schedule=schedule,
        )

    assert actions[-3:] == [
        ("clear", "unit-a"),
        ("mode", "open"),
        ("restore", ("idle", "output-a", "output-b")),
    ]


def test_run_monitor_schedule_defaults_to_live_instrument_center_frequency(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """Missing frequency should use the live instrument's range center."""
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
        schedule.add("output-a", Arbitrary([1 + 0j], sampling_period=0.4))

    captured = controller.run_monitor_schedule(pulse_schedule=schedule)

    assert np.array_equal(captured["output-a"], [[1 + 2j, 3 + 4j]])
    assert manager.request is not None
    timelines = manager.request.payload.fixed_timelines
    monitor_alias = next(
        alias for alias, timeline in timelines.items() if timeline.capture_windows
    )
    assert timelines["output-a"].frequency_hz == pytest.approx(5e9)
    assert timelines[monitor_alias].frequency_hz == pytest.approx(5e9)
    assert schedule.get_frequency("output-a") is None


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
        schedule.add("output-a", Arbitrary([1 + 0j], sampling_period=0.4))
    schedule.set_frequency("output-a", frequency_ghz)

    with pytest.raises(ValueError, match="frequency"):
        controller.run_monitor_schedule(
            pulse_schedule=schedule,
        )

    assert actions == []


def test_run_monitor_schedule_rejects_missing_target_before_deletion(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """An unknown schedule target should leave the unit's instruments untouched."""
    controller, manager, actions = monitor_schedule_runtime
    controller._instrument_cache.replace_all(
        instrument_infos=(_instrument_info("missing", "tx_p00"),)
    )
    original_snapshot = controller._instrument_cache.snapshot()
    with PulseSchedule() as schedule:
        schedule.add("missing", Arbitrary([1 + 0j], sampling_period=0.4))
    schedule.set_frequency("missing", 5.0)

    with pytest.raises(ValueError, match="has no instrument"):
        controller.run_monitor_schedule(
            pulse_schedule=schedule,
        )

    assert actions == []
    assert manager.request is None
    assert controller._instrument_cache.snapshot() == original_snapshot


def test_run_monitor_schedule_rejects_invalid_capture_before_deletion(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """Invalid capture settings should not trigger instrument deletion."""
    controller, _, actions = monitor_schedule_runtime
    with PulseSchedule() as schedule:
        schedule.add("output-a", Arbitrary([1 + 0j], sampling_period=0.4))
    schedule.set_frequency("output-a", 5.0)

    with pytest.raises(ValueError, match="n_iterations"):
        controller.run_monitor_schedule(
            pulse_schedule=schedule,
            n_iterations=0,
        )

    assert actions == []


def test_monitor_tool_runs_schedule_without_controller_dependency(
    monitor_schedule_runtime: tuple[
        Quel3BackendController, _MonitorExecutionManager, list[tuple[object, ...]]
    ],
) -> None:
    """A standalone monitor tool should infer the unit and restore its instruments."""
    controller, manager, actions = monitor_schedule_runtime
    tool = Quel3MonitorTool(
        configuration_manager=controller.configuration_manager,
        execution_manager=cast(Any, manager),
        resource_reader=controller.resource_reader,
        instrument_cache=controller._instrument_cache,
    )
    with PulseSchedule() as schedule:
        schedule.add("output-a", Arbitrary([1 + 0j], sampling_period=0.4))

    captured = tool.run_schedule(pulse_schedule=schedule)

    assert np.array_equal(captured["output-a"], [[1 + 2j, 3 + 4j]])
    assert actions[-3:] == [
        ("clear", "unit-a"),
        ("mode", "open"),
        ("restore", ("idle", "output-a", "output-b")),
    ]


@pytest.mark.parametrize(
    ("controller_method", "tool_method", "arguments", "expected"),
    [
        (
            "run_monitor_schedule",
            "run_schedule",
            {
                "pulse_schedule": PulseSchedule(),
                "capture_start_ns": 4.0,
                "capture_length_ns": 8.0,
                "n_iterations": 2,
                "shot_interval_ns": 16.0,
                "parallel": False,
            },
            {"output-a": np.array([[1 + 2j]])},
        ),
    ],
)
def test_controller_delegates_monitor_operations(
    monkeypatch: pytest.MonkeyPatch,
    controller_method: str,
    tool_method: str,
    arguments: dict[str, Any],
    expected: object,
) -> None:
    """Controller monitor methods should forward arguments and tool results."""
    calls: list[dict[str, object]] = []

    def delegate(self: Quel3MonitorTool, **kwargs: object) -> object:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(Quel3MonitorTool, tool_method, delegate)
    controller = Quel3BackendController()

    result = getattr(controller, controller_method)(**arguments)

    assert result is expected
    assert calls == [arguments]
