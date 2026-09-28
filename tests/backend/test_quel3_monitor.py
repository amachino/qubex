# ruff: noqa: SLF001

"""Tests for QuEL-3 monitor configuration and IQ capture."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

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


def _service_for_controller(controller: Quel3BackendController) -> Quel3MonitorService:
    """Bind a standalone service to the test controller's supplied components."""
    return Quel3MonitorService(
        configuration_manager=controller.configuration_manager,
        execution_manager=controller.execution_manager,
        hardware_state_reader=controller.hardware_state_reader,
        instrument_cache=controller._instrument_cache,
    )
