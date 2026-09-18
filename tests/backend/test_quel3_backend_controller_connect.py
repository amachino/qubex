"""Tests for instrument cache initialization during QuEL-3 connection."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel3 import Quel3BackendController
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol


def _info(resource_id: str) -> InstrumentInfoProtocol:
    return cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id=resource_id,
            port_id="unit-a:tx_p01",
            definition=SimpleNamespace(
                alias="unit-a:Q00",
                role="TRANSMITTER",
                mode="FIXED_TIMELINE",
                profile=SimpleNamespace(
                    frequency_range_min=4e9,
                    frequency_range_max=6e9,
                ),
            ),
            config=SimpleNamespace(sampling_period_fs=400_000, samples_per_tick=4),
        ),
    )


class _ConnectionManager:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls
        self.is_connected = False
        self.fail = False

    def connect(
        self,
        unit_labels: str | list[str] | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        self.calls.append(("connect", unit_labels, parallel))
        if self.fail:
            raise RuntimeError("connection failed")
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False


class _Reader:
    def __init__(
        self, calls: list[object], infos: tuple[InstrumentInfoProtocol, ...]
    ) -> None:
        self.calls = calls
        self.infos = infos
        self.fail = False

    def read_instrument_infos(
        self,
        *,
        unit_labels: Sequence[str] = (),
        port_ids: Sequence[str] = (),
        parallel: bool = True,
    ) -> tuple[InstrumentInfoProtocol, ...]:
        self.calls.append(("read", tuple(unit_labels), tuple(port_ids), parallel))
        if self.fail:
            raise RuntimeError("read failed")
        return self.infos


@pytest.mark.parametrize("selection", ["unit-a", ["unit-a"]])
@pytest.mark.parametrize("parallel", [None, False, True])
def test_connect_makes_existing_instruments_available_to_execution(
    parallel: bool | None,
    selection: str | list[str],
) -> None:
    """Connecting should make observed instruments executable without a separate refresh."""
    calls: list[object] = []
    info = _info("unit-a:live")
    connection = _ConnectionManager(calls)
    reader = _Reader(calls, (info,))
    observed: list[InstrumentInfoProtocol] = []

    def execute_sync(
        *, request: object, instrument_cache: InstrumentCache, parallel: bool
    ) -> str:
        observed.append(instrument_cache.get("Q00"))
        return "executed"

    controller = Quel3BackendController(
        connection_manager=cast(Any, connection),
        hardware_state_reader=cast(Any, reader),
        execution_manager=cast(
            Any, SimpleNamespace(sampling_period_ns=0.4, execute_sync=execute_sync)
        ),
    )

    controller.connect(selection, parallel=parallel)
    result = controller.execute_sync(request=BackendExecutionRequest(payload=object()))

    assert controller.is_connected
    assert result == "executed"
    assert observed == [info]
    assert observed[0] is info
    assert calls == [
        ("connect", ["unit-a"], parallel),
        ("read", ("unit-a",), (), True if parallel is None else parallel),
    ]


@pytest.mark.parametrize("failure", ["connection", "read", "validation"])
def test_failed_connect_discards_stale_cache_and_connection_status(
    failure: str,
) -> None:
    """Connection or cache initialization failure should leave no stale executable state."""
    calls: list[object] = []
    connection = _ConnectionManager(calls)
    reader = _Reader(calls, (_info("unit-a:old"),))
    controller = Quel3BackendController(
        connection_manager=cast(Any, connection),
        hardware_state_reader=cast(Any, reader),
    )
    controller.refresh_instrument_cache()
    connection.is_connected = True
    connection.fail = failure == "connection"
    reader.fail = failure == "read"
    if failure == "validation":
        reader.infos = (_info(""),)

    with pytest.raises((RuntimeError, ValueError)):
        controller.connect()

    assert not controller.is_connected
    assert controller.get_instrument_configuration().instruments == ()


def test_repeated_connect_reloads_instruments_from_hardware() -> None:
    """Connecting again should replace the previous connection's instrument definitions."""
    calls: list[object] = []
    reader = _Reader(calls, (_info("unit-a:old"),))
    controller = Quel3BackendController(
        connection_manager=cast(Any, _ConnectionManager(calls)),
        hardware_state_reader=cast(Any, reader),
    )

    controller.connect()
    assert len(controller.get_instrument_configuration().instruments) == 1
    reader.infos = ()
    controller.connect()

    assert controller.is_connected
    assert controller.get_instrument_configuration().instruments == ()
    assert calls == [
        ("connect", None, None),
        ("read", (), (), True),
        ("connect", None, None),
        ("read", (), (), True),
    ]


@pytest.mark.parametrize("already_connected", [False, True])
def test_connect_rejects_missing_units_before_reading_instruments(
    monkeypatch: pytest.MonkeyPatch,
    already_connected: bool,
) -> None:
    """Missing units should reject initial and repeated connections without reading instruments."""
    from qubex.backend.quel3.managers.connection_manager import Quel3ConnectionManager

    connection = Quel3ConnectionManager()
    available = ["unit-a"]

    async def probe() -> list[str]:
        return available

    monkeypatch.setattr(connection, "_probe_quelware_connection", probe)
    calls: list[object] = []
    controller = Quel3BackendController(
        connection_manager=connection,
        hardware_state_reader=cast(Any, _Reader(calls, (_info("unit-a:old"),))),
    )
    if already_connected:
        controller.connect("unit-a")
    calls.clear()

    with pytest.raises(ValueError, match="unit-missing"):
        controller.connect(["unit-a", "unit-missing"])

    assert not controller.is_connected
    assert controller.get_instrument_configuration().instruments == ()
    assert calls == []


@pytest.mark.parametrize("selection", [None, [], "unit-a", ["unit-a", "unit-a"]])
def test_connection_manager_validates_selected_units(
    monkeypatch: pytest.MonkeyPatch,
    selection: str | list[str] | None,
) -> None:
    """Valid or unspecified selections should connect after discovering units."""
    from qubex.backend.quel3.managers.connection_manager import Quel3ConnectionManager

    connection = Quel3ConnectionManager()
    probes: list[bool] = []

    async def probe() -> list[str]:
        probes.append(True)
        return ["unit-a", "unit-b"]

    monkeypatch.setattr(connection, "_probe_quelware_connection", probe)
    connection.connect(selection)

    assert connection.is_connected
    assert probes == [True]


@pytest.mark.parametrize("port_id", ["unit-a:tx_p01", "unit-b:tx_p01"])
@pytest.mark.parametrize("parallel", [False, True])
def test_connect_warns_and_executes_last_duplicate_alias(
    port_id: str, parallel: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Connecting should warn on duplicate aliases and execute the last observed instrument."""
    first = _info("unit-a:first")
    last = cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id=f"{port_id.partition(':')[0]}:last",
            port_id=port_id,
            definition=SimpleNamespace(alias=f"{port_id.partition(':')[0]}: Q00 "),
            config=SimpleNamespace(sampling_period_fs=400_000, samples_per_tick=4),
        ),
    )
    calls: list[object] = []
    observed: list[InstrumentInfoProtocol] = []

    def execute_sync(
        *, request: object, instrument_cache: InstrumentCache, parallel: bool
    ) -> str:
        observed.append(instrument_cache.get("Q00"))
        return "executed"

    controller = Quel3BackendController(
        connection_manager=cast(Any, _ConnectionManager(calls)),
        hardware_state_reader=cast(Any, _Reader(calls, (first, last))),
        execution_manager=cast(
            Any, SimpleNamespace(sampling_period_ns=0.4, execute_sync=execute_sync)
        ),
    )

    controller.connect(parallel=parallel)
    assert (
        controller.execute_sync(request=BackendExecutionRequest(payload=object()))
        == "executed"
    )

    assert controller.is_connected
    assert observed == [last]
    assert observed[0] is last
    assert "Q00" in caplog.text
    assert first.id in caplog.text
    assert last.id in caplog.text
    assert first.port_id in caplog.text
    assert last.port_id in caplog.text
    assert any(record.levelname == "WARNING" for record in caplog.records)
    with pytest.raises(ValueError, match="Duplicate instrument alias"):
        controller.refresh_instrument_cache()


def test_connect_with_empty_selection_clears_cache_without_reading() -> None:
    """An empty unit selection should connect without reading any instruments."""
    calls: list[object] = []
    controller = Quel3BackendController(
        connection_manager=cast(Any, _ConnectionManager(calls)),
        hardware_state_reader=cast(Any, _Reader(calls, (_info("unit-a:old"),))),
    )
    controller.refresh_instrument_cache()
    calls.clear()

    controller.connect([])

    assert controller.is_connected
    assert calls == [("connect", [], None)]
    assert controller.get_instrument_configuration().instruments == ()
