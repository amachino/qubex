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
        box_names: str | list[str] | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        self.calls.append(("connect", box_names, parallel))
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


@pytest.mark.parametrize("parallel", [None, False, True])
def test_connect_makes_existing_instruments_available_to_execution(
    parallel: bool | None,
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

    controller.connect(["BOX1"], parallel=parallel)
    result = controller.execute_sync(request=BackendExecutionRequest(payload=object()))

    assert controller.is_connected
    assert result == "executed"
    assert observed == [info]
    assert observed[0] is info
    assert calls == [
        ("connect", ["BOX1"], parallel),
        ("read", (), (), True if parallel is None else parallel),
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
        reader.infos = (_info("unit-a:new-1"), _info("unit-a:new-2"))

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
