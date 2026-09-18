"""Tests for deleting instruments from one QuEL-3 unit."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel3 import Quel3BackendController
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol
from qubex.backend.quel3.managers import Quel3ConfigurationManager


def _info(unit: str) -> InstrumentInfoProtocol:
    return cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id=f"{unit}:instrument",
            port_id=f"{unit}:p0",
            definition=SimpleNamespace(
                alias=unit,
                role="TRANSMITTER",
                mode="FIXED_TIMELINE",
                profile=SimpleNamespace(
                    frequency_range_min=4e9, frequency_range_max=6e9
                ),
            ),
        ),
    )


class _Client:
    def __init__(self) -> None:
        self.ports = ["unit-a:p0", "unit-a:p1", "unit-ab:p0", "unit-b:p0"]
        self.started: list[str] = []
        self.finished: list[str] = []
        self.sessions: list[tuple[str, ...]] = []
        self.closed = False
        self.finished_at_close: tuple[str, ...] = ()
        self.fail = False
        self.session_error: RuntimeError | None = None
        self.discard_error = RuntimeError("discard failed")

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def list_unit_labels(self) -> list[str]:
        return ["unit-a", "unit-ab", "unit-b"]

    async def list_resource_infos(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(id=port, category=SimpleNamespace(name="PORT"))
            for port in self.ports
        ] + [SimpleNamespace(id="unit-a:instrument", category="INSTRUMENT")]

    def create_session(self, resource_ids: Any, **kwargs: Any) -> Any:
        self.sessions.append(tuple(resource_ids))
        client = self

        class _SessionContext:
            async def __aenter__(self) -> _Client:
                if client.session_error is not None:
                    raise client.session_error
                return client

            async def __aexit__(self, *args: object) -> None:
                client.closed = True
                client.finished_at_close = tuple(client.finished)

        return _SessionContext()

    async def discard_instruments(self, port_id: str) -> None:
        self.started.append(port_id)
        if self.fail and port_id == "unit-a:p0":
            raise self.discard_error
        await asyncio.sleep(0)
        self.finished.append(port_id)


@pytest.fixture
def runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Quel3ConfigurationManager, _Client]:
    """Provide a configuration manager backed by an in-memory quelware client."""
    client = _Client()
    manager = Quel3ConfigurationManager()
    monkeypatch.setattr(
        manager, "_load_quelware_client_factory", lambda: lambda *args: client
    )
    return manager, client


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("has_ports", [False, True])
def test_clear_only_discards_selected_unit_and_preserves_other_cache(
    runtime: tuple[Quel3ConfigurationManager, _Client], parallel: bool, has_ports: bool
) -> None:
    """Clearing should discard every selected unit port and preserve all other units."""
    manager, client = runtime
    if not has_ports:
        client.ports = ["unit-ab:p0", "unit-b:p0"]
    cache = InstrumentCache()
    other = _info("unit-ab")
    cache.replace_all(instrument_infos=(_info("unit-a"), other))

    assert (
        manager.clear_instruments(
            unit_label="unit-a", instrument_cache=cache, parallel=parallel
        )
        is None
    )

    expected = ["unit-a:p0", "unit-a:p1"] if has_ports else []
    assert client.started == expected
    assert client.finished == expected
    assert client.sessions == ([tuple(expected)] if has_ports else [])
    assert client.closed is has_ports
    assert cache.snapshot() == {"unit-ab": other}


@pytest.mark.parametrize("unit_label", ["", "   ", "missing"])
def test_invalid_unit_leaves_hardware_and_cache_unchanged(
    runtime: tuple[Quel3ConfigurationManager, _Client], unit_label: str
) -> None:
    """An empty or undiscovered unit should fail without deleting instruments or cache."""
    manager, client = runtime
    cache = InstrumentCache()
    old = _info("unit-a")
    cache.replace_all(instrument_infos=(old,))

    with pytest.raises(ValueError, match=r"must not be empty|was not discovered"):
        manager.clear_instruments(unit_label=unit_label, instrument_cache=cache)

    assert cache.snapshot() == {"unit-a": old}
    assert client.started == []
    assert client.sessions == []


@pytest.mark.parametrize("parallel", [False, True])
def test_failed_clear_invalidates_unit_and_closes_after_pending_requests(
    runtime: tuple[Quel3ConfigurationManager, _Client],
    parallel: bool,
) -> None:
    """A failed deletion should invalidate only its unit and finish pending work before cleanup."""
    manager, client = runtime
    client.fail = True
    cache = InstrumentCache()
    other = _info("unit-b")
    cache.replace_all(instrument_infos=(_info("unit-a"), other))

    with pytest.raises(RuntimeError, match="discard failed") as caught:
        manager.clear_instruments(
            unit_label="unit-a", instrument_cache=cache, parallel=parallel
        )

    assert caught.value is client.discard_error
    assert cache.snapshot() == {"unit-b": other}
    assert client.closed
    assert client.started == (["unit-a:p0", "unit-a:p1"] if parallel else ["unit-a:p0"])
    assert client.finished_at_close == (("unit-a:p1",) if parallel else ())
    assert len(client.sessions) == 1


def test_clear_propagates_session_failure_without_retry(
    runtime: tuple[Quel3ConfigurationManager, _Client],
) -> None:
    """Session acquisition should fail once and propagate the original error."""
    manager, client = runtime
    client.session_error = RuntimeError("resource busy")
    cache = InstrumentCache()
    other = _info("unit-b")
    cache.replace_all(instrument_infos=(_info("unit-a"), other))

    with pytest.raises(RuntimeError, match="resource busy") as caught:
        manager.clear_instruments(unit_label="unit-a", instrument_cache=cache)

    assert caught.value is client.session_error
    assert len(client.sessions) == 1
    assert client.started == []
    assert cache.snapshot() == {"unit-b": other}


def test_controller_clear_updates_its_execution_cache(
    runtime: tuple[Quel3ConfigurationManager, _Client],
) -> None:
    """Controller deletion should remove only the selected unit from its execution cache."""
    manager, client = runtime
    infos = (_info("unit-a"), _info("unit-b"))
    controller = Quel3BackendController(
        configuration_manager=manager,
        hardware_state_reader=cast(
            Any, SimpleNamespace(read_instrument_infos=lambda **kwargs: infos)
        ),
    )
    controller.refresh_instrument_cache()
    controller.clear_instruments(unit_label="unit-a", parallel=False)

    assert [
        spec.alias for spec in controller.get_instrument_configuration().instruments
    ] == ["unit-b"]
    assert client.started == ["unit-a:p0", "unit-a:p1"]
