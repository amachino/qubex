"""Tests for single-instrument addition and replacement through QuEL-3 APIs."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel3 import InstrumentSpec, Quel3BackendController
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol
from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
    Quel3HardwareStateReader,
    configuration_manager as configuration_module,
)
from qubex.backend.quel3.managers.session_workarounds import QuelwareSessionError


def _spec(alias: str = "Q02", port_id: str = "unit-a:tx_p01") -> InstrumentSpec:
    return InstrumentSpec(
        port_id=port_id,
        alias=alias,
        role="TRANSMITTER",
        frequency_range_min_hz=4e9,
        frequency_range_max_hz=6e9,
    )


def _info(
    alias: str, port_id: str = "unit-a:tx_p01", *, resource_id: str | None = None
) -> InstrumentInfoProtocol:
    return cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id=resource_id or f"{port_id}:{alias}",
            port_id=port_id,
            definition=SimpleNamespace(
                alias=f"{port_id.partition(':')[0]}:{alias}",
                role="TRANSMITTER",
                mode="FIXED_TIMELINE",
                profile=SimpleNamespace(
                    frequency_range_min=4e9, frequency_range_max=6e9
                ),
            ),
            config=SimpleNamespace(sampling_period_fs=400_000),
        ),
    )


class _Hardware:
    def __init__(self) -> None:
        self.infos = [_info("Q00"), _info("Q01", "unit-b:tx_p02")]
        self.deploy_calls: list[tuple[str, bool]] = []
        self.read_calls: list[tuple[str, ...]] = []
        self.closed_sessions = 0
        self.fail_after_write = False
        self.fail_read_number: int | None = None
        self.omit_added = False

    async def __aenter__(self) -> _Hardware:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    def create_session(self, resource_ids: Sequence[str], **kwargs: object) -> object:
        hardware = self

        class _Session:
            token = "single-deploy-session"  # noqa: S105

            async def __aenter__(self) -> _Session:
                return self

            async def __aexit__(self, *_: object) -> None:
                hardware.closed_sessions += 1

            async def deploy_instruments(
                self,
                port_id: str,
                *,
                definitions: list[Any],
                append: bool,
            ) -> list[InstrumentInfoProtocol]:
                hardware.deploy_calls.append((port_id, append))
                if not append:
                    hardware.infos = [
                        info for info in hardware.infos if info.port_id != port_id
                    ]
                else:
                    aliases = {definition.alias for definition in definitions}
                    hardware.infos = [
                        info
                        for info in hardware.infos
                        if info.port_id != port_id
                        or InstrumentCache.alias_for(info) not in aliases
                    ]
                if not hardware.omit_added:
                    hardware.infos.extend(
                        _info(
                            definition.alias,
                            port_id,
                            resource_id=f"{port_id}:{definition.alias}:deployment-{len(hardware.deploy_calls)}",
                        )
                        for definition in definitions
                    )
                if hardware.fail_after_write:
                    raise RuntimeError("response lost after deployment")
                # The manager must use the subsequent full-port read, not this response.
                return []

        return _Session()

    def read_instrument_infos(
        self,
        *,
        port_ids: Sequence[str] = (),
        parallel: bool = True,
        unit_labels: Sequence[str] = (),
    ) -> tuple[InstrumentInfoProtocol, ...]:
        self.read_calls.append(tuple(port_ids))
        if self.fail_read_number == len(self.read_calls):
            raise RuntimeError("read failed")
        return tuple(
            info for info in self.infos if not port_ids or info.port_id in port_ids
        )


@pytest.fixture
def deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache]:
    """Provide a real manager and cache backed by in-memory hardware IO."""
    hardware = _Hardware()
    manager = Quel3ConfigurationManager()
    monkeypatch.setattr(
        manager, "_load_quelware_client_factory", lambda: lambda *args: hardware
    )
    entities = configuration_module._QuelwareInstrumentEntities(  # noqa: SLF001
        fixed_timeline_profile_factory=cast(Any, SimpleNamespace),
        instrument_definition_factory=cast(Any, SimpleNamespace),
        instrument_mode_namespace=cast(
            Any, SimpleNamespace(FIXED_TIMELINE="FIXED_TIMELINE")
        ),
        instrument_role_namespace=cast(Any, SimpleNamespace(TRANSMITTER="TRANSMITTER")),
    )
    monkeypatch.setattr(manager, "_load_instrument_entities", lambda: entities)
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=hardware.infos)
    return manager, hardware, cache


@pytest.mark.parametrize("append", [True, False])
def test_single_deploy_updates_the_full_port_and_returns_one_readback(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
    append: bool,
) -> None:
    """Single deployment should honor append and preserve other ports in the cache."""
    manager, hardware, cache = deployment
    untouched = cache.get("Q01")

    result = manager.deploy_instrument(
        instrument=_spec(),
        append=append,
        instrument_cache=cache,
        hardware_state_reader=cast(Quel3HardwareStateReader, hardware),
        parallel=False,
    )

    assert hardware.deploy_calls == [("unit-a:tx_p01", append)]
    assert hardware.read_calls == [("unit-a:tx_p01",)]
    assert cache.get("Q02") is result
    assert cache.get("Q01") is untouched
    assert set(cache.snapshot()) == (
        {"Q00", "Q01", "Q02"} if append else {"Q01", "Q02"}
    )


def test_controller_single_deploy_defaults_to_append(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
) -> None:
    """The controller should add one instrument by default and expose the updated configuration."""
    manager, hardware, _ = deployment
    controller = Quel3BackendController(
        configuration_manager=manager,
        hardware_state_reader=cast(Quel3HardwareStateReader, hardware),
    )
    controller.refresh_instrument_cache()

    result = controller.deploy_instrument(instrument=_spec())

    assert result.id == "unit-a:tx_p01:Q02:deployment-1"
    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert {
        spec.alias for spec in controller.get_instrument_configuration().instruments
    } == {"Q00", "Q01", "Q02"}


@pytest.mark.parametrize("cache_populated", [True, False])
def test_append_replaces_existing_alias_without_a_preflight_read(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
    cache_populated: bool,
) -> None:
    """Append should let hardware replace an existing alias and cache the full port readback."""
    manager, hardware, cache = deployment
    old = hardware.infos[0]
    sibling = _info("Q03")
    hardware.infos.append(sibling)
    cache.replace_all(instrument_infos=hardware.infos if cache_populated else ())

    result = manager.deploy_instrument(
        instrument=_spec(alias="Q00"),
        instrument_cache=cache,
        hardware_state_reader=cast(Quel3HardwareStateReader, hardware),
    )

    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert hardware.read_calls == [("unit-a:tx_p01",)]
    assert result.id != old.id
    assert cache.get("Q00") is result
    assert cache.get("Q03") is sibling
    assert set(cache.snapshot()) == (
        {"Q00", "Q01", "Q03"} if cache_populated else {"Q00", "Q03"}
    )


def test_append_does_not_retry_after_an_uncertain_write(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
) -> None:
    """An uncertain append should run once and invalidate only the touched port."""
    manager, hardware, cache = deployment
    hardware.fail_after_write = True
    untouched = cache.get("Q01")

    with pytest.raises(QuelwareSessionError, match="without retry"):
        manager.deploy_instrument(
            instrument=_spec(),
            instrument_cache=cache,
            hardware_state_reader=cast(Quel3HardwareStateReader, hardware),
        )

    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert hardware.closed_sessions == 1
    assert cache.snapshot() == {"Q01": untouched}


@pytest.mark.parametrize("failure", ["readback", "missing"])
def test_failed_append_readback_does_not_publish_partial_information(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
    failure: str,
) -> None:
    """Failed append readback should leave the touched port absent from the cache."""
    manager, hardware, cache = deployment
    hardware.fail_read_number = 1 if failure == "readback" else None
    hardware.omit_added = failure == "missing"
    untouched = cache.get("Q01")

    with pytest.raises((ValueError, RuntimeError)):
        manager.deploy_instrument(
            instrument=_spec(),
            instrument_cache=cache,
            hardware_state_reader=cast(Quel3HardwareStateReader, hardware),
        )

    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert cache.snapshot() == {"Q01": untouched}
