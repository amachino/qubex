"""Tests for single-instrument addition and replacement through QuEL-3 APIs."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel3 import (
    InstrumentConfiguration,
    InstrumentSpec,
    Quel3BackendController,
)
from qubex.backend.quel3.infra import Quel3ResourceReader
from qubex.backend.quel3.infra.quelware_imports import QuelwareInstrumentEntities
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol
from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
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


def test_modify_target_frequencies_preserves_instrument_width_and_other_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frequency changes should append matching aliases with their existing range widths."""
    manager = Quel3ConfigurationManager()
    controller = Quel3BackendController(configuration_manager=manager)
    original = _spec("Q00")
    sibling = _spec("Q01")
    monkeypatch.setattr(
        controller,
        "get_instrument_configuration",
        lambda: InstrumentConfiguration(instruments=(original, sibling)),
    )
    deployed: list[InstrumentSpec] = []

    def deploy(*, instrument: InstrumentSpec, **kwargs: Any) -> Any:
        deployed.append(instrument)
        return None

    monkeypatch.setattr(manager, "replace_instruments", deploy)
    controller.modify_target_frequencies({"Q00": 7.0})

    assert deployed == [
        InstrumentSpec(
            alias="Q00",
            port_id=original.port_id,
            role=original.role,
            frequency_range_min_hz=6e9,
            frequency_range_max_hz=8e9,
        )
    ]


def _info(
    alias: str,
    port_id: str = "unit-a:tx_p01",
    *,
    resource_id: str | None = None,
    frequency_range_min_hz: float = 4e9,
    frequency_range_max_hz: float = 6e9,
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
                    frequency_range_min=frequency_range_min_hz,
                    frequency_range_max=frequency_range_max_hz,
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
                            frequency_range_min_hz=definition.profile.frequency_range_min,
                            frequency_range_max_hz=definition.profile.frequency_range_max,
                        )
                        for definition in definitions
                    )
                if hardware.fail_after_write:
                    raise RuntimeError("response lost after deployment")
                return [
                    info
                    for info in hardware.infos
                    if info.port_id == port_id
                    and InstrumentCache.alias_for(info)
                    in {definition.alias for definition in definitions}
                ]

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
    entities = QuelwareInstrumentEntities(
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


@pytest.mark.parametrize("parallel", [False, True])
def test_frequency_changes_only_read_changed_instruments(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
    monkeypatch: pytest.MonkeyPatch,
    parallel: bool,
) -> None:
    """Frequency changes should read new target IDs without querying unrelated instruments."""
    manager, hardware, cache = deployment
    sibling = _info("Q03")
    hardware.infos.append(sibling)
    cache.replace_all(instrument_infos=hardware.infos)
    untouched = cache.get("Q01")
    requested: list[str] = []

    async def get_info(resource_id: str) -> InstrumentInfoProtocol:
        requested.append(resource_id)
        return next(info for info in hardware.infos if info.id == resource_id)

    reader = Quel3ResourceReader()
    monkeypatch.setattr(
        reader, "_load_quelware_client_factory", lambda: lambda *args: hardware
    )
    monkeypatch.setattr(hardware, "get_instrument_info", get_info, raising=False)

    manager.replace_instruments(
        instrument=_spec("Q00"),
        instrument_cache=cache,
        resource_reader=reader,
        parallel=parallel,
    )

    updated = cache.get("Q00")
    assert updated.id == "unit-a:tx_p01:Q00:deployment-1"
    assert requested == [updated.id]
    assert hardware.read_calls == []
    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert cache.get("Q03") is sibling
    assert cache.get("Q01") is untouched


@pytest.mark.parametrize(
    "failure",
    ["write", "response", "readback", "wrong_port", "wrong_alias", "wrong_id"],
)
def test_frequency_change_failure_preserves_unrelated_cached_instruments(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Failed target redeployment should invalidate only that alias and preserve siblings."""
    manager, hardware, cache = deployment
    sibling = _info("Q03")
    hardware.infos.append(sibling)
    cache.replace_all(instrument_infos=hardware.infos)
    untouched = cache.get("Q01")
    hardware.fail_after_write = failure == "write"
    hardware.omit_added = failure == "response"

    def read_ids(*args: object, **kwargs: object) -> tuple[InstrumentInfoProtocol, ...]:
        if failure == "readback":
            raise RuntimeError("instrument unavailable")
        updated = next(
            info for info in hardware.infos if InstrumentCache.alias_for(info) == "Q00"
        )
        return (
            _info(
                "Q99" if failure == "wrong_alias" else "Q00",
                "unit-a:tx_p02" if failure == "wrong_port" else updated.port_id,
                resource_id="wrong-id" if failure == "wrong_id" else updated.id,
            ),
        )

    reader = Quel3ResourceReader()
    monkeypatch.setattr(reader, "read_instrument_infos_by_ids", read_ids)

    with pytest.raises((QuelwareSessionError, ValueError, RuntimeError)):
        manager.replace_instruments(
            instrument=_spec("Q00"), instrument_cache=cache, resource_reader=reader
        )

    assert cache.snapshot() == {"Q01": untouched, "Q03": sibling}
    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert hardware.closed_sessions == 1


def test_controller_frequency_change_uses_only_target_readback(
    deployment: tuple[Quel3ConfigurationManager, _Hardware, InstrumentCache],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controller frequency changes should preserve the range width and unrelated aliases."""
    manager, hardware, _ = deployment
    reader = Quel3ResourceReader()
    monkeypatch.setattr(
        reader, "_load_quelware_client_factory", lambda: lambda *args: hardware
    )
    requested: list[str] = []

    async def get_info(resource_id: str) -> InstrumentInfoProtocol:
        requested.append(resource_id)
        return next(info for info in hardware.infos if info.id == resource_id)

    monkeypatch.setattr(hardware, "get_instrument_info", get_info, raising=False)
    monkeypatch.setattr(
        hardware,
        "read_instrument_infos_by_ids",
        reader.read_instrument_infos_by_ids,
        raising=False,
    )
    controller = Quel3BackendController(
        configuration_manager=manager,
        resource_reader=cast(Quel3ResourceReader, hardware),
    )
    controller.refresh_instrument_cache()
    hardware.read_calls.clear()

    controller.modify_target_frequencies({"Q00": 7.0})

    assert requested == ["unit-a:tx_p01:Q00:deployment-1"]
    assert hardware.read_calls == []
    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    specs = {
        spec.alias: spec
        for spec in controller.get_instrument_configuration().instruments
    }
    assert set(specs) == {"Q00", "Q01"}
    assert specs["Q00"].frequency_range_min_hz == 6e9
    assert specs["Q00"].frequency_range_max_hz == 8e9


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
        resource_reader=cast(Quel3ResourceReader, hardware),
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
        resource_reader=cast(Quel3ResourceReader, hardware),
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
        resource_reader=cast(Quel3ResourceReader, hardware),
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
            resource_reader=cast(Quel3ResourceReader, hardware),
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
            resource_reader=cast(Quel3ResourceReader, hardware),
        )

    assert hardware.deploy_calls == [("unit-a:tx_p01", True)]
    assert cache.snapshot() == {"Q01": untouched}
