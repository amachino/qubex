"""Tests for QuEL-3 backend configuration manager behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
    configuration_manager as configuration_manager_module,
)
from qubex.backend.quel3.models import InstrumentDeployRequest


def test_deploy_instruments_calls_session_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given deploy requests, backend configuration manager should call session deploy."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    class _Profile:
        def __init__(self, *, frequency_range_min: float, frequency_range_max: float):
            self.frequency_range_min = frequency_range_min
            self.frequency_range_max = frequency_range_max

    class _Definition:
        def __init__(self, *, alias: str, mode: object, role: object, profile: object):
            self.alias = alias
            self.mode = mode
            self.role = role
            self.profile = profile

    class _Mode:
        FIXED_TIMELINE = "fixed_timeline"

    class _Role:
        TRANSMITTER = "transmitter"
        TRANSCEIVER = "transceiver"

    deploy_calls: list[tuple[str, list[object]]] = []
    create_session_calls: list[tuple[str, ...]] = []

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[object],
            append: bool = False,
        ) -> list[object]:
            assert append is False
            deploy_calls.append((port_id, definitions))
            return [
                SimpleNamespace(
                    id=f"id:{port_id}",
                    port_id=port_id,
                    definition=definitions[0],
                )
            ]

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            create_session_calls.append(tuple(resource_ids))
            return _FakeSession()

    fake_client = _FakeClient()
    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: (lambda endpoint, port: fake_client),
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: (_Profile, _Definition, _Mode, _Role),
    )

    request = InstrumentDeployRequest(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="inst_transmitter_quel3-02-a01_tx_p02_q00",
        target_labels=("Q00",),
    )

    deployed = manager.deploy_instruments(requests=(request,))

    assert create_session_calls == [("quel3-02-a01:tx_p02",)]
    assert len(deploy_calls) == 1
    port_id, definitions = deploy_calls[0]
    assert port_id == "quel3-02-a01:tx_p02"
    definition = cast(Any, definitions[0])
    assert definition.mode == "fixed_timeline"
    assert definition.role == "transmitter"
    assert definition.profile.frequency_range_min == pytest.approx(4.1e9)
    assert definition.profile.frequency_range_max == pytest.approx(4.3e9)
    assert definition.alias == "inst_transmitter_quel3-02-a01_tx_p02_q00"
    assert manager.target_alias_map == {"Q00": definition.alias}
    assert definition.alias in deployed


def test_deploy_instruments_clears_cache_for_empty_requests() -> None:
    """Given empty requests, backend configuration manager should clear deployment cache."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )
    request = InstrumentDeployRequest(
        port_id="quel3-02-a01:tx_p02",
        role="TRANSMITTER",
        frequency_range_min_hz=4.1e9,
        frequency_range_max_hz=4.3e9,
        alias="inst_transmitter_quel3-02-a01_tx_p02_q00",
        target_labels=("Q00",),
    )
    manager._last_deployed_instrument_infos = {request.alias: (object(),)}  # noqa: SLF001
    manager._target_alias_map = {"Q00": request.alias}  # noqa: SLF001

    deployed = manager.deploy_instruments(requests=())

    assert deployed == {}
    assert manager.last_deployed_instrument_infos == {}
    assert manager.target_alias_map == {}


def test_deploy_instruments_groups_requests_by_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given same-port requests, backend configuration manager should append instruments on that port."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )

    class _Profile:
        def __init__(self, *, frequency_range_min: float, frequency_range_max: float):
            self.frequency_range_min = frequency_range_min
            self.frequency_range_max = frequency_range_max

    class _Definition:
        def __init__(self, *, alias: str, mode: object, role: object, profile: object):
            self.alias = alias
            self.mode = mode
            self.role = role
            self.profile = profile

    class _Mode:
        FIXED_TIMELINE = "fixed_timeline"

    class _Role:
        TRANSMITTER = "transmitter"

    deploy_calls: list[tuple[str, list[object], bool]] = []
    create_session_calls: list[tuple[str, ...]] = []

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        async def deploy_instruments(
            self,
            port_id: str,
            *,
            definitions: list[object],
            append: bool = False,
        ) -> list[object]:
            deploy_calls.append((port_id, definitions, append))
            return [
                SimpleNamespace(
                    id=f"id:{port_id}:{index}",
                    port_id=port_id,
                    definition=definition,
                )
                for index, definition in enumerate(definitions)
            ]

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> None:
            _ = (exc_type, exc, tb)

        def create_session(self, resource_ids: list[str]) -> _FakeSession:
            create_session_calls.append(tuple(resource_ids))
            return _FakeSession()

    fake_client = _FakeClient()
    monkeypatch.setattr(
        manager,
        "_load_quelware_client_factory",
        lambda: (lambda endpoint, port: fake_client),
    )
    monkeypatch.setattr(
        manager,
        "_load_instrument_entities",
        lambda: (_Profile, _Definition, _Mode, _Role),
    )

    requests = (
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p04",
            role="TRANSMITTER",
            frequency_range_min_hz=4.1e9,
            frequency_range_max_hz=4.3e9,
            alias="inst_q00",
            target_labels=("Q00",),
        ),
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p04",
            role="TRANSMITTER",
            frequency_range_min_hz=4.2e9,
            frequency_range_max_hz=4.4e9,
            alias="inst_q00_cr",
            target_labels=("Q00-CR",),
        ),
    )

    deployed = manager.deploy_instruments(requests=requests)

    assert create_session_calls == [("quel3-02-a01:tx_p04",)]
    assert len(deploy_calls) == 2
    assert deploy_calls[0][0] == "quel3-02-a01:tx_p04"
    assert deploy_calls[0][2] is False
    assert [cast(Any, definition).alias for definition in deploy_calls[0][1]] == [
        "inst_q00",
    ]
    assert deploy_calls[1][0] == "quel3-02-a01:tx_p04"
    assert deploy_calls[1][2] is True
    assert [cast(Any, definition).alias for definition in deploy_calls[1][1]] == [
        "inst_q00_cr",
    ]
    assert manager.target_alias_map == {
        "Q00": "inst_q00",
        "Q00-CR": "inst_q00_cr",
    }
    assert set(deployed) == {"inst_q00", "inst_q00_cr"}
    assert cast(Any, deployed["inst_q00"][0]).id == "id:quel3-02-a01:tx_p04:0"
    assert cast(Any, deployed["inst_q00_cr"][0]).id == "id:quel3-02-a01:tx_p04:0"


def test_load_client_factory_uses_configured_client_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given standalone runtime options, client factory loading should use that runtime."""
    captured: dict[str, object] = {}
    fake_client_factory = object()
    monkeypatch.setattr(
        configuration_manager_module,
        "load_quelware_client_factory",
        lambda *, client_mode, standalone_unit_label: (
            captured.update(
                {
                    "client_mode": client_mode,
                    "standalone_unit_label": standalone_unit_label,
                }
            )
            or fake_client_factory
        ),
    )
    manager = Quel3ConfigurationManager(
        quelware_endpoint="worker-host",
        quelware_port=61000,
        client_mode="standalone",
        standalone_unit_label="quel3-02-a01",
    )

    client_factory = manager._load_quelware_client_factory()  # noqa: SLF001

    assert client_factory is fake_client_factory
    assert captured == {
        "client_mode": "standalone",
        "standalone_unit_label": "quel3-02-a01",
    }
