"""Tests for QuEL-3 backend configuration manager behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel3.managers import Quel3ConfigurationManager
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
