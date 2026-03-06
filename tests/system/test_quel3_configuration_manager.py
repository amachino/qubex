"""Tests for QuEL-3 configuration manager and synchronizer integration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.system.quel3 import Quel3ConfigurationManager, Quel3SystemSynchronizer
from qubex.system.target import TargetType


def test_build_instrument_deploy_requests_creates_one_request_per_target() -> None:
    """Given selected boxes, when building requests, then QuEL-3 creates one instrument per target."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )
    read_out_port = SimpleNamespace(id="BOX1.READ0.OUT", box_id="BOX1", number=1)
    ctrl_port = SimpleNamespace(id="BOX1.CTRL0", box_id="BOX1", number=2)
    other_box_port = SimpleNamespace(id="BOX2.CTRL0", box_id="BOX2", number=9)
    mux0 = SimpleNamespace(index=0)
    read_in_port = SimpleNamespace(id="BOX1.READ0.IN", box_id="BOX1", number=0)

    gen_targets = {
        "RQ00": SimpleNamespace(
            label="RQ00",
            frequency=6.0,
            type=TargetType.READ,
            channel=SimpleNamespace(port=read_out_port),
        ),
        "Q00": SimpleNamespace(
            label="Q00",
            frequency=4.20,
            type=TargetType.CTRL_GE,
            channel=SimpleNamespace(port=ctrl_port),
        ),
        "Q00-CR": SimpleNamespace(
            label="Q00-CR",
            frequency=4.35,
            type=TargetType.CTRL_CR,
            channel=SimpleNamespace(port=ctrl_port),
        ),
        "Q01": SimpleNamespace(
            label="Q01",
            frequency=4.50,
            type=TargetType.CTRL_GE,
            channel=SimpleNamespace(port=other_box_port),
        ),
    }
    experiment_system = SimpleNamespace(
        gen_targets=gen_targets,
        wiring_info=SimpleNamespace(read_in=[(mux0, read_in_port)]),
        control_params=SimpleNamespace(
            get_frequency_margin=lambda _target_type: 0.1,
        ),
        get_box=lambda box_id: SimpleNamespace(
            id=box_id,
            name="quel3-02-a01" if box_id == "BOX1" else "quel3-02-a02",
        ),
        get_mux_by_readout_port=lambda port: mux0 if port is read_out_port else None,
    )

    requests = manager._build_instrument_deploy_requests(  # noqa: SLF001
        experiment_system=cast(Any, experiment_system),
        box_ids=["BOX1"],
    )

    request_by_target = {request.target_labels[0]: request for request in requests}
    assert set(request_by_target) == {"RQ00", "Q00", "Q00-CR"}

    read_request = request_by_target["RQ00"]
    assert read_request.port_id == "quel3-02-a01:trx_p00p01"
    assert read_request.role == "TRANSCEIVER"
    assert read_request.frequency_range_min_hz == pytest.approx(5.9e9)
    assert read_request.frequency_range_max_hz == pytest.approx(6.1e9)
    assert read_request.target_labels == ("RQ00",)

    ctrl_request = request_by_target["Q00"]
    assert ctrl_request.port_id == "quel3-02-a01:tx_p02"
    assert ctrl_request.role == "TRANSMITTER"
    assert ctrl_request.frequency_range_min_hz == pytest.approx(4.10e9)
    assert ctrl_request.frequency_range_max_hz == pytest.approx(4.30e9)
    assert ctrl_request.target_labels == ("Q00",)

    cr_request = request_by_target["Q00-CR"]
    assert cr_request.port_id == "quel3-02-a01:tx_p02"
    assert cr_request.role == "TRANSMITTER"
    assert cr_request.frequency_range_min_hz == pytest.approx(4.25e9)
    assert cr_request.frequency_range_max_hz == pytest.approx(4.45e9)
    assert cr_request.target_labels == ("Q00-CR",)


def test_build_instrument_deploy_requests_filters_by_target_labels() -> None:
    """Given selected targets, when building requests, then only requested targets are included."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )
    ctrl_port = SimpleNamespace(id="BOX1.CTRL0", box_id="BOX1", number=2)
    experiment_system = SimpleNamespace(
        gen_targets={
            "Q00": SimpleNamespace(
                label="Q00",
                frequency=4.20,
                type=TargetType.CTRL_GE,
                channel=SimpleNamespace(port=ctrl_port),
            ),
            "Q00-CR": SimpleNamespace(
                label="Q00-CR",
                frequency=4.35,
                type=TargetType.CTRL_CR,
                channel=SimpleNamespace(port=ctrl_port),
            ),
        },
        wiring_info=SimpleNamespace(read_in=[]),
        control_params=SimpleNamespace(
            get_frequency_margin=lambda _target_type: 0.1,
        ),
        get_box=lambda _box_id: SimpleNamespace(id="BOX1", name="quel3-02-a01"),
        get_mux_by_readout_port=lambda _port: None,
    )

    requests = manager._build_instrument_deploy_requests(  # noqa: SLF001
        experiment_system=cast(Any, experiment_system),
        box_ids=["BOX1"],
        target_labels=["Q00"],
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.target_labels == ("Q00",)
    assert request.frequency_range_min_hz == pytest.approx(4.10e9)
    assert request.frequency_range_max_hz == pytest.approx(4.30e9)


def test_deploy_instruments_from_target_registry_calls_session_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given grouped targets, when deploying, then deploy_instruments is called per deploy request."""
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

    ctrl_port = SimpleNamespace(id="BOX1.CTRL0", box_id="BOX1", number=2)
    target = SimpleNamespace(
        label="Q00",
        frequency=4.2,
        type=TargetType.CTRL_GE,
        channel=SimpleNamespace(port=ctrl_port),
    )
    experiment_system = SimpleNamespace(
        gen_targets={"Q00": target},
        wiring_info=SimpleNamespace(read_in=[]),
        control_params=SimpleNamespace(
            get_frequency_margin=lambda _target_type: 0.1,
        ),
        get_box=lambda _box_id: SimpleNamespace(id="BOX1", name="quel3-02-a01"),
        get_mux_by_readout_port=lambda _port: None,
    )

    deployed = manager.deploy_instruments_from_target_registry(
        experiment_system=cast(Any, experiment_system),
        box_ids=["BOX1"],
    )

    assert create_session_calls == [("quel3-02-a01:tx_p02",)]
    assert len(deploy_calls) == 1
    port_id, definitions = deploy_calls[0]
    assert port_id == "quel3-02-a01:tx_p02"
    definition = cast(Any, definitions[0])
    assert definition.mode == "fixed_timeline"
    assert definition.role == "transmitter"
    assert definition.profile.frequency_range_min == pytest.approx(4.1e9)
    assert definition.profile.frequency_range_max == pytest.approx(4.3e9)
    assert definition.alias.startswith("inst_transmitter_quel3-02-a01_tx_p02_q00")
    assert manager.target_alias_map == {"Q00": definition.alias}
    assert definition.alias in deployed


def test_build_instrument_deploy_requests_raises_when_frequency_margin_reaches_nyquist() -> (
    None
):
    """Given an oversized margin, when building requests, then QuEL-3 fails fast before deploy."""
    manager = Quel3ConfigurationManager(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )
    ctrl_port = SimpleNamespace(id="BOX1.CTRL0", box_id="BOX1", number=2)
    experiment_system = SimpleNamespace(
        gen_targets={
            "Q00": SimpleNamespace(
                label="Q00",
                frequency=4.20,
                type=TargetType.CTRL_GE,
                channel=SimpleNamespace(port=ctrl_port),
            ),
        },
        wiring_info=SimpleNamespace(read_in=[]),
        control_params=SimpleNamespace(
            get_frequency_margin=lambda _target_type: 1.25,
        ),
        get_box=lambda _box_id: SimpleNamespace(id="BOX1", name="quel3-02-a01"),
        get_mux_by_readout_port=lambda _port: None,
    )

    with pytest.raises(
        ValueError, match="frequency_margin must be smaller than Nyquist"
    ):
        manager._build_instrument_deploy_requests(  # noqa: SLF001
            experiment_system=cast(Any, experiment_system),
            box_ids=["BOX1"],
        )


def test_quel3_synchronizer_deploys_from_last_synced_experiment_system() -> None:
    """Given synchronized experiment system, when pushing hardware sync, then synchronizer delegates deployment."""
    calls: list[tuple[object, tuple[str, ...], tuple[str, ...]]] = []

    class _FakeConfigurationManager:
        def deploy_instruments_from_target_registry(
            self,
            *,
            experiment_system: object,
            box_ids: list[str],
            target_labels: list[str] | None = None,
        ) -> dict[str, tuple[object, ...]]:
            calls.append(
                (experiment_system, tuple(box_ids), tuple(target_labels or ()))
            )
            return {}

    backend_controller = SimpleNamespace(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )
    synchronizer = Quel3SystemSynchronizer(
        backend_controller=cast(Any, backend_controller),
        configuration_manager=cast(Any, _FakeConfigurationManager()),
    )

    experiment_system = object()
    synchronizer.sync_experiment_system_to_backend_controller(
        cast(Any, experiment_system)
    )
    boxes = [SimpleNamespace(id="BOX1"), SimpleNamespace(id="BOX2")]

    synchronizer.sync_experiment_system_to_hardware(
        boxes=cast(Any, boxes),
        parallel=True,
        target_labels=["Q00", "RQ00"],
    )

    assert calls == [(experiment_system, ("BOX1", "BOX2"), ("Q00", "RQ00"))]


def test_quel3_synchronizer_raises_if_push_called_before_load() -> None:
    """Given unsynchronized state, when pushing on QuEL-3, then synchronizer raises RuntimeError."""
    backend_controller = SimpleNamespace(
        quelware_endpoint="localhost",
        quelware_port=50051,
    )
    synchronizer = Quel3SystemSynchronizer(
        backend_controller=cast(Any, backend_controller),
    )

    with pytest.raises(RuntimeError, match="Call load\\(\\) before push\\(\\)"):
        synchronizer.sync_experiment_system_to_hardware(
            boxes=cast(Any, [SimpleNamespace(id="BOX1")]),
            parallel=None,
        )
