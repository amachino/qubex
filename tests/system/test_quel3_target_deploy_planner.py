"""Tests for QuEL-3 deploy planning and synchronizer integration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel3.models import InstrumentDeployRequest
from qubex.system.quel3 import Quel3SystemSynchronizer, Quel3TargetDeployPlanner
from qubex.system.target_type import TargetType


def test_build_deploy_requests_creates_one_request_per_target() -> None:
    """Given selected boxes, planner should create one deploy request per target."""
    planner = Quel3TargetDeployPlanner()
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

    requests = planner.build_deploy_requests(
        experiment_system=cast(Any, experiment_system),
        box_ids=["BOX1"],
    )

    request_by_target = {request.target_labels[0]: request for request in requests}
    assert set(request_by_target) == {"RQ00", "Q00", "Q00-CR"}

    read_request = request_by_target["RQ00"]
    assert read_request.port_id == "quel3-02-a01:trx_p00p01"
    assert read_request.role == "TRANSCEIVER"
    assert read_request.alias == "RQ00"
    assert read_request.frequency_range_min_hz == pytest.approx(5.9e9)
    assert read_request.frequency_range_max_hz == pytest.approx(6.1e9)
    assert read_request.target_labels == ("RQ00",)

    ctrl_request = request_by_target["Q00"]
    assert ctrl_request.port_id == "quel3-02-a01:tx_p02"
    assert ctrl_request.role == "TRANSMITTER"
    assert ctrl_request.alias == "Q00"
    assert ctrl_request.frequency_range_min_hz == pytest.approx(4.10e9)
    assert ctrl_request.frequency_range_max_hz == pytest.approx(4.30e9)
    assert ctrl_request.target_labels == ("Q00",)

    cr_request = request_by_target["Q00-CR"]
    assert cr_request.port_id == "quel3-02-a01:tx_p02"
    assert cr_request.role == "TRANSMITTER"
    assert cr_request.alias == "Q00-CR"
    assert cr_request.frequency_range_min_hz == pytest.approx(4.25e9)
    assert cr_request.frequency_range_max_hz == pytest.approx(4.45e9)
    assert cr_request.target_labels == ("Q00-CR",)


def test_build_deploy_requests_filters_by_target_labels() -> None:
    """Given selected targets, planner should include only requested labels."""
    planner = Quel3TargetDeployPlanner()
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

    requests = planner.build_deploy_requests(
        experiment_system=cast(Any, experiment_system),
        box_ids=["BOX1"],
        target_labels=["Q00"],
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.target_labels == ("Q00",)
    assert request.frequency_range_min_hz == pytest.approx(4.10e9)
    assert request.frequency_range_max_hz == pytest.approx(4.30e9)


def test_build_deploy_requests_raises_when_frequency_margin_reaches_nyquist() -> None:
    """Given an oversized margin, planner should fail fast before deploy."""
    planner = Quel3TargetDeployPlanner()
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
        planner.build_deploy_requests(
            experiment_system=cast(Any, experiment_system),
            box_ids=["BOX1"],
        )


def test_quel3_synchronizer_plans_then_deploys_from_hardware_sync_input() -> None:
    """Given push input, synchronizer should plan requests from that experiment system."""
    planner_calls: list[tuple[object, tuple[str, ...], tuple[str, ...]]] = []
    deploy_calls: list[tuple[tuple[InstrumentDeployRequest, ...], bool]] = []
    requests = (
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p02",
            role="TRANSMITTER",
            frequency_range_min_hz=4.1e9,
            frequency_range_max_hz=4.3e9,
            alias="Q00",
            target_labels=("Q00",),
        ),
    )

    class _FakeDeployPlanner:
        def build_deploy_requests(
            self,
            *,
            experiment_system: object,
            box_ids: list[str],
            target_labels: list[str] | None = None,
        ) -> tuple[InstrumentDeployRequest, ...]:
            planner_calls.append(
                (experiment_system, tuple(box_ids), tuple(target_labels or ()))
            )
            return requests

    class _FakeBackendController:
        def deploy_instruments(
            self,
            *,
            requests: tuple[InstrumentDeployRequest, ...],
            parallel: bool = True,
        ) -> dict[str, tuple[object, ...]]:
            deploy_calls.append((requests, parallel))
            return {}

    backend_controller = cast(Any, _FakeBackendController())
    synchronizer = Quel3SystemSynchronizer(
        backend_controller=backend_controller,
        deploy_planner=cast(Any, _FakeDeployPlanner()),
    )

    experiment_system = object()
    boxes = [SimpleNamespace(id="BOX1"), SimpleNamespace(id="BOX2")]

    synchronizer.sync_experiment_system_to_hardware(
        experiment_system=cast(Any, experiment_system),
        boxes=cast(Any, boxes),
        parallel=True,
        target_labels=["Q00", "RQ00"],
    )

    assert planner_calls == [(experiment_system, ("BOX1", "BOX2"), ("Q00", "RQ00"))]
    assert deploy_calls == [(requests, True)]


def test_quel3_synchronizer_defaults_parallel_deploy_to_true() -> None:
    """Given missing parallel flag, synchronizer should default deploy to parallel true."""
    deploy_calls: list[bool] = []
    expected_requests = (
        InstrumentDeployRequest(
            port_id="quel3-02-a01:tx_p02",
            role="TRANSMITTER",
            frequency_range_min_hz=4.1e9,
            frequency_range_max_hz=4.3e9,
            alias="Q00",
            target_labels=("Q00",),
        ),
    )

    class _FakeDeployPlanner:
        def build_deploy_requests(
            self,
            *,
            experiment_system: object,
            box_ids: list[str],
            target_labels: list[str] | None = None,
        ) -> tuple[InstrumentDeployRequest, ...]:
            del experiment_system, box_ids, target_labels
            return expected_requests

    class _FakeBackendController:
        def deploy_instruments(
            self,
            *,
            requests: tuple[InstrumentDeployRequest, ...],
            parallel: bool = True,
        ) -> dict[str, tuple[object, ...]]:
            assert requests == expected_requests
            deploy_calls.append(parallel)
            return {}

    synchronizer = Quel3SystemSynchronizer(
        backend_controller=cast(Any, _FakeBackendController()),
        deploy_planner=cast(Any, _FakeDeployPlanner()),
    )

    synchronizer.sync_experiment_system_to_hardware(
        experiment_system=cast(Any, object()),
        boxes=cast(Any, [SimpleNamespace(id="BOX1")]),
        parallel=None,
    )

    assert deploy_calls == [True]


def test_quel3_synchronizer_does_not_cache_experiment_system_for_push() -> None:
    """Given controller-sync input, QuEL-3 push should use the experiment system passed to hardware sync."""
    planner_calls: list[object] = []

    class _FakeDeployPlanner:
        def build_deploy_requests(
            self,
            *,
            experiment_system: object,
            box_ids: list[str],
            target_labels: list[str] | None = None,
        ) -> tuple[InstrumentDeployRequest, ...]:
            del box_ids, target_labels
            planner_calls.append(experiment_system)
            return ()

    class _FakeBackendController:
        def deploy_instruments(
            self,
            *,
            requests: tuple[InstrumentDeployRequest, ...],
            parallel: bool = True,
        ) -> dict[str, tuple[object, ...]]:
            _ = parallel
            assert requests == ()
            return {}

    synchronizer = Quel3SystemSynchronizer(
        backend_controller=cast(Any, _FakeBackendController()),
        deploy_planner=cast(Any, _FakeDeployPlanner()),
    )
    synced_experiment_system = object()
    pushed_experiment_system = object()
    synchronizer.sync_experiment_system_to_backend_controller(
        cast(Any, synced_experiment_system)
    )

    synchronizer.sync_experiment_system_to_hardware(
        experiment_system=cast(Any, pushed_experiment_system),
        boxes=cast(Any, [SimpleNamespace(id="BOX1")]),
        parallel=None,
    )

    assert planner_calls == [pushed_experiment_system]
