"""Tests for QuEL-3 deploy planning and synchronizer integration."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel3.models import InstrumentConfiguration, InstrumentSpec
from qubex.system.quel3 import Quel3SystemSynchronizer, Quel3TargetDeployPlanner
from qubex.system.target_type import TargetType


def test_build_configuration_creates_one_instrument_per_target() -> None:
    """Selected boxes should produce one deployable instrument specification per target."""
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
        "Q00-Q01-CUSTOM": SimpleNamespace(
            label="Q00-Q01-CUSTOM",
            frequency=4.37,
            type=TargetType.CTRL_2Q,
            channel=SimpleNamespace(port=ctrl_port),
        ),
        "Q00-Q01-bSWAP": SimpleNamespace(
            label="Q00-Q01-bSWAP",
            frequency=4.40,
            type=TargetType.CTRL_2Q,
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

    configuration = planner.build_configuration(
        experiment_system=cast(Any, experiment_system),
        box_ids=["BOX1"],
    )

    spec_by_target = {spec.alias: spec for spec in configuration.instruments}
    assert set(spec_by_target) == {
        "RQ00",
        "Q00",
        "Q00-CR",
        "Q00-Q01-CUSTOM",
        "Q00-Q01-bSWAP",
    }

    read_spec = spec_by_target["RQ00"]
    assert read_spec.port_id == "quel3-02-a01:trx_p00p01"
    assert read_spec.role == "TRANSCEIVER"
    assert read_spec.alias == "RQ00"
    assert read_spec.frequency_range_min_hz == pytest.approx(5.9e9)
    assert read_spec.frequency_range_max_hz == pytest.approx(6.1e9)

    ctrl_spec = spec_by_target["Q00"]
    assert ctrl_spec.port_id == "quel3-02-a01:tx_p02"
    assert ctrl_spec.role == "TRANSMITTER"
    assert ctrl_spec.alias == "Q00"
    assert ctrl_spec.frequency_range_min_hz == pytest.approx(4.10e9)
    assert ctrl_spec.frequency_range_max_hz == pytest.approx(4.30e9)

    cr_spec = spec_by_target["Q00-CR"]
    assert cr_spec.port_id == "quel3-02-a01:tx_p02"
    assert cr_spec.role == "TRANSMITTER"
    assert cr_spec.alias == "Q00-CR"
    assert cr_spec.frequency_range_min_hz == pytest.approx(4.25e9)
    assert cr_spec.frequency_range_max_hz == pytest.approx(4.45e9)

    custom_2q_spec = spec_by_target["Q00-Q01-CUSTOM"]
    assert custom_2q_spec.port_id == "quel3-02-a01:tx_p02"
    assert custom_2q_spec.role == "TRANSMITTER"
    assert custom_2q_spec.alias == "Q00-Q01-CUSTOM"
    assert custom_2q_spec.frequency_range_min_hz == pytest.approx(4.27e9)
    assert custom_2q_spec.frequency_range_max_hz == pytest.approx(4.47e9)

    bswap_spec = spec_by_target["Q00-Q01-bSWAP"]
    assert bswap_spec.port_id == "quel3-02-a01:tx_p02"
    assert bswap_spec.role == "TRANSMITTER"
    assert bswap_spec.alias == "Q00-Q01-bSWAP"
    assert bswap_spec.frequency_range_min_hz == pytest.approx(4.30e9)
    assert bswap_spec.frequency_range_max_hz == pytest.approx(4.50e9)


def test_build_configuration_filters_by_target_labels() -> None:
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

    configuration = planner.build_configuration(
        experiment_system=cast(Any, experiment_system),
        box_ids=["BOX1"],
        target_labels=["Q00"],
    )

    assert len(configuration.instruments) == 1
    spec = configuration.instruments[0]
    assert spec.alias == "Q00"
    assert spec.frequency_range_min_hz == pytest.approx(4.10e9)
    assert spec.frequency_range_max_hz == pytest.approx(4.30e9)


def test_build_configuration_skips_targets_with_non_finite_frequency() -> None:
    """Given non-finite target frequency, planner should skip that target and continue."""
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
                frequency=math.nan,
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

    configuration = planner.build_configuration(
        experiment_system=cast(Any, experiment_system),
        box_ids=["BOX1"],
    )

    assert len(configuration.instruments) == 1
    spec = configuration.instruments[0]
    assert spec.alias == "Q00"
    assert spec.frequency_range_min_hz == pytest.approx(4.10e9)
    assert spec.frequency_range_max_hz == pytest.approx(4.30e9)


def test_build_configuration_raises_when_frequency_margin_reaches_nyquist() -> None:
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
        planner.build_configuration(
            experiment_system=cast(Any, experiment_system),
            box_ids=["BOX1"],
        )


def test_quel3_synchronizer_plans_then_deploys_from_hardware_sync_input() -> None:
    """Push should deploy the configuration planned from its selected targets and boxes."""
    planner_calls: list[tuple[object, tuple[str, ...], tuple[str, ...]]] = []
    deploy_calls: list[tuple[InstrumentConfiguration, bool]] = []
    expected_configuration = InstrumentConfiguration(
        instruments=(
            InstrumentSpec(
                port_id="quel3-02-a01:tx_p02",
                alias="Q00",
                role="TRANSMITTER",
                frequency_range_min_hz=4.1e9,
                frequency_range_max_hz=4.3e9,
            ),
        )
    )

    class _FakeDeployPlanner:
        def build_configuration(
            self,
            *,
            experiment_system: object,
            box_ids: list[str],
            target_labels: list[str] | None = None,
        ) -> InstrumentConfiguration:
            planner_calls.append(
                (experiment_system, tuple(box_ids), tuple(target_labels or ()))
            )
            return expected_configuration

    class _FakeBackendController:
        def deploy_instruments(
            self,
            *,
            configuration: InstrumentConfiguration,
            parallel: bool = True,
        ) -> None:
            deploy_calls.append((configuration, parallel))

    synchronizer = Quel3SystemSynchronizer(
        backend_controller=cast(Any, _FakeBackendController()),
        deploy_planner=cast(Any, _FakeDeployPlanner()),
    )
    experiment_system = object()

    synchronizer.sync_experiment_system_to_hardware(
        experiment_system=cast(Any, experiment_system),
        boxes=cast(Any, [SimpleNamespace(id="BOX1"), SimpleNamespace(id="BOX2")]),
        parallel=True,
        target_labels=["Q00", "RQ00"],
    )

    assert planner_calls == [(experiment_system, ("BOX1", "BOX2"), ("Q00", "RQ00"))]
    assert deploy_calls == [(expected_configuration, True)]


def test_quel3_synchronizer_defaults_parallel_deploy_to_true() -> None:
    """Push should default to parallel deployment when its parallel option is omitted."""
    deploy_calls: list[bool] = []
    expected_configuration = InstrumentConfiguration()

    class _FakeDeployPlanner:
        def build_configuration(
            self,
            *,
            experiment_system: object,
            box_ids: list[str],
            target_labels: list[str] | None = None,
        ) -> InstrumentConfiguration:
            return expected_configuration

    class _FakeBackendController:
        def deploy_instruments(
            self,
            *,
            configuration: InstrumentConfiguration,
            parallel: bool = True,
        ) -> None:
            assert configuration == expected_configuration
            deploy_calls.append(parallel)

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
    """Push should plan from its current experiment system without persisting a prior model."""
    planner_calls: list[object] = []

    class _FakeDeployPlanner:
        def build_configuration(
            self,
            *,
            experiment_system: object,
            box_ids: list[str],
            target_labels: list[str] | None = None,
        ) -> InstrumentConfiguration:
            planner_calls.append(experiment_system)
            return InstrumentConfiguration()

    class _FakeBackendController:
        def deploy_instruments(
            self,
            *,
            configuration: InstrumentConfiguration,
            parallel: bool = True,
        ) -> None:
            assert configuration.instruments == ()

    synchronizer = Quel3SystemSynchronizer(
        backend_controller=cast(Any, _FakeBackendController()),
        deploy_planner=cast(Any, _FakeDeployPlanner()),
    )
    pushed_experiment_system = object()
    synchronizer.sync_experiment_system_to_backend_controller(cast(Any, object()))

    synchronizer.sync_experiment_system_to_hardware(
        experiment_system=cast(Any, pushed_experiment_system),
        boxes=cast(Any, [SimpleNamespace(id="BOX1")]),
        parallel=None,
    )

    assert planner_calls == [pushed_experiment_system]


def test_quel3_synchronizer_projects_hardware_state_to_backend_settings() -> None:
    """Given selected boxes, synchronizer should fetch backend settings from hardware state."""
    calls: list[dict[str, object]] = []

    class _FakeHardwareStateReader:
        def fetch_backend_settings_from_hardware(
            self,
            *,
            unit_labels_by_box_id: dict[str, str],
            parallel: bool | None = None,
        ) -> dict[str, dict]:
            calls.append(
                {
                    "unit_labels_by_box_id": unit_labels_by_box_id,
                    "parallel": parallel,
                }
            )
            return {
                "BOX1": {
                    "instruments": {
                        "Q00": {
                            "resource_id": "unit-a:inst-q00",
                            "port_id": "unit-a:tx_p01",
                            "role": "TRANSMITTER",
                        }
                    }
                }
            }

    class _FakeBackendController:
        hardware_state_reader = _FakeHardwareStateReader()

    experiment_system = SimpleNamespace(
        get_box=lambda box_id: SimpleNamespace(id=box_id, name="unit-a"),
    )
    synchronizer = Quel3SystemSynchronizer(
        backend_controller=cast(Any, _FakeBackendController()),
    )

    fetched = synchronizer.fetch_backend_settings_from_hardware(
        experiment_system=cast(Any, experiment_system),
        box_ids=("BOX1",),
        parallel=False,
    )

    assert fetched == {
        "BOX1": {
            "instruments": {
                "Q00": {
                    "resource_id": "unit-a:inst-q00",
                    "port_id": "unit-a:tx_p01",
                    "role": "TRANSMITTER",
                }
            }
        }
    }
    assert calls == [
        {
            "unit_labels_by_box_id": {"BOX1": "unit-a"},
            "parallel": False,
        }
    ]


def test_quel3_synchronizer_does_not_restore_instruments_from_settings() -> None:
    """Settings synchronization should preserve existing instrument-cache contents."""
    cache = {"Q00": object()}
    controller = SimpleNamespace(_instrument_cache=cache)
    synchronizer = Quel3SystemSynchronizer(backend_controller=cast(Any, controller))

    synchronizer.sync_backend_settings_to_backend_controller(
        backend_settings={"BOX1": {"instruments": {"Q01": {"resource_id": "saved"}}}},
    )

    assert list(cache) == ["Q00"]
