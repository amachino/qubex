"""Tests for QuEL-1 sideband defaults and push behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.system.control_system import Box, CapPort, GenPort, PortType
from qubex.system.quel1.quel1_system_synchronizer import Quel1SystemSynchronizer


def test_mntr_out_sideband_defaults_to_none() -> None:
    """Given a QuEL-1 box, when building ports, then MNTR_OUT sideband defaults to None."""
    box = Box.new(
        id="B0",
        name="BOX0",
        type="quel1-a",
        address="127.0.0.1",
        adapter="A0",
    )
    mntr_out = next(port for port in box.ports if port.type == PortType.MNTR_OUT)
    assert isinstance(mntr_out, GenPort)

    assert mntr_out.sideband is None


def test_sync_mntr_in_port_passes_sideband_none_to_backend() -> None:
    """Given QuEL-1 MNTR_IN push, when syncing, then sideband=None is forwarded."""

    class _BackendController:
        def __init__(self) -> None:
            self.config_port_calls: list[dict[str, Any]] = []
            self.config_channel_calls: list[dict[str, Any]] = []
            self.config_runit_calls: list[dict[str, Any]] = []

        def config_port(self, **kwargs: Any) -> None:
            self.config_port_calls.append(dict(kwargs))

        def config_channel(self, **kwargs: Any) -> None:
            self.config_channel_calls.append(dict(kwargs))

        def config_runit(self, **kwargs: Any) -> None:
            self.config_runit_calls.append(dict(kwargs))

    backend_controller = _BackendController()
    synchronizer = Quel1SystemSynchronizer(
        backend_controller=cast(Any, backend_controller)
    )

    box = Box.new(
        id="B0",
        name="BOX0",
        type="quel1-a",
        address="127.0.0.1",
        adapter="A0",
        port_numbers=[1],
    )
    port = next(port for port in box.ports if port.type == PortType.MNTR_IN)
    assert isinstance(port, CapPort)
    port.lo_freq = None
    port.cnco_freq = None
    port.rfswitch = "open"
    for channel in port.channels:
        channel.fnco_freq = 0

    synchronizer.sync_box_to_hardware(cast(Any, box))

    mntr_call = next(
        call
        for call in backend_controller.config_port_calls
        if call["port"] == port.number
    )
    assert "sideband" in mntr_call
    assert mntr_call["sideband"] is None


def test_sync_read_out_port_passes_sideband_to_backend() -> None:
    """Given QuEL-1 READ_OUT push, when syncing, then sideband is forwarded."""

    class _BackendController:
        def __init__(self) -> None:
            self.config_port_calls: list[dict[str, Any]] = []
            self.config_channel_calls: list[dict[str, Any]] = []
            self.config_runit_calls: list[dict[str, Any]] = []

        def config_port(self, **kwargs: Any) -> None:
            self.config_port_calls.append(dict(kwargs))

        def config_channel(self, **kwargs: Any) -> None:
            self.config_channel_calls.append(dict(kwargs))

        def config_runit(self, **kwargs: Any) -> None:
            self.config_runit_calls.append(dict(kwargs))

    backend_controller = _BackendController()
    synchronizer = Quel1SystemSynchronizer(
        backend_controller=cast(Any, backend_controller)
    )

    box = Box.new(
        id="B0",
        name="BOX0",
        type="quel1-a",
        address="127.0.0.1",
        adapter="A0",
        port_numbers=[1],
    )
    port = box.get_port(1)
    assert isinstance(port, GenPort)
    port.lo_freq = 10_000_000_000
    port.cnco_freq = 1_500_000_000
    port.vatt = None
    port.sideband = "U"
    port.fullscale_current = None
    port.rfswitch = "pass"
    for channel in port.channels:
        channel.fnco_freq = 0

    synchronizer.sync_box_to_hardware(cast(Any, box))

    read_out_call = next(
        call
        for call in backend_controller.config_port_calls
        if call["port"] == port.number
    )
    assert read_out_call["sideband"] == "U"


def test_sync_capture_port_passes_sideband_none_to_backend() -> None:
    """Given QuEL-1 capture push, when syncing one port, then sideband=None is forwarded."""

    class _BackendController:
        def __init__(self) -> None:
            self.config_port_calls: list[dict[str, Any]] = []
            self.config_channel_calls: list[dict[str, Any]] = []
            self.config_runit_calls: list[dict[str, Any]] = []

        def config_port(self, **kwargs: Any) -> None:
            self.config_port_calls.append(dict(kwargs))

        def config_channel(self, **kwargs: Any) -> None:
            self.config_channel_calls.append(dict(kwargs))

        def config_runit(self, **kwargs: Any) -> None:
            self.config_runit_calls.append(dict(kwargs))

    backend_controller = _BackendController()
    synchronizer = Quel1SystemSynchronizer(
        backend_controller=cast(Any, backend_controller)
    )

    box = Box.new(
        id="B0",
        name="BOX0",
        type="quel1-a",
        address="127.0.0.1",
        adapter="A0",
        port_numbers=[0],
    )
    port = box.get_port(0)
    assert isinstance(port, CapPort)
    port.lo_freq = None
    port.cnco_freq = None
    port.rfswitch = "open"
    for channel in port.channels:
        channel.fnco_freq = 0

    synchronizer.sync_box_to_hardware(cast(Any, box))

    read_in_call = next(
        call
        for call in backend_controller.config_port_calls
        if call["port"] == port.number
    )
    assert "sideband" in read_in_call
    assert read_in_call["sideband"] is None


def test_sync_model_skips_clockmaster_for_single_box_without_address() -> None:
    """Given one QuEL-1 box without clock master, when syncing model, then clock master setup is skipped."""

    class _BackendController:
        def __init__(self) -> None:
            self.define_clockmaster_calls: list[dict[str, Any]] = []

        def define_clockmaster(self, **kwargs: Any) -> None:
            self.define_clockmaster_calls.append(dict(kwargs))

        def set_box_options(self, *_: Any, **__: Any) -> None:
            return None

        def define_box(self, **_: Any) -> None:
            return None

        def define_port(self, **_: Any) -> None:
            return None

        def define_channel(self, **_: Any) -> None:
            return None

        def add_channel_target_relation(self, **_: Any) -> None:
            return None

        def define_target(self, **_: Any) -> None:
            return None

        def clear_command_queue(self) -> None:
            return None

        def clear_cache(self) -> None:
            return None

    backend_controller = _BackendController()
    synchronizer = Quel1SystemSynchronizer(
        backend_controller=cast(Any, backend_controller)
    )
    box = Box.new(
        id="B0",
        name="BOX0",
        type="quel1-a",
        address="127.0.0.1",
        adapter="A0",
        port_numbers=[],
    )
    experiment_system = SimpleNamespace(
        control_system=SimpleNamespace(clock_master_address=None, boxes=[box]),
        control_params=SimpleNamespace(),
        all_targets=[],
    )

    synchronizer.sync_experiment_system_to_backend_controller(
        cast(Any, experiment_system)
    )

    assert backend_controller.define_clockmaster_calls == []


def test_sync_model_requires_clockmaster_for_multiple_boxes() -> None:
    """Given multiple QuEL-1 boxes without clock master, when syncing model, then ValueError is raised."""

    class _BackendController:
        def define_clockmaster(self, **_: Any) -> None:
            return None

    synchronizer = Quel1SystemSynchronizer(
        backend_controller=cast(Any, _BackendController())
    )
    box_a = Box.new(
        id="B0",
        name="BOX0",
        type="quel1-a",
        address="127.0.0.1",
        adapter="A0",
        port_numbers=[],
    )
    box_b = Box.new(
        id="B1",
        name="BOX1",
        type="quel1-a",
        address="127.0.0.2",
        adapter="A1",
        port_numbers=[],
    )
    experiment_system = SimpleNamespace(
        control_system=SimpleNamespace(clock_master_address=None, boxes=[box_a, box_b]),
        control_params=SimpleNamespace(),
        all_targets=[],
    )

    with pytest.raises(
        ValueError,
        match=r"Clock master address is required for multi-box QuEL-1 synchronization\.",
    ):
        synchronizer.sync_experiment_system_to_backend_controller(
            cast(Any, experiment_system)
        )
