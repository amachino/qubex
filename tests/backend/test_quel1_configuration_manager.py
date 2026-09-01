"""Tests for QuEL-1 configuration manager behavior."""

from __future__ import annotations

from typing import Any, cast

import pytest

from qubex.backend.quel1.managers.configuration_manager import Quel1ConfigurationManager
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContext

Port = int | tuple[int, int]


class _FakeBox:
    def __init__(
        self,
        *,
        boxtype: str = "quel1se-riken8",
        input_ports: tuple[Port, ...] = (),
    ) -> None:
        self.boxtype = boxtype
        self._input_ports = input_ports
        self.config_port_calls: list[dict[str, Any]] = []
        self.dump_port_calls: list[Port] = []

    def get_input_ports(self) -> tuple[Port, ...]:
        """Return fixed input ports."""
        return self._input_ports

    def config_port(self, **kwargs: Any) -> None:
        """Record config_port kwargs."""
        self.config_port_calls.append(kwargs)

    def dump_port(self, port: Port) -> dict[str, Any]:
        """Record dump_port calls."""
        self.dump_port_calls.append(port)
        return {}


class _BoxPoolStub:
    def __init__(self, box: _FakeBox) -> None:
        self._boxes = {"B0": (box,)}


class _RuntimeContextStub:
    is_connected = True

    def __init__(self, box: _FakeBox) -> None:
        self.boxpool = _BoxPoolStub(box)
        self.validated_box_names: list[str] = []

    def validate_box_availability(self, box_name: str) -> None:
        """Record box availability checks."""
        self.validated_box_names.append(box_name)


def _configure_r8_port(
    *,
    port: Port,
    input_ports: tuple[Port, ...] = (),
) -> tuple[_RuntimeContextStub, _FakeBox]:
    box = _FakeBox(input_ports=input_ports)
    runtime_context = _RuntimeContextStub(box)
    manager = Quel1ConfigurationManager(
        runtime_context=cast(Quel1RuntimeContext, runtime_context)
    )

    manager.config_port(
        box_name="B0",
        port=port,
        lo_freq_hz=5_000_000_000,
        cnco_freq_hz=100_000_000,
        cnco_locked_with=None,
        vatt=2048,
        sideband="U",
        fullscale_current=16383,
        rfswitch="pass",
    )

    return runtime_context, box


@pytest.mark.parametrize(
    (
        "port",
        "input_ports",
        "expected_lo_freq",
        "expected_vatt",
        "expected_sideband",
    ),
    [
        (1, (), 5_000_000_000, 2048, "U"),
        (0, (0,), 5_000_000_000, None, None),
        (3, (), None, None, None),
    ],
)
def test_r8_config_port_filters_only_unsupported_mixer_fields(
    port: Port,
    input_ports: tuple[Port, ...],
    expected_lo_freq: int | None,
    expected_vatt: int | None,
    expected_sideband: str | None,
) -> None:
    """Given R8 port traits, when configuring a port, then unsupported mixer fields are dropped."""
    runtime_context, box = _configure_r8_port(port=port, input_ports=input_ports)

    assert runtime_context.validated_box_names == ["B0"]
    assert box.dump_port_calls == []
    assert box.config_port_calls == [
        {
            "port": port,
            "lo_freq": expected_lo_freq,
            "cnco_freq": 100_000_000,
            "cnco_locked_with": None,
            "vatt": expected_vatt,
            "sideband": expected_sideband,
            "fullscale_current": 16383,
            "rfswitch": "pass",
        }
    ]
