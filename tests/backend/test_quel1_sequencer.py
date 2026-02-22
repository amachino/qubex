"""Tests for QuEL-1 sequencer compatibility behavior."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import qubex.backend.quel1.compat.quel1_sequencer as sequencer_module
from qubex.backend.quel1.compat.quel1_sequencer import Quel1Sequencer


@dataclass
class _FakeBoxSetting:
    box_name: str


@dataclass
class _FakePortSetting:
    port: int


class _FakeBox:
    def get_input_ports(self) -> set[int]:
        """Return input-port numbers."""
        return {0}

    def get_output_ports(self) -> set[int]:
        """Return output-port numbers."""
        return {1}


class _FakeBoxPool:
    def __init__(self) -> None:
        self._box = _FakeBox()

    def get_box(self, box_name: str) -> tuple[_FakeBox, object]:
        """Return one box tuple for a given box name."""
        _ = box_name
        return self._box, object()


class _FakePortConfigAcquirer:
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        """Record constructor kwargs for assertions."""
        self.__class__.calls.append(kwargs)


class _FakeConverter:
    cap_kwargs: ClassVar[dict[str, Any]] = {}
    gen_kwargs: ClassVar[dict[str, Any]] = {}

    @classmethod
    def convert_to_cap_device_specific_sequence(
        cls, **kwargs: Any
    ) -> dict[tuple[str, int, int], str]:
        """Return fixed capture settings and record kwargs."""
        cls.cap_kwargs = kwargs
        return {("B0", 0, 0): "cap"}

    @classmethod
    def convert_to_gen_device_specific_sequence(
        cls, **kwargs: Any
    ) -> dict[tuple[str, int, int], str]:
        """Return fixed generation settings and record kwargs."""
        cls.gen_kwargs = kwargs
        return {("B0", 1, 0): "gen"}


@dataclass
class _FakeDriverSymbols:
    package_name: str = "qxdriver_quel"
    Converter: type[_FakeConverter] = _FakeConverter


def test_generate_e7_settings_uses_boxpool_port_config_path(monkeypatch: Any) -> None:
    """Given sequencer wrapper, when generating settings, then PortConfigAcquirer is built without driver."""
    monkeypatch.setattr(sequencer_module, "driver", _FakeDriverSymbols())
    monkeypatch.setattr(
        sequencer_module,
        "import_module",
        lambda _name: SimpleNamespace(PortConfigAcquirer=_FakePortConfigAcquirer),
    )
    _FakePortConfigAcquirer.calls = []
    _FakeConverter.cap_kwargs = {}
    _FakeConverter.gen_kwargs = {}

    sequencer = cast(Any, object.__new__(Quel1Sequencer))
    sequencer.resource_map = {
        "RQ00": [
            {
                "box": _FakeBoxSetting("B0"),
                "port": _FakePortSetting(0),
                "channel_number": 0,
                "target": {"frequency": 6.0},
            },
            {
                "box": _FakeBoxSetting("B0"),
                "port": _FakePortSetting(1),
                "channel_number": 0,
                "target": {"frequency": 6.0},
            },
        ]
    }
    sequencer.cap_sampled_sequence = {"RQ00": object()}
    sequencer.gen_sampled_sequence = {"RQ00": object()}
    sequencer.repeats = 64
    sequencer.interval = 128
    sequencer.integral_mode = "integral"
    sequencer.dsp_demodulation = True
    sequencer.software_demodulation = False
    sequencer.enable_sum = False
    sequencer.enable_classification = False
    sequencer.line_param0 = (1.0, 0.0, 0.0)
    sequencer.line_param1 = (0.0, 1.0, 0.0)

    cap_settings, gen_settings, cap_resource_map = cast(
        Any, sequencer
    ).generate_e7_settings(cast(Any, _FakeBoxPool()))

    assert cap_settings == {("B0", 0, 0): "cap"}
    assert gen_settings == {("B0", 1, 0): "gen"}
    assert cap_resource_map["RQ00"]["port"].port == 0
    assert _FakeConverter.cap_kwargs["resource_map"]["RQ00"]["port"].port == 0
    assert _FakeConverter.gen_kwargs["resource_map"]["RQ00"]["port"].port == 1
    assert {call["port"] for call in _FakePortConfigAcquirer.calls} == {0, 1}
    assert all("driver" not in call for call in _FakePortConfigAcquirer.calls)
