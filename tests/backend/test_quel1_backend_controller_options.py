# ruff: noqa: SLF001

"""Tests for option-driven config options in Quel1BackendController."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel1.managers.configuration_manager import (
    Quel1ConfigurationManager,
)
from qubex.backend.quel1.quel1_backend_constants import (
    DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT,
    DEFAULT_BACKGROUND_NOISE_THRESHOLD_RELINKUP,
)
from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController


class _FakeQuel1ConfigOption(str, Enum):
    SE8_MXFE1_AWG1331 = "se8_mxfe1_awg1331"
    SE8_MXFE1_AWG2222 = "se8_mxfe1_awg2222"
    REFCLK_CORRECTED_MXFE1 = "refclk_corrected_mxfe1"


class _FakeBox:
    def __init__(self, boxtype: str, status: dict[int, bool]) -> None:
        self.boxtype = boxtype
        self._status = status
        self.relinkup_calls: list[dict[str, Any]] = []
        self.reconnect_calls: list[dict[str, Any]] = []

    def link_status(self) -> dict[int, bool]:
        """Return a fixed link status."""
        return self._status

    def relinkup(self, **kwargs: Any) -> None:
        """Record relinkup kwargs."""
        self.relinkup_calls.append(kwargs)

    def reconnect(self, **kwargs: Any) -> None:
        """Accept reconnect calls."""
        self.reconnect_calls.append(kwargs)


def _make_controller() -> Quel1BackendController:
    controller = Quel1BackendController()
    cast(Any, controller)._qubecalib = object()
    cast(Any, controller)._runtime_context._qubecalib = cast(Any, controller)._qubecalib
    return controller


def _override_driver_classes(
    controller: Quel1BackendController, **overrides: Any
) -> None:
    """Replace selected driver classes in one controller instance."""
    driver = replace(cast(Any, controller.driver), **overrides)
    cast(Any, controller)._runtime_context._driver = driver


def test_constructor_rejects_config_path_argument() -> None:
    """Given legacy config_path kwarg, when constructing Quel1BackendController, then TypeError is raised."""
    with pytest.raises(TypeError, match="config_path"):
        cast(Any, Quel1BackendController)(config_path="dummy")


def test_constructor_allows_runtime_context_injection() -> None:
    """Given injected runtime context, when constructing Quel1BackendController, then default managers share that context."""
    runtime_context = cast(Any, object())

    controller = cast(Any, Quel1BackendController)(runtime_context=runtime_context)

    assert controller._runtime_context is runtime_context
    assert controller._connection_manager._runtime_context is runtime_context
    assert controller._clock_manager._runtime_context is runtime_context
    assert controller._execution_manager._runtime_context is runtime_context
    assert controller._configuration_manager._runtime_context is runtime_context
    assert controller._skew_manager._runtime_context is runtime_context


def test_constructor_allows_manager_injection() -> None:
    """Given injected managers, when constructing Quel1BackendController, then the provided manager instances are used."""
    runtime_context = cast(Any, object())
    connection_manager = cast(Any, object())
    clock_manager = cast(Any, object())
    execution_manager = cast(Any, object())
    configuration_manager = cast(Any, object())
    skew_manager = cast(Any, object())

    controller = cast(Any, Quel1BackendController)(
        runtime_context=runtime_context,
        connection_manager=connection_manager,
        clock_manager=clock_manager,
        execution_manager=execution_manager,
        configuration_manager=configuration_manager,
        skew_manager=skew_manager,
    )

    assert controller._runtime_context is runtime_context
    assert controller._connection_manager is connection_manager
    assert controller._clock_manager is clock_manager
    assert controller._execution_manager is execution_manager
    assert controller._configuration_manager is configuration_manager
    assert controller._skew_manager is skew_manager


def test_config_port_forwards_cnco_locked_with_to_configuration_manager() -> None:
    """Given a CNCO lock target, when configuring a port, then it is forwarded."""
    calls: list[dict[str, Any]] = []

    class _ConfigurationManager:
        def config_port(self, **kwargs: Any) -> None:
            calls.append(dict(kwargs))

    controller = cast(Any, Quel1BackendController)(
        runtime_context=cast(Any, object()),
        connection_manager=cast(Any, object()),
        clock_manager=cast(Any, object()),
        execution_manager=cast(Any, object()),
        configuration_manager=_ConfigurationManager(),
        skew_manager=cast(Any, object()),
    )

    controller.config_port(
        "B0",
        port=4,
        lo_freq_hz=8_500_000_000,
        cnco_locked_with=1,
        rfswitch="loop",
    )

    assert calls == [
        {
            "box_name": "B0",
            "port": 4,
            "lo_freq_hz": 8_500_000_000,
            "cnco_freq_hz": None,
            "cnco_locked_with": 1,
            "vatt": None,
            "sideband": None,
            "fullscale_current": None,
            "rfswitch": "loop",
        }
    ]


def test_configuration_manager_passes_cnco_locked_with_to_box() -> None:
    """Given a CNCO lock target, when configuring hardware, then box config receives it."""

    class _Box:
        boxtype = "quel1-a"

        def __init__(self) -> None:
            self.config_port_calls: list[dict[str, Any]] = []

        def config_port(self, **kwargs: Any) -> None:
            self.config_port_calls.append(dict(kwargs))

    box = _Box()
    runtime_context = SimpleNamespace(
        is_connected=True,
        boxpool=SimpleNamespace(_boxes={"B0": (box,)}),
        validate_box_availability=lambda _box_name: None,
    )
    manager = Quel1ConfigurationManager(runtime_context=cast(Any, runtime_context))

    manager.config_port(
        box_name="B0",
        port=4,
        lo_freq_hz=8_500_000_000,
        cnco_freq_hz=None,
        cnco_locked_with=1,
        vatt=None,
        sideband=None,
        fullscale_current=None,
        rfswitch="loop",
    )

    assert box.config_port_calls == [
        {
            "port": 4,
            "lo_freq": 8_500_000_000,
            "cnco_freq": None,
            "cnco_locked_with": 1,
            "vatt": None,
            "sideband": None,
            "fullscale_current": None,
            "rfswitch": "loop",
        }
    ]


def test_configuration_manager_returns_loopback_source_ports() -> None:
    """Given an input port, when resolving loopbacks, then output ports are returned."""

    class _Box:
        def get_loopbacks_of_port(self, port: int) -> set[int]:
            assert port == 4
            return {1, 2}

    runtime_context = SimpleNamespace(
        is_connected=True,
        boxpool=SimpleNamespace(_boxes={"B0": (_Box(),)}),
        validate_box_availability=lambda _box_name: None,
    )
    manager = Quel1ConfigurationManager(runtime_context=cast(Any, runtime_context))

    loopbacks = manager.get_loopbacks_of_port(box_name="B0", port_number=4)

    assert loopbacks == {1, 2}


def test_default_relinkup_noise_threshold_is_1024() -> None:
    """Given backend constants, when reading relinkup threshold, then it is 1024."""
    assert DEFAULT_BACKGROUND_NOISE_THRESHOLD_RELINKUP == 1024.0


def test_relinkup_uses_default_awg2222_for_r8(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given R8 box without options, when relinkup runs, then default awg2222 is used."""
    controller = _make_controller()
    fake_box = _FakeBox("quel1se-riken8", {0: False})
    _override_driver_classes(controller, Quel1ConfigOption=_FakeQuel1ConfigOption)
    monkeypatch.setattr(
        controller._runtime_context, "validate_box_availability", lambda _: None
    )
    monkeypatch.setattr(
        controller._connection_manager,
        "_get_existing_or_create_box",
        lambda **kwargs: fake_box,
    )

    controller.relinkup("B0")

    relinkup_kwargs = fake_box.relinkup_calls[0]
    assert relinkup_kwargs["background_noise_threshold"] == (
        DEFAULT_BACKGROUND_NOISE_THRESHOLD_RELINKUP
    )
    assert relinkup_kwargs["config_options"] == [
        _FakeQuel1ConfigOption.SE8_MXFE1_AWG2222
    ]
    assert fake_box.reconnect_calls == [
        {"background_noise_threshold": DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT}
    ]


def test_relinkup_maps_explicit_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given explicit options, when relinkup runs, then options are converted and passed."""
    controller = _make_controller()
    fake_box = _FakeBox("quel1se-riken8", {0: False})
    _override_driver_classes(controller, Quel1ConfigOption=_FakeQuel1ConfigOption)
    monkeypatch.setattr(
        controller._runtime_context, "validate_box_availability", lambda _: None
    )
    monkeypatch.setattr(
        controller._connection_manager,
        "_get_existing_or_create_box",
        lambda **kwargs: fake_box,
    )
    controller.set_box_options(
        {
            "B0": (
                "se8_mxfe1_awg1331",
                "refclk_corrected_mxfe1",
            )
        }
    )

    controller.relinkup("B0")

    relinkup_kwargs = fake_box.relinkup_calls[0]
    assert relinkup_kwargs["config_options"] == [
        _FakeQuel1ConfigOption.SE8_MXFE1_AWG1331,
        _FakeQuel1ConfigOption.REFCLK_CORRECTED_MXFE1,
    ]


def test_relinkup_rejects_conflicting_awg_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given conflicting awg options, when relinkup runs, then ValueError is raised."""
    controller = _make_controller()
    fake_box = _FakeBox("quel1se-riken8", {0: False})
    _override_driver_classes(controller, Quel1ConfigOption=_FakeQuel1ConfigOption)
    monkeypatch.setattr(
        controller._runtime_context, "validate_box_availability", lambda _: None
    )
    monkeypatch.setattr(
        controller._connection_manager,
        "_get_existing_or_create_box",
        lambda **kwargs: fake_box,
    )
    controller.set_box_options({"B0": ("se8_mxfe1_awg1331", "se8_mxfe1_awg2222")})

    with pytest.raises(ValueError, match="Multiple AWG options are not allowed"):
        controller.relinkup("B0")


def test_relinkup_keeps_explicit_noise_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given explicit threshold, when relinkup runs, then provided threshold is used."""
    controller = _make_controller()
    fake_box = _FakeBox("quel1se-riken8", {0: False})
    _override_driver_classes(controller, Quel1ConfigOption=_FakeQuel1ConfigOption)
    monkeypatch.setattr(
        controller._runtime_context, "validate_box_availability", lambda _: None
    )
    monkeypatch.setattr(
        controller._connection_manager,
        "_get_existing_or_create_box",
        lambda **kwargs: fake_box,
    )

    controller.relinkup("B0", noise_threshold=12345)

    assert fake_box.relinkup_calls[0]["background_noise_threshold"] == 12345
    assert fake_box.reconnect_calls == [{"background_noise_threshold": 12345}]


def test_linkup_uses_default_reconnect_noise_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no threshold, when linkup runs, then reconnect threshold is used."""
    controller = _make_controller()
    fake_box = _FakeBox("quel1se-riken8", {0: True})
    reconnect_calls: list[dict[str, Any]] = []

    def _fake_reconnect(**kwargs: Any) -> None:
        reconnect_calls.append(kwargs)

    fake_box.reconnect = _fake_reconnect  # type: ignore[method-assign]

    monkeypatch.setattr(
        controller._runtime_context, "validate_box_availability", lambda _: None
    )
    monkeypatch.setattr(
        controller._connection_manager,
        "_get_existing_or_create_box",
        lambda **kwargs: fake_box,
    )

    controller.linkup("B0")

    assert reconnect_calls
    assert reconnect_calls[0]["background_noise_threshold"] == (
        DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT
    )


def test_linkup_keeps_explicit_noise_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given explicit threshold, when linkup runs, then provided threshold is used."""
    controller = _make_controller()
    fake_box = _FakeBox("quel1se-riken8", {0: True})
    reconnect_calls: list[dict[str, Any]] = []

    def _fake_reconnect(**kwargs: Any) -> None:
        reconnect_calls.append(kwargs)

    fake_box.reconnect = _fake_reconnect  # type: ignore[method-assign]

    monkeypatch.setattr(
        controller._runtime_context, "validate_box_availability", lambda _: None
    )
    monkeypatch.setattr(
        controller._connection_manager,
        "_get_existing_or_create_box",
        lambda **kwargs: fake_box,
    )

    controller.linkup("B0", noise_threshold=12345)

    assert reconnect_calls
    assert reconnect_calls[0]["background_noise_threshold"] == 12345
