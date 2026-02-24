# ruff: noqa: SLF001

"""Tests for option-driven config options in Quel1BackendController."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Any, cast

import pytest

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

    def link_status(self) -> dict[int, bool]:
        """Return a fixed link status."""
        return self._status

    def relinkup(self, **kwargs: Any) -> None:
        """Record relinkup kwargs."""
        self.relinkup_calls.append(kwargs)

    def reconnect(self, **kwargs: Any) -> None:
        """Accept reconnect calls."""


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
    """Given legacy config_path kwarg, constructor raises TypeError."""
    with pytest.raises(TypeError, match="config_path"):
        cast(Any, Quel1BackendController)(config_path="dummy")


def test_constructor_allows_runtime_context_injection() -> None:
    """Given injected runtime context, default managers share the same context."""
    runtime_context = cast(Any, object())

    controller = cast(Any, Quel1BackendController)(runtime_context=runtime_context)

    assert controller._runtime_context is runtime_context
    assert controller._connection_manager._runtime_context is runtime_context
    assert controller._clock_manager._runtime_context is runtime_context
    assert controller._execution_manager._runtime_context is runtime_context
    assert controller._configuration_manager._runtime_context is runtime_context
    assert controller._skew_manager._runtime_context is runtime_context


def test_constructor_allows_manager_injection() -> None:
    """Given injected managers, constructor uses provided manager instances."""
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
    assert relinkup_kwargs["config_options"] == [
        _FakeQuel1ConfigOption.SE8_MXFE1_AWG2222
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


def test_linkup_uses_relaxed_noise_threshold_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no threshold, when linkup runs, then relaxed threshold is used."""
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
    assert reconnect_calls[0]["background_noise_threshold"] == 10000


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
