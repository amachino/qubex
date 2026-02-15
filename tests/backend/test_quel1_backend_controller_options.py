# ruff: noqa: SLF001

"""Tests for option-driven config options in Quel1BackendController."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Any, cast

import pytest

from qubex.backend.quel1 import quel1_backend_controller as module
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
    return controller


def _override_driver_classes(
    controller: Quel1BackendController, **overrides: Any
) -> None:
    """Replace selected driver classes in one controller instance."""
    cast(Any, controller)._driver = replace(cast(Any, controller)._driver, **overrides)


def test_relinkup_uses_default_awg2222_for_r8(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given R8 box without options, when relinkup runs, then default awg2222 is used."""
    controller = _make_controller()
    fake_box = _FakeBox("quel1se-riken8", {0: False})
    _override_driver_classes(controller, Quel1ConfigOption=_FakeQuel1ConfigOption)
    monkeypatch.setattr(controller, "_check_box_availability", lambda _: None)
    monkeypatch.setattr(controller, "_create_box", lambda *args, **kwargs: fake_box)

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
    monkeypatch.setattr(controller, "_check_box_availability", lambda _: None)
    monkeypatch.setattr(controller, "_create_box", lambda *args, **kwargs: fake_box)
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
    monkeypatch.setattr(controller, "_check_box_availability", lambda _: None)
    monkeypatch.setattr(controller, "_create_box", lambda *args, **kwargs: fake_box)
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

    monkeypatch.setattr(controller, "_check_box_availability", lambda _: None)
    monkeypatch.setattr(
        controller, "_get_existing_or_create_box", lambda *args, **kwargs: fake_box
    )

    controller.linkup("B0")

    assert reconnect_calls
    assert (
        reconnect_calls[0]["background_noise_threshold"]
        == module._RELAXED_NOISE_THRESHOLD
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

    monkeypatch.setattr(controller, "_check_box_availability", lambda _: None)
    monkeypatch.setattr(
        controller, "_get_existing_or_create_box", lambda *args, **kwargs: fake_box
    )

    controller.linkup("B0", noise_threshold=12345)

    assert reconnect_calls
    assert reconnect_calls[0]["background_noise_threshold"] == 12345
