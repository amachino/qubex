"""Tests for parallel sequencer execution in Quel1BackendController."""

from __future__ import annotations

from typing import Any

import pytest

from qubex.backend.quel1.execution import SequencerExecutionEngine
from qubex.backend.quel1.quel1_backend_controller import (
    Quel1BackendController,
    Quel1BackendRawResult,
)


def test_execute_sequencer_parallel_delegates_to_execution_engine(monkeypatch) -> None:
    """Given parallel mode, execute_sequencer_parallel delegates to engine and wraps result."""

    class _Sequencer:
        interval = 256

        def set_measurement_option(self, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

    called: dict[str, Any] = {}

    controller = Quel1BackendController()
    sequencer = _Sequencer()
    controller.__dict__["_boxpool"] = object()
    controller.__dict__["_quel1system"] = object()

    def _fake_execute_parallel(
        **kwargs: object,
    ) -> tuple[dict[str, str], dict[str, list[str]], dict]:
        called.update(kwargs)
        return {"Q00": "OK"}, {"Q00": ["RAW", "CPRM"]}, {}

    monkeypatch.setattr(
        SequencerExecutionEngine,
        "execute_parallel",
        staticmethod(_fake_execute_parallel),
    )

    result = controller.execute_sequencer_parallel(
        sequencer=sequencer,  # type: ignore[arg-type]
        repeats=16,
    )

    assert isinstance(result, Quel1BackendRawResult)
    assert result.status == {"Q00": "OK"}
    assert result.data == {"Q00": ["RAW", "CPRM"]}
    assert result.config == {}
    assert called["sequencer"] is sequencer
    assert called["boxpool"] is controller.boxpool
    assert called["system"] is controller.quel1system


def test_initialize_awg_and_capunits_parallel_calls_each_box(monkeypatch) -> None:
    """Given parallel init, when initializing boxes, then each box is processed once."""
    controller = Quel1BackendController()
    controller.__dict__["_quel1system"] = object()
    called: list[str] = []

    def _fake_initialize(box_name: str) -> None:
        called.append(box_name)

    monkeypatch.setattr(
        controller, "_initialize_box_awg_and_capunits", _fake_initialize
    )

    controller.initialize_awg_and_capunits(["A", "B"], parallel=True)

    assert set(called) == {"A", "B"}
    assert len(called) == 2


def test_initialize_awg_and_capunits_parallel_deduplicates_boxes(monkeypatch) -> None:
    """Given duplicate boxes, each box is initialized only once."""
    controller = Quel1BackendController()
    controller.__dict__["_quel1system"] = object()
    called: list[str] = []

    def _fake_initialize(box_name: str) -> None:
        called.append(box_name)

    monkeypatch.setattr(
        controller, "_initialize_box_awg_and_capunits", _fake_initialize
    )

    controller.initialize_awg_and_capunits(["A", "A", "B"], parallel=True)

    assert set(called) == {"A", "B"}
    assert len(called) == 2


def test_initialize_awg_and_capunits_raises_when_not_connected() -> None:
    """Given disconnected controller, initialize_awg_and_capunits raises before box access."""
    controller = Quel1BackendController()

    with pytest.raises(ValueError, match="Boxes not connected"):
        controller.initialize_awg_and_capunits(["A"], parallel=False)


def test_linkup_boxes_parallel_collects_successes(monkeypatch) -> None:
    """Given parallel linkup, when one box fails, then successful boxes are returned."""
    controller = Quel1BackendController()
    called: list[str] = []

    def _fake_linkup(box_name: str, **kwargs: object) -> object:
        _ = kwargs
        called.append(box_name)
        if box_name == "B":
            raise RuntimeError("boom")
        return object()

    monkeypatch.setattr(controller, "linkup", _fake_linkup)

    boxes = controller.linkup_boxes(["A", "B"], parallel=True)

    assert set(called) == {"A", "B"}
    assert list(boxes) == ["A"]


def test_linkup_boxes_parallel_deduplicates_box_names(monkeypatch) -> None:
    """Given duplicate box names, parallel linkup calls each box only once."""
    controller = Quel1BackendController()
    called: list[str] = []

    def _fake_linkup(box_name: str, **kwargs: object) -> object:
        _ = kwargs
        called.append(box_name)
        return object()

    monkeypatch.setattr(controller, "linkup", _fake_linkup)

    boxes = controller.linkup_boxes(["A", "A", "B"], parallel=True)

    assert set(called) == {"A", "B"}
    assert len(called) == 2
    assert list(boxes) == ["A", "B"]


def test_relinkup_boxes_parallel_calls_each_box(monkeypatch) -> None:
    """Given parallel relinkup, when executing, then relinkup is called for each box."""
    controller = Quel1BackendController()
    called: list[str] = []

    def _fake_relinkup(box_name: str, noise_threshold: int | None = None) -> None:
        _ = noise_threshold
        called.append(box_name)

    monkeypatch.setattr(controller, "relinkup", _fake_relinkup)

    controller.relinkup_boxes(["A", "B"], parallel=True)

    assert set(called) == {"A", "B"}


def test_relinkup_boxes_parallel_deduplicates_box_names(monkeypatch) -> None:
    """Given duplicate box names, parallel relinkup calls each box only once."""
    controller = Quel1BackendController()
    called: list[str] = []

    def _fake_relinkup(box_name: str, noise_threshold: int | None = None) -> None:
        _ = noise_threshold
        called.append(box_name)

    monkeypatch.setattr(controller, "relinkup", _fake_relinkup)

    controller.relinkup_boxes(["A", "A", "B"], parallel=True)

    assert set(called) == {"A", "B"}
    assert len(called) == 2


def test_linkup_uses_existing_pooled_box_without_recreating(monkeypatch) -> None:
    """Given pooled box, linkup reuses it and does not create another proxy."""

    class _FakeBox:
        boxtype = "quel1-a"

        def link_status(self) -> dict[int, bool]:
            return {0: True}

        def reconnect(self, **kwargs: object) -> None:
            _ = kwargs

    class _FakeBoxPool:
        def __init__(self, box: object) -> None:
            self._boxes = {"A": (box, object())}

    controller = Quel1BackendController()
    fake_box = _FakeBox()
    controller.__dict__["_boxpool"] = _FakeBoxPool(fake_box)

    monkeypatch.setattr(controller, "_check_box_availability", lambda _: None)
    monkeypatch.setattr(
        controller,
        "_create_box",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not create a new box")
        ),
    )

    linked_box = controller.linkup("A")

    assert linked_box is fake_box


def test_relinkup_uses_existing_pooled_box_without_recreating(monkeypatch) -> None:
    """Given pooled box, relinkup reuses it and does not create another proxy."""

    class _FakeBox:
        boxtype = "quel1-a"

        def relinkup(self, **kwargs: object) -> None:
            _ = kwargs

        def reconnect(self, **kwargs: object) -> None:
            _ = kwargs

    class _FakeBoxPool:
        def __init__(self, box: object) -> None:
            self._boxes = {"A": (box, object())}

    controller = Quel1BackendController()
    fake_box = _FakeBox()
    controller.__dict__["_boxpool"] = _FakeBoxPool(fake_box)

    monkeypatch.setattr(controller, "_check_box_availability", lambda _: None)
    monkeypatch.setattr(controller, "_resolve_config_options", lambda **_: [])
    monkeypatch.setattr(
        controller,
        "_create_box",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not create a new box")
        ),
    )

    controller.relinkup("A")
