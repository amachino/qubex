# ruff: noqa: SLF001

"""Tests for backend-resource cleanup in Quel1BackendController."""

from __future__ import annotations

from typing import Any, cast

from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController


class _Closable:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        """Record one close call."""
        self.close_calls += 1


class _Terminable:
    def __init__(self) -> None:
        self.terminate_calls = 0

    def terminate(self) -> None:
        """Record one terminate call."""
        self.terminate_calls += 1


class _FailingClosable:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        """Raise an error after recording one close call."""
        self.close_calls += 1
        raise RuntimeError("close failed")


class _FakeQuel1System:
    def __init__(self, *, clockmaster: Any, boxes: dict[str, Any]) -> None:
        self._clockmaster = clockmaster
        self.boxes = boxes
        self.config_cache: dict[str, dict[str, Any]] = {}
        self.config_fetched_at = None


class _FakeBoxPool:
    def __init__(self, *, boxes: dict[str, Any], clock_master: Any) -> None:
        self._boxes = {name: (box, object()) for name, box in boxes.items()}
        self._clock_master = clock_master


def test_disconnect_closes_clockmaster_and_boxes_and_clears_state() -> None:
    """Given connected state, disconnect closes held resources and clears caches."""
    controller = Quel1BackendController()

    system_master = _Closable()
    pool_master = _Terminable()
    shared_box = _Closable()
    pool_only_box = _Terminable()
    system = _FakeQuel1System(
        clockmaster=system_master,
        boxes={
            "A": shared_box,
        },
    )
    boxpool = _FakeBoxPool(
        boxes={
            "A": shared_box,
            "B": pool_only_box,
        },
        clock_master=pool_master,
    )

    controller._connection_manager.set_connected_state(
        boxpool=cast(Any, boxpool),
        quel1system=cast(Any, system),
        cap_resource_map={"cap": {}},
        gen_resource_map={"gen": {}},
    )

    controller.disconnect()

    assert system_master.close_calls == 1
    assert pool_master.terminate_calls == 1
    assert shared_box.close_calls == 1
    assert pool_only_box.terminate_calls == 1
    assert controller._connection_manager.quel1system is None
    assert controller._connection_manager.boxpool is None
    assert controller._connection_manager.cap_resource_map is None
    assert controller._connection_manager.gen_resource_map is None
    assert controller.is_connected is False


def test_disconnect_continues_when_resource_disconnect_fails(caplog) -> None:
    """Given disconnect failure, disconnect logs and continues cleanup."""
    controller = Quel1BackendController()
    failing_master = _FailingClosable()
    box = _Closable()
    system = _FakeQuel1System(clockmaster=failing_master, boxes={"A": box})

    controller._connection_manager.set_quel1system(cast(Any, system))

    controller.disconnect()

    assert failing_master.close_calls == 1
    assert box.close_calls == 1
    assert controller._connection_manager.quel1system is None
    assert "Failed to disconnect backend resource" in caplog.text
