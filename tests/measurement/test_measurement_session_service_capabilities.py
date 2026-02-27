"""Tests for optional backend capability handling in session service."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import qubex.measurement.services.measurement_session_service as session_module
from qubex.measurement.services.measurement_session_service import (
    MeasurementSessionService,
)


class _BackendWithoutOptionalCapabilities:
    hash = 0
    is_connected = True

    def __init__(self) -> None:
        self.connect_calls: list[tuple[list[str] | str | None, bool | None]] = []
        self.disconnect_calls = 0

    async def execute(self, *, request: object) -> object:
        _ = request
        raise AssertionError("execute is not used in these tests.")

    def connect(
        self,
        box_names: str | list[str] | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        self.connect_calls.append((box_names, parallel))

    def disconnect(self) -> None:
        self.disconnect_calls += 1


class _BackendWithLinkCapability(_BackendWithoutOptionalCapabilities):
    def __init__(self, *, link_status_by_box: dict[str, dict[int, bool]]) -> None:
        super().__init__()
        self._link_status_by_box = link_status_by_box
        self.link_status_calls: list[str] = []

    def link_status(self, box_name: str) -> dict[int, bool]:
        self.link_status_calls.append(box_name)
        return self._link_status_by_box[box_name]


class _SystemManagerStub:
    def __init__(self) -> None:
        self.pull_calls: list[tuple[list[str], bool | None]] = []

    def pull(self, box_ids: list[str], *, parallel: bool | None = None) -> None:
        self.pull_calls.append((box_ids, parallel))


class _ContextStub:
    def __init__(
        self,
        *,
        backend_controller: object,
        box_ids: list[str],
        config_path: Path,
    ) -> None:
        self.backend_controller = backend_controller
        self.box_ids = box_ids
        self.experiment_system = object()
        self.config_loader = type(
            "_ConfigLoader",
            (),
            {
                "config_path": config_path,
            },
        )()


def _make_session_service(
    *,
    backend_controller: object,
    tmp_path: Path,
    box_ids: list[str] | None = None,
) -> tuple[MeasurementSessionService, _SystemManagerStub]:
    system_manager = _SystemManagerStub()
    context = _ContextStub(
        backend_controller=backend_controller,
        box_ids=["A"] if box_ids is None else box_ids,
        config_path=tmp_path,
    )
    service = MeasurementSessionService(
        system_manager=system_manager,  # type: ignore[arg-type]
        context=context,  # type: ignore[arg-type]
    )
    return service, system_manager


def test_connect_skips_resync_when_backend_does_not_support_it(tmp_path: Path) -> None:
    """Given backend without resync capability, when connect runs, then connect and pull still succeed."""
    backend = _BackendWithoutOptionalCapabilities()
    service, system_manager = _make_session_service(
        backend_controller=backend,
        tmp_path=tmp_path,
        box_ids=["A"],
    )

    service.connect(sync_clocks=True)

    assert backend.connect_calls == [(["A"], None)]
    assert system_manager.pull_calls == [(["A"], None)]


def test_connect_runs_pull_when_all_links_are_up(tmp_path: Path) -> None:
    """Given backend with healthy links, when connect runs, then pull still executes."""
    backend = _BackendWithLinkCapability(link_status_by_box={"A": {0: True}})
    service, system_manager = _make_session_service(
        backend_controller=backend,
        tmp_path=tmp_path,
        box_ids=["A"],
    )

    service.connect(sync_clocks=False)

    assert backend.connect_calls == [(["A"], None)]
    assert backend.link_status_calls == ["A"]
    assert system_manager.pull_calls == [(["A"], None)]


def test_connect_raises_without_pull_when_some_links_are_down(tmp_path: Path) -> None:
    """Given backend with down links, when connect runs, then pull is skipped with error."""
    backend = _BackendWithLinkCapability(link_status_by_box={"A": {0: False}})
    service, system_manager = _make_session_service(
        backend_controller=backend,
        tmp_path=tmp_path,
        box_ids=["A"],
    )

    with pytest.raises(ConnectionError, match="linkup"):
        service.connect(sync_clocks=False)

    assert backend.connect_calls == [(["A"], None)]
    assert backend.link_status_calls == ["A"]
    assert system_manager.pull_calls == []


def test_check_link_status_raises_not_implemented_without_link_capability(
    tmp_path: Path,
) -> None:
    """Given backend without link capability, when checking link status, then NotImplementedError is raised."""
    backend = _BackendWithoutOptionalCapabilities()
    service, _ = _make_session_service(backend_controller=backend, tmp_path=tmp_path)

    with pytest.raises(NotImplementedError, match="link status"):
        service.check_link_status(["A"])


def test_check_link_status_uses_parallel_map_when_parallel_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given parallel flag, when checking link status, then link checks run via parallel map."""
    backend = _BackendWithLinkCapability(
        link_status_by_box={
            "A": {0: True},
            "B": {0: True},
        }
    )
    service, _ = _make_session_service(
        backend_controller=backend,
        tmp_path=tmp_path,
        box_ids=["A", "B"],
    )
    captured_boxes: list[str] = []

    def _fake_run_parallel_map(
        items: Sequence[str],
        worker: Callable[[str], dict[int, bool]],
        *,
        key: Callable[[str], str],
        max_workers: int | None = None,
        as_completed_order: bool = False,
        on_error: Callable[[str, BaseException], dict[int, bool]] | None = None,
    ) -> dict[str, dict[int, bool]]:
        del max_workers, as_completed_order, on_error
        captured_boxes.extend(items)
        return {key(item): worker(item) for item in items}

    monkeypatch.setattr(session_module, "run_parallel_map", _fake_run_parallel_map)

    result = service.check_link_status(["A", "B"], parallel=True)

    assert captured_boxes == ["A", "B"]
    assert backend.link_status_calls == ["A", "B"]
    assert result == {
        "status": True,
        "links": {
            "A": {0: True},
            "B": {0: True},
        },
    }


def test_connect_forwards_parallel_flag_to_link_status_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given connect parallel flag, when connect runs, then link checks use the same parallel mode."""
    backend = _BackendWithLinkCapability(
        link_status_by_box={
            "A": {0: True},
            "B": {0: True},
        }
    )
    service, system_manager = _make_session_service(
        backend_controller=backend,
        tmp_path=tmp_path,
        box_ids=["A", "B"],
    )
    captured_boxes: list[str] = []

    def _fake_run_parallel_map(
        items: Sequence[str],
        worker: Callable[[str], dict[int, bool]],
        *,
        key: Callable[[str], str],
        max_workers: int | None = None,
        as_completed_order: bool = False,
        on_error: Callable[[str, BaseException], dict[int, bool]] | None = None,
    ) -> dict[str, dict[int, bool]]:
        del max_workers, as_completed_order, on_error
        captured_boxes.extend(items)
        return {key(item): worker(item) for item in items}

    monkeypatch.setattr(session_module, "run_parallel_map", _fake_run_parallel_map)

    service.connect(sync_clocks=False, parallel=True)

    assert captured_boxes == ["A", "B"]
    assert system_manager.pull_calls == [(["A", "B"], True)]


def test_check_clock_status_raises_not_implemented_without_clock_capability(
    tmp_path: Path,
) -> None:
    """Given backend without clock capability, when checking clock status, then NotImplementedError is raised."""
    backend = _BackendWithoutOptionalCapabilities()
    service, _ = _make_session_service(backend_controller=backend, tmp_path=tmp_path)

    with pytest.raises(NotImplementedError, match="clock status"):
        service.check_clock_status(["A"])


def test_linkup_raises_not_implemented_without_linkup_capability(
    tmp_path: Path,
) -> None:
    """Given backend without linkup capability, when linkup runs, then NotImplementedError is raised."""
    backend = _BackendWithoutOptionalCapabilities()
    service, _ = _make_session_service(backend_controller=backend, tmp_path=tmp_path)

    with pytest.raises(NotImplementedError, match="linkup"):
        service.linkup(["A"])


def test_relinkup_raises_not_implemented_without_relinkup_capability(
    tmp_path: Path,
) -> None:
    """Given backend without relinkup capability, when relinkup runs, then NotImplementedError is raised."""
    backend = _BackendWithoutOptionalCapabilities()
    service, _ = _make_session_service(backend_controller=backend, tmp_path=tmp_path)

    with pytest.raises(NotImplementedError, match="relinkup"):
        service.relinkup(["A"])
