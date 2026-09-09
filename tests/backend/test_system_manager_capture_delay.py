"""Tests for temporary capture-delay overrides."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qubex.backend.quel1 import Quel1BackendController
from qubex.backend.quel3 import Quel3BackendController
from qubex.system.quel1 import Quel1SystemSynchronizer
from qubex.system.quel3 import Quel3SystemSynchronizer
from qubex.system.system_manager import SystemManager


def _make_manager(monkeypatch: pytest.MonkeyPatch, backend: str):
    manager = SystemManager.shared()
    controller = Mock(
        spec=Quel1BackendController if backend == "quel1" else Quel3BackendController
    )
    controller.set_capture_delay = Mock(return_value=8)
    port = SimpleNamespace(id="capture-port")
    channels = [
        SimpleNamespace(id=f"capture-{i}", port=port, number=i, ndelay=8)
        for i in range(2)
    ]
    port.channels = channels
    system = SimpleNamespace(
        control_params=SimpleNamespace(
            # Native config units: ndelay for QuEL-1, ns for QuEL-3.
            capture_delay={0: 8, 1: 16},
            capture_delay_word=(
                {0: 2, 1: 0} if backend == "quel1" else {0: None, 1: None}
            ),
        ),
        resolve_qubit_label=lambda label: "Q00",
        get_mux_by_qubit=lambda label: SimpleNamespace(index=0),
        wiring_info=SimpleNamespace(read_in=[(SimpleNamespace(index=0), port)]),
    )
    monkeypatch.setattr(manager, "_backend_controller", controller)
    monkeypatch.setattr(
        manager,
        "_system_synchronizer",
        (
            Quel1SystemSynchronizer(backend_controller=controller)
            if isinstance(controller, Quel1BackendController)
            else Quel3SystemSynchronizer(backend_controller=controller)
        ),
    )
    monkeypatch.setattr(manager, "_experiment_system", system)
    return manager, system, controller, channels


def test_capture_delay_reports_uninitialized_backend(monkeypatch) -> None:
    """Overrides report an uninitialized backend without changing capture settings."""
    manager, system, controller, channels = _make_manager(monkeypatch, "quel1")
    monkeypatch.setattr(manager, "_backend_controller", None)
    with (
        pytest.raises(
            RuntimeError,
            match=r"^Cannot override capture delay: backend controller is not initialized\.$",
        ),
        manager.modified_capture_delay({0: 24}),
    ):
        pytest.fail("Override accepted without an initialized backend")
    assert system.control_params.capture_delay == {0: 8, 1: 16}
    assert [channel.ndelay for channel in channels] == [8, 8]
    assert controller.mock_calls == []


@pytest.mark.parametrize(
    ("backend", "delay"),
    [
        ("quel1", 0),
        ("quel1", 776.0),
        ("quel3", 0),
        ("quel3", 24.0),
        ("quel3", 0.8),
        ("quel3", 3 * 0.8),
    ],
)
@pytest.mark.parametrize("fail", [False, True])
def test_capture_delay_is_temporary(monkeypatch, backend, delay, fail) -> None:
    """Capture delay overrides affect selected muxes and restore after success or failure."""
    manager, system, controller, channels = _make_manager(monkeypatch, backend)
    original = system.control_params.capture_delay
    expected_words = {0: 2, 1: 0} if backend == "quel1" else {0: None, 1: None}
    assert system.control_params.capture_delay_word == expected_words

    def run() -> None:
        with manager.modified_capture_delay({0: delay}):
            expected = int(delay // 128) if backend == "quel1" else delay
            assert system.control_params.capture_delay == {0: expected, 1: 16}
            if backend == "quel1":
                assert [channel.ndelay for channel in channels] == [expected, expected]
                assert controller.set_capture_delay.call_count == 2
                assert (
                    controller.set_capture_delay.call_args.kwargs["capture_delay"]
                    == expected
                )
            if backend == "quel3":
                assert system.control_params.capture_delay_word == {0: None, 1: None}
            if fail:
                raise RuntimeError("measurement failed")

    if fail:
        with pytest.raises(RuntimeError, match="measurement failed"):
            run()
    else:
        run()

    assert system.control_params.capture_delay is original
    assert original == {0: 8, 1: 16}
    assert system.control_params.capture_delay_word == expected_words
    assert [channel.ndelay for channel in channels] == [8, 8]
    if backend == "quel1":
        assert controller.set_capture_delay.call_count == 4
        assert controller.set_capture_delay.call_args.kwargs["capture_delay"] == 8


@pytest.mark.parametrize(
    ("backend", "delay", "error"),
    [
        ("quel1", 1.5, ValueError),
        ("quel1", -1, ValueError),
        ("quel3", -2, ValueError),
        ("quel3", float("nan"), ValueError),
        ("quel3", float("inf"), ValueError),
        ("quel3", 1.0, ValueError),
    ],
)
def test_invalid_capture_delay_leaves_settings_unchanged(
    monkeypatch, backend, delay, error
) -> None:
    """Invalid capture delays fail before changing configuration or controller state."""
    manager, system, controller, channels = _make_manager(monkeypatch, backend)
    with pytest.raises(error), manager.modified_capture_delay({0: delay}):
        pytest.fail("Invalid delay was accepted")
    assert system.control_params.capture_delay == {0: 8, 1: 16}
    assert [channel.ndelay for channel in channels] == [8, 8]
    assert controller.mock_calls == []


def test_capture_delay_restores_after_controller_failure(monkeypatch) -> None:
    """A partial controller update restores the original delay on every affected channel."""
    manager, system, controller, channels = _make_manager(monkeypatch, "quel1")
    controller.set_capture_delay.side_effect = [
        8,
        RuntimeError("update failed"),
        24,
    ]
    with (
        pytest.raises(RuntimeError, match="update failed"),
        manager.modified_capture_delay({0: 24}),
    ):
        pytest.fail("Controller failure was ignored")
    assert system.control_params.capture_delay == {0: 8, 1: 16}
    assert [channel.ndelay for channel in channels] == [8, 8]
    assert controller.set_capture_delay.call_count == 3
    assert system.control_params.capture_delay_word == {0: 2, 1: 0}


@pytest.mark.parametrize("backend", ["quel1", "quel3"])
def test_capture_delay_accepts_distinct_mux_values(monkeypatch, backend) -> None:
    """Mux-keyed overrides preserve distinct values and leave the input dictionary unchanged."""
    manager, system, controller, _channels = _make_manager(monkeypatch, backend)
    overrides = {0: 16, 1: 24}
    with manager.modified_capture_delay(overrides):
        assert system.control_params.capture_delay == (
            {0: 0, 1: 0} if backend == "quel1" else overrides
        )
    assert system.control_params.capture_delay == {0: 8, 1: 16}
    assert overrides == {0: 16, 1: 24}
    if backend == "quel1":
        controller.define_channel.assert_not_called()
    else:
        assert controller.mock_calls == []


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (10, TypeError),
        ({"0": 10}, TypeError),
        ({True: 10}, TypeError),
        ({9: 10}, ValueError),
    ],
)
def test_invalid_mux_overrides_do_not_mutate_state(
    monkeypatch, overrides, error
) -> None:
    """Scalar values, invalid mux keys and unknown muxes fail before any mutation."""
    manager, system, controller, _channels = _make_manager(monkeypatch, "quel1")
    with pytest.raises(error), manager.modified_capture_delay(overrides):
        pytest.fail("Invalid override accepted")
    assert system.control_params.capture_delay == {0: 8, 1: 16}
    assert controller.mock_calls == []


def test_mux_overrides_reach_existing_backend_ports(monkeypatch) -> None:
    """Distinct mux overrides update only their existing backend ports and restore exact values."""
    from typing import Any, cast

    from qubex.backend.quel1.managers.configuration_manager import (
        Quel1ConfigurationManager,
    )

    manager = SystemManager.shared()
    controller = object.__new__(Quel1BackendController)
    settings = {
        f"port-{i}": SimpleNamespace(ndelay_or_nwait=(8 + i,)) for i in range(3)
    }
    relations = [
        (f"channel-{i}", {"port_name": f"port-{i}", "channel_number": 0})
        for i in range(3)
    ]
    db = SimpleNamespace(_port_settings=settings, _relation_channel_port=relations)
    runtime = SimpleNamespace(qubecalib=SimpleNamespace(system_config_database=db))
    monkeypatch.setattr(
        controller,
        "_configuration_manager",
        Quel1ConfigurationManager(runtime_context=cast(Any, runtime)),
        raising=False,
    )
    ports = [
        SimpleNamespace(
            id=f"port-{i}", channels=[SimpleNamespace(number=0, ndelay=8 + i)]
        )
        for i in range(3)
    ]
    system = SimpleNamespace(
        control_params=SimpleNamespace(
            capture_delay={0: 8, 1: 9, 2: 10}, capture_delay_word={0: 0, 1: 0, 2: 0}
        ),
        wiring_info=SimpleNamespace(
            read_in=[(SimpleNamespace(index=i), port) for i, port in enumerate(ports)]
        ),
    )
    monkeypatch.setattr(manager, "_backend_controller", controller)
    monkeypatch.setattr(
        manager,
        "_system_synchronizer",
        (
            Quel1SystemSynchronizer(backend_controller=controller)
            if isinstance(controller, Quel1BackendController)
            else Quel3SystemSynchronizer(backend_controller=controller)
        ),
    )
    monkeypatch.setattr(manager, "_experiment_system", system)
    for _ in range(2):
        with manager.modified_capture_delay({0: 16 * 128, 1: 24 * 128}):
            assert [settings[f"port-{i}"].ndelay_or_nwait for i in range(3)] == [
                (16,),
                (24,),
                (10,),
            ]
            assert [port.channels[0].ndelay for port in ports] == [16, 24, 10]
            assert system.control_params.capture_delay == {0: 16, 1: 24, 2: 10}
        assert [settings[f"port-{i}"].ndelay_or_nwait for i in range(3)] == [
            (8,),
            (9,),
            (10,),
        ]
        assert [port.channels[0].ndelay for port in ports] == [8, 9, 10]
        assert system.control_params.capture_delay == {0: 8, 1: 9, 2: 10}
    assert len(relations) == 3


@pytest.mark.parametrize("failure", [None, "enter", "body"])
def test_capture_delay_uses_synchronizer_context(monkeypatch, failure) -> None:
    """Backend contexts enclose measurements and software delays restore on failure."""
    manager, system, controller, channels = _make_manager(monkeypatch, "quel1")
    events = []

    @contextmanager
    def modified_capture_delay(*, experiment_system, capture_delay):
        assert experiment_system is system
        assert capture_delay == {0: 24}
        events.append("enter")
        try:
            if failure == "enter":
                raise RuntimeError("enter")
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(
        manager,
        "_system_synchronizer",
        SimpleNamespace(modified_capture_delay=modified_capture_delay),
    )

    def run():
        with manager.modified_capture_delay({0: 24}):
            assert events == ["enter"]
            assert system.control_params.capture_delay == {0: 8, 1: 16}
            if failure == "body":
                raise RuntimeError("body")

    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run()
    else:
        run()
    assert events == ["enter", "exit"]
    assert system.control_params.capture_delay == {0: 8, 1: 16}
    assert [channel.ndelay for channel in channels] == [8, 8]
    assert controller.mock_calls == []


@pytest.mark.parametrize(
    ("delay", "ndelay", "word"),
    [(0, 0, 0), (8, 0, 1), (120, 0, 15), (128, 1, 0), (776.0, 6, 1), (2048, 16, 0)],
)
@pytest.mark.parametrize("fail", [False, True])
def test_ns_delay_splits_and_restores_words(monkeypatch, delay, ndelay, word, fail):
    """Nanosecond overrides split into canonical coarse and word settings and restore both."""
    manager, system, _controller, channels = _make_manager(monkeypatch, "quel1")
    original_words = system.control_params.capture_delay_word

    def run():
        with manager.modified_capture_delay({0: delay}):
            assert system.control_params.capture_delay == {0: ndelay, 1: 16}
            assert system.control_params.capture_delay_word == {0: word, 1: 0}
            assert [channel.ndelay for channel in channels] == [ndelay, ndelay]
            with manager.modified_capture_delay({0: 128}):
                assert system.control_params.capture_delay_word[0] == 0
            assert system.control_params.capture_delay_word[0] == word
            if fail:
                raise RuntimeError("measurement failed")

    if fail:
        with pytest.raises(RuntimeError, match="measurement failed"):
            run()
    else:
        run()
    assert system.control_params.capture_delay_word is original_words
    assert original_words == {0: 2, 1: 0}
    assert system.control_params.capture_delay == {0: 8, 1: 16}


@pytest.mark.parametrize(("backend", "step"), [("quel1", 8), ("quel3", 0.8)])
@pytest.mark.parametrize("value", [777.0, 8000001.0])
def test_ns_delay_rejects_off_grid_values_before_any_update(
    monkeypatch, backend, step, value
):
    """Off-grid values report backend resolution before any mux or controller changes."""
    manager, system, controller, _ = _make_manager(monkeypatch, backend)
    with (
        pytest.raises(ValueError, match=rf"multiple of {step}(?:\.0)? ns"),
        manager.modified_capture_delay({0: 776.0, 1: value}),
    ):
        pytest.fail("Off-grid delay accepted")
    assert system.control_params.capture_delay == {0: 8, 1: 16}
    assert system.control_params.capture_delay_word == (
        {0: 2, 1: 0} if backend == "quel1" else {0: None, 1: None}
    )
    assert controller.mock_calls == []
