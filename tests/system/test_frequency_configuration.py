"""System frequency override delegation contracts."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.system.quel1.quel1_system_synchronizer import Quel1SystemSynchronizer
from qubex.system.system_manager import SystemManager


def test_frequency_override_keeps_backend_calls_inside_sweep_scope():
    """Legacy frequency overrides still update and restore both model and backend."""
    calls = []
    target = SimpleNamespace(label="R00", frequency=6.0)

    def modify(values):
        target.frequency = values["R00"]

    controller = SimpleNamespace(
        modify_target_frequencies=lambda values: calls.append(values)
    )
    synchronizer = Quel1SystemSynchronizer(backend_controller=cast(Any, controller))
    manager = SimpleNamespace(
        experiment_system=SimpleNamespace(
            targets=[target], modify_target_frequencies=modify
        ),
        _resolve_system_synchronizer=lambda: synchronizer,
    )
    with SystemManager.modified_frequencies(cast(Any, manager), {"R00": 6.1}):
        assert target.frequency == 6.1
        assert calls == [{"R00": 6.1}]
    assert target.frequency == 6.0
    assert calls == [{"R00": 6.1}, {"R00": 6.0}]


def test_frequency_override_delegates_without_backend_type_checks():
    """Frequency scopes delegate model and backend ownership to the synchronizer."""
    calls = []
    system = object()

    @contextmanager
    def modified(*, experiment_system, target_frequencies):
        calls.append((experiment_system, target_frequencies))
        try:
            yield
        finally:
            calls.append("restored")

    manager = SimpleNamespace(
        experiment_system=system,
        _resolve_system_synchronizer=lambda: SimpleNamespace(
            modified_frequencies=modified
        ),
    )
    with (
        pytest.raises(RuntimeError, match="acquisition"),
        SystemManager.modified_frequencies(cast(Any, manager), {"R00": 6.1}),
    ):
        raise RuntimeError("acquisition")
    assert calls == [(system, {"R00": 6.1}), "restored"]
