"""Tests for mode switching in Quel1ExecutionManager."""

from __future__ import annotations

from typing import Any, cast

import pytest

import qubex.backend.quel1.managers.execution_manager as execution_manager_module
from qubex.backend import BackendExecutionRequest
from qubex.backend.quel1 import Quel1ExecutionPayload
from qubex.backend.quel1.managers.execution_manager import Quel1ExecutionManager


def _make_payload() -> Quel1ExecutionPayload:
    return Quel1ExecutionPayload(
        gen_sampled_sequence={"Q00": object()},
        cap_sampled_sequence={"RQ00": object()},
        resource_map={"Q00": [{}]},
        interval=128,
        repeats=10,
        integral_mode="integral",
        dsp_demodulation=True,
        enable_sum=False,
        enable_classification=False,
        line_param0=(1.0, 0.0, 0.0),
        line_param1=(0.0, 1.0, 0.0),
    )


def test_execute_uses_parallel_mode_by_default() -> None:
    """Given default mode, execute delegates to execute_sequencer_parallel."""
    called: dict[str, Any] = {}
    sequencer = object()

    class _ExecutionManager(Quel1ExecutionManager):
        def _create_quel1_sequencer(self, **kwargs: Any) -> object:
            called["create"] = kwargs
            return sequencer

        def _execute_sequencer(self, **kwargs: Any) -> str:
            called["serial"] = kwargs
            return "serial"

        def _execute_sequencer_parallel(self, **kwargs: Any) -> str:
            called["parallel"] = kwargs
            return "parallel"

    manager = _ExecutionManager(runtime_context=cast(Any, object()))
    result = manager.execute(request=BackendExecutionRequest(payload=_make_payload()))

    assert result == "parallel"
    assert called["parallel"]["sequencer"] is sequencer
    assert called["parallel"]["clock_health_checks"] is False
    assert "serial" not in called


def test_execute_uses_serial_mode_when_configured() -> None:
    """Given serial mode, execute delegates to execute_sequencer."""
    called: dict[str, Any] = {}
    sequencer = object()

    class _ExecutionManager(Quel1ExecutionManager):
        def _create_quel1_sequencer(self, **kwargs: Any) -> object:
            called["create"] = kwargs
            return sequencer

        def _execute_sequencer(self, **kwargs: Any) -> str:
            called["serial"] = kwargs
            return "serial"

        def _execute_sequencer_parallel(self, **kwargs: Any) -> str:
            called["parallel"] = kwargs
            return "parallel"

    manager = _ExecutionManager(runtime_context=cast(Any, object()))
    result = manager.execute(
        request=BackendExecutionRequest(payload=_make_payload()),
        execution_mode="serial",
    )

    assert result == "serial"
    assert called["serial"]["sequencer"] is sequencer
    assert "parallel" not in called


def test_init_raises_for_unknown_execution_mode() -> None:
    """Given unsupported mode, execute raises ValueError."""
    manager = Quel1ExecutionManager(runtime_context=cast(Any, object()))

    with pytest.raises(ValueError, match="Unsupported execution mode"):
        manager.execute(
            request=BackendExecutionRequest(payload=_make_payload()),
            execution_mode=cast(Any, "invalid"),
        )


def test_execute_raises_for_non_quel1_payload() -> None:
    """Given non-QuEL-1 payload, execute raises TypeError."""
    manager = Quel1ExecutionManager(runtime_context=cast(Any, object()))

    with pytest.raises(TypeError, match="expects `Quel1ExecutionPayload` payload"):
        manager.execute(request=BackendExecutionRequest(payload=object()))


def test_create_quel1_sequencer_passes_driver_for_constructor_compatibility(
    monkeypatch: Any,
) -> None:
    """Given manager sequencer creation, constructor receives driver and sysdb."""
    created_kwargs: dict[str, Any] = {}
    fake_system = object()
    fake_sysdb = object()

    class _FakeSequencer:
        def __init__(self, **kwargs: Any) -> None:
            created_kwargs.update(kwargs)

    class _RuntimeContext:
        @property
        def qubecalib(self) -> Any:
            return cast(Any, type("Q", (), {"sysdb": fake_sysdb})())

        @property
        def quel1system(self) -> Any:
            return fake_system

    monkeypatch.setattr(execution_manager_module, "Quel1Sequencer", _FakeSequencer)

    class _ExecutionManager(Quel1ExecutionManager):
        def _execute_sequencer(self, **kwargs: Any) -> str:
            _ = kwargs
            return "serial"

    execution_manager = _ExecutionManager(runtime_context=cast(Any, _RuntimeContext()))
    _ = execution_manager.execute(
        request=BackendExecutionRequest(payload=_make_payload()),
        execution_mode="serial",
    )

    assert created_kwargs["driver"] is fake_system
    assert created_kwargs["sysdb"] is fake_sysdb
