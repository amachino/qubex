"""Tests for mode switching in Quel1BackendExecutor."""

from __future__ import annotations

from typing import Any, cast

import pytest

import qubex.backend.quel1.managers.execution_manager as execution_manager_module
from qubex.backend import BackendExecutionRequest
from qubex.backend.quel1 import Quel1ExecutionPayload
from qubex.backend.quel1.managers.execution_manager import Quel1ExecutionManager
from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController
from qubex.backend.quel1.quel1_backend_executor import Quel1BackendExecutor


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

    class _Controller:
        def execute_sequencer(self, **kwargs: Any) -> str:
            called["serial"] = kwargs
            return "serial"

        def execute_sequencer_parallel(self, **kwargs: Any) -> str:
            called["parallel"] = kwargs
            return "parallel"

    class _ExecutionManager:
        def create_quel1_sequencer(self, **kwargs: Any) -> object:
            called["create"] = kwargs
            return sequencer

    executor = Quel1BackendExecutor(
        backend_controller=cast(Quel1BackendController, _Controller()),
        execution_manager=cast(Quel1ExecutionManager, _ExecutionManager()),
    )

    result = executor.execute(request=BackendExecutionRequest(payload=_make_payload()))

    assert result == "parallel"
    assert called["parallel"]["sequencer"] is sequencer
    assert "parallel" in called
    assert called["parallel"]["clock_health_checks"] is False
    assert "serial" not in called


def test_execute_uses_serial_mode_when_configured() -> None:
    """Given serial mode, execute delegates to execute_sequencer."""
    called: dict[str, Any] = {}
    sequencer = object()

    class _Controller:
        def execute_sequencer(self, **kwargs: Any) -> str:
            called["serial"] = kwargs
            return "serial"

        def execute_sequencer_parallel(self, **kwargs: Any) -> str:
            called["parallel"] = kwargs
            return "parallel"

    class _ExecutionManager:
        def create_quel1_sequencer(self, **kwargs: Any) -> object:
            called["create"] = kwargs
            return sequencer

    executor = Quel1BackendExecutor(
        backend_controller=cast(Quel1BackendController, _Controller()),
        execution_manager=cast(Quel1ExecutionManager, _ExecutionManager()),
        execution_mode="serial",
    )

    result = executor.execute(request=BackendExecutionRequest(payload=_make_payload()))

    assert result == "serial"
    assert called["serial"]["sequencer"] is sequencer
    assert "serial" in called
    assert "parallel" not in called


def test_init_raises_for_unknown_execution_mode() -> None:
    """Given unsupported mode, initializer raises ValueError."""

    class _Controller:
        def execute_sequencer(self, **kwargs: Any) -> str:
            return "serial"

    with pytest.raises(ValueError, match="Unsupported execution mode"):
        Quel1BackendExecutor(
            backend_controller=cast(Quel1BackendController, _Controller()),
            execution_manager=cast(Quel1ExecutionManager, object()),
            execution_mode=cast(Any, "invalid"),
        )


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
    execution_manager = Quel1ExecutionManager(
        runtime_context=cast(Any, _RuntimeContext())
    )

    execution_manager.create_quel1_sequencer(
        gen_sampled_sequence={"Q00": cast(Any, object())},
        cap_sampled_sequence={"RQ00": cast(Any, object())},
        resource_map={"Q00": [{}]},
        interval=128,
    )

    assert created_kwargs["driver"] is fake_system
    assert created_kwargs["sysdb"] is fake_sysdb
