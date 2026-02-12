"""Tests for mode switching in Quel1BackendExecutor."""

from __future__ import annotations

from typing import Any, cast

import pytest

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel1 import Quel1ExecutionPayload
from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController
from qubex.backend.quel1.quel1_backend_executor import Quel1BackendExecutor


def _make_payload() -> Quel1ExecutionPayload:
    return Quel1ExecutionPayload(
        sequencer=object(),
        repeats=10,
        integral_mode="integral",
        dsp_demodulation=True,
        enable_sum=False,
        enable_classification=False,
        line_param0=(1.0, 0.0, 0.0),
        line_param1=(0.0, 1.0, 0.0),
    )


def test_execute_uses_legacy_mode_by_default() -> None:
    """Given default mode, execute delegates to execute_sequencer."""
    called: dict[str, Any] = {}

    class _Controller:
        def execute_sequencer(self, **kwargs: object) -> str:  # type: ignore[no-untyped-def]
            called["legacy"] = kwargs
            return "legacy"

        def execute_sequencer_parallel(self, **kwargs: object) -> str:  # type: ignore[no-untyped-def]
            called["parallel"] = kwargs
            return "parallel"

    executor = Quel1BackendExecutor(
        backend_controller=cast(Quel1BackendController, _Controller())
    )

    result = executor.execute(request=BackendExecutionRequest(payload=_make_payload()))

    assert result == "legacy"
    assert "legacy" in called
    assert "parallel" not in called


def test_execute_uses_legacy_mode_when_configured() -> None:
    """Given legacy mode, execute delegates to execute_sequencer."""
    called: dict[str, Any] = {}

    class _Controller:
        def execute_sequencer(self, **kwargs: object) -> str:  # type: ignore[no-untyped-def]
            called["legacy"] = kwargs
            return "legacy"

        def execute_sequencer_parallel(self, **kwargs: object) -> str:  # type: ignore[no-untyped-def]
            called["parallel"] = kwargs
            return "parallel"

    executor = Quel1BackendExecutor(
        backend_controller=cast(Quel1BackendController, _Controller()),
        execution_mode="legacy",
    )

    result = executor.execute(request=BackendExecutionRequest(payload=_make_payload()))

    assert result == "legacy"
    assert "legacy" in called
    assert "parallel" not in called


def test_init_raises_for_unknown_execution_mode() -> None:
    """Given unsupported mode, initializer raises ValueError."""

    class _Controller:
        def execute_sequencer(self, **kwargs: object) -> str:  # type: ignore[no-untyped-def]
            return "legacy"

    with pytest.raises(ValueError, match="Unsupported execution mode"):
        Quel1BackendExecutor(
            backend_controller=cast(Quel1BackendController, _Controller()),
            execution_mode=cast(Any, "invalid"),
        )
