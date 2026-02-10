"""Tests for parallel sequencer execution in Quel1BackendController."""

from __future__ import annotations

import asyncio
from typing import Any

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


def test_dump_box_async_delegates_to_dump_box(monkeypatch) -> None:
    """Given async dump, when invoked, then sync dump_box result is returned."""
    controller = Quel1BackendController()
    called: list[str] = []

    def _fake_dump_box(box_name: str) -> dict[str, str]:
        called.append(box_name)
        return {"box": box_name}

    monkeypatch.setattr(controller, "dump_box", _fake_dump_box)

    result = asyncio.run(controller.dump_box_async("A"))

    assert result == {"box": "A"}
    assert called == ["A"]
