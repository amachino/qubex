"""Tests for backend package public API boundaries."""

from __future__ import annotations

import qubex.backend as backend
from qubex.backend.quel1 import (
    SAMPLING_PERIOD,
    DeviceController,
    Quel1BackendExecutor,
    Quel1BackendRawResult,
    Quel1ExecutionPayload,
)


def test_backend_module_hides_quel1_specific_symbols() -> None:
    """Given backend module, when checking QuEL-1 symbols, then they are not re-exported."""
    assert not hasattr(backend, "DeviceController")
    assert not hasattr(backend, "Quel1BackendExecutor")
    assert not hasattr(backend, "Quel1ExecutionPayload")
    assert not hasattr(backend, "Quel1BackendRawResult")
    assert not hasattr(backend, "SAMPLING_PERIOD")


def test_backend_quel1_module_exposes_quel1_specific_symbols() -> None:
    """Given backend.quel1 module, when importing symbols, then QuEL-1 symbols are exposed."""
    assert DeviceController.__name__ == "Quel1BackendController"
    assert Quel1BackendExecutor.__name__ == "Quel1BackendExecutor"
    assert Quel1ExecutionPayload.__name__ == "Quel1ExecutionPayload"
    assert Quel1BackendRawResult.__name__ == "Quel1BackendRawResult"
    assert isinstance(SAMPLING_PERIOD, float)
