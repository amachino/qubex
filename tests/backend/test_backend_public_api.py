"""Tests for backend package public API boundaries."""

from __future__ import annotations

from typing import get_args

import qubex.backend as backend
from qubex.backend.quel1 import (
    SAMPLING_PERIOD,
    ExecutionMode,
    Quel1BackendController,
    Quel1BackendResult,
    Quel1ExecutionPayload,
)


def test_backend_module_hides_quel1_specific_symbols() -> None:
    """Given backend module, when checking QuEL-1 symbols, then they are not re-exported."""
    assert not hasattr(backend, "Quel1BackendController")
    assert not hasattr(backend, "Quel1ExecutionPayload")
    assert not hasattr(backend, "Quel1BackendResult")
    assert not hasattr(backend, "SAMPLING_PERIOD")


def test_backend_quel1_module_exposes_quel1_specific_symbols() -> None:
    """Given backend.quel1 module, when importing symbols, then QuEL-1 symbols are exposed."""
    assert Quel1BackendController.__name__ == "Quel1BackendController"
    assert Quel1ExecutionPayload.__name__ == "Quel1ExecutionPayload"
    assert Quel1BackendResult.__name__ == "Quel1BackendResult"
    assert isinstance(SAMPLING_PERIOD, float)
    assert set(get_args(ExecutionMode)) == {"serial", "parallel"}
