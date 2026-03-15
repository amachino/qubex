"""QuEL-1 specific backend components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .quel1_backend_constants import (
    BLOCK_DURATION_NS,
    BLOCK_LENGTH,
    CAPTURE_DECIMATION_FACTOR,
    DEFAULT_CLOCK_HEALTH_CHECKS,
    DEFAULT_EXECUTION_MODE,
    SAMPLING_PERIOD_NS,
    WORD_DURATION_NS,
    WORD_LENGTH,
    ExecutionMode,
)
from .quel1_backend_controller import (
    Quel1BackendController,
)
from .quel1_backend_execution_result import Quel1BackendExecutionResult
from .quel1_execution_payload import Quel1ExecutionPayload

if TYPE_CHECKING:
    from .compat.sequencer import Quel1Sequencer

__all__ = [
    "BLOCK_DURATION_NS",
    "BLOCK_LENGTH",
    "CAPTURE_DECIMATION_FACTOR",
    "DEFAULT_CLOCK_HEALTH_CHECKS",
    "DEFAULT_EXECUTION_MODE",
    "SAMPLING_PERIOD_NS",
    "WORD_DURATION_NS",
    "WORD_LENGTH",
    "ExecutionMode",
    "Quel1BackendController",
    "Quel1BackendExecutionResult",
    "Quel1ExecutionPayload",
    "Quel1Sequencer",
]


def __getattr__(name: str):
    """Lazily import heavy QuEL-1 compatibility symbols."""
    if name == "Quel1Sequencer":
        from .compat.sequencer import Quel1Sequencer

        return Quel1Sequencer
    raise AttributeError(name)
