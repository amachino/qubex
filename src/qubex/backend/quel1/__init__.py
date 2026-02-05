"""QuEL1-specific backend components."""

from .quel1_backend_constants import (
    BLOCK_DURATION,
    BLOCK_LENGTH,
    EXTRA_CAPTURE_DURATION,
    EXTRA_CAPTURE_LENGTH,
    EXTRA_POST_BLANK_LENGTH,
    EXTRA_SUM_SECTION_LENGTH,
    SAMPLING_PERIOD,
    WORD_DURATION,
    WORD_LENGTH,
)
from .quel1_backend_controller import (
    DeviceController,
    Quel1BackendController,
    RawResult,
)
from .quel1_backend_executor import Quel1BackendExecutor, Quel1ExecutionPayload
from .quel1_sequencer import Quel1Sequencer, SequencerMod

__all__ = [
    "BLOCK_DURATION",
    "BLOCK_LENGTH",
    "EXTRA_CAPTURE_DURATION",
    "EXTRA_CAPTURE_LENGTH",
    "EXTRA_POST_BLANK_LENGTH",
    "EXTRA_SUM_SECTION_LENGTH",
    "SAMPLING_PERIOD",
    "WORD_DURATION",
    "WORD_LENGTH",
    "DeviceController",  # TODO: Remove this alias in future versions.
    "Quel1BackendController",
    "Quel1BackendExecutor",
    "Quel1ExecutionPayload",
    "Quel1Sequencer",
    "RawResult",
    "SequencerMod",  # TODO: Remove this alias in future versions.
]
