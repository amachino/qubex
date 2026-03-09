"""QuEL hardware-related timing constants."""

from __future__ import annotations

from typing import Final, Literal

ExecutionMode = Literal["serial", "parallel"]
DEFAULT_EXECUTION_MODE: Final[ExecutionMode] = "parallel"
DEFAULT_CLOCK_HEALTH_CHECKS: Final[bool] = False

SAMPLING_PERIOD_NS: Final[float] = 2.0
CAPTURE_DECIMATION_FACTOR: Final[int] = 4
WORD_LENGTH: Final[int] = 4  # samples
WORD_DURATION_NS: Final[float] = WORD_LENGTH * SAMPLING_PERIOD_NS
BLOCK_LENGTH: Final[int] = WORD_LENGTH * 16  # samples
BLOCK_DURATION_NS: Final[float] = BLOCK_LENGTH * SAMPLING_PERIOD_NS

RELAXED_NOISE_THRESHOLD: Final[int] = 50_000
