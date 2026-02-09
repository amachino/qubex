"""QuEL hardware-related timing constants."""

from __future__ import annotations

from typing import Final

SAMPLING_PERIOD: Final[float] = 2.0  # ns

WORD_LENGTH: Final[int] = 4  # samples
WORD_DURATION: Final[float] = WORD_LENGTH * SAMPLING_PERIOD  # ns

BLOCK_LENGTH: Final[int] = WORD_LENGTH * 16  # samples
BLOCK_DURATION: Final[float] = BLOCK_LENGTH * SAMPLING_PERIOD  # ns

EXTRA_SUM_SECTION_LENGTH: Final[int] = WORD_LENGTH * 4  # samples
EXTRA_POST_BLANK_LENGTH: Final[int] = WORD_LENGTH  # samples
EXTRA_CAPTURE_LENGTH: Final[int] = EXTRA_SUM_SECTION_LENGTH + EXTRA_POST_BLANK_LENGTH
EXTRA_CAPTURE_DURATION: Final[float] = EXTRA_CAPTURE_LENGTH * SAMPLING_PERIOD  # ns

DEFAULT_MONITOR_CAPTURE_DELAY: Final[int] = 7
