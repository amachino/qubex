"""QuEL hardware-related timing constants."""

from __future__ import annotations

from typing import Final, Literal

ExecutionMode = Literal["serial", "parallel"]
DEFAULT_EXECUTION_MODE: Final[ExecutionMode] = "parallel"
DEFAULT_CLOCK_HEALTH_CHECKS: Final[bool] = True

SAMPLING_PERIOD_NS: Final[float] = 2.0
CAPTURE_DECIMATION_FACTOR: Final[int] = 4
WORD_LENGTH: Final[int] = 4  # samples
WORD_DURATION_NS: Final[float] = WORD_LENGTH * SAMPLING_PERIOD_NS
# A capture word after decimation spans four input words (16 samples, 32 ns).
CAPTURE_DELAY_WORD_STEP: Final[int] = CAPTURE_DECIMATION_FACTOR
CAPTURE_DELAY_RESOLUTION_NS: Final[float] = CAPTURE_DELAY_WORD_STEP * WORD_DURATION_NS
BLOCK_LENGTH: Final[int] = WORD_LENGTH * 16  # samples
BLOCK_DURATION_NS: Final[float] = BLOCK_LENGTH * SAMPLING_PERIOD_NS

DEFAULT_BACKGROUND_NOISE_THRESHOLD_RELINKUP: Final[float] = 1024.0
DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT: Final[float] = 50000.0

# Keep LO/NCO frequencies on grids that preserve reproducible phase at QuBE
# execution boundaries. 500 MHz gives one full cycle in the 2 ns sampling
# period, and 23.4375 MHz gives three full cycles in the 128 ns minimum block.
# This prevents phase from depending on pulse placement within a block or across
# repeated shots. We do not currently use the finer 7.8125 MHz grid, even
# though it matches one cycle per 128 ns block, because older FTW handling does
# not represent it exactly and can introduce slow phase drift. When exact
# fractional FTW support is available end-to-end, this constraint may be
# relaxed.
LO_STEP_HZ: Final[int] = 500_000_000
NCO_STEP_HZ: Final[int] = 23_437_500
CNCO_CENTER_CTRL_HZ: Final[int] = 2_250_000_000

# AWG detuning limits on either side of the mixed carrier.
AWG_MAX_HZ: Final[int] = 250_000_000
AWG_SAFE_MAX_HZ: Final[int] = 200_000_000
