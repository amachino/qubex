"""QuEL-1 system-layer constants."""

from __future__ import annotations

from typing import Final

from qubex.backend.quel1.quel1_backend_constants import SAMPLING_PERIOD_NS, WORD_LENGTH

# Workaround capture length for strict QuEL-1 measurement paths.
# Keep 16 raw samples (4 capture words, 32 ns) so the first dummy capture
# absorbs the DSP demodulation/decimation pipeline transient; shorter lengths
# can lead to broken-data mismatches such as 64 captured samples vs 65 expected.
EXTRA_SUM_SECTION_LENGTH: Final[int] = WORD_LENGTH * 4  # samples
EXTRA_POST_BLANK_LENGTH: Final[int] = WORD_LENGTH  # samples
EXTRA_CAPTURE_LENGTH: Final[int] = EXTRA_SUM_SECTION_LENGTH + EXTRA_POST_BLANK_LENGTH
EXTRA_CAPTURE_DURATION_NS: Final[float] = EXTRA_CAPTURE_LENGTH * SAMPLING_PERIOD_NS

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
CNCO_CENTER_READ_HZ: Final[int] = 1_500_000_000
CNCO_CENTER_READ_R8_HZ: Final[int] = 2_250_000_000
FNCO_MAX_HZ: Final[int] = 750_000_000
AWG_MAX_HZ: Final[int] = 250_000_000

DEFAULT_LO_FREQUENCY_HZ: Final[int] = 9_000_000_000
DEFAULT_CNCO_FREQUENCY_HZ: Final[int] = 1_500_000_000
DEFAULT_FNCO_FREQUENCY_HZ: Final[int] = 0
DEFAULT_NWAIT: Final[int] = 0
