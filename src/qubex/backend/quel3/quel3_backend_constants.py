"""QuEL-3 backend-specific hardware constants."""

from __future__ import annotations

from typing import Final

SAMPLING_PERIOD_NS: Final[float] = 0.4
READOUT_SAMPLING_PERIOD_NS: Final[float] = 0.8
CAPTURE_DECIMATION_FACTOR: Final[int] = 1

# Full instrument spans used for temporary frequency sweep windows.
READOUT_FREQUENCY_SWEEP_WIDTH_HZ: Final[float] = 0.8e9
CONTROL_FREQUENCY_SWEEP_WIDTH_HZ: Final[float] = 1.8e9
DEFAULT_MIN_FREQUENCY_MARGIN_GHZ: Final[float] = 0.1

# Configurable amplitude multiplier applied when registering waveform events (-3 dB).
EVENT_GAIN_SCALE: float = 10 ** (-3 / 20)
