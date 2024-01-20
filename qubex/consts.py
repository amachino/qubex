"""
Constants used in the qubex package.
"""

from typing import Final

SAMPLES_PER_WORDS = 4
SAMPLING_PERIOD: Final[int] = 2  # [ns]
MIN_SAMPLE: Final[int] = 64  # min number of samples of e7awg
MIN_DURATION: Final[int] = MIN_SAMPLE * SAMPLING_PERIOD
T_CONTROL: Final[int] = 10 * 1024  # [ns]
T_READOUT: Final[int] = 1024  # [ns]
T_MARGIN: Final[int] = MIN_DURATION  # [ns]
