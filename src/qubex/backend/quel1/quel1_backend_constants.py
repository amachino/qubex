"""QuEL hardware-related timing constants."""

from __future__ import annotations

from typing import Final, Literal

ExecutionMode = Literal["serial", "parallel"]
DEFAULT_EXECUTION_MODE: Final[ExecutionMode] = "parallel"
DEFAULT_CLOCK_HEALTH_CHECKS: Final[bool] = False

SAMPLING_PERIOD: Final[float] = 2.0  # ns
CAPTURE_DECIMATION_FACTOR: Final[int] = 4
WORD_LENGTH: Final[int] = 4  # samples
WORD_DURATION: Final[float] = WORD_LENGTH * SAMPLING_PERIOD  # ns
BLOCK_LENGTH: Final[int] = WORD_LENGTH * 16  # samples
BLOCK_DURATION: Final[float] = BLOCK_LENGTH * SAMPLING_PERIOD  # ns

EXTRA_SUM_SECTION_LENGTH: Final[int] = WORD_LENGTH * 4  # samples
EXTRA_POST_BLANK_LENGTH: Final[int] = WORD_LENGTH  # samples
EXTRA_CAPTURE_LENGTH: Final[int] = EXTRA_SUM_SECTION_LENGTH + EXTRA_POST_BLANK_LENGTH
EXTRA_CAPTURE_DURATION: Final[float] = EXTRA_CAPTURE_LENGTH * SAMPLING_PERIOD  # ns

LO_STEP: Final[int] = 500_000_000
NCO_STEP: Final[int] = 23_437_500
CNCO_CENTER_CTRL: Final[int] = 2_250_000_000
CNCO_CENTER_READ: Final[int] = 1_500_000_000
CNCO_CENTER_READ_R8: Final[int] = 2_250_000_000
FNCO_MAX: Final[int] = 750_000_000
AWG_MAX: Final[int] = 250_000_000

DEFAULT_CLOCK_MASTER_ADDRESS: Final[str] = "10.3.0.255"
DEFAULT_LO_FREQ: Final[int] = 9_000_000_000
DEFAULT_CNCO_FREQ: Final[int] = 1_500_000_000
DEFAULT_FNCO_FREQ: Final[int] = 0
DEFAULT_VATT: Final[int] = 3072  # 0xC00
DEFAULT_FULLSCALE_CURRENT: Final[int] = 40527
DEFAULT_NDELAY: Final[int] = 7
DEFAULT_NWAIT: Final[int] = 0

DEFAULT_CONTROL_AMPLITUDE: Final[float] = 0.03
DEFAULT_READOUT_AMPLITUDE: Final[float] = 0.01
DEFAULT_CONTROL_VATT: Final[int] = 3072
DEFAULT_READOUT_VATT: Final[int] = 2048
DEFAULT_PUMP_VATT: Final[int] = 3072
DEFAULT_CONTROL_FSC: Final[int] = 40527
DEFAULT_READOUT_FSC: Final[int] = 40527
DEFAULT_PUMP_FSC: Final[int] = 40527
DEFAULT_CAPTURE_DELAY: Final[int] = 7
DEFAULT_CAPTURE_DELAY_WORD: Final[int] = 0
DEFAULT_PUMP_FREQUENCY: Final[float] = 10.0
DEFAULT_PUMP_AMPLITUDE: Final[float] = 0.0
DEFAULT_DC_VOLTAGE: Final[float] = 0.0
