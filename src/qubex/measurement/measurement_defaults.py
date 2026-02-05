"""Default values for measurement configuration."""

from __future__ import annotations

from typing import Final

from qubex.pulse import RampType

DEFAULT_SHOTS: Final[int] = 1024
DEFAULT_INTERVAL: Final[float] = 150.0 * 1024.0  # ns

DEFAULT_READOUT_DURATION: Final[float] = 384.0  # ns
DEFAULT_READOUT_RAMPTIME: Final[float] = 32.0  # ns
DEFAULT_READOUT_PRE_MARGIN: Final[float] = 32.0  # ns
DEFAULT_READOUT_POST_MARGIN: Final[float] = 128.0  # ns
DEFAULT_READOUT_DRAG_COEFF: Final[float] = 0.0
DEFAULT_READOUT_RAMP_TYPE: Final[RampType] = "RaisedCosine"

DEFAULT_ENABLE_DSP_DEMODULATION: Final[bool] = True
DEFAULT_ENABLE_DSP_SUM: Final[bool] = False
DEFAULT_ENABLE_DSP_CLASSIFICATION: Final[bool] = False
DEFAULT_LINE_PARAM0: Final[tuple[float, float, float]] = (1.0, 0.0, 0.0)
DEFAULT_LINE_PARAM1: Final[tuple[float, float, float]] = (0.0, 1.0, 0.0)

__all__ = [
    "DEFAULT_ENABLE_DSP_CLASSIFICATION",
    "DEFAULT_ENABLE_DSP_DEMODULATION",
    "DEFAULT_ENABLE_DSP_SUM",
    "DEFAULT_INTERVAL",
    "DEFAULT_LINE_PARAM0",
    "DEFAULT_LINE_PARAM1",
    "DEFAULT_READOUT_DRAG_COEFF",
    "DEFAULT_READOUT_DURATION",
    "DEFAULT_READOUT_POST_MARGIN",
    "DEFAULT_READOUT_PRE_MARGIN",
    "DEFAULT_READOUT_RAMPTIME",
    "DEFAULT_READOUT_RAMP_TYPE",
    "DEFAULT_SHOTS",
]
