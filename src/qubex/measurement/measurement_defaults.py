"""Default values for measurement configuration."""

from __future__ import annotations

from typing import Final

from qxpulse import RampType

DEFAULT_N_SHOTS: Final[int] = 1024
DEFAULT_SHOT_INTERVAL: Final[float] = 150.0 * 1024.0  # ns

# Backward compatibility aliases.
DEFAULT_SHOTS: Final[int] = DEFAULT_N_SHOTS
DEFAULT_INTERVAL: Final[float] = DEFAULT_SHOT_INTERVAL

DEFAULT_READOUT_DURATION: Final[float] = 384.0  # ns
DEFAULT_READOUT_RAMP_TIME: Final[float] = 32.0  # ns
DEFAULT_READOUT_PRE_MARGIN: Final[float] = 32.0  # ns
DEFAULT_READOUT_POST_MARGIN: Final[float] = 128.0  # ns
DEFAULT_READOUT_DRAG_COEFF: Final[float] = 0.0
DEFAULT_READOUT_RAMP_TYPE: Final[RampType] = "RaisedCosine"

DEFAULT_SHOT_AVERAGING: Final[bool] = True
DEFAULT_TIME_INTEGRATION: Final[bool] = False
DEFAULT_STATE_CLASSIFICATION: Final[bool] = False

__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_N_SHOTS",
    "DEFAULT_READOUT_DRAG_COEFF",
    "DEFAULT_READOUT_DURATION",
    "DEFAULT_READOUT_POST_MARGIN",
    "DEFAULT_READOUT_PRE_MARGIN",
    "DEFAULT_READOUT_RAMP_TIME",
    "DEFAULT_READOUT_RAMP_TYPE",
    "DEFAULT_SHOTS",
    "DEFAULT_SHOT_AVERAGING",
    "DEFAULT_SHOT_INTERVAL",
    "DEFAULT_STATE_CLASSIFICATION",
    "DEFAULT_TIME_INTEGRATION",
]
