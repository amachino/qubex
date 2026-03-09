"""QuEL-1 defaults for resolved control parameters."""

from __future__ import annotations

from typing import Final

from qubex.system.control_parameter_defaults import ControlParameterDefaults

DEFAULT_CONTROL_AMPLITUDE: Final[float] = 0.1
DEFAULT_READOUT_AMPLITUDE: Final[float] = 0.1
DEFAULT_CONTROL_VATT: Final[int] = 3072  # 0xC00
DEFAULT_READOUT_VATT: Final[int] = 2048
DEFAULT_PUMP_VATT: Final[int] = 3072  # 0xC00
DEFAULT_CONTROL_FSC: Final[int] = 40527
DEFAULT_READOUT_FSC: Final[int] = 40527
DEFAULT_PUMP_FSC: Final[int] = 40527
DEFAULT_CAPTURE_DELAY: Final[int] = 7
DEFAULT_CAPTURE_DELAY_WORD: Final[int] = 0
DEFAULT_PUMP_FREQUENCY_GHZ: Final[float] = 10.0
DEFAULT_PUMP_AMPLITUDE: Final[float] = 0.0
DEFAULT_DC_VOLTAGE: Final[float] = 0.0


class Quel1ControlParameterDefaults(ControlParameterDefaults):
    """Resolved control-parameter defaults for QuEL-1 systems."""

    def __init__(self) -> None:
        super().__init__(
            control_amplitude=DEFAULT_CONTROL_AMPLITUDE,
            readout_amplitude=DEFAULT_READOUT_AMPLITUDE,
            control_vatt=DEFAULT_CONTROL_VATT,
            readout_vatt=DEFAULT_READOUT_VATT,
            pump_vatt=DEFAULT_PUMP_VATT,
            control_fsc=DEFAULT_CONTROL_FSC,
            readout_fsc=DEFAULT_READOUT_FSC,
            pump_fsc=DEFAULT_PUMP_FSC,
            capture_delay=DEFAULT_CAPTURE_DELAY,
            capture_delay_word=DEFAULT_CAPTURE_DELAY_WORD,
            pump_frequency=DEFAULT_PUMP_FREQUENCY_GHZ,
            pump_amplitude=DEFAULT_PUMP_AMPLITUDE,
            dc_voltage=DEFAULT_DC_VOLTAGE,
        )
