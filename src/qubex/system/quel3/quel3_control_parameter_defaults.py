"""QuEL-3 defaults for resolved control parameters."""

from __future__ import annotations

from typing import Final

from qubex.system.control_parameter_defaults import ControlParameterDefaults

DEFAULT_FREQUENCY_MARGIN_GHZ: Final[dict[str, float]] = {
    "READ": 0.1,
    "CTRL_GE": 0.1,
    "CTRL_EF": 0.1,
    "CTRL_CR": 0.1,
    "PUMP": 0.1,
}

DEFAULT_CONTROL_AMPLITUDE: Final[float] = 0.03
DEFAULT_READOUT_AMPLITUDE: Final[float] = 0.01
DEFAULT_CONTROL_VATT: Final[None] = None
DEFAULT_READOUT_VATT: Final[None] = None
DEFAULT_PUMP_VATT: Final[None] = None
DEFAULT_CONTROL_FSC: Final[None] = None
DEFAULT_READOUT_FSC: Final[None] = None
DEFAULT_PUMP_FSC: Final[None] = None
DEFAULT_CAPTURE_DELAY: Final[None] = None
DEFAULT_CAPTURE_DELAY_WORD: Final[None] = None
DEFAULT_PUMP_FREQUENCY_GHZ: Final[float] = 10.0
DEFAULT_PUMP_AMPLITUDE: Final[float] = 0.0
DEFAULT_DC_VOLTAGE: Final[float] = 0.0


class Quel3ControlParameterDefaults(ControlParameterDefaults):
    """Resolved control-parameter defaults for QuEL-3 systems."""

    def __init__(self) -> None:
        super().__init__(
            frequency_margin_by_type=DEFAULT_FREQUENCY_MARGIN_GHZ,
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
