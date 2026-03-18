"""Shared constants for qubex package."""

from typing import Final

DEFAULT_CONFIG_DIR: Final = "/home/shared/qubex-config"
DEFAULT_RAWDATA_DIR: Final = ".rawdata"

CHIP_FILE: Final = "chip.yaml"
SYSTEM_FILE: Final = "system.yaml"
BOX_FILE: Final = "box.yaml"
WIRING_FILE: Final = "wiring.yaml"
PROPS_FILE: Final = "props.yaml"  # legacy
PARAMS_FILE: Final = "params.yaml"  # legacy
MEASUREMENT_DEFAULTS_FILE: Final = "measurement_defaults.yaml"

MUX_SIZE: Final = 4

PREFIX_QUBIT: Final = "Q"
PREFIX_RESONATOR: Final = "RQ"
PREFIX_MUX: Final = "MUX"
