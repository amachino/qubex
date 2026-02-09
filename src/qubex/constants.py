"""Shared constants for qubex package."""

from typing import Final

DEFAULT_CONFIG_DIR: Final = "/home/shared/qubex-config"
DEFAULT_RAWDATA_DIR: Final = ".rawdata"

CHIP_FILE: Final = "chip.yaml"
BOX_FILE: Final = "box.yaml"
WIRING_FILE: Final = "wiring.yaml"
PROPS_FILE: Final = "props.yaml"  # legacy
PARAMS_FILE: Final = "params.yaml"  # legacy

MUX_SIZE: Final = 4

PREFIX_QUBIT: Final = "Q"
PREFIX_RESONATOR: Final = "RQ"
PREFIX_MUX: Final = "MUX"
