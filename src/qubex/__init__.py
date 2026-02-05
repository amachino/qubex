"""Top-level package exports for qubex."""

from __future__ import annotations

import qubex.patches.quel_ic_config.linkup_fpga_mxfe_patch  # noqa: F401

from . import pulse
from .analysis import fitting as fit
from .analysis import visualization as viz
from .core import quantities, units
from .experiment import Experiment
from .logging import set_log_level
from .pulse import Blank, Pulse, PulseArray, PulseChannel, PulseSchedule, VirtualZ
from .style import apply_template

# Set default log level to INFO
set_log_level("INFO")

# Set default plotly template
apply_template("qubex")


__all__ = [
    "Blank",
    "Experiment",
    "Pulse",
    "PulseArray",
    "PulseChannel",
    "PulseSchedule",
    "VirtualZ",
    "fit",
    "pulse",
    "quantities",
    "set_log_level",
    "units",
    "viz",
]
