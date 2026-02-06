"""Top-level package exports for qubex."""

from __future__ import annotations

import qxcore as core
import qxpulse as pulse
from qxcore import quantities, units
from qxpulse import Blank, Pulse, PulseArray, PulseChannel, PulseSchedule, VirtualZ

import qubex.patches.quel_ic_config.linkup_fpga_mxfe_patch  # noqa: F401

from .analysis import fitting as fit
from .analysis import visualization as viz
from .experiment import Experiment
from .logging import set_log_level
from .measurement import MeasurementClient
from .style import apply_template

# Set default log level to INFO
set_log_level("INFO")

# Set default plotly template
apply_template("qubex")


__all__ = [
    "Blank",
    "Experiment",
    "MeasurementClient",
    "Pulse",
    "PulseArray",
    "PulseChannel",
    "PulseSchedule",
    "VirtualZ",
    "core",
    "fit",
    "pulse",
    "quantities",
    "set_log_level",
    "units",
    "viz",
]
