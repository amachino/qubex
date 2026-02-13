"""Top-level package exports for qubex."""

from __future__ import annotations

import qxpulse as pulse
from qxpulse import Blank, Pulse, PulseArray, PulseChannel, PulseSchedule, VirtualZ

import qubex.patches.quel_ic_config.disable_quelware_filelock_patch
import qubex.patches.quel_ic_config.linkup_fpga_mxfe_patch  # noqa: F401

from . import core
from .analysis import fitting as fit
from .analysis import visualization as viz
from .core import Frequency, FrequencyArray, Time, TimeArray, units
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
    "Frequency",
    "FrequencyArray",
    "MeasurementClient",
    "Pulse",
    "PulseArray",
    "PulseChannel",
    "PulseSchedule",
    "Time",
    "TimeArray",
    "VirtualZ",
    "core",
    "fit",
    "pulse",
    "set_log_level",
    "units",
    "viz",
]
