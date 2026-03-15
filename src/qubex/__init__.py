"""Top-level package exports for qubex."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import qxpulse as pulse
from qxpulse import (
    Blank,
    Pulse,
    PulseArray,
    PulseChannel,
    PulseSchedule,
    VirtualZ,
)

from .logging import set_log_level

if TYPE_CHECKING:
    from . import contrib, core, visualization as viz
    from .analysis import fitting as fit
    from .core import Frequency, FrequencyArray, Time, TimeArray, units
    from .experiment import Experiment
    from .measurement import Measurement

# Set default log level to INFO
set_log_level("INFO")


__all__ = [
    "Blank",
    "Experiment",
    "Frequency",
    "FrequencyArray",
    "Measurement",
    "Pulse",
    "PulseArray",
    "PulseChannel",
    "PulseSchedule",
    "Time",
    "TimeArray",
    "VirtualZ",
    "contrib",
    "core",
    "fit",
    "pulse",
    "set_log_level",
    "units",
    "viz",
]


_LAZY_ATTR_TO_MODULE: dict[str, str] = {
    "Experiment": "qubex.experiment",
    "Measurement": "qubex.measurement",
    "contrib": "qubex.contrib",
    "core": "qubex.core",
    "fit": "qubex.analysis.fitting",
    "viz": "qubex.visualization",
    "Frequency": "qubex.core",
    "FrequencyArray": "qubex.core",
    "Time": "qubex.core",
    "TimeArray": "qubex.core",
    "units": "qubex.core",
}


def __getattr__(name: str) -> Any:
    """Lazily load heavyweight top-level exports on first access."""
    module_name = _LAZY_ATTR_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = importlib.import_module(module_name)
    return getattr(module, name)
