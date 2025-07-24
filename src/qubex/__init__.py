import logging

from .style import apply_template

logger = logging.getLogger(__name__)

try:
    import qubex.patches.quel_ic_config.abstract_nco_ftw_patch  # noqa: F401
    import qubex.patches.quel_ic_config.linkup_fpga_mxfe_patch  # noqa: F401
except ImportError:
    logger.info("Failed to import abstract_nco_ftw_patch.")

try:
    from .experiment import Experiment
except ImportError:
    logger.info("Failed to import Experiment.")
    pass

try:
    from . import pulse
    from .analysis import fitting as fit
    from .analysis import visualization as viz
    from .pulse import Blank, Pulse, PulseArray, PulseChannel, PulseSchedule, VirtualZ
except ImportError as e:
    logger.error(f"Import error: {e}")

apply_template("qubex")


__all__ = [
    "fit",
    "viz",
    "pulse",
    "Experiment",
    "Blank",
    "Pulse",
    "PulseArray",
    "PulseChannel",
    "PulseSchedule",
    "VirtualZ",
]
