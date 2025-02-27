import logging

from .style import apply_template

logger = logging.getLogger(__name__)

try:
    from . import (
        api,
        backend,
        clifford,
        diagnostics,
        experiment,
        measurement,
        pulse,
        simulator,
    )
    from .experiment import Experiment
except ImportError as e:
    logger.debug(f"Import error: {e}")

apply_template("qubex")


__all__ = [
    "api",
    "backend",
    "clifford",
    "diagnostics",
    "experiment",
    "measurement",
    "pulse",
    "simulator",
    "Experiment",
]
