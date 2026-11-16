"""Validation and normalization of QuEL-3 capture timing offsets."""

import math

from qubex.backend.quel3.quel3_backend_constants import READOUT_SAMPLING_PERIOD_NS


def normalize_capture_delay_ns(delay_ns: float) -> float:
    """
    Snap a capture delay in ns onto the QuEL-3 readout sampling grid.

    Parameters
    ----------
    delay_ns : float
        Finite, non-negative capture delay in ns.

    Returns
    -------
    float
        Delay in ns as an exact multiple of the readout sampling period.

    Raises
    ------
    ValueError
        If the delay is not a multiple of the readout sampling period.
    """
    samples = delay_ns / READOUT_SAMPLING_PERIOD_NS
    if not math.isclose(samples, round(samples), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "QuEL-3 capture delay must be a multiple of "
            f"{READOUT_SAMPLING_PERIOD_NS} ns; got {delay_ns} ns."
        )
    return round(samples) * READOUT_SAMPLING_PERIOD_NS
