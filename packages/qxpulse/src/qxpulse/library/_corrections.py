"""Shared quadrature corrections for pulse envelopes."""

from numpy.typing import NDArray


def _cd_quadrature(
    envelope: NDArray,
    derivative: NDArray,
    *,
    delta: float,
    correction_factor: float,
) -> NDArray:
    """Return CD Q for transition-minus-drive delta and I+iQ angular-rate inputs."""
    return -(correction_factor * delta) / (delta**2 + envelope**2) * derivative


def _reject_legacy_factor(options: dict) -> None:
    """Reject the removed direct-SQUAD factor before it can be silently ignored."""
    if "factor" in options:
        raise TypeError(
            "factor has been removed; use correction_factor=-old_factor "
            "to preserve a legacy Squad waveform (old None/default means 1)."
        )
