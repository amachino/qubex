"""Validation and conversion of QuEL-1 capture timing offsets."""

import math

from qubex.backend.quel1.quel1_backend_constants import (
    BLOCK_LENGTH,
    CAPTURE_DELAY_RESOLUTION_NS,
    CAPTURE_DELAY_WORD_STEP,
    WORD_DURATION_NS,
    WORD_LENGTH,
)


def validate_capture_delay_word(value: object) -> None:
    """Require a non-negative integer offset aligned to decimated capture words."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value % CAPTURE_DELAY_WORD_STEP != 0
    ):
        raise ValueError(
            "QuEL-1 capture_delay_word must be a non-negative integer "
            f"multiple of {CAPTURE_DELAY_WORD_STEP} words (32 ns); got {value!r}."
        )


def split_capture_delay_ns(delay_ns: float) -> tuple[int, int]:
    """
    Split a total capture delay in ns into coarse `ndelay` and word offset.

    Parameters
    ----------
    delay_ns : float
        Finite, non-negative total capture delay in ns.

    Returns
    -------
    tuple[int, int]
        Coarse delay in 128 ns blocks and the remaining word offset
        (0, 4, 8 or 12).

    Raises
    ------
    ValueError
        If the delay is not a multiple of the 32 ns capture resolution.
    """
    steps = delay_ns / CAPTURE_DELAY_RESOLUTION_NS
    if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "QuEL-1 capture delay must be a multiple of "
            f"{CAPTURE_DELAY_RESOLUTION_NS} ns; got {delay_ns} ns."
        )
    return divmod(round(delay_ns / WORD_DURATION_NS), BLOCK_LENGTH // WORD_LENGTH)
