"""Validation of QuEL-1 capture timing offsets."""

from qubex.backend.quel1.quel1_backend_constants import CAPTURE_DELAY_WORD_STEP


def validate_capture_delay_word(value: object, *, mux: int) -> None:
    """Require a non-negative integer offset aligned to decimated capture words."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value % CAPTURE_DELAY_WORD_STEP != 0
    ):
        raise ValueError(
            f"QuEL-1 capture_delay_word for MUX{mux} must be a non-negative integer "
            f"multiple of {CAPTURE_DELAY_WORD_STEP} words (32 ns); got {value!r}."
        )
