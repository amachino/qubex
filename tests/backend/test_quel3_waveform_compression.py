"""Tests for block sharing while compiling QuEL-3 pulse schedules."""

import pytest

from qubex.backend.quel3.builders import Quel3PulseEventBuilder
from qubex.backend.quel3.models import Quel3WaveformCompression


@pytest.mark.parametrize(
    "settings",
    [
        {"target_block_samples": 0},
        {"min_chunk_samples": -1},
        {"min_chunk_samples": 5001},
        {"enabled": 1},
    ],
)
def test_invalid_compression_settings(settings):
    """Block lengths must be positive, ordered integers and enabled must be boolean."""
    with pytest.raises(ValueError, match=r"enabled|block_samples|chunk_samples"):
        Quel3WaveformCompression(**settings)


@pytest.mark.parametrize(
    ("length", "first", "stop", "alignment", "target", "minimum", "expected"),
    [
        (31, 1, 30, 3, 10, 4, (6, 9, 2)),
        (30, 0, 30, 6, 5, 3, None),
        (30, 0, 30, 6, 10, 7, None),
        (11, 0, 10, 1, 5, 2, None),
    ],
)
def test_integer_block_policy(
    length, first, stop, alignment, target, minimum, expected
):
    """Fixed blocks respect alignment, plateau margins and minimum fragment length."""
    block = Quel3PulseEventBuilder._choose_repeated_block(  # noqa: SLF001
        length=length,
        first=first,
        stop=stop,
        alignment=alignment,
        target=target,
        minimum=minimum,
    )
    assert ((block.start, block.size, block.count) if block else None) == expected
