"""Tests for block sharing while compiling QuEL-3 pulse schedules."""

import pytest

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
