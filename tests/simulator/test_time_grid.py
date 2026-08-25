"""Tests for simulator time-grid construction."""

import numpy as np
from qxsimulator.simulation._time_grid import create_integration_grid


def test_long_integration_grid_does_not_double_terminal_step() -> None:
    """Large durations should not turn the terminal interval into a double step."""
    duration = 10_000.0
    dt = 0.1

    times = create_integration_grid(duration=duration, dt=dt)

    assert times[-1] == duration
    assert np.max(np.diff(times)) <= dt + 1e-12
