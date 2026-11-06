"""Behavioral contracts for backend frequency-window planning."""

from itertools import pairwise

import pytest

from qubex.backend.frequency_sweep import plan_frequency_sweep


def window_for(frequencies: tuple[float, ...], _reference: float | None):
    """Return a quantized window with a bounded usable carrier range."""
    center = round((min(frequencies) + max(frequencies)) / 2, 1)
    return center - 0.2, center + 0.2, center, True


def test_nonuniform_points_respect_actual_window() -> None:
    """Nonuniform points are partitioned by achievable ranges, preserving indices."""
    plan = plan_frequency_sweep([5.0, 5.01, 5.02, 5.7], configure=window_for)
    assert [s.indices for s in plan] == [(0, 1, 2), (3,)]
    for segment in plan:
        assert all(abs(segment.configuration - f) < 0.2 for f in segment.values)


def test_overlap_is_covered_by_new_configuration() -> None:
    """Reference points remain representable under the following configuration."""
    plan = plan_frequency_sweep(
        [5.0, 5.1, 5.2, 5.3, 5.4],
        configure=window_for,
        subrange_width=0.2,
        overlap=True,
    )
    for previous, current in pairwise(plan):
        assert current.reference_value == previous.values[-1]
        assert current.reference_value is not None
        assert abs(current.configuration - current.reference_value) < 0.2


def test_impossible_overlap_fails() -> None:
    """Unbridgeable gaps fail instead of producing an invalid reference point."""
    with pytest.raises(ValueError, match="reference"):
        plan_frequency_sweep([5.0, 6.0], configure=window_for, overlap=True)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [[5.0]]])
def test_invalid_values_fail_before_configuration(values) -> None:
    """Invalid frequency arrays are rejected before asking for configurations."""

    def configure(_values, _reference):
        pytest.fail("Configuration must not be consulted")

    with pytest.raises(ValueError, match="one-dimensional"):
        plan_frequency_sweep(values, configure=configure)


def test_descending_and_repeated_points_preserve_order() -> None:
    """Descending and repeated frequencies retain their original ordering."""
    values = [5.5, 5.5, 5.2, 5.1, 4.8]
    plan = plan_frequency_sweep(values, configure=window_for)
    assert [f for segment in plan for f in segment.values] == values


def test_width_balances_points_without_shifting_centers_for_references():
    """Width-based groups stay balanced and references do not shift their settings."""
    values = [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6]

    def configure(points, _reference):
        center = (min(points) + max(points)) / 2
        return center - 0.5, center + 0.5, center, True

    plan = plan_frequency_sweep(
        values, configure=configure, subrange_width=0.4, overlap=True
    )
    assert [s.indices for s in plan] == [(0, 1, 2, 3), (4, 5, 6)]
    assert plan[1].configuration == 5.5
    assert plan[1].reference_value == 5.3
