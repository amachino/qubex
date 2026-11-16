"""Behavioral contracts for backend frequency-window planning."""

from itertools import pairwise

import numpy as np
import pytest

from qubex.backend.frequency_sweep import plan_frequency_sweep


def window_for(frequencies: tuple[float, ...]):
    """Return a quantized window with a bounded usable carrier range."""
    center = round((min(frequencies) + max(frequencies)) / 2, 1)
    if not all(center - 0.2 <= f <= center + 0.2 for f in frequencies):
        return None
    return center, True


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

    def configure(_values):
        pytest.fail("Configuration must not be consulted")

    with pytest.raises(ValueError, match="one-dimensional"):
        plan_frequency_sweep(values, configure=configure)


def test_descending_and_repeated_points_preserve_order() -> None:
    """Descending and repeated frequencies retain their original ordering."""
    values = [5.5, 5.5, 5.2, 5.1, 4.8]
    plan = plan_frequency_sweep(values, configure=window_for)
    assert [f for segment in plan for f in segment.values] == values


@pytest.mark.parametrize(
    ("overlap", "expected"), [(False, [(0, 1, 2), (3, 4)]), (True, [(0, 1, 2), (3, 4)])]
)
def test_greedy_width_includes_reference(overlap, expected):
    """Maximal prefixes include boundary references in their width budget."""
    values = [5.0, 5.125, 5.25, 5.375, 5.5]
    plan = plan_frequency_sweep(
        values,
        configure=lambda p: (p, True),
        subrange_width=0.25,
        overlap=overlap,
    )
    assert [s.indices for s in plan] == expected
    assert plan[1].configuration == tuple(values[2:] if overlap else values[3:])


@pytest.mark.parametrize(
    "values", [[5.0], [5.0, 5.0, 5.0], [5.25, 5.125, 5.0], [5.0, 5.25, 5.125]]
)
def test_closed_windows_preserve_input_positions(values):
    """Closed windows preserve singleton, duplicate, descending and nonmonotonic inputs."""
    plan = plan_frequency_sweep(
        values,
        configure=lambda p: (p, False),
        subrange_width=0.25,
        overlap=True,
    )
    assert [s.indices for s in plan] == [tuple(range(len(values)))]
    assert plan[0].values == tuple(values)


def test_overlap_gap_exceeding_width_cannot_advance():
    """A reference-only candidate fails instead of repeating the same position."""
    with pytest.raises(ValueError, match="reference"):
        plan_frequency_sweep(
            [5.0, 5.5],
            configure=lambda p: (p, False),
            subrange_width=0.25,
            overlap=True,
        )


@pytest.mark.parametrize(
    "values",
    [[5.0, 5.1, 5.2], [5.2, 5.1, 5.0], [-5.2, -5.1, -5.0], np.linspace(5.0, 5.2, 3)],
)
def test_overlap_width_allows_frequency_roundoff(values):
    """Decimal step boundaries retain overlap without rejecting round-off."""
    plan = plan_frequency_sweep(
        values,
        configure=window_for,
        subrange_width=0.1,
        overlap=True,
    )
    assert [segment.indices for segment in plan] == [(0, 1), (2,)]
    assert plan[1].reference_value == values[1]
    assert [f for segment in plan for f in segment.values] == list(values)


def test_overlap_width_rejects_real_excess():
    """A gap exceeding the width by one hertz still fails."""
    with pytest.raises(ValueError, match="reference"):
        plan_frequency_sweep(
            [5.0, 5.100000001],
            configure=window_for,
            subrange_width=0.1,
            overlap=True,
        )
