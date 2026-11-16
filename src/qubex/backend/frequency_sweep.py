"""Pure frequency-window planning shared by backend capability implementations."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike

ConfigurationT = TypeVar("ConfigurationT", covariant=True)


class SweepConfiguration:
    """
    Identify a backend-owned configuration for a temporary sweep scope.

    Concrete backends subclass this type and retain their native settings.
    Common system APIs pass configurations through unchanged; each backend
    synchronizer validates its own concrete type before applying settings.
    """


@dataclass(frozen=True)
class SweepSegment(Generic[ConfigurationT]):
    """Group input values that can execute under one backend configuration."""

    indices: tuple[int, ...]
    values: tuple[float, ...]
    configuration: ConfigurationT
    reference_value: float | None = None
    requires_reconfiguration: bool = False


def plan_frequency_sweep(
    frequencies: ArrayLike,
    *,
    configure: Callable[
        [tuple[float, ...]],
        tuple[ConfigurationT, bool] | None,
    ],
    subrange_width: float | None = None,
    overlap: bool = False,
) -> tuple[SweepSegment[ConfigurationT], ...]:
    """
    Partition carriers using the actual windows returned by a backend planner.

    Parameters
    ----------
    frequencies
        Nonempty finite one-dimensional carrier frequencies in GHz, in execution
        order. Duplicate and descending points are retained.
    configure
        Pure backend function selecting settings for measurement points and any
        boundary reference. Return the configuration and whether it changes the
        entry settings, or None if the candidate cannot fit. The backend checks
        every point against its closed interval in native units, including
        hardware quantization effects.
        The function must not perform hardware IO or mutate runtime state.
    subrange_width
        Optional positive maximum span in GHz, including the boundary reference.
        Maximal input prefixes are selected; backend constraints may shorten them.
        Width comparisons allow round-off at the input frequencies' precision.
    overlap
        Require the preceding segment's final carrier to be representable in
        each subsequent segment. References are not duplicated in input indices.

    Returns
    -------
    tuple[SweepSegment, ...]
        Immutable partition preserving input positions.

    Raises
    ------
    ValueError
        Input is invalid or a point and its required reference cannot fit.

    Notes
    -----
    Windows cover carrier frequencies, not arbitrary waveform spectra or phase
    continuity. Configuration selection exceptions propagate without retries.
    """
    values = np.asarray(frequencies, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Frequencies must be a nonempty finite one-dimensional array.")
    if subrange_width is not None and (
        not math.isfinite(subrange_width) or subrange_width <= 0
    ):
        raise ValueError("Subrange width must be finite and positive.")
    points = tuple(float(value) for value in values)
    segments: list[SweepSegment[ConfigurationT]] = []
    start = 0
    while start < len(points):
        reference = points[start - 1] if overlap and start else None
        accepted: tuple[ConfigurationT, bool] | None = None
        stop = start
        limit = start
        lower_point = upper_point = points[start] if reference is None else reference
        for point in points[start:]:
            lower_point = min(lower_point, point)
            upper_point = max(upper_point, point)
            if (
                subrange_width is not None
                and upper_point - lower_point
                > subrange_width + math.ulp(lower_point) + math.ulp(upper_point)
            ):
                break
            limit += 1
        for end in range(limit, start, -1):
            group = points[start:end]
            required = group if reference is None else (reference, *group)
            candidate = configure(required)
            if candidate is None:
                continue
            accepted = candidate
            stop = end
            break
        if accepted is None:
            raise ValueError(
                f"Frequency at index {start} and its reference cannot fit a backend window."
            )
        segments.append(
            SweepSegment(
                indices=tuple(range(start, stop)),
                values=points[start:stop],
                configuration=accepted[0],
                requires_reconfiguration=accepted[1],
                reference_value=reference,
            )
        )
        start = stop
    return tuple(segments)
