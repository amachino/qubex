"""Pure frequency-window planning shared by backend capability implementations."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike

ConfigurationT = TypeVar("ConfigurationT", covariant=True)


ValueT = TypeVar("ValueT", covariant=True)


class SweepConfiguration:
    """
    Identify a backend-owned configuration for a temporary sweep scope.

    Concrete backends subclass this type and retain their native settings.
    Common system APIs pass configurations through unchanged; each backend
    synchronizer validates its own concrete type before applying settings.
    """


@dataclass(frozen=True)
class SweepSegment(Generic[ValueT, ConfigurationT]):
    """Group input values that can execute under one backend configuration."""

    indices: tuple[int, ...]
    values: tuple[ValueT, ...]
    configuration: ConfigurationT
    reference_value: ValueT | None = None
    requires_reconfiguration: bool = False


def plan_frequency_sweep(
    frequencies: ArrayLike,
    *,
    configure: Callable[
        [tuple[float, ...], float | None],
        tuple[float, float, ConfigurationT, bool],
    ],
    subrange_width: float | None = None,
    overlap: bool = False,
    include_endpoints: bool = False,
) -> tuple[SweepSegment[float, ConfigurationT], ...]:
    """
    Partition carriers using the actual windows returned by a backend planner.

    Parameters
    ----------
    frequencies
        Nonempty finite one-dimensional carrier frequencies in GHz, in execution
        order. Duplicate and descending points are retained.
    configure
        Pure backend function selecting settings and their usable interval for
        a proposed group of measurement points and a separate optional boundary
        reference. Both must fit the selected interval. Keeping the reference
        separate lets backends preserve measurement-centered tuning when possible.
        Returned limits include hardware quantization effects.
        The function must not perform hardware IO or mutate runtime state.
    subrange_width
        Optional positive maximum span of measurement points in GHz. Initial
        groups distribute points evenly; hardware constraints may split them further.
    overlap
        Require the preceding segment's final carrier to be representable in
        each subsequent segment. References are not duplicated in input indices.
    include_endpoints
        Whether the backend permits carriers at the returned interval endpoints.
        Defaults to strict limits; inclusive limits also support zero-width ranges.

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
    segments: list[SweepSegment[float, ConfigurationT]] = []
    count = (
        1
        if subrange_width is None
        else min(len(points), int((max(points) - min(points)) // subrange_width) + 1)
    )
    boundaries = np.cumsum([len(group) for group in np.array_split(values, count)])
    start = 0
    while start < len(points):
        reference = points[start - 1] if overlap and start else None
        accepted: tuple[float, float, ConfigurationT, bool] | None = None
        stop = start
        limit = next(int(boundary) for boundary in boundaries if boundary > start)
        for end in range(limit, start, -1):
            group = points[start:end]
            if subrange_width is not None and max(group) - min(group) > subrange_width:
                continue
            required = group if reference is None else (reference, *group)
            candidate = configure(group, reference)
            lower, upper = candidate[:2]
            if not all(
                lower <= value <= upper if include_endpoints else lower < value < upper
                for value in required
            ):
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
                configuration=accepted[2],
                requires_reconfiguration=accepted[3],
                reference_value=reference,
            )
        )
        start = stop
    return tuple(segments)
