"""Time-grid construction for quantum simulations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .control import Control


def create_integration_grid(
    controls: list[Control],
    max_step: float,
) -> npt.NDArray[np.float64]:
    """
    Combine a uniform maximum-step grid with all control boundaries.

    Parameters
    ----------
    controls : list[Control]
        Nonempty controls with a common duration.
    max_step : float
        Maximum interval of the uniform grid in ns. Must be finite and
        positive.

    Returns
    -------
    npt.NDArray[np.float64]
        Strictly increasing integration times in ns, starting at zero and
        ending at the common control duration.

    Raises
    ------
    ValueError
        If `max_step` is not finite and positive.

    Notes
    -----
    Control boundaries can split a uniform interval, so adjacent returned
    times may be separated by less than `max_step`.
    """
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("dt must be finite and greater than zero.")

    duration = controls[0].duration
    uniform_times = _create_uniform_times(duration, max_step)
    return _merge_times(
        [uniform_times, *(control.times for control in controls)],
        duration,
    )


def create_control_boundary_times(
    controls: list[Control],
) -> npt.NDArray[np.float64]:
    """
    Return the sorted union of all control segment boundaries.

    Parameters
    ----------
    controls : list[Control]
        Nonempty controls with a common duration.

    Returns
    -------
    npt.NDArray[np.float64]
        Strictly increasing boundary times in ns, starting at zero and ending
        at the common control duration.
    """
    duration = controls[0].duration
    return _merge_times(
        [control.times for control in controls],
        duration,
    )


def create_uniform_output_times(
    duration: float,
    n_samples: int,
) -> npt.NDArray[np.float64]:
    """
    Create uniformly spaced trajectory output times.

    Parameters
    ----------
    duration : float
        Terminal time in ns.
    n_samples : int
        Requested number of output points. Must be at least 2.

    Returns
    -------
    npt.NDArray[np.float64]
        Uniform times from zero through `duration`. A zero-duration trajectory
        contains the single unique time zero.

    Raises
    ------
    ValueError
        If `n_samples` is less than 2.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2 when specified.")
    if duration == 0:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(0.0, duration, n_samples, dtype=np.float64)


def create_evolution_times(
    checkpoint_times: npt.NDArray[np.float64],
    output_times: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Combine mandatory evolution checkpoints with requested output times.

    Parameters
    ----------
    checkpoint_times : npt.NDArray[np.float64]
        Sorted times where integration must stop, such as control boundaries.
    output_times : npt.NDArray[np.float64]
        Sorted requested public output times.

    Returns
    -------
    npt.NDArray[np.float64]
        Strictly increasing evolution grid containing every checkpoint and
        each output time that is not numerically coincident with a checkpoint.

    Notes
    -----
    A requested output within `1e-12` ns of a checkpoint reuses the exact
    checkpoint. States are continuous at a Hamiltonian discontinuity, and this
    avoids introducing a numerically tiny integration interval while never
    shifting a control boundary.
    """
    additional_outputs = np.array(
        [
            time
            for time in output_times
            if not np.any(np.isclose(checkpoint_times, time, rtol=0.0, atol=1e-12))
        ],
        dtype=np.float64,
    )
    return np.sort(np.concatenate((checkpoint_times, additional_outputs)))


def find_time_indices(
    times: npt.NDArray[np.float64],
    selected_times: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """
    Locate selected times in a sorted time grid.

    Parameters
    ----------
    times : npt.NDArray[np.float64]
        Sorted source time grid.
    selected_times : npt.NDArray[np.float64]
        Sorted times to locate exactly or within `1e-12` ns.

    Returns
    -------
    npt.NDArray[np.int64]
        Integer indices of `selected_times` in `times`.

    Raises
    ------
    ValueError
        If no source time is within `1e-12` ns of a selected time.
    """
    right_indices = np.searchsorted(times, selected_times)
    right_indices = np.minimum(right_indices, len(times) - 1)
    left_indices = np.maximum(right_indices - 1, 0)
    use_left = np.abs(times[left_indices] - selected_times) < np.abs(
        times[right_indices] - selected_times
    )
    indices = np.where(use_left, left_indices, right_indices)
    if not np.allclose(
        times[indices],
        selected_times,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Selected output times are absent from the solver grid.")
    return indices.astype(np.int64, copy=False)


def _merge_times(
    time_arrays: list[npt.NDArray[np.float64]],
    duration: float,
) -> npt.NDArray[np.float64]:
    """
    Merge time arrays while coalescing nearly equal boundaries.

    Parameters
    ----------
    time_arrays : list[npt.NDArray[np.float64]]
        Nonempty one-dimensional arrays of times in ns.
    duration : float
        Common terminal time in ns.

    Returns
    -------
    npt.NDArray[np.float64]
        Sorted times with adjacent candidates within `1e-12` ns coalesced.

    Notes
    -----
    The later candidate in each near-equal group is retained so a rounded
    uniform-grid point cannot precede the corresponding control boundary. The
    first and last values are set exactly to zero and `duration`.
    """
    candidates = np.sort(np.concatenate(time_arrays))

    # Keep the later representative so a rounded uniform point cannot precede
    # a nearly equal control boundary.
    keep = np.append(np.diff(candidates) > 1e-12, True)
    times = candidates[keep]
    times[0] = 0.0
    times[-1] = duration
    return times


def _create_uniform_times(
    duration: float,
    max_step: float,
) -> npt.NDArray[np.float64]:
    """
    Create uniform times from zero through a finite duration.

    Parameters
    ----------
    duration : float
        Terminal time in ns.
    max_step : float
        Uniform interval in ns.

    Returns
    -------
    npt.NDArray[np.float64]
        Times whose final element is exactly `duration`.
    """
    times = np.arange(0, duration, max_step)

    # Handle potential floating point overshoot from arange.
    if len(times) > 0 and times[-1] > duration:
        times = times[:-1]

    if len(times) == 0 or not np.isclose(times[-1], duration, atol=1e-12):
        times = np.append(times, duration)
    else:
        times[-1] = duration

    return times
