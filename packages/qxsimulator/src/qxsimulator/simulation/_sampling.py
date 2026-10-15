"""Sampling helpers for quantum simulations."""

from __future__ import annotations

from typing import TypeVar, overload

import numpy as np
import numpy.typing as npt

T = TypeVar("T")


@overload
def downsample(
    data: npt.NDArray,
    n_samples: int | None,
) -> npt.NDArray: ...


@overload
def downsample(
    data: list[T],
    n_samples: int | None,
) -> list[T]: ...


def downsample(
    data: npt.NDArray | list[T],
    n_samples: int | None,
) -> npt.NDArray | list[T]:
    """
    Downsample a trajectory to at most the requested number of samples.

    Parameters
    ----------
    data : npt.NDArray | list[T]
        Array sampled along its first axis, or a list of samples.
    n_samples : int | None
        Non-negative maximum number of samples. If `None`, retain all samples.

    Returns
    -------
    npt.NDArray | list[T]
        Downsampled data with the input container type preserved.

    Raises
    ------
    ValueError
        If `n_samples` is negative and downsampling is required.

    Notes
    -----
    If no downsampling is needed, the original object is returned. Otherwise,
    indices are spaced uniformly over the sample index range. Requests for at
    least two samples retain both endpoints; a request for one sample retains
    only the first point, and a request for zero returns an empty container.
    """
    if n_samples is None:
        return data
    if len(data) <= n_samples:
        return data
    indices = np.linspace(0, len(data) - 1, n_samples).astype(int)
    if isinstance(data, np.ndarray):
        return data[indices]
    return [data[index] for index in indices]
