"""Pulse base class and helpers."""

# ruff: noqa: SLF001
from __future__ import annotations

from copy import deepcopy
from typing import Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, deprecated

from .waveform import Waveform


class Pulse(Waveform):
    """
    Pulse base class backed by sampled complex values.

    `Pulse` stores complex I/Q samples and applies waveform modifiers
    (`scale`, `detuning`, and `phase`) when `values` is accessed.

    Parameters
    ----------
    values : ArrayLike | None, optional
        Complex I/Q samples. If `None`, samples are materialized lazily from
        `_sample_values()`.
    duration : float | None, optional
        Pulse duration in ns. Required to lazily sample non-empty pulses.
    scale : float, optional
        Multiplicative amplitude scale. Defaults to `1.0`.
    detuning : float, optional
        Frequency detuning in GHz. Defaults to `0.0`.
    phase : float, optional
        Phase offset in radians. Defaults to `0.0`.
    sampling_period : float, optional
        Sampling period in ns. Defaults to `Waveform.SAMPLING_PERIOD`.
    lazy : bool, optional
        If `True`, defer sampling until `values` is accessed. Subclasses can
        call `_finalize_initialization()` to materialize during initialization
        when this flag is `False`. Defaults to `True`.
    """

    def __init__(
        self,
        values: npt.ArrayLike | None = None,
        *,
        duration: float | None = None,
        scale: float | None = None,
        detuning: float | None = None,
        phase: float | None = None,
        sampling_period: float | None = None,
        lazy: bool = True,
        **kwargs,
    ):
        super().__init__(
            scale=scale,
            detuning=detuning,
            phase=phase,
            sampling_period=sampling_period,
            **kwargs,
        )
        self._initialize_sampling(
            lazy=lazy,
            values=values,
            duration=duration,
        )

    @staticmethod
    def _validate_init_arguments(
        *,
        values: npt.ArrayLike | None,
        duration: float | None,
    ) -> None:
        """Validate constructor arguments for sampled initialization."""
        if values is not None and duration is not None:
            raise ValueError("Specify either values or duration, not both.")

    def _initialize_sampling(
        self,
        *,
        lazy: bool,
        values: npt.ArrayLike | None,
        duration: float | None,
    ) -> None:
        """Initialize sampling-related state and optional sampled values cache."""
        self._validate_init_arguments(values=values, duration=duration)
        self._length = 0
        self._values: npt.NDArray[np.complex128] | None = None
        self._lazy = lazy
        if duration is not None:
            self._length = self._number_of_samples(duration)
        if values is not None:
            sampled_values = np.asarray(values, dtype=np.complex128)
            self._values = sampled_values
            self._length = len(sampled_values)

    def _set_sampled_values(self, values: npt.ArrayLike | None) -> None:
        """Set sampled values cache and synchronize pulse length."""
        if values is None:
            self._values = None
            return
        array = np.asarray(values, dtype=np.complex128)
        self._values = array
        self._length = len(array)

    def _sample_values(self) -> npt.ArrayLike:
        """Return sampled complex values for this pulse."""
        return np.zeros(self._length, dtype=np.complex128)

    def _materialize_values(self) -> npt.NDArray[np.complex128]:
        """Return sampled values, materializing and caching as needed."""
        values = self._values
        if values is None:
            sampled = np.asarray(self._sample_values(), dtype=np.complex128)
            if len(sampled) != self._length:
                raise ValueError(
                    f"Pulse sampler returned {len(sampled)} samples, expected {self._length}."
                )
            self._set_sampled_values(sampled)
            values = sampled
        return values

    def _finalize_initialization(self) -> None:
        """Materialize sampled values at init when lazy mode is disabled."""
        if self._values is None and not self._lazy:
            self._materialize_values()

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"{self.name}({self.length})"

    @property
    def length(self) -> int:
        """Return the length of the pulse in samples."""
        return self._length

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """Return the I/Q values of the pulse."""
        values = self._materialize_values()
        return (
            values
            * self._scale
            * np.exp(-1j * (2 * np.pi * self._detuning * self.times - self._phase))
        )

    def copy(self, reset_cached_duration: bool = False) -> Self:
        """Return a copy of the pulse."""
        pulse = deepcopy(self)
        if reset_cached_duration:
            pulse.reset_cached_duration()
        return pulse

    def padded(
        self,
        total_duration: float,
        pad_side: Literal["right", "left"] = "right",
    ) -> Self:
        """
        Return a copy of the pulse with zero padding.

        Parameters
        ----------
        total_duration : float
            Total duration of the pulse in ns.
        pad_side : {"right", "left"}, optional
            Side of the zero padding.
        """
        N = self._number_of_samples(total_duration)
        values = self._materialize_values()
        if pad_side == "right":
            values = np.pad(values, (0, N - self.length), mode="constant")
        elif pad_side == "left":
            values = np.pad(values, (N - self.length, 0), mode="constant")
        else:
            raise ValueError("pad_side must be either 'right' or 'left'.")
        new_pulse = self.copy(reset_cached_duration=True)
        new_pulse._set_sampled_values(values)
        return new_pulse

    def scaled(self, scale: float) -> Self:
        """Return a copy of the pulse scaled by the given factor."""
        if scale == 1:
            return self
        new_pulse = self.copy()
        new_pulse._scale *= scale
        return new_pulse

    def detuned(self, detuning: float) -> Self:
        """Return a copy of the pulse detuned by the given frequency."""
        if detuning == 0:
            return self
        new_pulse = self.copy()
        new_pulse._detuning += detuning
        return new_pulse

    def shifted(self, phase: float) -> Self:
        """Return a copy of the pulse shifted by the given phase."""
        if phase == 0:
            return self
        new_pulse = self.copy()
        new_pulse._phase += phase
        return new_pulse

    def repeated(self, n: int) -> Self:
        """Return a copy of the pulse repeated n times."""
        if n == 1:
            return self
        new_pulse = self.copy(reset_cached_duration=True)
        new_pulse._set_sampled_values(np.tile(self._materialize_values(), n))
        return new_pulse

    @deprecated(
        "The `reversed` method is deprecated, use `inverted` instead.",
    )
    def reversed(self) -> Self:
        """Return a copy of the pulse with the time inverted."""
        return self.inverted()

    def inverted(self) -> Self:
        """Return a copy of the pulse with the time inverted."""
        new_pulse = self.copy()
        new_pulse._set_sampled_values(np.flip(-1 * self._materialize_values()))
        return new_pulse
