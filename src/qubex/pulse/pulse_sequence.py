from __future__ import annotations

from copy import deepcopy
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from .pulse import Blank, Pulse
from .waveform import Waveform


class PhaseShift:
    def __init__(self, theta: float):
        self.theta = theta


class VirtualZ(PhaseShift):
    def __init__(self, theta: float):
        super().__init__(-theta)


class PulseSequence(Waveform):
    """
    A class to represent a pulse sequence.

    Parameters
    ----------
    sequences : Sequence[Waveform | PhaseShift]
        A list of Waveform or PhaseShift objects.
    scale : float, optional
        Scaling factor of the pulse sequence.
    detuning : float, optional
        Detuning of the pulse sequence in GHz.
    phase_shift : float, optional
        Phase shift of the pulse sequence in rad.

    Examples
    --------
    >>> seq = PulseSequence([
    ...     Rect(duration=10, amplitude=1.0),
    ...     Blank(duration=10),
    ...     PhaseShift(theta=np.pi/2),
    ...     Gauss(duration=10, amplitude=1.0, sigma=2.0),
    ... ])
    """

    def __init__(
        self,
        sequence: Sequence = [],
        *,
        scale: float = 1.0,
        detuning: float = 0.0,
        phase_shift: float = 0.0,
    ):
        super().__init__(
            scale=scale,
            detuning=detuning,
            phase_shift=phase_shift,
        )
        self._sequence = list(sequence)

    @property
    def waveforms(self) -> Sequence[Waveform]:
        """Returns the list of waveforms in the pulse sequence."""
        waveforms = []
        current_phase = 0.0
        for obj in self._sequence:
            if isinstance(obj, PulseSequence):
                waveforms.append(obj.shifted(current_phase))
                current_phase += obj.total_virtual_phase
            elif isinstance(obj, Pulse):
                waveforms.append(obj.shifted(current_phase))
            elif isinstance(obj, PhaseShift):
                current_phase += obj.theta
        return waveforms

    @property
    def length(self) -> int:
        """Returns the total length of the pulse sequence in samples."""
        return sum([w.length for w in self.waveforms])

    @property
    def values(self) -> npt.NDArray[np.complex128]:
        """Returns the concatenated values of the pulse sequence."""
        if len(self.waveforms) == 0:
            return np.array([])

        concat_values = np.concatenate([w.values for w in self.waveforms])
        values = (
            concat_values
            * self._scale
            * np.exp(1j * (2 * np.pi * self._detuning * self.times + self._phase_shift))
        )
        return values

    @property
    def virtual_phases(self) -> npt.NDArray[np.float64]:
        """Returns the virtual phases of the pulse sequence."""
        phases = []
        current_phase = 0.0
        for obj in self._sequence:
            if isinstance(obj, PhaseShift):
                current_phase += obj.theta
            elif isinstance(obj, Waveform):
                phases += [current_phase] * obj.length
        return np.array(phases)

    @property
    def total_virtual_phase(self) -> float:
        """Returns the total virtual phase of the pulse sequence."""
        phase_shift = 0.0
        for obj in self._sequence:
            if isinstance(obj, PhaseShift):
                phase_shift += obj.theta
        return phase_shift

    def add(self, obj: Waveform | PhaseShift) -> None:
        """Adds the given waveform or phase shift to the pulse sequence."""
        self._sequence.append(obj)

    def pad(
        self,
        total_duration: float,
        pad_side: Literal["right", "left"] = "right",
    ) -> None:
        """
        Adds zero padding to the pulse sequence.

        Parameters
        ----------
        total_duration : float
            Total duration of the pulse sequence in ns.
        pad_side : {"right", "left"}, optional
            Side of the zero padding.
        """
        duration = total_duration - self.duration
        if duration < 0:
            raise ValueError(
                f"Total duration ({total_duration}) must be greater than the current duration ({self.duration})."
            )
        blank = Blank(duration)
        if pad_side == "right":
            self._sequence.append(blank)
        elif pad_side == "left":
            self._sequence.insert(0, blank)
        else:
            raise ValueError("pad_side must be either 'right' or 'left'.")

    def copy(self) -> PulseSequence:
        """Returns a copy of the pulse sequence."""
        return deepcopy(self)

    def padded(
        self,
        total_duration: float,
        pad_side: Literal["right", "left"] = "right",
    ) -> PulseSequence:
        """
        Returns a copy of the pulse sequence with zero padding.

        Parameters
        ----------
        total_duration : float
            Total duration of the pulse sequence in ns.
        pad_side : {"right", "left"}, optional
            Side of the zero padding.
        """
        duration = total_duration - self.duration
        if duration < 0:
            raise ValueError(
                f"Total duration ({total_duration}) must be greater than the current duration ({self.duration})."
            )
        new_sequence = deepcopy(self)
        blank = Blank(duration=duration)
        if pad_side == "right":
            new_waveforms = new_sequence._sequence + [blank]
        elif pad_side == "left":
            new_waveforms = [blank] + new_sequence._sequence
        else:
            raise ValueError("pad_side must be either 'right' or 'left'.")
        new_sequence._sequence = new_waveforms  # type: ignore
        return new_sequence

    def scaled(self, scale: float) -> PulseSequence:
        """Returns a copy of the pulse sequence scaled by the given factor."""
        new_sequence = deepcopy(self)
        new_sequence._scale *= scale
        return new_sequence

    def detuned(self, detuning: float) -> PulseSequence:
        """Returns a copy of the pulse sequence detuned by the given frequency."""
        new_sequence = deepcopy(self)
        new_sequence._detuning += detuning
        return new_sequence

    def shifted(self, phase: float) -> PulseSequence:
        """Returns a copy of the pulse sequence shifted by the given phase."""
        new_sequence = deepcopy(self)
        new_sequence._phase_shift += phase
        return new_sequence

    def repeated(self, n: int) -> PulseSequence:
        """Returns a copy of the pulse sequence repeated n times."""
        new_sequence = deepcopy(self)
        new_sequence._sequence = list(new_sequence._sequence) * n
        return new_sequence

    def reversed(self) -> PulseSequence:
        """Returns a copy of the pulse sequence with the order of the waveforms reversed."""
        new_sequence = deepcopy(self)
        new_sequence._sequence = list(reversed(new_sequence.waveforms))
        return new_sequence

    def added(self, obj: Waveform | PhaseShift) -> PulseSequence:
        """Returns a copy of the pulse sequence with the given waveform or phase shift added."""
        new_sequence = deepcopy(self)
        new_sequence._sequence.append(obj)
        return new_sequence

    def __repr__(self) -> str:
        pulses = ", ".join([pulse.__class__.__name__ for pulse in self._sequence])
        return f"PulseSequence([{pulses})]"
