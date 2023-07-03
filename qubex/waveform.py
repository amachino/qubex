"""
A module for representing a waveform.
"""
import numpy as np

SAMPLING_TIME: int = 2  # ns


class Waveform:
    """
    A class to represent a waveform.

    Attributes
    ----------
    iq : np.ndarray[complex]
        A NumPy array of complex numbers representing the waveform.
    time : np.ndarray
        Time array of the waveform in ns.

    Properties
    ----------
    duration : int
        Duration of the waveform in ns.
    real : np.ndarray
        Real part of the waveform.
    imag : np.ndarray
        Imaginary part of the waveform.
    ampl : np.ndarray
        Amplitude of the waveform, calculated as the absolute value.
    phase : np.ndarray
        Phase of the waveform, calculated as the angle.
    """

    @classmethod
    def concat(cls, *waveforms):
        """
        Concatenates the given waveforms into a single waveform.

        Parameters
        ----------
        waveforms : Waveform
            The waveforms to concatenate.

        Returns
        -------
        Waveform
            The concatenated waveform.
        """
        waveform = np.concatenate(waveforms)
        return cls(waveform)

    def __init__(self, iq):
        if isinstance(iq, np.ndarray):
            self.iq = iq
        elif isinstance(iq, list):
            self.iq = np.array(iq)
        else:
            raise TypeError("waveform must be a NumPy array or a list.")
        self.time = np.arange(len(self.iq)) * SAMPLING_TIME

    def set_time(self, time: np.ndarray):
        """Sets the time array of the waveform."""
        if len(time) != len(self.iq):
            raise ValueError("time and iq arrays must have the same length.")
        self.time = time

    @property
    def duration(self) -> int:
        """Returns the duration of the waveform in ns."""
        return len(self.iq) * SAMPLING_TIME

    @property
    def real(self) -> np.ndarray:
        """Returns the real part of the waveform."""
        return np.real(self.iq)

    @property
    def imag(self) -> np.ndarray:
        """Returns the imaginary part of the waveform."""
        return np.imag(self.iq)

    @property
    def ampl(self) -> np.ndarray:
        """Calculates and returns the amplitude of the waveform."""
        return np.abs(self.iq)

    @property
    def phase(self) -> np.ndarray:
        """Calculates and returns the phase of the waveform."""
        return np.angle(self.iq)
