"""
A module for representing a pulse.
"""
import numpy as np

SAMPLING_TIME: int = 2  # ns


class Pulse:
    """
    A class to represent a pulse.

    Attributes
    ----------
    waveform : np.ndarray[complex]
        A NumPy array of complex numbers representing the pulse.
    time : np.ndarray
        Time array of the pulse in ns.

    Properties
    ----------
    duration : int
        Duration of the pulse in ns.
    real : np.ndarray
        Real part of the pulse.
    imag : np.ndarray
        Imaginary part of the pulse.
    ampl : np.ndarray
        Amplitude of the pulse, calculated as the absolute value.
    phase : np.ndarray
        Phase of the pulse, calculated as the angle.
    """

    @classmethod
    def concat(cls, *pulses):
        """
        Concatenates the given pulses into a single pulse.

        Parameters
        ----------
        pulses : Pulse
            The pulses to concatenate.

        Returns
        -------
        Pulse
            The concatenated pulse.
        """
        waveform = np.concatenate([pulse.waveform for pulse in pulses])
        return cls(waveform)

    def __init__(self, waveform):
        if isinstance(waveform, np.ndarray):
            self.waveform = waveform
        elif isinstance(waveform, list):
            self.waveform = np.array(waveform)
        else:
            raise TypeError("waveform must be a NumPy array or a list.")
        self.time = np.arange(len(self.waveform)) * SAMPLING_TIME

    def set_time(self, time: np.ndarray):
        """Sets the time array of the pulse."""
        if len(time) != len(self.waveform):
            raise ValueError("time and waveform must have the same length.")
        self.time = time

    @property
    def duration(self) -> int:
        """Returns the duration of the pulse in ns."""
        return len(self.waveform) * SAMPLING_TIME

    @property
    def real(self) -> np.ndarray:
        """Returns the real part of the pulse."""
        return np.real(self.waveform)

    @property
    def imag(self) -> np.ndarray:
        """Returns the imaginary part of the pulse."""
        return np.imag(self.waveform)

    @property
    def ampl(self) -> np.ndarray:
        """Calculates and returns the amplitude of the pulse."""
        return np.abs(self.waveform)

    @property
    def phase(self) -> np.ndarray:
        """Calculates and returns the phase of the pulse."""
        return np.angle(self.waveform)


class Rcft(Pulse):
    """
    A subclass of the Pulse class to represent a Raised Cosine Flat Top (RCFT) pulse.

    Attributes
    ----------
    waveform : np.ndarray
        Complex-valued waveform of the pulse.
    """

    def __init__(self, ampl=1.0, rise=10, flat=10, fall=10):
        """
        Initializes a new RCFT pulse with given parameters.

        Parameters
        ----------
        ampl : float, optional
            Peak amplitude of the pulse. Defaults to 1.0.
        rise : int, optional
            Duration of the pulse's rise from zero to peak. Defaults to 10.
        flat : int, optional
            Duration of the pulse's peak. Defaults to 10.
        fall : int, optional
            Duration of the pulse's fall from peak to zero. Defaults to 10.
        """
        t_rise = np.arange(0, rise, SAMPLING_TIME)
        t_flat = np.arange(0, flat, SAMPLING_TIME)
        t_fall = np.arange(0, fall, SAMPLING_TIME)

        rise_waveform = 0.5 * ampl * (1 - np.cos(np.pi * t_rise / rise))
        flat_waveform = ampl * np.ones_like(t_flat)
        fall_waveform = 0.5 * ampl * (1 + np.cos(np.pi * t_fall / fall))

        waveform = np.concatenate((rise_waveform, flat_waveform, fall_waveform))

        super().__init__(waveform)


# class Drag(Pulse):
#     """
#     A subclass of the Pulse class to represent a Derivative Removal by Adiabatic Gate (DRAG) pulse.

#     Attributes
#     ----------
#     waveform : np.ndarray
#         Complex-valued waveform of the pulse.
#     """

#     def __init__(self, duration=30, ampl=1.0, beta=1.0):
#         """
#         Initializes a new DRAG pulse with given parameters.

#         Parameters
#         ----------
#         duration : int
#             Duration of the pulse in ns.
#         ampl : float
#             Peak amplitude of the pulse.
#         beta : float
#             DRAG coefficient that characterizes the amplitude of the correction term.
#         """
#         t = np.arange(0, duration, SAMPLING_TIME)
#         sigma = duration * 0.5
#         envelope = ampl * (np.exp(-0.5 * (t / sigma) ** 2) - np.exp(0.5))
#         correction = beta * (-t / (sigma**2)) * np.exp(-0.5 * (t / sigma) ** 2)

#         waveform = envelope + 1j * correction

#         super().__init__(waveform)


class Drag(Pulse):
    """
    A subclass of the Pulse class to represent a Derivative Removal by Adiabatic Gate (DRAG) pulse.

    Attributes
    ----------
    waveform : np.ndarray
        Complex-valued waveform of the pulse.
    """

    def __init__(self, duration=30, ampl=1.0, beta=1.0):
        """
        Initializes a new DRAG pulse with given parameters.

        Parameters
        ----------
        duration : int
            Duration of the pulse in ns.
        ampl : float
            Peak amplitude of the pulse.
        beta : float
            DRAG coefficient that characterizes the amplitude of the correction term.
        """
        t = np.arange(0, duration, SAMPLING_TIME)
        envelope = (1.0 - np.cos(2 * np.pi * t / duration)) / 2
        correction = beta * np.sin(2 * np.pi * t / duration) / 2

        waveform = ampl * (envelope + 1j * correction)

        super().__init__(waveform)


class QctrlPi(Pulse):
    """
    A subclass of the Pulse class to represent an optimized pi pulse.
    """

    def __init__(self):
        waveform = np.array(
            [
                0.03481618 + 5.94384947e-03j,
                0.03265834 + 9.58042570e-03j,
                0.02718015 + 4.56709459e-03j,
                0.03008459 + 1.07180817e-02j,
                -0.00023517 + 1.06311655e-04j,
                0.02231457 + 1.06699124e-03j,
                0.02500387 + 9.39997498e-03j,
                -0.00029007 - 1.53053753e-04j,
                0.03373711 - 3.31980694e-03j,
                0.02424721 + 1.32870650e-04j,
                0.00661651 + 5.76160314e-03j,
                0.01289346 + 3.03768648e-03j,
                0.02347185 + 5.50176897e-03j,
                0.03545285 + 2.31146867e-02j,
                0.02416318 - 6.78417157e-04j,
                0.03165002 + 2.80328007e-03j,
                0.00455699 + 1.62375270e-02j,
                0.02475384 - 1.86465684e-03j,
                0.00508424 - 4.86456888e-03j,
                0.01028951 - 7.75430524e-03j,
                0.01918622 - 1.04645872e-02j,
                0.0099093 - 4.08165474e-03j,
                0.03882687 + 2.27914374e-02j,
                0.0283639 + 2.94183055e-02j,
                0.0152782 + 6.45479207e-03j,
                -0.01709611 - 8.82758026e-03j,
                0.04212434 + 1.40447913e-02j,
                0.04487165 - 3.15664494e-03j,
                -0.00468664 + 6.60658952e-03j,
                0.0131243 + 5.73832368e-05j,
            ]
        )
        super().__init__(waveform)
