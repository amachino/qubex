from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RabiParam:
    """
    Represents the parameters for a Rabi oscillation experiment.

    ```
    rabi_1d = amplitude * cos(2π * frequency * time + phase) + offset)
    rabi_2d = (distance + 1j * rabi_1d) * exp(1j * angle) ± noise
    ```

    Attributes
    ----------
    target : str
        Identifier of the target.
    amplitude : float
        Amplitude of the Rabi oscillation.
    frequency : float
        Frequency of the Rabi oscillation.
    phase : float
        Phase of the Rabi oscillation in radians.
    offset : float
        Offset of the Rabi oscillation.
    noise : float
        Noise level of the Rabi oscillation.
    angle : float
        Angle of the Rabi oscillation in radians.
    distance : float
        Distance of the Rabi oscillation in the complex plane.
    r2 : float
        Coefficient of determination.
    reference_phase : float
        Reference phase for the Rabi oscillation, used for normalization.
    """

    target: str
    amplitude: float
    frequency: float
    phase: float
    offset: float
    noise: float
    angle: float
    distance: float
    r2: float
    reference_phase: float

    @classmethod
    def nan(cls, target: str) -> RabiParam:
        return cls(
            target=target,
            amplitude=np.nan,
            frequency=np.nan,
            phase=np.nan,
            offset=np.nan,
            noise=np.nan,
            angle=np.nan,
            distance=np.nan,
            r2=np.nan,
            reference_phase=np.nan,
        )

    @property
    def endpoints(self) -> tuple[complex, complex]:
        rotated_0 = complex(self.distance, self.offset + self.amplitude)
        rotated_1 = complex(self.distance, self.offset - self.amplitude)
        iq_0 = complex(rotated_0 * np.exp(1j * self.angle))
        iq_1 = complex(rotated_1 * np.exp(1j * self.angle))
        return iq_0, iq_1

    def correct(self, new_reference_phase: float):
        """
        Corrects the angle according to a new reference phase.

        Parameters
        ----------
        new_reference_phase : float
            The new reference phase in radians.
        """
        if self.reference_phase is None:
            self.reference_phase = new_reference_phase
            return
        self.angle += new_reference_phase - self.reference_phase
        self.reference_phase = new_reference_phase

    def normalize(
        self,
        values: NDArray,
    ) -> NDArray:
        """
        Normalizes the measured I/Q values.

        Parameters
        ----------
        values : NDArray
            Measured I/Q values.

        Returns
        -------
        NDArray
            Normalized I/Q values.
        """
        rotated = values * np.exp(-1j * self.angle)
        normalized = (np.imag(rotated) - self.offset) / self.amplitude
        return normalized
