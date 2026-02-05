from __future__ import annotations

from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..pulse import Pulse
from .bump import Bump
from .gaussian import Gaussian
from .multi_derivative import MultiDerivative
from .raised_cosine import RaisedCosine
from .ramp_type import RampType
from .sintegral import Sintegral
from .squad import Squad


class FlatTop(Pulse):
    """
    A class to represent a raised cosine flat-top pulse.

    Parameters
    ----------
    duration : float
        Duration of the pulse in ns.
    amplitude : float
        Amplitude of the pulse.
    tau : float
        Rise and fall time of the pulse in ns.
    beta : float, optional
        DRAG correction coefficient. Default is None.

    Examples
    --------
    >>> pulse = FlatTop(
    ...     duration=100,
    ...     amplitude=1.0,
    ...     tau=10,
    ... )

    Notes
    -----
    flat-top period = duration - 2 * tau
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        tau: float,
        beta: float | None = None,
        delta: float | None = None,
        type: RampType | None = None,
        correction_type: Literal["DRAG", "CD"] | None = None,
        correction_factor: float | None = None,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.tau: Final = tau
        self.beta: Final = beta
        self.delta: Final = delta
        self.type: Final = type
        self.correction_type: Final = correction_type
        self.correction_factor: Final = correction_factor

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
                tau=tau,
                beta=beta,
                delta=delta,
                type=type,
                correction_type=correction_type,
                correction_factor=correction_factor,
                **kwargs,
            )

        super().__init__(values, **kwargs)

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        tau: float,
        beta: float | None = None,
        delta: float | None = None,
        type: RampType | None = None,
        correction_type: Literal["DRAG", "CD"] | None = None,
        correction_factor: float | None = None,
        **kwargs,
    ) -> NDArray:
        """
        Flat-top pulse function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the pulse in ns.
        amplitude : float
            Amplitude of the pulse.
        tau : float
            Rise and fall time of the pulse in ns.
        beta : float, optional
            DRAG correction coefficient. Default is None.
        type : RampType | None, optional
            Type of the pulse. Default is "RaisedCosine".

        Returns
        -------
        NDArray
            Flat-top pulse values.
        """
        if type is None:
            type = "RaisedCosine"

        t = np.asarray(t)
        T = 2 * tau
        flattime = duration - T

        if flattime < 0:
            raise ValueError("duration must be greater than `2 * tau`.")

        I = np.zeros_like(t, dtype=np.complex128)

        if duration <= 0:
            return I

        mask_up = (t >= 0.0) & (t < tau)
        if np.any(mask_up):
            I[mask_up] = _ramp_func(
                t=t[mask_up],
                duration=T,
                amplitude=amplitude,
                delta=delta,
                type=type,
                **kwargs,
            )

        mask_flat = (t >= tau) & (t <= duration - tau)
        if np.any(mask_flat):
            I[mask_flat] = amplitude

        mask_down = (t > duration - tau) & (t <= duration)
        if np.any(mask_down):
            u = duration - t[mask_down]
            I[mask_down] = _ramp_func(
                t=u,
                duration=T,
                amplitude=amplitude,
                delta=delta,
                type=type,
                **kwargs,
            )

        if beta is not None:
            # print(
            #     "Warning: beta parameter is deprecated. Use correction_type and correction_factor instead."
            # )
            if correction_type is None and correction_factor is None:
                dI = np.gradient(I, t)
                Q = beta * dI
                return I + 1j * Q

        if correction_type is None:
            return I

        if delta is None:
            raise ValueError("delta must be provided for pulse correction.")

        if correction_factor is None:
            correction_factor = 1.0

        dI = np.gradient(I, t)
        Q = np.zeros_like(t, dtype=np.complex128)
        if correction_type == "DRAG":
            Q = -(correction_factor / delta) * dI
        elif correction_type == "CD":
            Q = -(correction_factor * delta) / (delta**2 + I**2) * dI
        else:
            raise ValueError(f"Unknown correction type: {correction_type}")
        return I + 1j * Q


def _ramp_func(
    t: ArrayLike,
    *,
    duration: float,
    amplitude: float,
    delta: float | None = None,
    type: RampType | None = None,
    **kwargs,
) -> NDArray:
    if type is None:
        type = "RaisedCosine"

    t = np.asarray(t, dtype=np.float64)

    if duration <= 0:
        return np.zeros_like(t, dtype=np.complex128)

    if type == "Gaussian":
        return Gaussian.func(
            t=t,
            duration=duration,
            amplitude=amplitude,
            sigma=duration / 4,
            zero_bounds=True,
        )
    elif type == "RaisedCosine":
        return RaisedCosine.func(
            t=t,
            duration=duration,
            amplitude=amplitude,
        )
    elif type == "Sintegral":
        return Sintegral.func(
            t=t,
            duration=duration,
            amplitude=amplitude,
            power=2,
        )
    elif type == "Bump":
        return Bump.func(
            t=t,
            duration=duration,
            amplitude=amplitude,
        )
    elif type == "Squad":
        assert delta is not None, "delta must be provided for Squad ramp."
        return Squad.func(
            t=t,
            duration=duration,
            amplitude=amplitude,
            tau=(duration * 0.5) // Pulse.SAMPLING_PERIOD * Pulse.SAMPLING_PERIOD,
            delta=delta,
            factor=0,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown ramp type: {type}")


class MultiDerivativeFlatTop(Pulse):
    """
    A class to represent a raised cosine flat-top pulse.

    Parameters
    ----------
    duration : float
        Duration of the pulse in ns.
    amplitude : float
        Amplitude of the pulse.
    tau : float
        Rise and fall time of the pulse in ns.
    betas : dict[int, float], optional
        multi-Derivative pulse correction coefficients. Default is None.

    Examples
    --------
    >>> pulse = FlatTop(
    ...     duration=100,
    ...     amplitude=1.0,
    ...     tau=10,
    ... )

    Notes
    -----
    flat-top period = duration - 2 * tau
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        tau: float,
        betas: dict[int, float] | None = None,
        power: int = 2,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.tau: Final = tau
        self.betas: Final = betas
        self.power: Final = power

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
                tau=tau,
                betas=betas,
                power=power,
            )

        super().__init__(values, **kwargs)

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        tau: float,
        betas: dict[int, float] | None = None,
        power: int = 2,
    ) -> NDArray:
        """
            Flat-top pulse function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the pulse in ns.
        amplitude : float
            Amplitude of the pulse.
        tau : float
            Rise and fall time of the pulse in ns.
        betas : dict[int, float], optional
            multi-Derivative pulse correction coefficients. Default is None.
        power : int
            Power of the sine integral function.

        Returns
        -------
        NDArray
            Flat-top pulse values.
        """
        t = np.asarray(t)
        T = 2 * tau
        flattime = duration - T

        if flattime < 0:
            raise ValueError("duration must be greater than `2 * tau`.")

        v_rise = MultiDerivative.func(
            t=t,
            duration=T,
            amplitude=amplitude,
            betas=betas,
            power=power,
        )
        v_flat = amplitude * np.ones_like(t)
        v_fall = MultiDerivative.func(
            t=t - flattime,
            duration=T,
            amplitude=amplitude,
            betas=betas,
            power=power,
        )
        return np.where(
            (t >= 0) & (t <= duration),
            np.where(
                (t >= tau) & (t <= duration - tau),
                v_flat,
                np.where(
                    (t < tau),
                    v_rise,
                    v_fall,
                ),
            ),
            0,
        ).astype(np.complex128)
