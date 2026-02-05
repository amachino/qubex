from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import comb

from ..pulse import Pulse


class Sintegral(Pulse):
    """
    A class representing a sine integral pulse.

    Parameters
    ----------
    duration : float
        Duration of the pulse in ns.
    amplitude : float
        Amplitude of the pulse.
    beta : float, optional
        DRAG correction coefficient. Default is None.
    power : int
        Power of the sine integral function.

    Examples
    --------
    >>> pulse = Sintegral(
    ...     duration=100,
    ...     amplitude=1.0,
    ...     power=2,
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        power: int = 2,
        beta: float | None = None,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.power: Final = power
        self.beta: Final = beta

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
                beta=beta,
                power=power,
            )

        super().__init__(values, **kwargs)

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        power: int = 2,
        beta: float | None = None,
    ) -> NDArray:
        """
        Evaluate the sine integral function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the pulse in ns.
        amplitude : float
            Amplitude of the pulse.
        power : int
            Power of the sine integral function.
        beta : float, optional
            DRAG correction coefficient. Default is None.
        """
        if duration == 0:
            raise ValueError("Duration cannot be zero.")
        t = np.asarray(t)

        Omega = sin_pow_integral(
            2 * np.pi * t / duration,
            n=power,
        )
        Omega -= sin_pow_integral(0, n=power)
        scale = amplitude / (
            sin_pow_integral(np.pi, n=power) - sin_pow_integral(0, n=power)
        )
        Omega *= scale
        if beta is None:
            values = Omega
        else:
            dOmega = sin_pow_derivative(
                2 * np.pi * t / duration,
                n=power,
                m=0,
            )
            dOmega *= scale * 2 * np.pi / duration
            values = Omega + 1j * beta * dOmega

        is_odd = power % 2 == 1
        return np.where(
            (t >= 0) & (t <= duration),
            np.where(
                (
                    t
                    <= (duration * 0.5) // Pulse.SAMPLING_PERIOD * Pulse.SAMPLING_PERIOD
                ),
                values,
                values if is_odd else 2 * amplitude - values,
            ),
            0,
        ).astype(np.complex128)


def sin_pow_integral(
    x: ArrayLike,
    n: int,
) -> NDArray:
    x = np.asarray(x, dtype=float)

    if n < 0 or not isinstance(n, int):
        raise ValueError("n must be a non-negative integer")

    if n % 2 == 1:
        # n is odd: use explicit sum formula
        m = n // 2
        result = np.zeros_like(x)
        for k in range(m + 1):
            coeff = (-1) ** k * comb(m, k) / (2 * k + 1)
            result -= coeff * np.cos(x) ** (2 * k + 1)
        return result
    else:
        # n is even: reduce via recurrence
        def recursive_even_integral(x, m):
            if m == 0:
                return x
            term1 = (-(np.sin(x) ** (m - 1)) * np.cos(x)) / m
            return term1 + ((m - 1) / m) * recursive_even_integral(x, m - 2)

        return recursive_even_integral(x, n)


def sin_pow_derivative(
    x: ArrayLike,
    n: int,
    m: int,
) -> NDArray:
    if n < 0 or not isinstance(n, int):
        raise ValueError("n must be a non-negative integer")
    if m < 0 or not isinstance(m, int):
        raise ValueError("m must be a non-negative integer")

    x = np.asarray(x, dtype=float)

    # initial: f(x) = sin^n x
    terms = [(1.0, n, 0)]  # (coefficient, sin_pow, cos_pow)

    for _ in range(m):
        new_terms = []
        for coeff, s_pow, c_pow in terms:
            if s_pow > 0:
                new_terms.append(
                    (coeff * s_pow, s_pow - 1, c_pow + 1)
                )  # d/dx[sin^n] = n sin^{n-1} cos
            if c_pow > 0:
                new_terms.append(
                    (-coeff * c_pow, s_pow + 1, c_pow - 1)
                )  # d/dx[cos^n] = -n sin cos^{n-1}
        terms = new_terms

    result = np.zeros_like(x)
    for coeff, s_pow, c_pow in terms:
        result += coeff * (np.sin(x) ** s_pow) * (np.cos(x) ** c_pow)
    return result


class MultiDerivativeSintegral(Pulse):
    """
    A class representing a sine integral pulse.

    Parameters
    ----------
    duration : float
        Duration of the pulse in ns.
    amplitude : float
        Amplitude of the pulse.
    betas : float, optional
        multi-Derivative pulse correction coefficients. Default is None.
    power : int
        Power of the sine integral function.

    Examples
    --------
    >>> pulse = Sintegral(
    ...     duration=100,
    ...     amplitude=1.0,
    ...     power=2,
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        betas: dict[int, float] | None = None,
        power: int = 2,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.power: Final = power
        self.betas: Final = betas

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            values = self.func(
                t=self._sampling_points(duration),
                duration=duration,
                amplitude=amplitude,
                power=power,
                betas=betas,
            )

        super().__init__(values, **kwargs)

    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        betas: dict[int, float] | None = None,
        power: int,
    ) -> NDArray:
        """
        Evaluate the sine integral function.

        Parameters
        ----------
        t : ArrayLike
            Time points at which to evaluate the pulse.
        duration : float
            Duration of the pulse in ns.
        amplitude : float
            Amplitude of the pulse.
        betas : dict[int, float], optional
            multi-Derivative pulse correction coefficients. Default is None.
        power : int
            Power of the sine integral function.

        """
        if duration == 0:
            raise ValueError("Duration cannot be zero.")

        if betas is None:
            betas = {}

        t = np.asarray(t)

        Omega = sin_pow_integral(
            2 * np.pi * t / duration,
            n=power,
        )
        Omega -= sin_pow_integral(0, n=power)
        scale = amplitude / (
            sin_pow_integral(np.pi, n=power) - sin_pow_integral(0, n=power)
        )
        Omega *= scale

        pulse_terms = {}
        for order, beta in betas.items():
            dOmega = sin_pow_derivative(
                2 * np.pi * t / duration,
                n=power,
                m=order - 1,
            )
            dOmega *= scale * (2 * np.pi / duration) ** order
            pulse_terms[order] = beta * dOmega

        real_part = Omega
        imag_part = np.zeros_like(Omega)
        for pdx, pulse_term in pulse_terms.items():
            if pdx % 2 == 0:
                real_part += pulse_term
            else:
                imag_part += pulse_term

        values = real_part + 1j * imag_part

        is_odd = power % 2 == 1
        return np.where(
            (t >= 0) & (t <= duration),
            np.where(
                (
                    t
                    <= (duration * 0.5) // Pulse.SAMPLING_PERIOD * Pulse.SAMPLING_PERIOD
                ),
                values,
                values if is_odd else 2 * amplitude - values,
            ),
            0,
        ).astype(np.complex128)
