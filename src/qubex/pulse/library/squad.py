from __future__ import annotations

from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc

from ..pulse import Pulse

SmoothingType = Literal["none", "hann", "beta"]


class Squad(Pulse):
    """
    Smooth quasi-adiabatic (SQUAD) pulse.

    The pulse consists of:
    - ramp-up on [0, tau]
    - flat top on [tau, duration - tau]
    - ramp-down on [duration - tau, duration]

    Window types
    ------------
    - "none" : constant-adiabatic ramp (FAQUAD-like, ε(t) = const)
    - "hann" : smooth adiabatic ramp using a sin^2(pi u) window
               (integrated to g(u) = u - sin(2πu)/(2π)).
    - "beta" : smooth adiabatic ramp using the regularized incomplete
               beta function I_u(α, β) with fixed shape parameters.

    Parameters
    ----------
    duration : float
        Total duration of the pulse in ns.
    amplitude : float
        Flat-top amplitude of the pulse.
    delta : float
        Detuning parameter for Counter-Diabatic term in GHz.
    tau : float
        Rise and fall time (each side) in ns.
    factor : float, optional
        Strength of the quadrature (Q) component. If 0, no CD term.
        If None, defaults to 1.0.
    window : {"none", "hann", "beta"}, optional
        Window type for the SQUAD ramp. Default is "hann".
    beta_mode : float, optional
        Mode of the beta distribution for window="beta". Default is 1/3.
    beta_sum : float, optional
        Sum of alpha and beta parameters for window="beta". Default is 5.0.

    Notes
    -----
    flat-top period = duration - 2 * tau
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        delta: float,
        tau: float,
        factor: float | None = None,
        window: SmoothingType | None = None,
        beta_mode: float = 1.0 / 3.0,
        beta_sum: float = 5.0,
        **kwargs,
    ):
        self.amplitude: Final = amplitude
        self.delta: Final = delta
        self.tau: Final = tau
        self.factor: Final = factor
        self.window: Final = window
        self.beta_mode: Final = beta_mode
        self.beta_sum: Final = beta_sum

        if duration == 0:
            values = np.array([], dtype=np.complex128)
        else:
            t = self._sampling_points(duration)
            values = self.func(
                t=t,
                duration=duration,
                amplitude=amplitude,
                delta=delta,
                tau=tau,
                factor=factor,
                window=window,
                beta_mode=beta_mode,
                beta_sum=beta_sum,
            )

        super().__init__(values, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _squad_ramp(
        t: NDArray,
        *,
        tau: float,
        amplitude: float,
        delta: float,
        window: SmoothingType | None = None,
        beta_mode: float = 1.0 / 3.0,
        beta_sum: float = 5.0,
    ) -> NDArray:
        """
        Rising (or falling, if t is time-reversed) SQUAD ramp.

        Implements different "g(u)" mappings of the adiabatic angle:
            sin θ(t) = sin θ_max * g(u),   u = t / tau ∈ [0,1],
        for each window type.
        """
        if window is None:
            window = "hann"

        t = np.asarray(t, dtype=float)
        values = np.zeros_like(t, dtype=float)

        if tau <= 0:
            return values

        Ω_max: float = amplitude
        Δ = delta

        # If detuning is zero, we cannot define the FAQUAD-like mapping.
        # In that case, just return zeros to avoid division by zero.
        if Δ == 0.0:
            return values

        # Normalized time u
        u = t / tau
        mask = (u >= 0.0) & (u <= 1.0)
        if not np.any(mask):
            return values

        u_sel = u[mask]

        # Common scale: target final angle
        θ_max = np.arctan(Ω_max / Δ)

        if window == "none":
            # Constant-adiabatic ramp: g(u) = u
            g_u = u_sel

        elif window == "hann":
            # Smooth ε(t) ∝ sin^2(πu) → g(u) = ∫ ε ∝ u - sin(2πu)/(2π)
            g_u = u_sel - np.sin(2.0 * np.pi * u_sel) / (2.0 * np.pi)

        elif window == "beta":
            # Beta-shaped smooth ramp:
            # use regularized incomplete beta I_u(α,β) as g(u)
            alpha = beta_mode * (beta_sum - 2.0) + 1.0
            beta_param = beta_sum - alpha
            g_u = betainc(alpha, beta_param, u_sel)

        else:
            raise ValueError(f"Invalid window type: {window}")

        # Map g(u) to sin θ(t), then back to Ω(t)
        s_t = np.sin(θ_max) * g_u
        # Avoid numerical overflow when s_t → ±1
        s_t = np.clip(s_t, -0.999999999, 0.999999999)
        Ω_t = Δ * s_t / np.sqrt(1.0 - s_t**2)

        values[mask] = Ω_t
        return values

    @staticmethod
    def _squad_flat_top_envelope(
        t: NDArray,
        *,
        duration: float,
        amplitude: float,
        delta: float,
        tau: float,
        window: SmoothingType | None = None,
        beta_mode: float = 1.0 / 3.0,
        beta_sum: float = 5.0,
    ) -> NDArray:
        """
        Flat-top constant-adiabaticity pulse envelope (I component only).
        """
        t = np.asarray(t, dtype=float)
        values = np.zeros_like(t, dtype=np.complex128)

        if duration <= 0:
            return values

        flattime = duration - 2.0 * tau
        if flattime < 0.0:
            raise ValueError("duration must be greater than `2 * tau`.")

        # Regions:
        #  - ramp-up:   0 <= t < tau
        #  - flat:      tau <= t <= duration - tau
        #  - ramp-down: duration - tau < t <= duration

        # Rising ramp
        mask_up = (t >= 0.0) & (t < tau)
        if np.any(mask_up):
            values[mask_up] = Squad._squad_ramp(
                t[mask_up],
                tau=tau,
                amplitude=amplitude,
                delta=delta,
                window=window,
                beta_mode=beta_mode,
                beta_sum=beta_sum,
            )

        # Flat-top
        mask_flat = (t >= tau) & (t <= duration - tau)
        if np.any(mask_flat):
            values[mask_flat] = amplitude

        # Falling ramp: time-reversed ramp
        mask_down = (t > duration - tau) & (t <= duration)
        if np.any(mask_down):
            u = duration - t[mask_down]
            values[mask_down] = Squad._squad_ramp(
                u,
                tau=tau,
                amplitude=amplitude,
                delta=delta,
                window=window,
                beta_mode=beta_mode,
                beta_sum=beta_sum,
            )

        return values

    # ------------------------------------------------------------------
    # Public pulse function
    # ------------------------------------------------------------------
    @staticmethod
    def func(
        t: ArrayLike,
        *,
        duration: float,
        amplitude: float,
        tau: float,
        delta: float,
        factor: float | None = None,
        window: SmoothingType | None = None,
        beta_mode: float = 1.0 / 3.0,
        beta_sum: float = 5.0,
    ) -> NDArray:
        """
        Full complex SQUAD pulse:
            I(t) + i Q(t),

        where I(t) is the flat-top envelope with chosen SQUAD ramps,
        and Q(t) is the (scaled) counter-diabatic quadrature.
        """
        t = np.asarray(t, dtype=float)

        if duration <= 0:
            return np.zeros_like(t, dtype=np.complex128)

        if factor is None:
            factor = 1.0

        # In-phase component
        I = Squad._squad_flat_top_envelope(
            t,
            duration=duration,
            amplitude=amplitude,
            delta=delta,
            tau=tau,
            window=window,
            beta_mode=beta_mode,
            beta_sum=beta_sum,
        )

        if factor == 0:
            return I.astype(np.complex128)

        # Numerical derivative dI/dt
        dI_dt = np.gradient(I.real, t)  # I is real-valued envelope here

        # Counter-diabatic quadrature
        Δ = delta
        denom = Δ**2 + I.real**2
        # Avoid division by zero if Δ=0 and I=0 everywhere
        Q = np.zeros_like(I.real)
        nonzero = denom != 0.0
        Q[nonzero] = (factor * Δ * dI_dt[nonzero]) / denom[nonzero]

        return I.real + 1j * Q
