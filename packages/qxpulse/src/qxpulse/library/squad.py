"""SQUAD pulse shape helpers."""

from __future__ import annotations

from numbers import Real
from typing import Final, Literal, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import NotRequired, override

from qxpulse.pulse import Pulse

from ._corrections import _cd_quadrature, _reject_legacy_factor

SmoothingType = Literal["none", "hann", "tukey", "beta"]


class SimpleWindowConfig(TypedDict):
    """Parameter-free SQUAD ramp window configuration."""

    type: Literal["none", "hann"]


class TukeyWindowConfig(TypedDict):
    """Tukey window positions in normalized ramp time, each defaulting to 0.5."""

    type: Literal["tukey"]
    rise_end: NotRequired[float]
    fall_start: NotRequired[float]


class BetaWindowConfig(TypedDict):
    """Beta window mode and parameter sum, defaulting to 1/3 and 5 respectively."""

    type: Literal["beta"]
    mode: NotRequired[float]
    sum: NotRequired[float]


WindowConfig: TypeAlias = SimpleWindowConfig | TukeyWindowConfig | BetaWindowConfig
WindowSpec: TypeAlias = SmoothingType | WindowConfig


def _reject_removed_window_options(options: dict) -> None:
    """Reject removed standalone window parameters before lazy sampling."""
    removed = {
        "beta_mode",
        "beta_sum",
        "tukey_rise_end",
        "tukey_fall_start",
    }.intersection(options)
    if removed:
        raise TypeError(
            f"Unexpected SQUAD options {sorted(removed)}; use a window dictionary."
        )


def _real_window_parameter(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"window {name} must be a real number.")
    return float(value)


def _resolve_window(
    window: object,
) -> tuple[SmoothingType, float, float, float, float]:
    """Resolve a window without modifying the caller's configuration."""
    beta_mode = 1.0 / 3.0
    beta_sum = 5.0
    if window is None:
        return "hann", beta_mode, beta_sum, 0.5, 0.5
    if isinstance(window, str):
        if window not in ("none", "hann", "tukey", "beta"):
            raise ValueError(f"Invalid window type: {window}")
        return cast(SmoothingType, window), beta_mode, beta_sum, 0.5, 0.5
    if not isinstance(window, dict):
        raise TypeError("window must be a string or dictionary.")

    kind = window.get("type")
    if not isinstance(kind, str) or kind not in ("none", "hann", "tukey", "beta"):
        raise ValueError("window dictionary requires a valid 'type'.")
    allowed = {"type"}
    if kind == "tukey":
        allowed.update(("rise_end", "fall_start"))
    elif kind == "beta":
        allowed.update(("mode", "sum"))
    if any(key not in allowed for key in window):
        raise ValueError(f"Unexpected window keys for {kind}: {set(window) - allowed}")
    rise_end = fall_start = 0.5
    if kind == "tukey":
        rise_end = _real_window_parameter(window.get("rise_end", 0.5), "rise_end")
        fall_start = _real_window_parameter(window.get("fall_start", 0.5), "fall_start")
        if not (0.0 <= rise_end <= fall_start <= 1.0):
            raise ValueError(
                "window positions must satisfy 0 <= rise_end <= fall_start <= 1."
            )
    elif kind == "beta":
        beta_mode = _real_window_parameter(window.get("mode", 1.0 / 3.0), "mode")
        beta_sum = _real_window_parameter(window.get("sum", 5.0), "sum")
        if not (0.0 <= beta_mode <= 1.0 and np.isfinite(beta_sum) and beta_sum > 2.0):
            raise ValueError(
                "window beta mode must be in [0, 1] and sum must be finite and > 2."
            )

    return cast(SmoothingType, kind), beta_mode, beta_sum, rise_end, fall_start


def _integrated_tukey(u: NDArray, *, rise_end: float, fall_start: float) -> NDArray:
    """Return the normalized Tukey integral for normalized times in [0, 1]."""
    # The unit-height window has area a/2 + (b-a) + (1-b)/2.
    area = (1.0 + fall_start - rise_end) / 2.0
    integral = u - rise_end / 2.0

    if rise_end > 0.0:
        rising = u < rise_end
        x = u[rising]
        integral[rising] = (x - rise_end / np.pi * np.sin(np.pi * x / rise_end)) / 2.0

    if fall_start < 1.0:
        falling = u > fall_start
        x = 1.0 - u[falling]
        width = 1.0 - fall_start
        integral[falling] = area - (x - width / np.pi * np.sin(np.pi * x / width)) / 2.0

    return integral / area


class Squad(Pulse):
    """
    SQUAD flat-top envelope with counterdiabatic (CD) quadrature.

    Parameters
    ----------
    duration : float
        Complete pulse duration in ns, including both ramps.
    amplitude : float
        Flat-top in-phase amplitude in rad/ns, before Pulse scaling.
    delta : float
        Signed design detuning in rad/ns: transition frequency minus drive
        frequency. Must be nonzero for a meaningful SQUAD ramp. This shapes
        the envelope; it does not set the carrier frequency.
    tau : float
        Duration of each ramp in ns; the flat top lasts duration - 2*tau.
    correction_factor : float, optional
        Dimensionless CD strength. None defaults to 1.0; 0 disables CD and
        0.5 halves it. Negative values reverse the correction.
    window : str or WindowConfig, optional
        Adiabaticity window, default "hann". String choices are "none",
        "hann", "tukey", and "beta". A dictionary requires a `type` key.
        Tukey accepts `rise_end` and `fall_start` (default 0.5 each),
        with `0 <= rise_end <= fall_start <= 1`. Beta accepts `mode`
        (default 1/3, range [0, 1]) and `sum` (default 5, finite and > 2).
        Other windows accept only `type`. Dictionaries are copied.

    Raises
    ------
    TypeError
        If removed standalone options are supplied or window settings have
        non-real numeric values.
    ValueError
        If a window has missing/unknown keys or invalid values.

    Notes
    -----
    With time in ns and the envelope `I + i*Q`, both SQUAD APIs use
    `Q = -correction_factor * delta * dI/dt / (delta**2 + I**2)`.
    The derivative is evaluated on the supplied sampling grid. This follows
    the drive-frame convention `H/hbar = (-delta*Z + I*X + Q*Y)/2`.
    `FlatTop(type="Squad", correction_type="CD")` uses this same convention;
    the coefficient never needs an API-dependent sign change.

    For command amplitude A and Rabi scale r in GHz/command, set
    `K = 2*pi*r`, pass `amplitude=K*A` and
    `delta=2*pi*(f_transition-f_drive)`, then use `scale=1/K` to convert
    the completed I/Q back to command units. Keep `correction_factor`
    dimensionless; no Rabi calibration or unit conversion is inferred here.

    The window is integrated and normalized before the SQUAD mapping.
    Positions describe normalized time inside each ramp, not outer pulse
    durations. The falling ramp is the time reverse of the rising ramp.
    Tukey (0.5, 0.5) reproduces Hann and (0, 1) reproduces "none".

    Configure the carrier separately. Recompute delta when intentionally
    redesigning the waveform for each carrier, or hold design delta fixed
    when scanning one fixed waveform. These are different experiments.

    Examples
    --------
    >>> pulse = Squad(
    ...     duration=40, amplitude=0.6, delta=-0.8, tau=12,
    ...     correction_factor=1.0,
    ...     window={"type": "beta", "mode": 0.4, "sum": 6},
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        delta: float,
        tau: float,
        correction_factor: float | None = None,
        window: WindowSpec | None = None,
        **kwargs,
    ):
        _reject_legacy_factor(kwargs)
        _reject_removed_window_options(kwargs)
        _resolve_window(window)
        super().__init__(
            duration=duration,
            **kwargs,
        )

        self.amplitude: Final = amplitude
        self.delta: Final = delta
        self.tau: Final = tau
        self.correction_factor: Final = correction_factor
        self.window: Final = window.copy() if isinstance(window, dict) else window
        self._finalize_initialization()

    @override
    def _sample_values(self) -> NDArray[np.complex128]:
        """Return sampled values for the SQUAD pulse."""
        if self.length == 0:
            return np.array([], dtype=np.complex128)
        duration = self.duration
        return Squad.func(
            t=self._sampling_points(duration),
            duration=duration,
            amplitude=self.amplitude,
            delta=self.delta,
            tau=self.tau,
            correction_factor=self.correction_factor,
            window=self.window,
        )

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
        window: WindowSpec | None = None,
    ) -> NDArray:
        """
        Rising (or falling, if t is time-reversed) SQUAD ramp.

        Implements different "g(u)" mappings of the adiabatic angle:
            sin θ(t) = sin θ_max * g(u),   u = t / tau ∈ [0,1],
        for each window type.
        """
        window, beta_mode, beta_sum, rise_end, fall_start = _resolve_window(window)

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

        elif window == "tukey":
            g_u = _integrated_tukey(u_sel, rise_end=rise_end, fall_start=fall_start)

        elif window == "beta":
            # Beta-shaped smooth ramp:
            # use regularized incomplete beta I_u(α,β) as g(u)
            from scipy.special import betainc  # lazy import

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
        window: WindowSpec | None = None,
    ) -> NDArray:
        """Compute the flat-top constant-adiabaticity pulse envelope (I component only)."""
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
        correction_factor: float | None = None,
        window: WindowSpec | None = None,
    ) -> NDArray:
        """
        Sample the SQUAD envelope as I + i*Q, before generic Pulse transforms.

        `t`, `duration`, and `tau` are in ns. `amplitude` and signed
        `delta` (transition minus drive) are in rad/ns. `correction_factor`
        is dimensionless: None means 1, 0 disables CD, and 0.5 halves it.
        See `Squad` for the shared CD formula, windows and hardware scaling.
        """
        _resolve_window(window)
        t = np.asarray(t, dtype=float)

        if duration <= 0:
            return np.zeros_like(t, dtype=np.complex128)

        if correction_factor is None:
            correction_factor = 1.0

        # In-phase component
        I = Squad._squad_flat_top_envelope(
            t,
            duration=duration,
            amplitude=amplitude,
            delta=delta,
            tau=tau,
            window=window,
        )

        if correction_factor == 0:
            return I.astype(np.complex128)

        # Numerical derivative dI/dt
        dI_dt = np.gradient(I.real, t)  # I is real-valued envelope here

        # Counter-diabatic quadrature
        Δ = delta
        denom = Δ**2 + I.real**2
        # Avoid division by zero if Δ=0 and I=0 everywhere
        Q = np.zeros_like(I.real)
        nonzero = denom != 0.0
        Q[nonzero] = _cd_quadrature(
            I.real[nonzero],
            dI_dt[nonzero],
            delta=Δ,
            correction_factor=correction_factor,
        )

        return I.real + 1j * Q
