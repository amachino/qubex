"""SQUAD pulse shape helpers."""

from __future__ import annotations

from numbers import Real
from typing import Final, Literal, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import NotRequired, override

from qxpulse.pulse import Pulse

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
    - "tukey" : normalized integral of a Tukey window with independent
                rise-end and fall-start positions within each ramp.
    - "beta" : smooth adiabatic ramp using the regularized incomplete
               beta function I_u(α, β) with fixed shape parameters.

    Parameters
    ----------
    duration : float
        Total duration of the pulse in ns.
    amplitude : float
        Flat-top in-phase envelope level, not the carrier frequency. Use the
        same units as `delta`; there is no automatic unit conversion.
    delta : float
        Signed design detuning, transition frequency minus drive frequency,
        expressed in the same units as `amplitude`. Shapes the envelope and
        CD quadrature; does not set or shift the carrier. See Notes for units.
    tau : float
        Rise and fall time (each side) in ns.
    factor : float, optional
        Signed CD coefficient; 0 disables CD and None defaults to 1.0.
        Its magnitude depends on the amplitude units. Its sign is opposite
        to `FlatTop.correction_factor` for equivalent quadrature (see Notes).
    window : str or WindowConfig, optional
        Window type or dictionary with a required `type` key. Default is
        "hann". String choices are "none", "hann", "tukey", and "beta".
        Tukey dictionaries accept `rise_end` and `fall_start`, both defaulting
        to 0.5, with `0 <= rise_end <= fall_start <= 1`. These normalized
        positions end the window's rise and begin its fall. Both at 0.5
        reproduce Hann; (0, 1) reproduces "none". Beta dictionaries accept
        `mode` (default 1/3, range [0, 1]) and `sum` (default 5, finite and > 2).
        Other windows accept only `type`. Unknown keys are rejected.
        Dictionaries are copied on construction. Standalone `beta_mode` and
        `beta_sum` arguments have been removed; use the dictionary keys.

    Raises
    ------
    TypeError
        If a dictionary's numeric settings are not real numbers, or removed
        standalone window parameters are supplied.
    ValueError
        If a window dictionary has missing/unknown keys or invalid values.

    Notes
    -----
    flat-top period = duration - 2 * tau

    Before any generic Pulse scale/phase/detuning transforms, the envelope
    is `I + i*Q` with `Q = factor * delta * dI/dt / (delta**2 + I**2)`.
    `FlatTop(type="Squad", correction_type="CD")` uses the opposite sign:
    set `factor = -correction_factor` to match it on the same sampling grid.

    Time is in ns. If K is the angular Rabi rate in rad/ns per numeric
    amplitude unit, the analytic CD coefficient magnitude is `1/K`.
    For angular-rate inputs (rad/ns), use magnitude 1. For cyclic-rate
    inputs (GHz = cycles/ns), use `1/(2*pi)`. For command amplitudes with
    calibrated Rabi conversion r in GHz per command unit, pass
    `delta = (f_transition - f_drive)/r` and use magnitude `1/(2*pi*r)`.
    For example, amplitude 0.99 in command units must not be combined with
    an unconverted GHz detuning. Qubit Rabi conversion is a model input,
    not a guarantee of strong-drive hardware response. qxsimulator expects
    angular-rate envelopes even though Control.frequency uses cyclic GHz.

    The carrier is configured separately. For carrier-adaptive pulse design,
    recompute `delta` from each drive frequency and regenerate the waveform.
    Holding a design delta fixed instead scans the carrier of a fixed
    waveform; these are different experiments. In particular, adapting delta
    changes the waveform throughout a chevron and its fit interpretation.

    Window positions refer to normalized time `u=t/tau` inside the rising
    SQUAD ramp, not to the full pulse. The falling SQUAD ramp is its time
    reverse, including the window. Thus an asymmetric Tukey window does not
    change the equal outer ramp durations `tau` or the pulse's flat top.
    The window is integrated and normalized before the SQUAD mapping; it
    does not multiply the final I/Q envelope.

    Examples
    --------
    >>> pulse = Squad(
    ...     duration=40, amplitude=0.6, delta=0.8, tau=12,
    ...     window={"type": "tukey", "rise_end": 0.2, "fall_start": 0.7},
    ... )
    """

    def __init__(
        self,
        *,
        duration: float,
        amplitude: float,
        delta: float,
        tau: float,
        factor: float | None = None,
        window: WindowSpec | None = None,
        **kwargs,
    ):
        _reject_removed_window_options(kwargs)
        _resolve_window(window)
        super().__init__(
            duration=duration,
            **kwargs,
        )

        self.amplitude: Final = amplitude
        self.delta: Final = delta
        self.tau: Final = tau
        self.factor: Final = factor
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
            factor=self.factor,
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
        factor: float | None = None,
        window: WindowSpec | None = None,
    ) -> NDArray:
        """
        Compute the full complex SQUAD pulse.

        Returns `I(t) + i Q(t)`, where `I(t)` is the flat-top envelope with
        chosen SQUAD ramps and `Q(t)` is the scaled counter-diabatic quadrature.

        A window dictionary such as
        `{"type": "tukey", "rise_end": 0.2, "fall_start": 0.7}` sets the
        normalized positions within each ramp. Omitted positions default to
        0.5 (Hann). See `Squad` for validation and time-reversal conventions.

        `t`, `duration`, and `tau` are in ns. `amplitude` and signed `delta`
        (transition minus drive) must share units. Delta does not shift the
        carrier. `Q = factor * delta * dI/dt / (delta**2 + I**2)`; see `Squad`
        for the unit-dependent factor and its sign relative to `FlatTop`.
        Beta settings use `window={"type": "beta", "mode": 0.4, "sum": 6}`;
        standalone beta arguments are not accepted.
        """
        _resolve_window(window)
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
