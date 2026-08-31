"""
Pure JAZZ schedules, signed-quadrature analysis, and model-derived exchange.

All frequencies are cyclic GHz and times are ns. Define energy ZZ as
`(E11-E10-E01+E00)/h`, so the computational Hamiltonian contains
`+zz_energy_ghz*ZZ/4` in `H/h`. With the emitted X/Y phases below and
`T=2*tau`, `-z_X + 1j*z_Y` winds at `zz_energy_ghz/2-rotation_frequency_ghz`.
The legacy service's nonnegative cosine-frequency `xi`/`zeta` aliases are not
this signed energy difference and are deliberately not accepted here.

No function connects, measures, writes files, updates parameters, or changes
classifiers. Acquisition notebooks must retain per-shot IQ and supply fresh
production DRAG pulses. A fitted decay/ZZ describes the stated model; thermal
spectator mixtures, relaxation, pulse errors and extra modes can violate it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qxpulse import Blank, PulseSchedule, Waveform
from scipy.optimize import least_squares
from scipy.stats import chi2


def _number(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _half_waits(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("tau_ns must be finite and nonnegative")
    if not np.allclose(result / 2, np.rint(result / 2), atol=1e-9, rtol=0):
        raise ValueError("tau_ns must use the native 2 ns grid")
    return result


def build_jazz_schedule(
    target_qubit: str,
    spectator_qubit: str,
    tau_ns: float,
    *,
    x90: Mapping[str, Waveform],
    xpi: Mapping[str, Waveform],
    analysis_axis: Literal["X", "Y"] = "Y",
    rotation_frequency_ghz: float = 0.0002,
) -> PulseSchedule:
    """
    Build one JAZZ schedule from supplied, freshly calibrated DRAG pulses.

    Parameters
    ----------
    target_qubit, spectator_qubit : str
        Distinct measured and echoed spectator labels.
    tau_ns : float
        Each free-evolution half, nonnegative on the 2 ns grid. The fitted
        free-evolution time is `2*tau_ns`, not the complete schedule duration.
    x90, xpi : Mapping[str, Waveform]
        Production DRAG X90 and Xpi waveforms. Echo Xpi durations must match.
    analysis_axis : {"X", "Y"}, optional
        Legacy JAZZ phase label. X emits phase `pi-2*pi*f_rotation*2*tau`;
        Y emits `-pi/2-2*pi*f_rotation*2*tau`. Both are measured in Z.
    rotation_frequency_ghz : float, optional
        Finite programmed phase-reference rate, default 0.0002 GHz.

    Returns
    -------
    PulseSchedule
        Preparation, two waits, simultaneous echoes and final analysis.

    Raises
    ------
    ValueError
        Invalid labels, timing, analysis axis, sampling or command headroom.

    Notes
    -----
    Pulse phases multiply I+iQ by exp(+i*phase). Ideal rotations follow
    exp[-i*angle*(cos(phase)*X+sin(phase)*Y)/2]. Finite pulse evolution can
    affect the fitted intercept/model; it is not counted as free time.
    This function neither acquires data nor silently substitutes non-DRAG gates.
    """
    if not target_qubit or not spectator_qubit or target_qubit == spectator_qubit:
        raise ValueError("JAZZ requires two distinct qubit labels")
    tau = float(_half_waits(tau_ns))
    reference = _number(rotation_frequency_ghz, "rotation_frequency_ghz")
    if analysis_axis not in ("X", "Y"):
        raise ValueError("analysis_axis must be X or Y")
    first, echo_target, echo_spectator = (
        x90[target_qubit],
        xpi[target_qubit],
        xpi[spectator_qubit],
    )
    for pulse in (first, echo_target, echo_spectator):
        values = np.asarray(pulse.values)
        if pulse.sampling_period != 2.0 or pulse.duration <= 0:
            raise ValueError("Production DRAG pulses must use the native 2 ns grid")
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.0 + 1e-12:
            raise ValueError("Production DRAG waveform exceeds command headroom")
    if echo_target.duration != echo_spectator.duration:
        raise ValueError("JAZZ echo durations must be equal")
    phase = (
        np.pi if analysis_axis == "X" else -np.pi / 2
    ) - 2 * np.pi * reference * 2 * tau
    with PulseSchedule([target_qubit, spectator_qubit]) as schedule:
        schedule.add(target_qubit, first)
        schedule.add(target_qubit, Blank(tau, sampling_period=2))
        schedule.barrier()
        schedule.add(target_qubit, echo_target)
        schedule.add(spectator_qubit, echo_spectator)
        schedule.barrier()
        schedule.add(target_qubit, Blank(tau, sampling_period=2))
        schedule.add(target_qubit, first.shifted(float(phase)))
        schedule.barrier()
    return schedule


def _fit_input(
    times: ArrayLike, x: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    arrays = tuple(np.asarray(value, dtype=float) for value in (times, x, y))
    time, zx, zy = arrays
    if (
        time.ndim != 1
        or len(time) < 8
        or zx.shape != time.shape
        or zy.shape != time.shape
    ):
        raise ValueError(
            "Require at least eight equally shaped one-dimensional time/X/Y points"
        )
    if not all(np.isfinite(value).all() for value in arrays) or np.any(time < 0):
        raise ValueError(
            "Time and quadratures must be finite; time must be nonnegative"
        )
    steps = np.diff(time)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], atol=1e-9, rtol=1e-9):
        raise ValueError(
            "Signed JAZZ fit requires a strictly increasing uniform time grid"
        )
    return time, zx, zy


def fit_jazz_quadratures(
    total_free_time_ns: ArrayLike,
    z_x: ArrayLike,
    z_y: ArrayLike,
    *,
    rotation_frequency_ghz: float,
    standard_error_x: ArrayLike | None = None,
    standard_error_y: ArrayLike | None = None,
    frequency_bounds_ghz: tuple[float, float] | None = None,
    min_r2: float = 0.9,
    min_contrast: float = 0.05,
    max_reduced_chi2: float = 5.0,
) -> dict[str, Any]:
    """
    Fit signed JAZZ quadratures with a decaying complex oscillation.

    Parameters
    ----------
    total_free_time_ns : ArrayLike
        Uniform, increasing free times `T=2*tau`, with at least eight points.
    z_x, z_y : ArrayLike
        Target Z expectation values from the two explicit analysis phases.
    rotation_frequency_ghz : float
        Programmed phase-reference rate used when building the schedules.
    standard_error_x, standard_error_y : ArrayLike, optional
        Positive marginal standard errors. Supply both or neither.
    frequency_bounds_ghz : tuple[float, float], optional
        Signed beat-search interval strictly inside the grid's Nyquist band.
        The default searches both signs across that band.
    min_r2, min_contrast : float, optional
        Diagnostic qualification thresholds, default 0.9 and 0.05. They are
        not gate-fidelity criteria and never alter the fitted observations.
    max_reduced_chi2 : float, optional
        Weighted single-mode lack-of-fit diagnostic limit, default 5. Applied
        only when both quadrature standard errors are supplied. The nominal
        Gaussian p-value is approximate for estimated binomial errors.

    Returns
    -------
    dict[str, Any]
        Signed beat, `zz_energy_ghz=2*(beat+rotation)`, uncertainties, fitted
        quadratures, residuals, all observations, search starts and reasons.
        An unidentified trace has `qualified=False`; a flat trace has no ZZ.

    Raises
    ------
    ValueError
        Nonfinite/mismatched input, nonuniform grid or invalid Nyquist bounds.

    Notes
    -----
    Fit `C=-z_x+i*z_y=b+a*exp(-gamma*T)*exp(i*2*pi*f_beat*T)` without taking
    an absolute frequency. Finite pulse errors, warm spectators and relaxation
    can require a richer model. Statistical errors do not cover those effects.
    No legacy xi/zeta alias is converted or accepted. No input is modified.
    """
    time, zx, zy = _fit_input(total_free_time_ns, z_x, z_y)
    reference = _number(rotation_frequency_ghz, "rotation_frequency_ghz")
    r2_limit = _number(min_r2, "min_r2")
    contrast_limit = _number(min_contrast, "min_contrast")
    chi2_limit = _number(max_reduced_chi2, "max_reduced_chi2")
    if not 0 <= r2_limit <= 1 or contrast_limit < 0 or chi2_limit <= 0:
        raise ValueError("Invalid fit diagnostic thresholds")
    if (standard_error_x is None) != (standard_error_y is None):
        raise ValueError("Supply both quadrature standard errors or neither")
    weighted = standard_error_x is not None
    sx = (
        np.ones_like(time)
        if standard_error_x is None
        else np.broadcast_to(np.asarray(standard_error_x, dtype=float), time.shape)
    )
    sy = (
        np.ones_like(time)
        if standard_error_y is None
        else np.broadcast_to(np.asarray(standard_error_y, dtype=float), time.shape)
    )
    if (
        not np.isfinite(sx).all()
        or not np.isfinite(sy).all()
        or np.any(sx <= 0)
        or np.any(sy <= 0)
    ):
        raise ValueError("Standard errors must be positive and finite")
    nyquist = 1 / (2 * (time[1] - time[0]))
    bounds_ghz = (
        (-nyquist * (1 - 1e-8), nyquist * (1 - 1e-8))
        if frequency_bounds_ghz is None
        else frequency_bounds_ghz
    )
    low, high = (_number(value, "frequency bound") for value in bounds_ghz)
    if not -nyquist < low < high < nyquist:
        raise ValueError(
            "Frequency bounds must lie strictly inside the signed Nyquist band"
        )
    observed = -zx + 1j * zy
    result: dict[str, Any] = {
        "qualified": False,
        "status": "unqualified",
        "reasons": [],
        "total_free_time_ns": time.tolist(),
        "raw_z_x": zx.tolist(),
        "raw_z_y": zy.tolist(),
        "rotation_frequency_ghz": reference,
        "nyquist_frequency_ghz": float(nyquist),
        "frequency_bounds_ghz": [low, high],
        "signed_beat_frequency_ghz": None,
        "zz_energy_ghz": None,
        "zz_energy_standard_error_ghz": None,
        "convention": "C=-z_X+i*z_Y; f_beat=zz_energy/2-f_rotation; H/h contains +zz_energy*ZZ/4",
        "scope": "signed single-mode JAZZ fit; no microscopic or gate-fidelity inference",
        "weighted_fit": weighted,
        "max_reduced_chi2": chi2_limit,
    }
    if float(np.sum(np.abs(observed - observed.mean()) ** 2)) <= 1e-20:
        result["reasons"] = ["No resolved quadrature variation"]
        return result
    # Use microseconds/MHz internally for a well-conditioned numerical fit.
    t = (time - time[0]) * 1e-3
    step = float(t[1] - t[0])
    span = float(t[-1])
    frequencies = np.fft.fftfreq(max(512, 8 * len(t)), step)
    power = np.abs(np.fft.fft(observed - observed.mean(), len(frequencies)))
    candidates: list[float] = []
    for index in np.argsort(power)[::-1]:
        frequency = float(frequencies[index])
        if low * 1000 < frequency < high * 1000 and all(
            abs(frequency - old) >= 0.5 / span for old in candidates
        ):
            candidates.append(frequency)
        if len(candidates) == 4:
            break
    if not candidates:
        candidates = [(low + high) * 500]
    mirrored = -candidates[0]
    if low * 1000 < mirrored < high * 1000:
        candidates.append(mirrored)
    if low < 0 < high:
        candidates.append(0.0)

    def model(parameters: NDArray[np.float64]) -> NDArray[np.complex128]:
        offset = parameters[0] + 1j * parameters[1]
        amplitude = parameters[2] + 1j * parameters[3]
        return offset + amplitude * np.exp(
            (-parameters[5] + 2j * np.pi * parameters[4]) * t
        )

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        difference = model(parameters) - observed
        return np.concatenate((difference.real / sx, difference.imag / sy))

    lower = [-np.inf] * 4 + [low * 1000, 0.0]
    upper = [np.inf] * 4 + [high * 1000, 10 / step]
    fits = []
    for seed in candidates:
        exponential = np.exp((-2 / span + 2j * np.pi * seed) * t)
        offset, amplitude = np.linalg.lstsq(
            np.column_stack((np.ones_like(t), exponential)), observed, rcond=None
        )[0]
        initial = np.array(
            [offset.real, offset.imag, amplitude.real, amplitude.imag, seed, 2 / span]
        )
        fit = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            x_scale=1.0,
            max_nfev=3000,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        if fit.success and np.isfinite(fit.x).all() and np.isfinite(fit.fun).all():
            fits.append(fit)
    if not fits:
        result["reasons"] = ["Signed multistart fit did not converge"]
        return result
    fit = min(fits, key=lambda item: float(np.dot(item.fun, item.fun)))
    parameters = np.asarray(fit.x, dtype=float)
    predicted = model(parameters)
    residual_complex = predicted - observed
    dof = 2 * len(t) - len(parameters)
    reduced = float(np.dot(fit.fun, fit.fun) / dof)
    scale = max(1.0, reduced) if weighted else reduced
    jacobian = np.asarray(fit.jac, dtype=float)
    rank = int(np.linalg.matrix_rank(jacobian))
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * scale
    frequency = float(parameters[4] * 1e-3)
    frequency_se = float(np.sqrt(max(0.0, covariance[4, 4])) * 1e-3)
    r2 = 1 - float(
        np.sum(np.abs(residual_complex) ** 2)
        / np.sum(np.abs(observed - observed.mean()) ** 2)
    )
    contrast = float(np.hypot(parameters[2], parameters[3]))
    reasons = []
    if rank < 6 or not np.isfinite(covariance).all() or not np.isfinite(frequency_se):
        reasons.append("Frequency covariance is rank-deficient or nonfinite")
    if (
        frequency - 1.96 * frequency_se <= low
        or frequency + 1.96 * frequency_se >= high
    ):
        reasons.append(
            "Frequency confidence interval touches the signed search/Nyquist boundary"
        )
    if r2 < r2_limit:
        reasons.append("Single-mode complex fit residual is too large")
    if weighted and reduced > chi2_limit:
        reasons.append("Weighted single-mode lack of fit exceeds max_reduced_chi2")
    if contrast < contrast_limit:
        reasons.append("Complex oscillation contrast is unresolved")
    result.update(
        qualified=not reasons,
        status="qualified" if not reasons else "unqualified",
        reasons=reasons,
        signed_beat_frequency_ghz=frequency,
        signed_beat_standard_error_ghz=frequency_se,
        zz_energy_ghz=2 * (frequency + reference),
        zz_energy_standard_error_ghz=2 * frequency_se,
        decay_time_ns=float(1000 / parameters[5]) if parameters[5] > 1e-12 else None,
        contrast_at_first_point=contrast,
        r2=r2,
        reduced_chi2=reduced if weighted else None,
        reduced_residual_sum_squares=reduced,
        nominal_gaussian_lack_of_fit_p_value=float(chi2.sf(reduced * dof, dof))
        if weighted
        else None,
        degrees_of_freedom=dof,
        parameter_names=[
            "offset_real",
            "offset_imag",
            "amplitude_real",
            "amplitude_imag",
            "signed_beat_mhz",
            "decay_rate_per_us",
        ],
        parameters=parameters.tolist(),
        parameter_covariance=covariance.tolist(),
        fitted_z_x=(-predicted.real).tolist(),
        fitted_z_y=predicted.imag.tolist(),
        residual_z_x=(zx + predicted.real).tolist(),
        residual_z_y=(zy - predicted.imag).tolist(),
        standard_error_x=sx.tolist() if weighted else None,
        standard_error_y=sy.tolist() if weighted else None,
        multistart_results=[{"seed_ghz": seed * 1e-3} for seed in candidates],
        converged_solutions=[
            {
                "signed_beat_ghz": float(item.x[4] * 1e-3),
                "weighted_sse": float(np.dot(item.fun, item.fun)),
            }
            for item in fits
        ],
    )
    return result


def analyze_jazz_counts(
    tau_ns: ArrayLike,
    counts_x: ArrayLike,
    counts_y: ArrayLike,
    *,
    target_index: Literal[0, 1],
    rotation_frequency_ghz: float,
    frequency_bounds_ghz: tuple[float, float] | None = None,
    min_r2: float = 0.9,
    max_reduced_chi2: float = 5.0,
) -> dict[str, Any]:
    """
    Retain raw joint counts and fit their signed target-qubit quadratures.

    Parameters
    ----------
    tau_ns : ArrayLike
        Free-evolution half waits, each on the native 2 ns grid.
    counts_x, counts_y : ArrayLike
        Nonnegative integer arrays shaped (number of waits, 4), ordered
        00, 01, 10, 11. Every row must contain at least one shot.
    target_index : {0, 1}
        Measured target bit in that explicit joint-count ordering.
    rotation_frequency_ghz : float
        Programmed analysis phase-reference rate.
    frequency_bounds_ghz : tuple[float, float], optional
        Optional signed beat-search bounds; never an absolute-frequency band.
    min_r2 : float, optional
        Single-mode complex-fit diagnostic threshold, default 0.9.
    max_reduced_chi2 : float, optional
        Weighted lack-of-fit diagnostic limit, default 5. Raw counts supply
        estimated marginal binomial standard errors, not exact Gaussian noise.

    Returns
    -------
    dict[str, Any]
        Fit diagnostics plus every input count and both half/total time axes.

    Notes
    -----
    Marginal Z errors use a binomial Jeffreys variance to avoid zero weights.
    These are unmitigated counts, not readout-error-free populations. Original
    per-shot IQ must also be saved by the caller before classification.
    """
    tau = _half_waits(tau_ns)
    if tau.ndim != 1 or target_index not in (0, 1):
        raise ValueError("Require one-dimensional tau and target_index 0 or 1")
    arrays = [np.asarray(counts, dtype=float) for counts in (counts_x, counts_y)]
    quadratures, standard_errors = [], []
    excited_indices = [2, 3] if target_index == 0 else [1, 3]
    for counts in arrays:
        if (
            counts.shape != (len(tau), 4)
            or not np.isfinite(counts).all()
            or np.any(counts < 0)
            or np.any(counts != np.rint(counts))
        ):
            raise ValueError(
                "JAZZ counts must be finite nonnegative integer (N,4) arrays"
            )
        shots = counts.sum(axis=1)
        if np.any(shots <= 0):
            raise ValueError("Every JAZZ counts row must have positive total shots")
        excited = counts[:, excited_indices].sum(axis=1)
        quadratures.append(1 - 2 * excited / shots)
        variance = (
            4
            * (excited + 0.5)
            * (shots - excited + 0.5)
            / ((shots + 1) ** 2 * (shots + 2))
        )
        standard_errors.append(np.sqrt(variance))
    result = fit_jazz_quadratures(
        2 * tau,
        *quadratures,
        rotation_frequency_ghz=rotation_frequency_ghz,
        standard_error_x=standard_errors[0],
        standard_error_y=standard_errors[1],
        frequency_bounds_ghz=frequency_bounds_ghz,
        min_r2=min_r2,
        max_reduced_chi2=max_reduced_chi2,
    )
    result.update(
        tau_ns=tau.tolist(),
        raw_counts_x=arrays[0].astype(int).tolist(),
        raw_counts_y=arrays[1].astype(int).tolist(),
        target_index=target_index,
        basis_order=["00", "01", "10", "11"],
        classification="raw binary all-shot; not leakage resolved",
    )
    return result


def infer_exchange_from_static_zz(
    zz_energy_ghz: float,
    *,
    frequency_1_ghz: float,
    frequency_2_ghz: float,
    anharmonicity_1_ghz: float,
    anharmonicity_2_ghz: float,
    zz_energy_standard_error_ghz: float | None = None,
    maximum_dispersive_ratio: float = 0.1,
) -> dict[str, Any]:
    """
    Infer an exchange magnitude from a signed two-Duffing static-ZZ model.

    Parameters
    ----------
    zz_energy_ghz : float
        Signed `(E11-E10-E01+E00)/h`, not the legacy cosine zeta alias.
    frequency_1_ghz, frequency_2_ghz : float
        Positive model qubit frequencies; retain their measured/configured
        provenance. Bare/dressed corrections are not automatically inferred.
    anharmonicity_1_ghz, anharmonicity_2_ghz : float
        Signed model anharmonicities, normally negative for transmons.
    zz_energy_standard_error_ghz : float, optional
        Nonnegative statistical uncertainty of energy ZZ only.
    maximum_dispersive_ratio : float, optional
        Diagnostic upper bound on exchange/level-separation ratios, default 0.1.

    Returns
    -------
    dict[str, Any]
        Coupling magnitude, uncertainty, dispersive ratios and qualification
        reasons. A sign-incompatible model returns no coupling, never abs().

    Notes
    -----
    For `H/h=sum[f_i*n_i+alpha_i*n_i*(n_i-1)/2]
    +g*(a1.dag()*a2+a1*a2.dag())`, second-order theory gives
    `zz=2*g**2*(alpha1+alpha2)/[(Delta+alpha1)*(Delta-alpha2)]` with
    `Delta=f1-f2`. Exchange sign is not identifiable from static ZZ.
    Extra coupler paths/direct ZZ can invalidate this inversion. Uncertainties
    exclude frequency/anharmonicity uncertainty and model systematics.
    """
    zz = _number(zz_energy_ghz, "zz_energy_ghz")
    f1, f2 = (
        _number(frequency_1_ghz, "frequency_1_ghz"),
        _number(frequency_2_ghz, "frequency_2_ghz"),
    )
    a1, a2 = (
        _number(anharmonicity_1_ghz, "anharmonicity_1_ghz"),
        _number(anharmonicity_2_ghz, "anharmonicity_2_ghz"),
    )
    limit = _number(maximum_dispersive_ratio, "maximum_dispersive_ratio")
    uncertainty = (
        None
        if zz_energy_standard_error_ghz is None
        else _number(zz_energy_standard_error_ghz, "zz_energy_standard_error_ghz")
    )
    if (
        f1 <= 0
        or f2 <= 0
        or limit <= 0
        or (uncertainty is not None and uncertainty < 0)
    ):
        raise ValueError(
            "Require positive model frequencies/ratio bound and nonnegative uncertainty"
        )
    delta = f1 - f2
    result: dict[str, Any] = dict(
        qualified=False,
        reasons=[],
        zz_energy_ghz=zz,
        frequency_1_ghz=f1,
        frequency_2_ghz=f2,
        anharmonicity_1_ghz=a1,
        anharmonicity_2_ghz=a2,
        delta_ghz=delta,
        g_magnitude_ghz=None,
        g_standard_error_ghz=None,
        scope="two-Duffing dispersive model-derived |g|; sign and extra coupling paths not inferred",
    )
    gaps = (delta, delta + a1, delta - a2)
    if any(abs(gap) < 1e-12 for gap in gaps) or abs(a1 + a2) < 1e-12:
        result["reasons"] = [
            "Degenerate levels or zero anharmonicity sum prevent dispersive inversion"
        ]
        return result
    coefficient = 2 * (a1 + a2) / ((delta + a1) * (delta - a2))
    squared = zz / coefficient
    result.update(zz_per_g_squared_per_ghz=coefficient, inferred_g_squared_ghz2=squared)
    if squared < 0:
        result["reasons"] = [
            "Measured ZZ sign is incompatible with the supplied two-Duffing model"
        ]
        return result
    g = float(np.sqrt(squared))
    ratios = [
        g / abs(delta),
        np.sqrt(2) * g / abs(delta + a1),
        np.sqrt(2) * g / abs(delta - a2),
    ]
    maximum = float(max(ratios))
    reasons = []
    resolved = uncertainty is None or abs(zz) > 1.96 * uncertainty
    if not resolved or g == 0:
        reasons.append("The signed nonzero ZZ shift is not statistically resolved")
    if maximum > limit:
        reasons.append(
            "Exchange is too large for the declared dispersive approximation"
        )
    result.update(
        qualified=not reasons,
        reasons=reasons,
        g_magnitude_ghz=g,
        maximum_dispersive_ratio=maximum,
        dispersive_ratios=ratios,
        signed_zz_resolved=resolved,
        zz_energy_standard_error_ghz=uncertainty,
        g_standard_error_ghz=uncertainty / (2 * abs(coefficient) * g)
        if uncertainty is not None and resolved and g > 0
        else None,
    )
    return result
