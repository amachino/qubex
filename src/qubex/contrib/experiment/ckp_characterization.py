"""
CKP (chi-kappa-power) characterization utilities.

This module provides measurement sequences and fitting routines for
extracting readout resonator parameters from CKP experiments.

Main features
-------------
- Generate CKP pulse sequences for |0> / |1> qubit states
- Perform 2D CKP scans versus qubit / resonator drive frequencies
- Extract qubit resonance shifts via Lorentzian fitting
- Measure no-drive reference frequencies for offset subtraction
- Fit filtered CKP traces including Purcell-filter effects
- Estimate readout parameters:

    omega_r_g, omega_r_e, omega_p, J, kappa, chi, |A|^2

Typical workflow
----------------
1. Run no-drive CKP measurement
2. Run driven CKP scans for |0> and |1>
3. Extract Stark shifts delta_g / delta_e
4. Estimate initial parameters
5. Perform simultaneous nonlinear fit
6. Return fitted readout parameters

Frequency unit is assumed to be GHz unless otherwise noted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from qxpulse import (
    Blank,
    FlatTop,
    PulseSchedule,
    Waveform,
)
from scipy.optimize import least_squares
from tqdm import tqdm

import qubex.visualization as viz
from qubex.analysis import fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_SHOTS
from qubex.experiment.models import Result


def ckp_sequence_v2(
    exp: Experiment,
    *,
    target: str,
    qubit_initial_state: str | None = None,
    qubit_drive_detuning: float | None = None,
    qubit_pi_pulse: Waveform | None = None,
    qubit_drive_scale: float | None = None,
    qubit_drive_duration: float | None = None,
    qubit_drive_ramptime: float | None = None,
    resonator_drive_detuning: float | None = None,
    resonator_drive_amplitude: float | None = None,
    resonator_drive_ramptime: float | None = None,
    resonator_settle_duration: float | None = None,
) -> PulseSchedule:
    """
    Generate a CKP measurement pulse sequence.

    This sequence applies simultaneous qubit-drive and resonator-drive tones,
    followed by a standard readout pulse, for CKP (chi-kappa-power)
    characterization of the readout resonator system.

    The qubit can be initialized in |0> or |1>, enabling separate
    measurements of state-dependent ac Stark shifts.

    Sequence structure
    ------------------
    1. Optional pi pulse to prepare qubit in |1>
    2. Simultaneous qubit drive + resonator drive
    3. Resonator settling delay
    4. Standard readout pulse

    The resonator drive uses a flat-top envelope during the qubit drive window.
    The qubit drive amplitude is calibrated from the supplied pi-pulse area.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing hardware context and pulse definitions.

    target : str
        Target qubit name or label.

    qubit_initial_state : {"0", "1"}, optional
        Initial qubit state before CKP drive.
        If None, defaults to "0".

    qubit_drive_detuning : float, optional
        Detuning of the qubit drive from the calibrated qubit frequency (GHz).
        If None, defaults to 0.

    qubit_pi_pulse : Waveform, optional
        Pulse used for qubit state preparation.
        If None, a default pi pulse is generated from two half-pi pulses.

    qubit_drive_scale : float, optional
        Relative pulse-area scale used to determine the CKP qubit-drive amplitude.
        If None, defaults to 0.8.

    qubit_drive_duration : float, optional
        Duration of the CKP qubit drive pulse (ns).
        If None, defaults to 128.

    qubit_drive_ramptime : float, optional
        Ramp time of the flat-top qubit drive pulse (ns).
        If None, chosen automatically from the pulse duration.

    resonator_drive_detuning : float, optional
        Detuning of the resonator drive from the calibrated readout frequency (GHz).
        If None, defaults to 0.

    resonator_drive_amplitude : float, optional
        Amplitude of the resonator drive pulse.
        If None, uses the standard readout amplitude.

    resonator_drive_ramptime : float, optional
        Ramp time of the flat-top resonator drive pulse (ns).
        If None, defaults to 32.

    resonator_settle_duration : float, optional
        Waiting time used to let the resonator response reach steady state
        around the CKP drive window.
        This duration is applied before the qubit-drive and
        after the qubit-drive, before the readout pulse.

        If None, defaults to 512 ns.

    Returns
    -------
    PulseSchedule
        Pulse schedule implementing the CKP measurement sequence.

    Notes
    -----
    Frequencies are assumed to use the same units as the experiment backend
    (typically GHz). Durations are in ns.
    """
    qubit = exp.ctx.qubits[target].label
    resonator = exp.ctx.resonators[target].label

    if qubit_initial_state is None:
        qubit_initial_state = "0"
    if qubit_drive_detuning is None:
        qubit_drive_detuning = 0.0
    if qubit_pi_pulse is None:
        qubit_pi_pulse = exp.pulse.get_hpi_pulse(target).repeated(2)
    if qubit_drive_scale is None:
        qubit_drive_scale = 0.8
    if qubit_drive_duration is None:
        qubit_drive_duration = 128
    if qubit_drive_ramptime is None:
        qubit_drive_ramptime = min(round(qubit_drive_duration * 3 / 8), 32)
    if resonator_drive_detuning is None:
        resonator_drive_detuning = 0.0
    if resonator_drive_amplitude is None:
        resonator_drive_amplitude = exp.ctx.params.get_readout_amplitude(
            exp.ctx.resolve_qubit_label(target)
        )
    if resonator_drive_ramptime is None:
        resonator_drive_ramptime = 32
    if resonator_settle_duration is None:
        resonator_settle_duration = 512

    resonator_drive_duration = (
        resonator_settle_duration + qubit_drive_duration + 2 * resonator_drive_ramptime
    )
    resonator_drive_pulse = FlatTop(
        duration=resonator_drive_duration,
        amplitude=resonator_drive_amplitude,
        tau=resonator_drive_ramptime,
    ).detuned(resonator_drive_detuning)

    qubit_pulse_area = (
        np.sum(qubit_pi_pulse.real) * qubit_pi_pulse.sampling_period * qubit_drive_scale
    )
    qubit_pulse_area = float(qubit_pulse_area)
    qubit_drive_amplitude = qubit_pulse_area / (
        qubit_drive_duration - qubit_drive_ramptime
    )
    qubit_drive_pulse = (
        FlatTop(
            duration=qubit_drive_duration,
            amplitude=qubit_drive_amplitude,
            tau=qubit_drive_ramptime,
        )
        .padded(
            resonator_drive_duration - resonator_drive_ramptime,
            pad_side="left",
        )
        .detuned(qubit_drive_detuning)
    )

    resonator_readout_pulse = exp.pulse.readout(target)

    with PulseSchedule() as seq:
        if qubit_initial_state == "1":
            seq.add(qubit, qubit_pi_pulse)
        seq.barrier()
        seq.add(qubit, qubit_drive_pulse)
        seq.add(resonator, resonator_drive_pulse)
        seq.add(resonator, Blank(resonator_settle_duration))
        seq.add(resonator, resonator_readout_pulse)
    return seq


def ckp_measurement_v2(
    exp: Experiment,
    *,
    target: str,
    qubit_initial_state: str,
    qubit_detuning_range: ArrayLike | None = None,
    qubit_pi_pulse: Waveform | None = None,
    qubit_drive_scale: float | None = None,
    qubit_drive_duration: float | None = None,
    resonator_detuning_range: ArrayLike | None = None,
    resonator_drive_amplitude: float | None = None,
    resonator_settle_duration: float | None = None,
    shots: int | None = None,
    plot: bool | None = None,
    verbose: bool | None = None,
    save_image: bool | None = None,
    enable_early_stop: bool | None = None,
    f0_lower_qubit_detuning_limit: float | None = None,
) -> Result:
    """
    Run a CKP measurement sweep and extract qubit resonance frequencies.

    This routine performs a two-dimensional CKP scan by sweeping:

    - qubit drive frequency
    - resonator drive frequency

    For each resonator drive point, qubit spectroscopy data are acquired
    over the specified qubit detuning range. Each 1D spectroscopy trace is
    fitted with a Lorentzian model to estimate the qubit resonance
    frequency shift.

    The extracted resonance frequencies are later used for filtered CKP
    fitting to estimate readout resonator parameters such as:

        omega_r_g, omega_r_e, omega_p, J, kappa, chi, |A|^2

    Invalid Lorentzian fits (fit failure, non-finite parameters, or poor
    goodness-of-fit) are skipped automatically.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing hardware context, pulse definitions,
        and measurement backends.

    target : str
        Target qubit name or label.

    qubit_initial_state : {"0", "1"}
        Initial qubit state used during the CKP measurement.

        - "0" : measure ground-state response
        - "1" : apply pi pulse before measurement

    qubit_detuning_range : ArrayLike, optional
        Sweep values of qubit drive detuning from the calibrated qubit
        frequency (GHz).

        If None, a default spectroscopy sweep range is used.

    qubit_pi_pulse : Waveform, optional
        Pulse used for preparing the qubit excited state.

        If None, the default calibrated pi pulse is used.

    qubit_drive_scale : float, optional
        Relative pulse-area scale of the CKP qubit drive.

    qubit_drive_duration : float, optional
        Duration of the CKP qubit drive pulse (ns).

    resonator_detuning_range : ArrayLike, optional
        Sweep values of resonator drive detuning from the calibrated
        readout resonator frequency (GHz).

        If None, a default non-uniform sweep concentrated near resonance
        is used.

    resonator_drive_amplitude : float, optional
        Amplitude of the resonator drive pulse.

        If None, the standard readout amplitude is used.

    resonator_settle_duration : float, optional
        Waiting time used to let the resonator response approach steady
        state before and after the qubit-drive interval (ns).

    shots : int, optional
        Number of measurement shots used to acquire a single sweep point.

        If None, uses DEFAULT_SHOTS.

    plot : bool, optional
        If True, display the measured CKP heatmap and extracted Lorentzian
        peak positions.

    verbose : bool, optional
        If True, show individual Lorentzian fitting plots for each
        resonator-frequency point.

    save_image : bool, optional
        If True, save the generated CKP heatmap figure.

    enable_early_stop : bool, optional
        If True, stop the resonator-frequency sweep early when the fitted
        qubit resonance frequency crosses the specified lower limit.

        If None, defaults to False.

    f0_lower_qubit_detuning_limit : float, optional
        Lower detuning threshold for early stop (GHz), measured from the
        calibrated qubit frequency.

        If None, a default value is used when early stop is enabled.

    Returns
    -------
    Result
        Result object containing measurement data and extracted resonance
        frequencies.

        Expected fields in ``data`` include:

        - ``qubit_frequency_range``
        - ``resonator_frequency_range``
        - ``qubit_detuning_range``
        - ``resonator_detuning_range``
        - ``valid_resonator_frequencies``
        - ``qubit_resonance_frequencies``
        - ``qubit_resonance_frequency_errors``
        - ``qubit_initial_state``
        - ``data`` (2D CKP map)
        - ``fit_results``

    Notes
    -----
    Frequencies are assumed to use the backend frequency unit
    (typically GHz). Durations are in ns.
    """
    if qubit_initial_state not in {"0", "1"}:
        qubit_initial_state = "0"
    if shots is None:
        shots = DEFAULT_SHOTS
    if plot is None:
        plot = True
    if verbose is None:
        verbose = False
    if save_image is None:
        save_image = True
    if qubit_detuning_range is None:
        left = np.linspace(-0.055, -0.04, 3, endpoint=False)
        center = np.linspace(-0.04, 0.01, 30, endpoint=False)
        right = np.linspace(0.01, 0.025, 3)
        qubit_detuning_range = np.concatenate([left, center, right])
    else:
        qubit_detuning_range = np.asarray(qubit_detuning_range)
    if resonator_detuning_range is None:
        u = np.linspace(-1, 1, 50)
        beta = 0.9
        resonator_detuning_range = 0.1 * np.arctanh(beta * u) / np.arctanh(beta)
    resonator_detuning_range = np.asarray(resonator_detuning_range)
    if resonator_drive_amplitude is None:
        resonator_drive_amplitude = exp.ctx.params.get_readout_amplitude(
            exp.ctx.resolve_qubit_label(target)
        )
    if qubit_drive_scale is None:
        qubit_drive_scale = 0.8
    if qubit_drive_duration is None:
        qubit_drive_duration = 128
    if enable_early_stop is None:
        enable_early_stop = False
    if enable_early_stop and f0_lower_qubit_detuning_limit is None:
        f0_lower_qubit_detuning_limit = -0.035

    qubit_label = exp.ctx.resolve_qubit_label(target)
    read_label = exp.ctx.resolve_read_label(target)
    f_qubit = exp.ctx.targets[qubit_label].frequency
    f_resonator = exp.ctx.targets[read_label].frequency
    qubit_frequency_range = qubit_detuning_range + f_qubit
    resonator_frequency_range = resonator_detuning_range + f_resonator

    if enable_early_stop and f0_lower_qubit_detuning_limit is not None:
        f0_lower_frequency_limit = f_qubit + f0_lower_qubit_detuning_limit
    else:
        f0_lower_frequency_limit = None
    early_stopped = False

    result2d = []
    valid_resonator_frequencies = []
    qubit_resonance_frequencies = []
    qubit_resonance_frequency_errors = []
    measured_resonator_frequencies = []
    fit_results = []
    exp.ctx.reset_awg_and_capunits(qubits=[qubit_label])
    for resonator_detuning in tqdm(resonator_detuning_range):
        measured_resonator_frequencies.append(float(f_resonator + resonator_detuning))
        result1d = []
        for qubit_detuning in qubit_detuning_range:
            result = exp.measurement_service.execute(
                ckp_sequence_v2(
                    exp=exp,
                    target=target,
                    qubit_initial_state=qubit_initial_state,
                    qubit_drive_detuning=qubit_detuning,
                    qubit_pi_pulse=qubit_pi_pulse,
                    qubit_drive_scale=qubit_drive_scale,
                    qubit_drive_duration=qubit_drive_duration,
                    resonator_drive_detuning=resonator_detuning,
                    resonator_drive_amplitude=resonator_drive_amplitude,
                    resonator_settle_duration=resonator_settle_duration,
                ),
                reset_awg_and_capunits=False,
                n_shots=shots,
            )
            data = result.data[target][-1]
            result1d.append(data.kerneled)

        result1d = exp.pulse.rabi_params[target].normalize(np.array(result1d))
        result2d.append(result1d)

        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        pad = len(kernel) // 2
        result1d_pad = np.pad(result1d, pad_width=pad, mode="edge")
        result1d_smooth = np.convolve(result1d_pad, kernel, mode="valid")
        if qubit_initial_state == "0":
            idx0 = np.argmin(result1d_smooth)
        else:
            idx0 = np.argmax(result1d_smooth)
        gamma_guess = 0.443 / qubit_drive_duration
        f0_guess = float(qubit_frequency_range[idx0])
        A_guess_mag = 2.0 * np.sin(0.5 * np.pi * qubit_drive_scale) ** 2

        if qubit_initial_state == "0":
            p0 = (-A_guess_mag, f0_guess, gamma_guess, 1.0)
            bounds = (
                (-np.inf, np.min(qubit_frequency_range), 0, -np.inf),
                (0, np.max(qubit_frequency_range), np.inf, np.inf),
            )
        else:
            p0 = (A_guess_mag, f0_guess, gamma_guess, -1.0)
            bounds = (
                (0, np.min(qubit_frequency_range), 0, -np.inf),
                (np.inf, np.max(qubit_frequency_range), np.inf, np.inf),
            )
        try:
            fit_result = fitting.fit_lorentzian(
                x=qubit_frequency_range,
                y=result1d,
                p0=p0,
                bounds=bounds,
                plot=verbose,
            )
        except Exception as exc:
            print(
                f"[WARN] Lorentzian fit raised exception at "
                f"resonator_detuning={resonator_detuning:.6f}: {exc}"
            )
            fit_results.append(None)
            continue
        fit_results.append(fit_result)
        f0 = fit_result.get("f0", np.nan)
        f0_err = fit_result.get("f0_err", np.nan)
        r2 = fit_result.get("r2", np.nan)
        if not (
            np.isfinite(f0) and np.isfinite(f0_err) and np.isfinite(r2) and r2 > 0.5
        ):
            print(f"[WARN] invalid fit at resonator_detuning={resonator_detuning:.6f}")
            print(f"f0={f0}, err={f0_err}, r2={r2:.3f}")
        else:
            valid_resonator_frequencies.append(f_resonator + resonator_detuning)
            qubit_resonance_frequencies.append(f0)
            qubit_resonance_frequency_errors.append(f0_err)

            if enable_early_stop and f0_lower_frequency_limit is not None:
                if f0 < f0_lower_frequency_limit:
                    early_stopped = True
                    break

    data = np.array(result2d)
    if qubit_initial_state == "1":
        data *= -1

    measured_resonator_frequency_range = np.asarray(
        measured_resonator_frequencies,
        dtype=float,
    )

    fig = viz.make_figure()
    fig.add_heatmap(
        z=data.T,
        x=measured_resonator_frequency_range,
        y=qubit_frequency_range,
        colorscale="Viridis",
        colorbar=dict(
            title=dict(
                text="Normalized signal",
                side="right",
            ),
        ),
    )
    fig.add_scatter(
        x=valid_resonator_frequencies,
        y=qubit_resonance_frequencies,
        mode="markers",
        name="Lorentzian fit f0",
        marker=dict(
            color="white",
            size=6,
            line=dict(color="black", width=1),
        ),
    )
    fig.update_layout(
        title=dict(
            text=f"CKP measurement : {target} : |{qubit_initial_state}〉",
            subtitle=dict(
                text=f"resonator_drive_amplitude={resonator_drive_amplitude:.6g}",
                font=dict(size=11, family="monospace"),
            ),
        ),
        xaxis_title="Resonator drive frequency (GHz)",
        yaxis_title="Qubit drive frequency (GHz)",
        width=600,
        height=400,
    )
    if plot:
        fig.show()

    if save_image:
        viz.save_figure(
            fig,
            name=f"ckp_measurement_{target}_{qubit_initial_state}",
            width=600,
            height=400,
        )

    return Result(
        data={
            "qubit_frequency_range": qubit_frequency_range,
            "qubit_detuning_range": qubit_detuning_range,
            "resonator_frequency_range": measured_resonator_frequency_range,
            "resonator_detuning_range": measured_resonator_frequency_range
            - f_resonator,
            "requested_resonator_frequency_range": resonator_frequency_range,
            "requested_resonator_detuning_range": resonator_detuning_range,
            "valid_resonator_frequencies": valid_resonator_frequencies,
            "qubit_resonance_frequencies": qubit_resonance_frequencies,
            "qubit_resonance_frequency_errors": qubit_resonance_frequency_errors,
            "qubit_initial_state": qubit_initial_state,
            "early_stop_mode": enable_early_stop,
            "f0_lower_frequency_limit": f0_lower_frequency_limit,
            "early_stopped": early_stopped,
            "data": data,
            "fit_results": fit_results,
        }
    )


def filtered_ckp_experiment(
    exp: Experiment,
    *,
    target: str,
    qubit_detuning_range: ArrayLike | None = None,
    qubit_pi_pulse: Waveform | None = None,
    qubit_drive_scale: float | None = None,
    qubit_drive_duration: float | None = None,
    resonator_detuning_range: ArrayLike | None = None,
    resonator_drive_amplitude: float | None = None,
    resonator_settle_duration: float | None = None,
    shots: int | None = None,
    plot: bool | None = None,
    show_animation: bool | None = None,
    save_image: bool | None = None,
    save_animation: bool | None = None,
    enable_rough_search: bool | None = None,
    f0_lower_qubit_detuning_limit: float | None = None,
    f0_upper_qubit_detuning_limit: float | None = None,
) -> Result:
    """
    Run a full filtered CKP experiment and estimate readout resonator parameters.

    This routine performs CKP measurements for both qubit initial states:

    - |0>
    - |1>

    It first acquires no-drive reference data (resonator drive amplitude = 0)
    to determine baseline qubit resonance frequencies:

        ωq^g, ωq^e

    Then standard CKP scans are performed with resonator drive applied.
    State-dependent ac Stark shifts are constructed as:

        delta_g = f0_g - ωq^g
        delta_e = f0_e - ωq^e

    The two traces are simultaneously fitted with the filtered CKP model,
    including Purcell-filter effects, to estimate:

        ωr^g, ωr^e, ωp, J, κ, χ, C, |A|²

    In addition, the estimated intracavity photon numbers for the |g> and |e>
    branches are evaluated.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing hardware context, pulse definitions,
        and measurement backend.

    target : str
        Target qubit name or label.

    qubit_detuning_range : ArrayLike, optional
        Sweep values of qubit drive detuning from the calibrated qubit
        frequency (GHz).

        If None, a default spectroscopy sweep range is used.

    qubit_pi_pulse : Waveform, optional
        Pulse used for excited-state preparation.

        If None, the default calibrated pi pulse is used.

    qubit_drive_scale : float, optional
        Relative pulse-area scale of the CKP qubit drive.

    qubit_drive_duration : float, optional
        Duration of the CKP qubit drive pulse (ns).

    resonator_detuning_range : ArrayLike, optional
        Sweep values of resonator drive detuning from the calibrated
        readout resonator frequency (GHz).

        If None, a default sweep range is used.

    resonator_drive_amplitude : float, optional
        Amplitude of the resonator drive pulse.

        If None, the standard readout amplitude is used.

    resonator_settle_duration : float, optional
        Waiting time used to let the resonator response approach steady
        state before and after the qubit-drive interval (ns).

    shots : int, optional
        Number of measurement shots used for one CKP sweep point.

        If None, uses DEFAULT_SHOTS.

    plot : bool, optional
        If True, display the fitted CKP traces and estimated parameters.

    show_animation : bool, optional
        If True, show the generated animation.

    save_image : bool, optional
        If True, save the generated fit figure.

    save_animation : bool, optional
        If True, save the generated animation.

    enable_rough_search : bool, optional
        If True, perform a coarse pre-scan to automatically adjust
        resonator drive amplitude before the main CKP measurement.

        If None, defaults to True.

    f0_lower_qubit_detuning_limit : float, optional
        Lower detuning threshold (GHz) used during rough search.

    f0_upper_qubit_detuning_limit : float, optional
        Upper detuning threshold (GHz) used during rough search.

    Returns
    -------
    Result
        Result object containing raw CKP measurements, fitted parameters,
        derived quantities, and intermediate fitting information.

        Expected fields in ``data`` include:

        Measurement data
        ----------------
        - ``result_0``
        - ``result_1``
        - ``result_0_no_drive``
        - ``result_1_no_drive``

        Processed traces
        ----------------
        - ``resonator_frequencies_g``
        - ``resonator_frequencies_e``
        - ``delta_g``
        - ``delta_e``

        Estimated parameters
        --------------------
        - ``omega_r_g``
        - ``omega_r_e``
        - ``omega_p``
        - ``J``
        - ``kappa``
        - ``chi``
        - ``C``
        - ``A2``  ( = |A|² )
        - ``A``

        Photon-number metrics
        ---------------------
        - ``ng_max``
        - ``ne_max``
        - ``x_ng_max``
        - ``x_ne_max``

        Fit diagnostics
        ---------------
        - ``initial_guess``
        - ``fit_result``

        Rough search results
        --------------------
        - ``resonator_drive_amplitude``
        - ``rough_search_results``
        - ``rough_search_reduces``
        - ``rough_search_increases``

        Optimized readout frequency
        ---------------
        - ``readout_frequency_opt``
        - ``readout_optimization_result``

    Notes
    -----
    Frequencies are assumed to use the backend frequency unit
    (typically GHz). Durations are in ns.

    This routine raises an exception when:

    - no valid no-drive Lorentzian fit is obtained
    - CKP fitting fails
    - estimated χ is too small
    - estimated |A|² becomes negative

    When rough search is enabled, a reduced sweep using every fourth
    resonator-frequency point and one-fourth shots is used.
    """
    if enable_rough_search is None:
        enable_rough_search = True
    if enable_rough_search:
        if f0_lower_qubit_detuning_limit is None:
            f0_lower_qubit_detuning_limit = -0.035
        if f0_upper_qubit_detuning_limit is None:
            f0_upper_qubit_detuning_limit = -0.01
    if shots is None:
        shots = DEFAULT_SHOTS
    if plot is None:
        plot = True
    if show_animation is None:
        show_animation = False
    if save_image is None:
        save_image = True
    if save_animation is None:
        save_animation = False
    if resonator_detuning_range is None:
        u = np.linspace(-1, 1, 50)
        beta = 0.9
        resonator_detuning_range = 0.1 * np.arctanh(beta * u) / np.arctanh(beta)
    resonator_detuning_range = np.asarray(resonator_detuning_range)
    if resonator_drive_amplitude is None:
        resonator_drive_amplitude = exp.ctx.params.get_readout_amplitude(
            exp.ctx.resolve_qubit_label(target)
        )
    verbose = False

    rough_search_results = []
    n_reduces = 0
    n_increases = 0

    if enable_rough_search:
        resonator_detuning_range_rough_search = resonator_detuning_range[::4]
        if len(resonator_detuning_range_rough_search) == 0:
            resonator_detuning_range_rough_search = resonator_detuning_range
        shots_rough_search = max(1, shots // 4)

        while True:
            result_0_rough_search = ckp_measurement_v2(
                exp=exp,
                target=target,
                qubit_initial_state="0",
                qubit_detuning_range=qubit_detuning_range,
                qubit_pi_pulse=qubit_pi_pulse,
                qubit_drive_scale=qubit_drive_scale,
                qubit_drive_duration=qubit_drive_duration,
                resonator_detuning_range=resonator_detuning_range_rough_search,
                resonator_drive_amplitude=resonator_drive_amplitude,
                resonator_settle_duration=resonator_settle_duration,
                shots=shots_rough_search,
                plot=False,
                verbose=verbose,
                save_image=False,
                enable_early_stop=True,
                f0_lower_qubit_detuning_limit=f0_lower_qubit_detuning_limit,
            )
            rough_search_results.append(result_0_rough_search)

            if result_0_rough_search.data["early_stopped"]:
                resonator_drive_amplitude *= 0.5
                n_reduces += 1
                print(
                    "[ROUGH SEARCH] ac Stark shift too large. "
                    f"Reduce resonator_drive_amplitude -> {resonator_drive_amplitude:.6g} "
                )
                if n_reduces >= 3:
                    print(
                        "[ROUGH SEARCH] Early stop still detected after max reduction trials. "
                        "Proceeding with current amplitude."
                    )
                    break
                continue

            f0_list = np.asarray(
                result_0_rough_search.data["qubit_resonance_frequencies"],
                dtype=float,
            )
            if len(f0_list) == 0:
                print("[ROUGH SEARCH] No valid f0 obtained. ")
                break
            f_qubit_rough = float(
                result_0_rough_search.data["qubit_frequency_range"][0]
                - result_0_rough_search.data["qubit_detuning_range"][0]
            )
            if f0_upper_qubit_detuning_limit is not None:
                f0_upper_frequency_limit = f_qubit_rough + f0_upper_qubit_detuning_limit
                has_below_upper = np.any(f0_list < f0_upper_frequency_limit)
                if not has_below_upper:
                    resonator_drive_amplitude *= 2.0
                    n_increases += 1
                    print(
                        "[ROUGH SEARCH] ac Stark shift too small. "
                        f"Increase resonator_drive_amplitude -> {resonator_drive_amplitude:.6g} "
                    )
                    if n_increases >= 1:
                        print(
                            "[ROUGH SEARCH] Max increase trial reached. "
                            "Proceeding with current amplitude."
                        )
                        break
                    continue

            print(
                "[ROUGH SEARCH] resonator_drive_amplitude accepted : "
                f"{resonator_drive_amplitude:.6g}"
            )
            break

    shots_no_drive = 4 * shots
    plot_no_drive = False
    save_image_no_drive = False

    result_0_no_drive = ckp_measurement_v2(
        exp=exp,
        target=target,
        qubit_initial_state="0",
        qubit_detuning_range=qubit_detuning_range,
        qubit_pi_pulse=qubit_pi_pulse,
        qubit_drive_scale=qubit_drive_scale,
        qubit_drive_duration=qubit_drive_duration,
        resonator_detuning_range=np.asarray([0.0]),
        resonator_drive_amplitude=0.0,
        resonator_settle_duration=resonator_settle_duration,
        shots=shots_no_drive,
        plot=plot_no_drive,
        verbose=verbose,
        save_image=save_image_no_drive,
    )
    result_1_no_drive = ckp_measurement_v2(
        exp=exp,
        target=target,
        qubit_initial_state="1",
        qubit_detuning_range=qubit_detuning_range,
        qubit_pi_pulse=qubit_pi_pulse,
        qubit_drive_scale=qubit_drive_scale,
        qubit_drive_duration=qubit_drive_duration,
        resonator_detuning_range=np.asarray([0.0]),
        resonator_drive_amplitude=0.0,
        resonator_settle_duration=resonator_settle_duration,
        shots=shots_no_drive,
        plot=plot_no_drive,
        verbose=verbose,
        save_image=save_image_no_drive,
    )

    freq_g = result_0_no_drive.data["qubit_resonance_frequencies"]
    freq_e = result_1_no_drive.data["qubit_resonance_frequencies"]
    if len(freq_g) == 0 or len(freq_e) == 0:
        raise ValueError(
            "No valid no-drive Lorentzian fit result. "
            "qubit_resonance_frequencies is empty."
        )
    offset_g = float(freq_g[0])
    offset_e = float(freq_e[0])
    print(f"ωq^g = {offset_g:.6f} GHz")
    print(f"ωq^e = {offset_e:.6f} GHz")

    result_0 = ckp_measurement_v2(
        exp=exp,
        target=target,
        qubit_initial_state="0",
        qubit_detuning_range=qubit_detuning_range,
        qubit_pi_pulse=qubit_pi_pulse,
        qubit_drive_scale=qubit_drive_scale,
        qubit_drive_duration=qubit_drive_duration,
        resonator_detuning_range=resonator_detuning_range,
        resonator_drive_amplitude=resonator_drive_amplitude,
        resonator_settle_duration=resonator_settle_duration,
        shots=shots,
        plot=plot,
        verbose=verbose,
        save_image=save_image,
    )
    result_1 = ckp_measurement_v2(
        exp=exp,
        target=target,
        qubit_initial_state="1",
        qubit_detuning_range=qubit_detuning_range,
        qubit_pi_pulse=qubit_pi_pulse,
        qubit_drive_scale=qubit_drive_scale,
        qubit_drive_duration=qubit_drive_duration,
        resonator_detuning_range=resonator_detuning_range,
        resonator_drive_amplitude=resonator_drive_amplitude,
        resonator_settle_duration=resonator_settle_duration,
        shots=shots,
        plot=plot,
        verbose=verbose,
        save_image=save_image,
    )

    x_g = np.asarray(result_0.data["valid_resonator_frequencies"], dtype=float)
    x_e = np.asarray(result_1.data["valid_resonator_frequencies"], dtype=float)
    f0_g = np.asarray(result_0.data["qubit_resonance_frequencies"], dtype=float)
    f0_e = np.asarray(result_1.data["qubit_resonance_frequencies"], dtype=float)
    err_g = np.asarray(result_0.data["qubit_resonance_frequency_errors"], dtype=float)
    err_e = np.asarray(result_1.data["qubit_resonance_frequency_errors"], dtype=float)

    delta_g = f0_g - offset_g
    delta_e = f0_e - offset_e

    eps = 1e-12
    weights_g = 1.0 / np.maximum(np.abs(err_g), eps)
    weights_e = 1.0 / np.maximum(np.abs(err_e), eps)

    init_g = estimate_filtered_ckp_initial_params(
        x=x_g,
        delta=delta_g,
    )
    init_e = estimate_filtered_ckp_initial_params(
        x=x_e,
        delta=delta_e,
    )

    C0 = 0.5 * (init_g.C + init_e.C)
    omega_r_g0 = init_g.omega_r
    omega_r_e0 = init_e.omega_r
    omega_p0 = 0.5 * (init_g.omega_p + init_e.omega_p)
    J0 = 0.5 * (init_g.J + init_e.J)
    kappa0 = 0.5 * (init_g.kappa + init_e.kappa)

    theta0 = np.array(
        [C0, omega_r_g0, omega_r_e0, omega_p0, J0, kappa0],
        dtype=float,
    )

    fit = fit_filtered_ckp_two_traces(
        x_g=x_g,
        x_e=x_e,
        delta_g=delta_g,
        delta_e=delta_e,
        weights_g=weights_g,
        weights_e=weights_e,
        theta0=theta0,
    )

    if not fit.success:
        raise RuntimeError(f"CKP fit failed: {fit.message}")

    omega_r_g = float(fit.omega_r_g)
    omega_r_e = float(fit.omega_r_e)
    omega_p = float(fit.omega_p)
    J = float(fit.J)
    kappa = float(fit.kappa)
    C = float(fit.C)

    chi = (omega_r_e - omega_r_g) / 2.0

    if np.isclose(chi, 0.0):
        raise ValueError("Estimated chi is too close to zero, cannot compute |A|^2.")

    A2 = C / (4.0 * chi)
    if A2 < 0:
        raise ValueError(
            f"Estimated |A|^2 became negative: C={C}, chi={chi}, C/(4chi)={A2}"
        )

    x_dense_min = min(np.min(x_g), np.min(x_e), omega_r_g, omega_r_e, omega_p) - 0.01
    x_dense_max = max(np.max(x_g), np.max(x_e), omega_r_g, omega_r_e, omega_p) + 0.01
    x_dense = np.linspace(x_dense_min, x_dense_max, 4000)

    fit_g = filtered_ckp_model(
        x_dense,
        C=C,
        omega_r=omega_r_g,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
    )
    fit_e = filtered_ckp_model(
        x_dense,
        C=C,
        omega_r=omega_r_e,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
    )

    ng_dense = filtered_ckp_photon_number(
        x_dense,
        omega_r=omega_r_g,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
        A2=A2,
    )
    ne_dense = filtered_ckp_photon_number(
        x_dense,
        omega_r=omega_r_e,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
        A2=A2,
    )

    ng_max = float(np.max(ng_dense))
    ne_max = float(np.max(ne_dense))
    x_ng_max = float(x_dense[np.argmax(ng_dense)])
    x_ne_max = float(x_dense[np.argmax(ne_dense)])

    if plot or save_image:
        fig = viz.make_figure()

        fig.add_scatter(
            x=x_g,
            y=delta_g,
            mode="markers",
            name="g-series data",
            error_y=dict(
                type="data",
                array=1.0 / weights_g,
                visible=True,
            ),
        )
        fig.add_scatter(
            x=x_dense,
            y=fit_g,
            mode="lines",
            name="g-series fit",
        )

        fig.add_scatter(
            x=x_e,
            y=delta_e,
            mode="markers",
            name="e-series data",
            error_y=dict(
                type="data",
                array=1.0 / weights_e,
                visible=True,
            ),
        )
        fig.add_scatter(
            x=x_dense,
            y=fit_e,
            mode="lines",
            name="e-series fit",
        )

        fig.update_layout(
            title=dict(
                text=f"Filtered CKP fit : {target}",
            ),
            xaxis_title="Resonator drive frequency (GHz)",
            yaxis_title="ac Stark shift (GHz)",
            width=800,
            height=500,
        )

        if plot:
            fig.show()
            print(f"=== Estimated Readout Parameters : {target} ===")
            print(f"ωr^g   = {omega_r_g:.6f} GHz")
            print(f"ωr^e   = {omega_r_e:.6f} GHz")
            print(f"ωp     = {omega_p:.6f} GHz")
            print(f"J      = {1000 * J:.3f} MHz")
            print(f"κ      = {1000 * kappa:.3f} MHz")
            print(f"χ      = {1000 * chi:.3f} MHz")
            print(f"|A|²   = {1000 * A2:.3f} MHz")
            print(f"n_g,max = {ng_max:.3f} @ {x_ng_max:.6f} GHz")
            print(f"n_e,max = {ne_max:.3f} @ {x_ne_max:.6f} GHz")

        if save_image:
            viz.save_figure(
                fig,
                name=f"filtered_ckp_fit_{target}",
                width=800,
                height=500,
            )

    readout_opt_result = estimate_optimal_readout_frequency_from_ckp(
        frequency_range=[x_dense_min, x_dense_max],
        omega_r_g=omega_r_g,
        omega_r_e=omega_r_e,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
        A=float(np.sqrt(A2)),
        plot=plot,
        show_animation=show_animation,
        save_image=save_image,
        save_animation=save_animation,
        name=f"readout_opt_{target}",
    )

    return Result(
        data={
            "target": target,
            "resonator_detuning_range": resonator_detuning_range,
            "resonator_drive_amplitude": resonator_drive_amplitude,
            "resonator_frequencies_g": x_g,
            "resonator_frequencies_e": x_e,
            "delta_g": delta_g,
            "delta_e": delta_e,
            "offset_g": offset_g,
            "offset_e": offset_e,
            "weights_g": weights_g,
            "weights_e": weights_e,
            "omega_r_g": omega_r_g,
            "omega_r_e": omega_r_e,
            "omega_p": omega_p,
            "J": J,
            "kappa": kappa,
            "chi": chi,
            "C": C,
            "A2": A2,
            "A": float(np.sqrt(A2)),
            "ng_max": ng_max,
            "ne_max": ne_max,
            "x_ng_max": x_ng_max,
            "x_ne_max": x_ne_max,
            "initial_guess": {
                "C0": C0,
                "omega_r_g0": omega_r_g0,
                "omega_r_e0": omega_r_e0,
                "omega_p0": omega_p0,
                "J0": J0,
                "kappa0": kappa0,
                "init_g": init_g,
                "init_e": init_e,
            },
            "fit_result": fit,
            "result_0": result_0,
            "result_1": result_1,
            "result_0_no_drive": result_0_no_drive,
            "result_1_no_drive": result_1_no_drive,
            "rough_search_results": rough_search_results,
            "rough_search_reduce_trials": n_reduces,
            "rough_search_increase_trials": n_increases,
            "readout_frequency_opt": readout_opt_result.data["readout_frequency_opt"],
            "readout_optimization_result": readout_opt_result,
        }
    )


@dataclass
class FilteredCKPInitialGuess:
    """Initial parameter estimates for filtered CKP model fitting."""

    C: float
    omega_r: float
    omega_p: float
    J: float
    kappa: float


def estimate_filtered_ckp_initial_params(
    x: ArrayLike,
    delta: ArrayLike,
) -> FilteredCKPInitialGuess:
    """
    Estimate initial parameters for the filtered CKP model.

    Use the moment method with a local quadratic fit around the centroid.

    Model:
        delta(x) = (C J^2 g) /
                (((x-r)(x-p)-J^2)^2 + g^2 (x-r)^2)

        where g = kappa / 2

    Theory:
        ∫ delta dx = pi C
        centroid = r
        variance = J^2

        delta(r) = C g / J^2
        delta'(r)/delta(r) = 2(r-p)/J^2

    Returns
    -------
    FilteredCKPInitialGuess
    """
    x = np.asarray(x, dtype=float).ravel()
    delta = np.asarray(delta, dtype=float).ravel()

    if x.size != delta.size:
        raise ValueError("x and delta must have same length.")
    if x.size < 5:
        raise ValueError("Need at least 5 points.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(delta)):
        raise ValueError("x or delta contains non-finite values.")
    if np.min(delta) >= 0:
        raise ValueError("delta does not contain a negative peak.")

    order = np.argsort(x)
    x = x[order]
    delta = delta[order]

    M0 = np.trapz(delta, x)
    M1 = np.trapz(x * delta, x)
    M2 = np.trapz((x**2) * delta, x)

    if np.abs(M0) < 1e-18:
        raise ValueError("Moment M0 is too small.")

    r0 = M1 / M0
    var0 = max(M2 / M0 - r0**2, 1e-12)
    J0 = np.sqrt(var0)

    C0 = M0 / np.pi

    idx = int(np.argmin(np.abs(x - r0)))
    left = max(0, min(idx - 2, x.size - 5))
    right = left + 5

    xx = x[left:right] - r0
    yy = delta[left:right]

    coef = np.polyfit(xx, yy, 2)
    _, a1, a0 = coef

    if np.abs(a0) < 1e-18:
        raise ValueError("delta(r) is too small.")

    g0 = (J0**2) * a0 / C0
    p0 = r0 - 0.5 * (J0**2) * (a1 / a0)
    kappa0 = 2.0 * g0

    if kappa0 <= 0:
        raise ValueError("Estimated kappa is non-positive.")

    return FilteredCKPInitialGuess(
        C=float(C0),
        omega_r=float(r0),
        omega_p=float(p0),
        J=float(J0),
        kappa=float(kappa0),
    )


@dataclass
class FilteredCKPFitResult:
    """Result of simultaneous filtered CKP fitting."""

    C: float
    omega_r_g: float
    omega_r_e: float
    omega_p: float
    J: float
    kappa: float
    success: bool
    cost: float
    message: str
    nfev: int
    result_raw: object


def filtered_ckp_model(
    x: ArrayLike,
    *,
    C: float,
    omega_r: float,
    omega_p: float,
    J: float,
    kappa: float,
) -> np.ndarray:
    """
    Filtered-CKP model for one series.

    Parameters
    ----------
    x
        Drive frequency array.
    C, omega_r, omega_p, J, kappa
        Model parameters in the same frequency unit as x.

    Returns
    -------
    np.ndarray
        Model values delta(x).
    """
    x = np.asarray(x, dtype=float)
    g = kappa / 2.0
    denom = ((x - omega_r) * (x - omega_p) - J**2) ** 2 + (g**2) * (x - omega_r) ** 2
    return (C * J**2 * g) / denom


def fit_filtered_ckp_two_traces(
    x_g: ArrayLike,
    x_e: ArrayLike,
    delta_g: ArrayLike,
    delta_e: ArrayLike,
    weights_g: ArrayLike,
    weights_e: ArrayLike,
    theta0: ArrayLike,
    *,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    loss: str = "linear",
    f_scale: float = 1.0,
    max_nfev: int = 10000,
) -> FilteredCKPFitResult:
    """
    Simultaneously fit g/e CKP traces with the filtered model.

    Parameters
    ----------
    x_g
        1D array of omega_{d,r} for the g-series.
    x_e
        1D array of omega_{d,r} for the e-series.
    delta_g
        1D array of delta^g(x).
    delta_e
        1D array of delta^e(x).
    weights_g
        1D array of weights for the g-series residuals.
    weights_e
        1D array of weights for the e-series residuals.
    theta0
        Initial parameters:
        [C0, omega_r_g0, omega_r_e0, omega_p0, J0, kappa0]
    bounds
        Optional bounds for least_squares, as (lower, upper).
        If omitted, loose default bounds are used.
    loss
        Loss function passed to scipy.optimize.least_squares.
    f_scale
        f_scale passed to least_squares.
    max_nfev
        Maximum number of function evaluations.

    Returns
    -------
    FilteredCKPFitResult
        Best-fit parameters and optimization metadata.

    Notes
    -----
    Residual vector is
        concat(
            weights_g * (model_g(x) - delta_g),
            weights_e * (model_e(x) - delta_e),
        )

    where
        model_g uses omega_r = omega_r_g
        model_e uses omega_r = omega_r_e
    and C, omega_p, J, kappa are shared.
    """
    x_g = np.asarray(x_g, dtype=float).ravel()
    x_e = np.asarray(x_e, dtype=float).ravel()
    delta_g = np.asarray(delta_g, dtype=float).ravel()
    delta_e = np.asarray(delta_e, dtype=float).ravel()
    weights_g = np.asarray(weights_g, dtype=float).ravel()
    weights_e = np.asarray(weights_e, dtype=float).ravel()
    theta0 = np.asarray(theta0, dtype=float).ravel()

    if not (x_g.size == delta_g.size == weights_g.size):
        raise ValueError("x_g, delta_g and weights_g must have the same length.")
    if not (x_e.size == delta_e.size == weights_e.size):
        raise ValueError("x_e, delta_e and weights_e must have the same length.")
    if theta0.size != 6:
        raise ValueError(
            "theta0 must have length 6: "
            "[C0, omega_r_g0, omega_r_e0, omega_p0, J0, kappa0]."
        )
    if x_g.size < 6 or x_e.size < 6:
        raise ValueError("At least 6 data points are required.")

    if not np.all(np.isfinite(x_g)):
        raise ValueError("x_g contains non-finite values.")
    if not np.all(np.isfinite(x_e)):
        raise ValueError("x_e contains non-finite values.")
    if not np.all(np.isfinite(delta_g)):
        raise ValueError("delta_g contains non-finite values.")
    if not np.all(np.isfinite(delta_e)):
        raise ValueError("delta_e contains non-finite values.")
    if not np.all(np.isfinite(weights_g)):
        raise ValueError("weights_g contains non-finite values.")
    if not np.all(np.isfinite(weights_e)):
        raise ValueError("weights_e contains non-finite values.")
    if not np.all(np.isfinite(theta0)):
        raise ValueError("theta0 contains non-finite values.")

    if np.any(weights_g < 0):
        raise ValueError("weights_g must be nonnegative.")
    if np.any(weights_e < 0):
        raise ValueError("weights_e must be nonnegative.")

    _, _, _, _, J0, kappa0 = theta0

    if bounds is None:
        xmin = float(min(np.min(x_g), np.min(x_e)))
        xmax = float(max(np.max(x_g), np.max(x_e)))
        xspan = max(xmax - xmin, 1e-6)

        lower = np.array(
            [
                -np.inf,  # C
                xmin - 2.0 * xspan,  # omega_r_g
                xmin - 2.0 * xspan,  # omega_r_e
                xmin - 2.0 * xspan,  # omega_p
                1e-12,  # J > 0
                1e-12,  # kappa > 0
            ],
            dtype=float,
        )
        upper = np.array(
            [
                0.0,  # often C < 0, allow up to 0
                xmax + 2.0 * xspan,
                xmax + 2.0 * xspan,
                xmax + 2.0 * xspan,
                10.0 * xspan + abs(J0) + 1.0,
                10.0 * xspan + abs(kappa0) + 1.0,
            ],
            dtype=float,
        )
        bounds = (lower, upper)
    else:
        lower = np.asarray(bounds[0], dtype=float).ravel()
        upper = np.asarray(bounds[1], dtype=float).ravel()
        if lower.size != 6 or upper.size != 6:
            raise ValueError("bounds must be a pair of length-6 arrays.")

    def residuals(theta: np.ndarray) -> np.ndarray:
        C, omega_r_g, omega_r_e, omega_p, J, kappa = theta

        model_g = filtered_ckp_model(
            x_g,
            C=C,
            omega_r=omega_r_g,
            omega_p=omega_p,
            J=J,
            kappa=kappa,
        )
        model_e = filtered_ckp_model(
            x_e,
            C=C,
            omega_r=omega_r_e,
            omega_p=omega_p,
            J=J,
            kappa=kappa,
        )

        resid_g = weights_g * (model_g - delta_g)
        resid_e = weights_e * (model_e - delta_e)
        return np.concatenate([resid_g, resid_e])

    result = least_squares(
        residuals,
        x0=theta0,
        bounds=bounds,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
    )

    C, omega_r_g, omega_r_e, omega_p, J, kappa = result.x

    return FilteredCKPFitResult(
        C=float(C),
        omega_r_g=float(omega_r_g),
        omega_r_e=float(omega_r_e),
        omega_p=float(omega_p),
        J=float(J),
        kappa=float(kappa),
        success=bool(result.success),
        cost=float(result.cost),
        message=str(result.message),
        nfev=int(result.nfev),
        result_raw=result,
    )


def filtered_ckp_photon_number(
    x: ArrayLike,
    *,
    omega_r: float,
    omega_p: float,
    J: float,
    kappa: float,
    A2: float,
) -> np.ndarray:
    """Photon number in the readout resonator."""
    x = np.asarray(x, dtype=float)
    delta_r = x - omega_r
    delta_p = x - omega_p
    denom = (delta_r * delta_p - J**2) ** 2 + ((delta_r * kappa / 2.0) ** 2)
    return (J**2 * kappa * A2) / denom


def filtered_ckp_alpha(
    x: ArrayLike,
    *,
    omega_r: float,
    omega_p: float,
    J: float,
    kappa: float,
    A: complex | float,
) -> np.ndarray:
    """Steady-state readout-resonator amplitude alpha_ss for the filtered CKP model."""
    x = np.asarray(x, dtype=float)

    delta_r = omega_r - x
    delta_p = omega_p - x

    denom = delta_r * (delta_p - 1j * kappa / 2.0) - J**2

    return -J * np.sqrt(kappa) * A / denom


def estimate_optimal_readout_frequency_from_ckp(
    *,
    frequency_range: ArrayLike,
    omega_r_g: float,
    omega_r_e: float,
    omega_p: float,
    J: float,
    kappa: float,
    A: complex | float,
    n_points: int = 4000,
    plot: bool = True,
    show_animation: bool = False,
    save_image: bool = True,
    save_animation: bool = False,
    output_dir: str | Path | None = None,
    name: str = "filtered_ckp_readout_frequency",
) -> Result:
    """
    Estimate the optimal readout frequency from fitted filtered-CKP parameters.

    The optimal frequency is defined as the frequency that maximizes
    |alpha_e - alpha_g|.

    Parameters
    ----------
    frequency_range : ArrayLike
        Two-element range [f_min, f_max] or an explicit frequency array.

    omega_r_g : float
        Readout-resonator frequency for the qubit ground state.

    omega_r_e : float
        Readout-resonator frequency for the qubit excited state.

    omega_p : float
        Purcell-filter frequency.

    J : float
        Coupling between readout resonator and Purcell filter.

    kappa : float
        Purcell-filter linewidth.

    A : complex or float, optional
        Input drive amplitude.

    n_points : int, optional
        Number of frequency points used when frequency_range is given as
        [f_min, f_max].

    plot : bool, optional
        If True, show |alpha_e-alpha_g| and IQ trajectory plots.

    show_animation : bool, optional
        If True, display the generated animation.

    save_image : bool, optional
        If True, save the generated figures.

    save_animation : bool, optional
        If True, save an HTML animation of alpha_g and alpha_e trajectories.

    output_dir : str or Path, optional
        Directory used for saving figures and animation.

    name : str, optional
        Base name for saved files.

    Returns
    -------
    Result
        Result object containing the optimal readout frequency and
        calculated alpha trajectories.
    """
    frequency_range = np.asarray(frequency_range, dtype=float).ravel()

    if frequency_range.size == 2:
        x = np.linspace(frequency_range[0], frequency_range[1], n_points)
    elif frequency_range.size >= 3:
        x = frequency_range
    else:
        raise ValueError("frequency_range must be [f_min, f_max] or an array.")

    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_g = filtered_ckp_alpha(
        x,
        omega_r=omega_r_g,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
        A=A,
    )
    alpha_e = filtered_ckp_alpha(
        x,
        omega_r=omega_r_e,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
        A=A,
    )

    alpha_diff = alpha_e - alpha_g
    separation = np.abs(alpha_diff)

    idx_opt = int(np.argmax(separation))
    readout_frequency_opt = float(x[idx_opt])
    max_separation = float(separation[idx_opt])

    fig_sep = viz.make_figure()
    fig_sep.add_scatter(
        x=x,
        y=separation,
        mode="lines",
        name="|alpha_e - alpha_g|",
    )
    fig_sep.add_scatter(
        x=[readout_frequency_opt],
        y=[max_separation],
        mode="markers",
        name="optimal readout frequency",
        marker=dict(size=9),
    )
    fig_sep.add_vline(
        x=readout_frequency_opt,
        line_dash="dash",
        annotation_text=f"opt = {readout_frequency_opt:.6f} GHz",
    )
    fig_sep.update_layout(
        title="Readout-frequency optimization from filtered CKP",
        xaxis_title="Readout drive frequency (GHz)",
        yaxis_title="|alpha_e - alpha_g|",
        width=800,
        height=450,
    )

    if plot:
        fig_sep.show()

    if save_image:
        viz.save_figure(
            fig_sep,
            name=f"{name}_separation",
            width=800,
            height=450,
        )

    fig_iq = viz.make_figure()
    fig_iq.add_scatter(
        x=np.real(alpha_g),
        y=np.imag(alpha_g),
        mode="lines",
        name="alpha_g trajectory",
    )
    fig_iq.add_scatter(
        x=np.real(alpha_e),
        y=np.imag(alpha_e),
        mode="lines",
        name="alpha_e trajectory",
    )
    fig_iq.add_scatter(
        x=[np.real(alpha_g[idx_opt])],
        y=[np.imag(alpha_g[idx_opt])],
        mode="markers",
        name="alpha_g at optimum",
        marker=dict(size=9),
    )
    fig_iq.add_scatter(
        x=[np.real(alpha_e[idx_opt])],
        y=[np.imag(alpha_e[idx_opt])],
        mode="markers",
        name="alpha_e at optimum",
        marker=dict(size=9),
    )
    fig_iq.add_scatter(
        x=[np.real(alpha_g[idx_opt]), np.real(alpha_e[idx_opt])],
        y=[np.imag(alpha_g[idx_opt]), np.imag(alpha_e[idx_opt])],
        mode="lines",
        name="max separation vector",
    )
    fig_iq.update_layout(
        title="Alpha trajectories in complex plane",
        xaxis_title="Re(alpha)",
        yaxis_title="Im(alpha)",
        width=600,
        height=600,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    if plot:
        fig_iq.show()

    if save_image:
        viz.save_figure(
            fig_iq,
            name=f"{name}_iq_trajectory",
            width=600,
            height=600,
        )

    animation_path = None

    if show_animation or save_animation:
        step = max(1, len(x) // 300)
        frame_indices = np.arange(0, len(x), step)
        if frame_indices[-1] != idx_opt:
            frame_indices = np.unique(np.append(frame_indices, idx_opt))

        frames = [
            go.Frame(
                data=[
                    go.Scatter(
                        x=np.real(alpha_g[: i + 1]),
                        y=np.imag(alpha_g[: i + 1]),
                        mode="lines",
                        name="alpha_g trajectory",
                    ),
                    go.Scatter(
                        x=np.real(alpha_e[: i + 1]),
                        y=np.imag(alpha_e[: i + 1]),
                        mode="lines",
                        name="alpha_e trajectory",
                    ),
                    go.Scatter(
                        x=[np.real(alpha_g[i])],
                        y=[np.imag(alpha_g[i])],
                        mode="markers",
                        name="alpha_g",
                        marker=dict(size=10),
                    ),
                    go.Scatter(
                        x=[np.real(alpha_e[i])],
                        y=[np.imag(alpha_e[i])],
                        mode="markers",
                        name="alpha_e",
                        marker=dict(size=10),
                    ),
                    go.Scatter(
                        x=[np.real(alpha_g[i]), np.real(alpha_e[i])],
                        y=[np.imag(alpha_g[i]), np.imag(alpha_e[i])],
                        mode="lines",
                        name="alpha_e - alpha_g",
                    ),
                ],
                name=str(i),
                layout=go.Layout(
                    title_text=(
                        "Alpha trajectories in complex plane "
                        f"(f = {x[i]:.6f} GHz, "
                        f"|diff| = {separation[i]:.4g})"
                    )
                ),
            )
            for i in frame_indices
        ]

        x_all = np.concatenate([np.real(alpha_g), np.real(alpha_e)])
        y_all = np.concatenate([np.imag(alpha_g), np.imag(alpha_e)])
        x_pad = 0.05 * max(np.ptp(x_all), 1e-12)
        y_pad = 0.05 * max(np.ptp(y_all), 1e-12)

        fig_anim = go.Figure(
            data=frames[0].data,
            frames=frames,
        )
        fig_anim.update_layout(
            title="Alpha trajectories in complex plane",
            xaxis_title="Re(alpha)",
            yaxis_title="Im(alpha)",
            width=650,
            height=650,
            xaxis=dict(
                range=[float(np.min(x_all) - x_pad), float(np.max(x_all) + x_pad)]
            ),
            yaxis=dict(
                range=[float(np.min(y_all) - y_pad), float(np.max(y_all) + y_pad)],
                scaleanchor="x",
                scaleratio=1,
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=40, redraw=True),
                                    fromcurrent=True,
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
        )

        if save_animation:
            animation_path = output_dir / f"{name}_iq_animation.html"

            fig_anim.write_html(
                str(animation_path),
                include_plotlyjs="cdn",
                auto_play=False,
            )

        if show_animation:
            fig_anim.show()

    return Result(
        data={
            "readout_frequency_opt": readout_frequency_opt,
            "max_alpha_separation": max_separation,
            "frequency_range": x,
            "alpha_g": alpha_g,
            "alpha_e": alpha_e,
            "alpha_diff": alpha_diff,
            "alpha_separation": separation,
            "omega_r_g": omega_r_g,
            "omega_r_e": omega_r_e,
            "omega_p": omega_p,
            "J": J,
            "kappa": kappa,
            "A": A,
            "fig_separation": fig_sep,
            "fig_iq": fig_iq,
            "animation_path": str(animation_path)
            if animation_path is not None
            else None,
        }
    )
