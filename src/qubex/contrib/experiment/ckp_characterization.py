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
        resonator_drive_amplitude = (
            exp.ctx.params.get_readout_amplitude(exp.ctx.resolve_qubit_label(target))
            / 2
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
    n_shots: int | None = None,
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
    fitting to estimate readout resonator parameters.

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

    n_shots : int, optional
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

        When early-stop mode is enabled but no early stop is triggered,
        additional resonator-frequency points are adaptively measured
        around the minimum fitted qubit resonance frequency.

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
        - ``heatmap_data``
        - ``fit_results``

    Notes
    -----
    Frequencies are assumed to use the backend frequency unit
    (typically GHz). Durations are in ns.
    """
    if qubit_initial_state not in {"0", "1"}:
        qubit_initial_state = "0"
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if plot is None:
        plot = True
    if verbose is None:
        verbose = False
    if save_image is None:
        save_image = True
    if qubit_detuning_range is None:
        left = np.linspace(-0.055, -0.04, 3, endpoint=False)
        medium_left = np.linspace(-0.04, -0.03, 4, endpoint=False)
        center = np.linspace(-0.03, 0.01, 24, endpoint=False)
        right = np.linspace(0.01, 0.025, 3)
        qubit_detuning_range = np.concatenate([left, medium_left, center, right])
    qubit_detuning_range = np.asarray(qubit_detuning_range, dtype=float)
    if resonator_detuning_range is None:
        u = np.linspace(-1, 1, 50)
        beta = 0.95
        resonator_detuning_range = 0.12 * np.arctanh(beta * u) / np.arctanh(beta)
    resonator_detuning_range = np.asarray(resonator_detuning_range, dtype=float)
    if resonator_drive_amplitude is None:
        resonator_drive_amplitude = (
            exp.ctx.params.get_readout_amplitude(exp.ctx.resolve_qubit_label(target))
            / 2
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

    def measure_one_resonator_point(resonator_detuning: float) -> None:
        nonlocal early_stopped

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
                n_shots=n_shots,
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
            return
        fit_results.append(fit_result)
        f0 = fit_result.get("f0", np.nan)
        f0_err = fit_result.get("f0_err", np.nan)
        r2 = fit_result.get("r2", np.nan)
        if not (
            np.isfinite(f0) and np.isfinite(f0_err) and np.isfinite(r2) and r2 > 0.5
        ):
            print(
                f"[WARN] invalid fit at resonator_detuning={resonator_detuning:.6f}"
                f"f0={f0}, err={f0_err}, r2={r2:.3f}"
            )
            return
        valid_resonator_frequencies.append(f_resonator + resonator_detuning)
        qubit_resonance_frequencies.append(f0)
        qubit_resonance_frequency_errors.append(f0_err)
        if enable_early_stop and f0_lower_frequency_limit is not None:
            if f0 < f0_lower_frequency_limit:
                early_stopped = True

    for resonator_detuning in tqdm(resonator_detuning_range):
        measure_one_resonator_point(float(resonator_detuning))
        if early_stopped:
            break

    if enable_early_stop and not early_stopped:
        if len(qubit_resonance_frequencies) >= 3:
            f0_arr = np.asarray(qubit_resonance_frequencies, dtype=float)
            fr_valid_arr = np.asarray(valid_resonator_frequencies, dtype=float)
            full_fr = np.asarray(resonator_frequency_range, dtype=float)
            idx_min = int(np.argmin(f0_arr))
            f_center = fr_valid_arr[idx_min]
            idx_center_full = int(np.argmin(np.abs(full_fr - f_center)))

            if (
                0 < idx_min < len(fr_valid_arr) - 1
                and 0 < idx_center_full < len(full_fr) - 1
            ):
                f_left_full = full_fr[idx_center_full - 1]
                f_right_full = full_fr[idx_center_full + 1]
                f0_left = f0_arr[idx_min - 1]
                f0_right = f0_arr[idx_min + 1]

                if f0_left <= f0_right:
                    f_neighbor = f_left_full
                    f_other = f_right_full
                else:
                    f_neighbor = f_right_full
                    f_other = f_left_full

                main_points = np.linspace(f_center, f_neighbor, 5)[1:3]
                other_point = np.linspace(f_center, f_other, 5)[1:2]
                extra_frequencies = np.concatenate([main_points, other_point])

                for f_extra in tqdm(extra_frequencies):
                    resonator_detuning_extra = float(f_extra - f_resonator)
                    measure_one_resonator_point(resonator_detuning_extra)
                    if early_stopped:
                        break

    data = np.array(result2d)
    if qubit_initial_state == "1":
        data *= -1

    measured_resonator_frequency_range = np.asarray(
        measured_resonator_frequencies,
        dtype=float,
    )

    idx_all = np.argsort(measured_resonator_frequency_range)
    measured_resonator_frequency_range = measured_resonator_frequency_range[idx_all]
    data = data[idx_all]
    fit_results = [fit_results[i] for i in idx_all]

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
            "heatmap_data": data,
            "fit_results": fit_results,
        },
        figure=fig,
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
    n_shots: int | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    enable_rough_search: bool | None = None,
    target_min_qubit_detuning: float | None = None,
    max_rough_search_reductions: int | None = None,
    max_rough_search_increases: int | None = None,
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

    After fitting, the routine estimates the optimal readout
    frequency by maximizing |beta_e-beta_g| with photon-number-limited drive amplitude.

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

    n_shots : int, optional
        Number of measurement shots used for one CKP sweep point.

        If None, uses DEFAULT_SHOTS.

    plot : bool, optional
        If True, display the fitted CKP traces and estimated parameters.

    save_image : bool, optional
        If True, save the generated fit figure.

    enable_rough_search : bool, optional
        If True, perform a coarse pre-scan to automatically adjust
        resonator drive amplitude before the main CKP measurement.

        If None, defaults to True.

    target_min_qubit_detuning : float, optional
        Target minimum qubit-frequency detuning (GHz) used to tune
        the resonator drive amplitude during rough search.

    max_rough_search_reductions : int, optional
        Maximum number of times the resonator drive amplitude
        may be reduced during rough search.

    max_rough_search_increases : int, optional
        Maximum number of times the resonator drive amplitude
        may be increased during rough search.

    Returns
    -------
    Result
        Result object containing raw CKP measurements, fitted parameters,
        derived quantities, readout optimization results, and generated figures.

        Expected fields in ``data`` include:

        Experiment configuration
        ------------------------
        - ``target``
        - ``resonator_detuning_range``
        - ``resonator_drive_amplitude``

        Rough-search results
        --------------------
        - ``rough_search_results``
        - ``rough_search_reduce_trials``
        - ``rough_search_increase_trials``

        Raw CKP measurement results
        ---------------------------
        - ``result_0``
        - ``result_1``
        - ``result_0_no_drive``
        - ``result_1_no_drive``

        Processed CKP traces
        --------------------
        - ``resonator_frequencies_g``
        - ``resonator_frequencies_e``
        - ``delta_g``
        - ``delta_e``
        - ``weights_g``
        - ``weights_e``

        Initial parameter estimates
        ---------------------------
        - ``initial_guess``

        Estimated readout parameters
        ----------------------------
        - ``omega_r_g``
        - ``omega_r_e``
        - ``omega_p``
        - ``J``
        - ``kappa``
        - ``chi``
        - ``C``
        - ``A2``  ( = |A|² )
        - ``A``
        - ``purcell_readout_detuning``

        Parameter uncertainties and fit statistics
        ------------------------------------------
        - ``omega_r_g_error``
        - ``omega_r_e_error``
        - ``omega_p_error``
        - ``J_error``
        - ``kappa_error``
        - ``C_error``
        - ``chi_error``
        - ``A2_error``
        - ``reduced_chi2``
        - ``r2``
        - ``covariance``
        - ``fit_result``

        Photon-number metrics
        ---------------------
        - ``ng_max``
        - ``ne_max``
        - ``x_ng_max``
        - ``x_ne_max``
        - ``n_crit``

        Readout optimization results
        ----------------------------
        - ``optimal_readout_frequency``
        - ``n_limited_A2_at_optimal_frequency``
        - ``n_limited_readout_amplitude``
        - ``readout_optimization_result``

        Figures
        -------
        The primary CKP fit figure is stored in ``figure``.

        Additional figures are stored in ``figures``:

        - ``ckp_fit``
        - ``ckp_heatmap_g``
        - ``ckp_heatmap_e``
        - ``readout_optimization``

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

    Approximate critical photon number estimated from
    n_crit ≈ Δ_qr / (4χ).
    """
    if enable_rough_search is None:
        enable_rough_search = True
    if enable_rough_search:
        if target_min_qubit_detuning is None:
            target_min_qubit_detuning = -0.02
        if target_min_qubit_detuning >= 0:
            raise ValueError("target_min_qubit_detuning must be negative.")
        f0_lower_qubit_detuning_limit = target_min_qubit_detuning * 1.5
        f0_upper_qubit_detuning_limit = target_min_qubit_detuning * 0.5
    if max_rough_search_reductions is None:
        max_rough_search_reductions = 4
    if max_rough_search_increases is None:
        max_rough_search_increases = 4
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if resonator_detuning_range is None:
        u = np.linspace(-1, 1, 50)
        beta = 0.95
        resonator_detuning_range = 0.12 * np.arctanh(beta * u) / np.arctanh(beta)
    resonator_detuning_range = np.asarray(resonator_detuning_range, dtype=float)
    if resonator_drive_amplitude is None:
        resonator_drive_amplitude = (
            exp.ctx.params.get_readout_amplitude(exp.ctx.resolve_qubit_label(target))
            / 2
        )
    verbose = False

    rough_search_results = []
    n_reduces = 0
    n_increases = 0

    if enable_rough_search:
        resonator_detuning_range_rough_search = resonator_detuning_range[::4]
        if len(resonator_detuning_range_rough_search) == 0:
            resonator_detuning_range_rough_search = resonator_detuning_range
        n_shots_rough_search = max(1, n_shots // 4)

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
                n_shots=n_shots_rough_search,
                plot=False,
                verbose=verbose,
                save_image=False,
                enable_early_stop=True,
                f0_lower_qubit_detuning_limit=f0_lower_qubit_detuning_limit,
            )
            rough_search_results.append(result_0_rough_search)

            if result_0_rough_search.data["early_stopped"]:
                resonator_drive_amplitude *= np.sqrt(0.5)
                n_reduces += 1
                print(
                    "[ROUGH SEARCH] ac Stark shift too large. "
                    f"Reduce resonator_drive_amplitude -> {resonator_drive_amplitude:.6g} "
                )
                if n_reduces >= max_rough_search_reductions:
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
            f0_upper_frequency_limit = f_qubit_rough + f0_upper_qubit_detuning_limit
            has_below_upper = np.any(f0_list < f0_upper_frequency_limit)
            if not has_below_upper:
                resonator_drive_amplitude *= np.sqrt(2.0)
                n_increases += 1
                print(
                    "[ROUGH SEARCH] ac Stark shift too small. "
                    f"Increase resonator_drive_amplitude -> {resonator_drive_amplitude:.6g} "
                )
                if n_increases >= max_rough_search_increases:
                    print(
                        "[ROUGH SEARCH] Max increase trial reached. "
                        "Proceeding with current amplitude."
                    )
                    break
                continue

            observed_min_qubit_detuning = float(np.min(f0_list) - f_qubit_rough)
            print(
                "[ROUGH SEARCH] Auto-adjust amplitude\n"
                f"    observed shift : {1000 * observed_min_qubit_detuning:.2f} MHz"
            )
            if (
                target_min_qubit_detuning is not None
                and resonator_drive_amplitude is not None
                and observed_min_qubit_detuning < 0
                and target_min_qubit_detuning < 0
            ):
                scale = np.sqrt(target_min_qubit_detuning / observed_min_qubit_detuning)
                scale = np.clip(scale, 0.5, 1.5)
                resonator_drive_amplitude *= float(scale)
                print(
                    f"    target shift   : {1000 * target_min_qubit_detuning:.2f} MHz\n"
                    f"    scale factor   : {scale:.3f}"
                )
            print(f"    amplitude      : {resonator_drive_amplitude:.6g}")
            break

    n_shots_no_drive = 4 * n_shots
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
        n_shots=n_shots_no_drive,
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
        n_shots=n_shots_no_drive,
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
        n_shots=n_shots,
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
        n_shots=n_shots,
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
    chi = float(fit.chi)
    A2 = float(fit.A2)
    purcell_readout_detuning = omega_p - 0.5 * (omega_r_g + omega_r_e)

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
    f_q = 0.5 * (offset_g + offset_e)
    f_r = 0.5 * (omega_r_g + omega_r_e)
    delta_qr = f_q - f_r
    if np.isclose(chi, 0.0):
        n_crit = np.nan
    else:
        n_crit = abs(delta_qr / (4.0 * chi))

    fig = None

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
            width=1000,
            height=500,
            margin=dict(l=85, r=230, t=60, b=70),
            legend=dict(
                x=1.12,
                y=1.0,
                xanchor="left",
                yanchor="top",
            ),
            title_x=0.43,
        )

        if plot:
            fig.show()
            print(f"=== Estimated Readout Parameters : {target} ===")
            print(f"ωr^g    = {omega_r_g:.6f} ± {fit.omega_r_g_error:.6f} GHz")
            print(f"ωr^e    = {omega_r_e:.6f} ± {fit.omega_r_e_error:.6f} GHz")
            print(f"ωp      = {omega_p:.6f} ± {fit.omega_p_error:.6f} GHz")
            print(f"J       = {1000 * J:.3f} ± {1000 * fit.J_error:.3f} MHz")
            print(f"κ       = {1000 * kappa:.3f} ± {1000 * fit.kappa_error:.3f} MHz")
            print(f"χ       = {1000 * chi:.3f} ± {1000 * fit.chi_error:.3f} MHz")
            print(f"|A|²    = {1000 * A2:.3f} ± {1000 * fit.A2_error:.3f} MHz")
            print(f"χ²_red  = {fit.reduced_chi2:.3g}")
            print(f"R²      = {fit.r2:.3g}")
            print(f"n_g,max = {ng_max:.3f} @ {x_ng_max:.6f} GHz")
            print(f"n_e,max = {ne_max:.3f} @ {x_ne_max:.6f} GHz")
            print(f"n_c     ≈ {n_crit:.3f}")

        if save_image:
            viz.save_figure(
                fig,
                name=f"filtered_ckp_fit_{target}",
                width=1000,
                height=500,
            )

    readout_opt_result = estimate_optimal_readout_frequency_from_ckp(
        frequency_range=[x_dense_min, x_dense_max],
        omega_r_g=omega_r_g,
        omega_r_e=omega_r_e,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
        n_crit=n_crit,
        plot=plot,
        save_image=save_image,
        name=f"readout_opt_{target}",
    )

    optimal_readout_frequency = readout_opt_result.data["optimal_readout_frequency"]
    n_limited_A2_at_optimal_frequency = readout_opt_result.data[
        "n_limited_A2_at_optimal_frequency"
    ]
    if A2 <= 0 or not np.isfinite(A2):
        n_limited_readout_amplitude = np.nan
    else:
        n_limited_readout_amplitude = resonator_drive_amplitude * np.sqrt(
            n_limited_A2_at_optimal_frequency / A2
        )

    figures: dict[str, go.Figure] = {}
    if fig is not None:
        figures["ckp_fit"] = fig
    if result_0.figure is not None:
        figures["ckp_heatmap_g"] = result_0.figure
    if result_1.figure is not None:
        figures["ckp_heatmap_e"] = result_1.figure
    if readout_opt_result.figure is not None:
        figures["readout_optimization"] = readout_opt_result.figure

    return Result(
        data={
            "target": target,
            "resonator_detuning_range": resonator_detuning_range,
            "resonator_drive_amplitude": resonator_drive_amplitude,
            "rough_search_results": rough_search_results,
            "rough_search_reduce_trials": n_reduces,
            "rough_search_increase_trials": n_increases,
            "result_0": result_0,
            "result_1": result_1,
            "result_0_no_drive": result_0_no_drive,
            "result_1_no_drive": result_1_no_drive,
            "resonator_frequencies_g": x_g,
            "resonator_frequencies_e": x_e,
            "delta_g": delta_g,
            "delta_e": delta_e,
            "weights_g": weights_g,
            "weights_e": weights_e,
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
            "omega_r_g": omega_r_g,
            "omega_r_e": omega_r_e,
            "omega_p": omega_p,
            "J": J,
            "kappa": kappa,
            "chi": chi,
            "C": C,
            "A2": A2,
            "A": fit.A,
            "purcell_readout_detuning": purcell_readout_detuning,
            "omega_r_g_error": fit.omega_r_g_error,
            "omega_r_e_error": fit.omega_r_e_error,
            "omega_p_error": fit.omega_p_error,
            "J_error": fit.J_error,
            "kappa_error": fit.kappa_error,
            "C_error": fit.C_error,
            "chi_error": fit.chi_error,
            "A2_error": fit.A2_error,
            "reduced_chi2": fit.reduced_chi2,
            "r2": fit.r2,
            "covariance": fit.covariance,
            "fit_result": fit,
            "ng_max": ng_max,
            "ne_max": ne_max,
            "x_ng_max": x_ng_max,
            "x_ne_max": x_ne_max,
            "n_crit": n_crit,
            "optimal_readout_frequency": optimal_readout_frequency,
            "n_limited_A2_at_optimal_frequency": n_limited_A2_at_optimal_frequency,
            "n_limited_readout_amplitude": n_limited_readout_amplitude,
            "readout_optimization_result": readout_opt_result,
        },
        figure=fig,
        figures=figures,
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
    chi: float
    A2: float
    A: float
    C_error: float
    omega_r_g_error: float
    omega_r_e_error: float
    omega_p_error: float
    J_error: float
    kappa_error: float
    chi_error: float
    A2_error: float
    success: bool
    cost: float
    chi2: float
    reduced_chi2: float
    r2: float
    dof: int
    message: str
    nfev: int
    result_raw: object
    covariance: np.ndarray | None = None


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

    chi = (omega_r_e - omega_r_g) / 2.0
    if np.isclose(chi, 0.0):
        A2 = np.nan
        A = np.nan
    else:
        A2 = C / (4.0 * chi)
        if A2 < 0:
            A = np.nan
        else:
            A = np.sqrt(A2)

    weighted_residuals = np.asarray(result.fun, dtype=float)
    n_data = weighted_residuals.size
    n_param = len(result.x)
    dof = max(n_data - n_param, 1)
    chi2 = float(np.sum(weighted_residuals**2))
    reduced_chi2 = chi2 / dof

    y_all = np.concatenate([delta_g, delta_e])
    w_all = np.concatenate([weights_g, weights_e])
    model_g = filtered_ckp_model(
        x_g, C=C, omega_r=omega_r_g, omega_p=omega_p, J=J, kappa=kappa
    )
    model_e = filtered_ckp_model(
        x_e, C=C, omega_r=omega_r_e, omega_p=omega_p, J=J, kappa=kappa
    )
    f_all = np.concatenate([model_g, model_e])
    y_mean = np.sum(w_all**2 * y_all) / np.sum(w_all**2)
    ss_res = np.sum((w_all * (y_all - f_all)) ** 2)
    ss_tot = np.sum((w_all * (y_all - y_mean)) ** 2)
    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    perr = np.full(6, np.nan, dtype=float)
    cov = None

    chi_error = np.nan
    A2_error = np.nan

    try:
        jac = result.jac
        n_data = jac.shape[0]
        n_param = jac.shape[1]
        fit_dof = max(0, n_data - n_param)

        if fit_dof > 0:
            resid = result.fun
            rss = np.sum(resid**2)
            s_sq = rss / fit_dof
            jtj = jac.T @ jac

            cov = np.linalg.pinv(jtj) * s_sq
            perr = np.sqrt(np.diag(cov))

            chi_error = 0.5 * np.sqrt(cov[1, 1] + cov[2, 2] - 2.0 * cov[1, 2])

            if np.isfinite(A2):
                grad_A2 = np.zeros(6, dtype=float)
                grad_A2[0] = 1.0 / (4.0 * chi)
                grad_A2[1] = C / (8.0 * chi**2)
                grad_A2[2] = -C / (8.0 * chi**2)

                A2_var = float(grad_A2 @ cov @ grad_A2)
                A2_error = np.sqrt(max(A2_var, 0.0))

    except np.linalg.LinAlgError:
        pass

    return FilteredCKPFitResult(
        C=float(C),
        omega_r_g=float(omega_r_g),
        omega_r_e=float(omega_r_e),
        omega_p=float(omega_p),
        J=float(J),
        kappa=float(kappa),
        chi=float(chi),
        chi_error=float(chi_error),
        A2=float(A2),
        A2_error=float(A2_error),
        A=float(A),
        C_error=float(perr[0]),
        omega_r_g_error=float(perr[1]),
        omega_r_e_error=float(perr[2]),
        omega_p_error=float(perr[3]),
        J_error=float(perr[4]),
        kappa_error=float(perr[5]),
        covariance=cov,
        success=bool(result.success),
        cost=float(result.cost),
        chi2=float(chi2),
        reduced_chi2=float(reduced_chi2),
        r2=float(r2),
        dof=int(dof),
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
    A: complex | float | ArrayLike,
) -> np.ndarray:
    """Steady-state readout-resonator amplitude alpha_ss for the filtered CKP model."""
    x = np.asarray(x, dtype=float)

    delta_r = omega_r - x
    delta_p = omega_p - x

    denom = delta_r * (delta_p - 1j * kappa / 2.0) - J**2
    A_arr = np.asarray(A, dtype=complex)

    return -J * np.sqrt(kappa) * A_arr / denom


def filtered_ckp_beta(
    x: ArrayLike,
    *,
    omega_r: float,
    omega_p: float,
    J: float,
    kappa: float,
    A: complex | float | ArrayLike,
) -> np.ndarray:
    """Steady-state Purcell-filter amplitude beta_ss for the filtered CKP model."""
    x = np.asarray(x, dtype=float)

    delta_r = omega_r - x
    delta_p = omega_p - x

    denom = delta_r * (delta_p - 1j * kappa / 2.0) - J**2
    A_arr = np.asarray(A, dtype=complex)

    return delta_r * np.sqrt(kappa) * A_arr / denom


def estimate_optimal_readout_frequency_from_ckp(
    *,
    frequency_range: ArrayLike,
    omega_r_g: float,
    omega_r_e: float,
    omega_p: float,
    J: float,
    kappa: float,
    n_crit: float,
    n_points: int = 4000,
    plot: bool = True,
    save_image: bool = True,
    name: str = "filtered_ckp_readout_frequency",
) -> Result:
    """
    Estimate the optimal readout frequency from fitted filtered-CKP parameters.

    The optimal readout frequency is defined as the frequency that maximizes

        |beta_e - beta_g|

    under the constraint

        max(|alpha_g|^2, |alpha_e|^2) = 0.1 * n_crit

    at each drive frequency.

    For every frequency point, the drive amplitude A is automatically
    rescaled so that the larger intracavity photon number of the
    ground-state and excited-state branches reaches the allowed maximum
    photon number.

    Parameters
    ----------
    frequency_range : ArrayLike
        Either:

        - a two-element range [f_min, f_max], or
        - an explicit frequency array.

        Frequencies are assumed to use the same unit as the fitted
        resonator parameters (typically GHz).

    omega_r_g : float
        Readout-resonator frequency for the qubit ground state.

    omega_r_e : float
        Readout-resonator frequency for the qubit excited state.

    omega_p : float
        Purcell-filter resonance frequency.

    J : float
        Coupling strength between the readout resonator and
        the Purcell filter.

    kappa : float
        Purcell-filter linewidth.

    n_crit : float
        Critical photon number of the readout resonator.

    n_points : int, optional
        Number of frequency points used when frequency_range is specified
        as [f_min, f_max].

    plot : bool, optional
        If True, display the readout optimization figure.

    save_image : bool, optional
        If True, save the generated optimization figure.

    name : str, optional
        File name used when saving the figure.

    Returns
    -------
    Result
        Result object containing:

        - ``optimal_readout_frequency``
            Frequency maximizing |beta_e - beta_g|.

        - ``max_beta_separation``
            Maximum value of |beta_e - beta_g|.

        - ``n_limited_A2_at_optimal_frequency``
            Allowed drive power |A|² at the optimal frequency under the
            photon-number constraint.

        - ``frequency_range``
            Frequency array used for optimization.

        - ``beta_g``, ``beta_e``
            Output-field responses for the ground-state and
            excited-state branches.

        - ``A_allowed``
            Frequency-dependent allowed drive amplitude satisfying

                max(|alpha_g|², |alpha_e|²) = 0.1 * n_crit

        - ``beta_separation``
            |beta_e - beta_g| evaluated over the frequency range.

        - ``n_max_allowed``
            Maximum allowed intracavity photon number
            (= 0.1 * n_crit).

    Notes
    -----
    The allowed drive amplitude is computed from

        A_allowed(f)
        =
        sqrt(
            n_max_allowed
            /
            max(|alpha_g(f)|², |alpha_e(f)|²)
        )

    so that the larger photon population between the two qubit states
    is limited to 0.1 * n_crit at every frequency point.
    """
    frequency_range = np.asarray(frequency_range, dtype=float).ravel()

    if frequency_range.size == 2:
        x = np.linspace(frequency_range[0], frequency_range[1], n_points)
    elif frequency_range.size >= 3:
        x = frequency_range
    else:
        raise ValueError("frequency_range must be [f_min, f_max] or an array.")

    if not np.isfinite(n_crit) or n_crit <= 0:
        raise ValueError("n_crit must be a positive finite value.")

    n_max_allowed = 0.1 * n_crit

    alpha_g_unit = filtered_ckp_alpha(
        x,
        omega_r=omega_r_g,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
        A=1.0,
    )
    alpha_e_unit = filtered_ckp_alpha(
        x,
        omega_r=omega_r_e,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
        A=1.0,
    )

    n_g_unit = np.abs(alpha_g_unit) ** 2
    n_e_unit = np.abs(alpha_e_unit) ** 2
    n_max_unit = np.maximum(n_g_unit, n_e_unit)

    eps = 1e-18
    A_allowed = np.sqrt(n_max_allowed / np.maximum(n_max_unit, eps))

    beta_g = filtered_ckp_beta(
        x,
        omega_r=omega_r_g,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
        A=A_allowed,
    )
    beta_e = filtered_ckp_beta(
        x,
        omega_r=omega_r_e,
        omega_p=omega_p,
        J=J,
        kappa=kappa,
        A=A_allowed,
    )

    separation = np.abs(beta_e - beta_g)

    idx_opt = int(np.argmax(separation))
    optimal_readout_frequency = float(x[idx_opt])
    max_separation = float(separation[idx_opt])

    fig_sep = viz.make_figure()

    fig_sep.add_scatter(
        x=x,
        y=separation,
        mode="lines",
        name="|beta_e - beta_g| (n limited)",
    )

    fig_sep.add_scatter(
        x=[optimal_readout_frequency],
        y=[max_separation],
        mode="markers",
        name="optimal readout frequency",
        marker=dict(size=9),
    )

    fig_sep.add_vline(
        x=optimal_readout_frequency,
        line_dash="dash",
        annotation_text=(f"opt = {optimal_readout_frequency:.6f} GHz"),
    )

    fig_sep.update_layout(
        title=("Readout optimization with max(|alpha_g|², |alpha_e|²) = 0.1 n_crit"),
        xaxis_title="Readout drive frequency (GHz)",
        yaxis=dict(
            title="|beta_e - beta_g|",
            range=[0, 1.1 * float(np.max(separation))],
        ),
        width=1000,
        height=500,
        margin=dict(l=85, r=230, t=60, b=70),
        legend=dict(
            x=1.12,
            y=1.0,
            xanchor="left",
            yanchor="top",
        ),
        title_x=0.43,
    )

    if plot:
        fig_sep.show()

    if save_image:
        viz.save_figure(
            fig_sep,
            name=name,
            width=1000,
            height=500,
        )

    return Result(
        data={
            "optimal_readout_frequency": optimal_readout_frequency,
            "max_beta_separation": max_separation,
            "n_limited_A2_at_optimal_frequency": float(A_allowed[idx_opt] ** 2),
            "frequency_range": x,
            "beta_g": beta_g,
            "beta_e": beta_e,
            "A_allowed": A_allowed,
            "beta_separation": separation,
            "omega_r_g": omega_r_g,
            "omega_r_e": omega_r_e,
            "omega_p": omega_p,
            "J": J,
            "kappa": kappa,
            "n_max_allowed": n_max_allowed,
        },
        figure=fig_sep,
    )
