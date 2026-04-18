"""Contributed GF calibration helper functions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

import qubex.visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_INTERVAL,
    DEFAULT_RABI_TIME_RANGE,
    DEFAULT_SHOTS,
    HPI_DURATION,
    HPI_RAMPTIME,
    PI_DURATION,
    PI_RAMPTIME,
)
from qubex.experiment.models.experiment_result import (
    AmplCalibData,
    ExperimentResult,
    RabiData,
    RabiParam,
    RamseyData,
)
from qubex.experiment.models.result import Result
from qubex.pulse import Blank, FlatTop, PulseSchedule
from qubex.system import Target

from ._deprecated_options import resolve_shot_options


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    if targets is None:
        return list(exp.ctx.qubit_labels)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def gf_rabi_experiment(
    exp: Experiment,
    *,
    amplitudes: dict[str, float],
    time_range: ArrayLike,
    ramptime: float | None = None,
    frequencies: dict[str, float] | None = None,
    detuning: float | None = None,
    is_damped: bool | None = None,
    fit_threshold: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    store_params: bool | None = None,
    **deprecated_options: Any,
) -> ExperimentResult[RabiData]:
    """
    Run a GF Rabi experiment and fit parameters.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurements.
    amplitudes
        GF drive amplitudes keyed by qubit or EF label.
    time_range
        Drive durations used for the sweep.
    ramptime
        Ramp time of the flat-top pulse in ns.
    frequencies
        Target EF frequencies keyed by EF label.
    detuning
        Optional detuning applied to target frequencies.
    is_damped
        Whether to fit with a damped cosine model.
    fit_threshold
        Minimum acceptable R² for storing a valid fit.
    n_shots
        Number of shots per sweep point.
    shot_interval
        Measurement interval in seconds.
    plot
        Whether to render plots.
    store_params
        Whether to store the fitted GF Rabi parameters.

    Returns
    -------
    ExperimentResult[RabiData]
        GF Rabi data and fitted parameters for each target.
    """
    if is_damped is None:
        is_damped = True
    if fit_threshold is None:
        fit_threshold = 0.5
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="gf_rabi_experiment",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if store_params is None:
        store_params = False

    normalized_amplitudes = {
        Target.ef_label(label): amplitude for label, amplitude in amplitudes.items()
    }
    ge_labels = [Target.ge_label(label) for label in normalized_amplitudes]
    ef_labels = [Target.ef_label(label) for label in normalized_amplitudes]

    time_values = np.asarray(time_range, dtype=np.float64)
    if ramptime is None:
        ramptime = 0.0
    effective_time_range = time_values + ramptime

    if frequencies is None:
        frequencies = {
            target: exp.ctx.targets[target].frequency
            for target in normalized_amplitudes
        }

    def gf_rabi_sequence(duration_ns: int) -> PulseSchedule:
        with PulseSchedule() as ps:
            for ge_label in ge_labels:
                ps.add(ge_label, exp.pulse.x180(ge_label))
            ps.barrier()
            for ef_label in ef_labels:
                ps.add(
                    ef_label,
                    FlatTop(
                        duration=duration_ns + 2 * ramptime,
                        amplitude=normalized_amplitudes[ef_label],
                        tau=ramptime,
                    ),
                )
            ps.barrier()
            for ge_label in ge_labels:
                ps.add(ge_label, exp.pulse.x180(ge_label))
        return ps

    if detuning is not None:
        frequencies = {
            target: frequencies[target] + detuning for target in normalized_amplitudes
        }

    sweep_result = exp.measurement_service.sweep_parameter(
        sequence=gf_rabi_sequence,
        sweep_range=time_values,
        frequencies=frequencies,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
    )

    gf_rabi_params: dict[str, RabiParam] = {}
    gf_rabi_data: dict[str, RabiData] = {}
    for qubit, sweep_data in sweep_result.data.items():
        ef_label = Target.ef_label(qubit)
        ge_label = Target.ge_label(qubit)
        gf_label = f"{ge_label}_{ef_label}"
        ge_rabi_param = exp.ge_rabi_params[qubit]
        iq_e = ge_rabi_param.endpoints[0]
        fit_result = fitting.fit_rabi(
            target=qubit,
            times=effective_time_range,
            data=sweep_data.data,
            reference_point=iq_e,
            plot=plot,
            is_damped=is_damped,
        )

        if fit_result.status is FitStatus.ERROR or fit_result["r2"] < fit_threshold:
            gf_rabi_params[gf_label] = RabiParam.nan(target=gf_label)
        else:
            gf_rabi_params[gf_label] = RabiParam(
                target=gf_label,
                amplitude=fit_result["amplitude"],
                frequency=fit_result["frequency"],
                phase=fit_result["phase"],
                offset=fit_result["offset"],
                noise=fit_result["noise"],
                angle=fit_result["angle"],
                distance=fit_result["distance"],
                r2=fit_result["r2"],
                reference_phase=fit_result["reference_phase"],
            )

        gf_rabi_data[gf_label] = RabiData(
            target=gf_label,
            data=sweep_data.data,
            time_range=effective_time_range,
            rabi_param=gf_rabi_params[gf_label],
        )

    if store_params:
        exp.store_rabi_params(gf_rabi_params)

    return ExperimentResult(
        data=gf_rabi_data,
        rabi_params=gf_rabi_params,
    )


def obtain_gf_rabi_params(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    frequencies: dict[str, float] | None = None,
    is_damped: bool | None = None,
    fit_threshold: float | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    store_params: bool | None = None,
    **deprecated_options: Any,
) -> ExperimentResult[RabiData]:
    """
    Estimate GF Rabi parameters for the specified targets.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurements.
    targets
        Target qubits to characterize.
    time_range
        Time sweep range for the GF Rabi experiment.
    ramptime
        Ramp time of the flat-top pulse in ns.
    frequencies
        Target EF frequencies keyed by EF label.
    is_damped
        Whether to fit with a damped cosine model.
    fit_threshold
        Minimum acceptable R² for storing a valid fit.
    n_shots
        Number of shots per sweep point.
    shot_interval
        Measurement interval in seconds.
    plot
        Whether to render plots.
    store_params
        Whether to store the fitted GF Rabi parameters.

    Returns
    -------
    ExperimentResult[RabiData]
        GF Rabi data and fitted parameters for each target.
    """
    if time_range is None:
        time_range = DEFAULT_RABI_TIME_RANGE
    if is_damped is None:
        is_damped = True
    if fit_threshold is None:
        fit_threshold = 0.5
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="obtain_gf_rabi_params",
    )
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if store_params is None:
        store_params = False

    target_list = _normalize_targets(exp, targets)
    time_values = np.asarray(time_range)

    if ramptime is None:
        ramptime = 32 - 12

    amplitudes = {
        target: exp.params.get_ef_control_amplitude(target) for target in target_list
    }

    rabi_data: dict[str, RabiData] = {}
    rabi_params: dict[str, RabiParam] = {}
    for target in target_list:
        ge_label = Target.ge_label(target)
        ef_label = Target.ef_label(target)
        gf_label = f"{ge_label}_{ef_label}"
        data = gf_rabi_experiment(
            exp=exp,
            amplitudes={target: amplitudes[target]},
            time_range=time_values,
            ramptime=ramptime,
            frequencies=frequencies,
            is_damped=is_damped,
            fit_threshold=fit_threshold,
            n_shots=n_shots,
            shot_interval=shot_interval,
            store_params=store_params,
            plot=plot,
        ).data[gf_label]
        rabi_data[gf_label] = data
        rabi_params[gf_label] = data.rabi_param

    return ExperimentResult(
        data=rabi_data,
        rabi_params=rabi_params,
    )


def gf_chevron_pattern(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    detuning_range: ArrayLike | None = None,
    time_range: ArrayLike | None = None,
    frequencies: dict[str, float] | None = None,
    amplitudes: dict[str, float] | None = None,
    rabi_params: dict[str, RabiParam] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure GF chevron patterns for the specified targets.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurements.
    targets
        Target qubits to characterize.
    detuning_range
        Frequency detuning sweep range in GHz.
    time_range
        Drive duration sweep range in ns.
    frequencies
        Target EF frequencies keyed by EF label.
    amplitudes
        GF drive amplitudes keyed by EF label.
    rabi_params
        Pre-computed GF Rabi parameters keyed by GF label.
    n_shots
        Number of shots per sweep point.
    shot_interval
        Measurement interval in seconds.
    plot
        Whether to render plots.
    save_image
        Whether to save generated figures.

    Returns
    -------
    Result
        Chevron data, fitted resonant frequencies, and figures.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="gf_chevron_pattern",
    )
    if time_range is None:
        time_range = DEFAULT_RABI_TIME_RANGE
    if detuning_range is None:
        detuning_range = np.linspace(-0.05, 0.05, 51)
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True

    target_list = _normalize_targets(exp, targets)
    detuning_values = np.asarray(detuning_range, dtype=np.float64)
    time_values = np.asarray(time_range, dtype=np.float64)

    if frequencies is None:
        ef_targets = [Target.ef_label(target) for target in target_list]
        frequencies = {target: exp.targets[target].frequency for target in ef_targets}

    if amplitudes is None:
        ef_labels = [Target.ef_label(target) for target in target_list]
        ef_targets = [exp.targets[ef_label] for ef_label in ef_labels]
        amplitudes = {
            ef.label: exp.params.get_ef_control_amplitude(ef.qubit) for ef in ef_targets
        }

    ramptime = 32 - 12

    shared_rabi_params: dict[str, RabiParam]
    if rabi_params is None:
        print("Obtaining Rabi parameters between g and f...")
        shared_rabi_params = dict(
            obtain_gf_rabi_params(
                exp=exp,
                targets=target_list,
                time_range=time_values,
                fit_threshold=0.0,
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=False,
                store_params=False,
            ).rabi_params
            or {}
        )
    else:
        shared_rabi_params = rabi_params

    rabi_rates: dict[str, NDArray] = {}
    chevron_data: dict[str, NDArray] = {}
    resonant_frequencies: dict[str, float] = {}

    print(f"Targets : {target_list}")
    subgroups = exp.util.create_qubit_subgroups(target_list)
    figs: dict[str, go.Figure] = {}

    for idx, subgroup in enumerate(subgroups):
        if len(subgroup) == 0:
            continue

        print(f"Subgroup ({idx + 1}/{len(subgroups)}) : {subgroup}")

        rabi_rates_buffer: dict[str, list[float]] = defaultdict(list)
        chevron_data_buffer: dict[str, list[NDArray]] = defaultdict(list)
        ef_subgroup = [Target.ef_label(target) for target in subgroup]

        for detuning in tqdm(detuning_values, leave=False):
            with exp.util.no_output():

                def gf_rabi_sequence(
                    duration_ns: int,
                    _subgroup: Collection[str] = subgroup,
                    _ef_subgroup: Collection[str] = ef_subgroup,
                ) -> PulseSchedule:
                    with PulseSchedule() as ps:
                        for ge_label in _subgroup:
                            ps.add(ge_label, exp.pulse.x180(ge_label))
                        ps.barrier()
                        for ef_label in _ef_subgroup:
                            ps.add(
                                ef_label,
                                FlatTop(
                                    duration=duration_ns + 2 * ramptime,
                                    amplitude=amplitudes[ef_label],
                                    tau=ramptime,
                                ),
                            )
                        for ge_label in _subgroup:
                            ps.add(ge_label, exp.pulse.x180(ge_label))
                        return ps

                sweep_result = exp.measurement_service.sweep_parameter(
                    sequence=gf_rabi_sequence,
                    sweep_range=time_values,
                    frequencies={
                        label: frequencies[label] + detuning for label in ef_subgroup
                    },
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                )

                for target, sweep_data in sweep_result.data.items():
                    fit_result = fitting.fit_rabi(
                        target=sweep_data.target,
                        times=sweep_data.sweep_range,
                        data=sweep_data.data,
                        plot=False,
                    )
                    rabi_rates_buffer[target].append(
                        float(fit_result.get("frequency", np.nan))
                    )
                    ge_label = Target.ge_label(target)
                    ef_label = Target.ef_label(target)
                    gf_label = f"{ge_label}_{ef_label}"
                    sweep_data.rabi_param = shared_rabi_params[gf_label]
                    chevron_data_buffer[target].append(sweep_data.normalized)

        for target in subgroup:
            rabi_rates[target] = np.asarray(rabi_rates_buffer[target])
            chevron_data[target] = np.asarray(chevron_data_buffer[target]).T
            ef_label = Target.ef_label(target)

            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=detuning_values + frequencies[ef_label],
                    y=time_values,
                    z=chevron_data[target],
                    colorscale="Viridis",
                )
            )
            fig.update_layout(
                title=dict(
                    text=f"Chevron pattern : {target}",
                    subtitle=dict(
                        text=f"control_amplitude={amplitudes[ef_label]:.6g}",
                        font=dict(size=13, family="monospace"),
                    ),
                ),
                xaxis_title="Drive frequency (GHz)",
                yaxis_title="Time (ns)",
                width=600,
                height=400,
                margin=dict(t=80),
            )
            figs[target] = fig
            if plot:
                fig.show()

            fit_result = fitting.fit_detuned_rabi(
                target=target,
                control_frequencies=detuning_values + frequencies[ef_label],
                rabi_frequencies=rabi_rates[target],
                plot=plot,
            )
            resonant_frequencies[target] = fit_result["f_resonance"]

            if save_image:
                viz.save_figure(
                    fig,
                    name=f"chevron_pattern_{target}",
                    width=600,
                    height=400,
                )
                fig_fit = fit_result.figure
                if fig_fit is not None:
                    viz.save_figure(
                        fig_fit,
                        name=f"chevron_pattern_fit_{target}",
                        width=600,
                        height=300,
                    )

    rabi_rates = dict(sorted(rabi_rates.items()))
    chevron_data = dict(sorted(chevron_data.items()))
    resonant_frequencies = dict(sorted(resonant_frequencies.items()))

    return Result(
        data={
            "time_range": time_values,
            "detuning_range": detuning_values,
            "frequencies": frequencies,
            "chevron_data": chevron_data,
            "rabi_rates": rabi_rates,
            "resonant_frequencies": resonant_frequencies,
            "fig": figs,
        },
        figures=figs,
    )


def calibrate_gf_pulse(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    pulse_type: Literal["pi", "hpi"],
    duration: float | None = None,
    ramptime: float | None = None,
    n_points: int | None = None,
    n_rotations: int | None = None,
    r2_threshold: float | None = None,
    plot: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> ExperimentResult[AmplCalibData]:
    """Calibrate GF pulse amplitude for the specified targets."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="calibrate_gf_pulse",
    )
    if n_points is None:
        n_points = 20
    if n_rotations is None:
        n_rotations = 1
    if r2_threshold is None:
        r2_threshold = 0.5
    if plot is None:
        plot = True
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    target_list = _normalize_targets(exp, targets)
    gf_rabi_params = exp.calib_note.rabi_params
    if gf_rabi_params is None:
        raise ValueError("Rabi parameters are not stored.")
    sampling_period_ns = exp.ctx.util.resolve_sampling_period(
        exp.ctx.measurement.sampling_period
    )

    def calibrate(target: str) -> AmplCalibData:
        ge_label = Target.ge_label(target)
        ef_label = Target.ef_label(target)
        gf_label = f"{ge_label}_{ef_label}"

        if pulse_type == "hpi":
            pulse = FlatTop(
                duration=duration if duration is not None else HPI_DURATION,
                amplitude=1,
                tau=ramptime if ramptime is not None else HPI_RAMPTIME,
            )
            area = pulse.real.sum() * sampling_period_ns
            rabi_rate = 0.25 / area
        elif pulse_type == "pi":
            pulse = FlatTop(
                duration=duration if duration is not None else PI_DURATION,
                amplitude=1,
                tau=ramptime if ramptime is not None else PI_RAMPTIME,
            )
            area = pulse.real.sum() * sampling_period_ns
            rabi_rate = 0.5 / area
        else:
            raise ValueError("Invalid pulse type.")

        gf_rabi_param = gf_rabi_params.get(gf_label)
        if gf_rabi_param is None:
            raise ValueError(f"GF Rabi parameters are not stored for `{gf_label}`.")

        default_amplitude = exp.params.get_ef_control_amplitude(target)
        ampl = rabi_rate * default_amplitude / gf_rabi_param["frequency"]

        ampl_min = ampl * (1 - 0.8 / n_rotations)
        ampl_max = ampl * (1 + 0.5 / n_rotations)
        ampl_min = np.clip(ampl_min, 0, 1)
        ampl_max = np.clip(ampl_max, 0, 1)
        if ampl_min == ampl_max:
            ampl_min = 0
            ampl_max = 1
        ampl_range = np.linspace(ampl_min, ampl_max, n_points)

        n_per_rotation = 2 if pulse_type == "pi" else 4
        repetitions = n_per_rotation * n_rotations

        def sequence(x: float) -> PulseSchedule:
            with PulseSchedule() as ps:
                ps.add(ge_label, exp.pulse.x180(ge_label))
                ps.barrier()
                ps.add(ef_label, pulse.scaled(x).repeated(repetitions))
                ps.barrier()
                ps.add(ge_label, exp.pulse.x180(ge_label))
            return ps

        sweep_data = exp.measurement_service.sweep_parameter(
            sequence=sequence,
            sweep_range=ampl_range,
            repetitions=1,
            rabi_level="ef",
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
        ).data[ge_label]

        fit_result = fitting.fit_ampl_calib_data(
            target=target,
            amplitude_range=ampl_range,
            data=sweep_data.data,
            plot=plot,
            title=f"gf {pulse_type} pulse calibration",
            ylabel="Normalized signal",
        )

        r2 = fit_result["r2"]
        if r2 > r2_threshold:
            if pulse_type == "hpi":
                exp.calib_note.update_hpi_param(
                    ef_label,
                    {
                        "target": ef_label,
                        "duration": pulse.duration,
                        "amplitude": fit_result["amplitude"],
                        "tau": pulse.tau,
                    },
                )
            else:
                exp.calib_note.update_pi_param(
                    ef_label,
                    {
                        "target": ef_label,
                        "duration": pulse.duration,
                        "amplitude": fit_result["amplitude"],
                        "tau": pulse.tau,
                    },
                )
        else:
            print(f"Error: R² value is too low ({r2:.3f})")
            print(f"Calibration data not stored for {ef_label}.")

        return AmplCalibData.new(
            sweep_data=sweep_data,
            calib_value=fit_result["amplitude"],
            r2=r2,
        )

    data: dict[str, AmplCalibData] = {}
    for target in target_list:
        data[target] = calibrate(target)

    print("")
    print(f"Calibration results for {pulse_type} pulse:")
    for target, calib_data in data.items():
        print(f"  {target}: {calib_data.calib_value:.6f}")

    return ExperimentResult(data=data)


def calibrate_gf_hpi_pulse(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    duration: float | None = None,
    ramptime: float | None = None,
    n_points: int | None = None,
    n_rotations: int | None = None,
    r2_threshold: float | None = None,
    plot: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> ExperimentResult[AmplCalibData]:
    """Calibrate GF half-pi pulse amplitude for the specified targets."""
    return calibrate_gf_pulse(
        exp,
        targets=targets,
        pulse_type="hpi",
        duration=duration,
        ramptime=ramptime,
        n_points=n_points,
        n_rotations=n_rotations,
        r2_threshold=r2_threshold,
        plot=plot,
        n_shots=n_shots,
        shot_interval=shot_interval,
        **deprecated_options,
    )


def calibrate_gf_pi_pulse(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    duration: float | None = None,
    ramptime: float | None = None,
    n_points: int | None = None,
    n_rotations: int | None = None,
    r2_threshold: float | None = None,
    plot: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> ExperimentResult[AmplCalibData]:
    """Calibrate GF pi pulse amplitude for the specified targets."""
    return calibrate_gf_pulse(
        exp,
        targets=targets,
        pulse_type="pi",
        duration=duration,
        ramptime=ramptime,
        n_points=n_points,
        n_rotations=n_rotations,
        r2_threshold=r2_threshold,
        plot=plot,
        n_shots=n_shots,
        shot_interval=shot_interval,
        **deprecated_options,
    )


def gf_ramsey_experiment(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    time_range: ArrayLike | None = None,
    detuning: float | None = None,
    second_rotation_axis: Literal["X", "Y"] | None = None,
    spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> ExperimentResult[RamseyData]:
    """Run a GF Ramsey experiment."""
    if second_rotation_axis is None:
        second_rotation_axis = "Y"
    if spectator_state is None:
        spectator_state = "0"
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="gf_ramsey_experiment",
    )
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False

    target_list = _normalize_targets(exp, targets)

    if time_range is None:
        time_range = np.arange(0, 10001, 100)
    sweep_range = exp.ctx.util.discretize_time_range(
        np.asarray(time_range),
        sampling_period=exp.ctx.measurement.sampling_period,
    )

    if detuning is None:
        detuning = 0.001

    exp.pulse.validate_rabi_params(target_list)

    ef_hpi_pulses = exp.ef_hpi_pulse
    missing_ef_hpi = [
        Target.ef_label(target)
        for target in target_list
        if Target.ef_label(target) not in ef_hpi_pulses
    ]
    if missing_ef_hpi:
        raise ValueError(
            f"EF half-pi pulse parameters are not stored for {missing_ef_hpi}."
        )

    gf_rabi_params = {
        target: exp.get_rabi_param(
            f"{Target.ge_label(target)}_{Target.ef_label(target)}"
        )
        for target in target_list
    }
    missing_gf_rabi = [
        target for target, rabi_param in gf_rabi_params.items() if rabi_param is None
    ]
    if missing_gf_rabi:
        raise ValueError(f"GF Rabi parameters are not stored for {missing_gf_rabi}.")

    target_groups = exp.util.create_qubit_subgroups(target_list)
    spectator_groups = reversed(target_groups)  # TODO: make it more general

    data: dict[str, RamseyData] = {}

    for target_qubits, spectator_qubits in zip(
        target_groups, spectator_groups, strict=True
    ):
        active_targets = (
            target_qubits + spectator_qubits
            if spectator_state != "0"
            else target_qubits
        )

        if len(active_targets) == 0:
            continue

        print(f"Target qubits: {target_qubits}")
        print(f"Spectator qubits: {spectator_qubits}")

        def gf_ramsey_sequence(
            t_ns: int,
            _spectator_qubits: Collection[str] = spectator_qubits,
            _target_qubits: Collection[str] = target_qubits,
        ) -> PulseSchedule:
            with PulseSchedule() as ps:
                # Excite spectator qubits if needed
                if spectator_state != "0":
                    for spectator in _spectator_qubits:
                        if spectator in exp.ctx.qubit_labels:
                            pulse = exp.get_pulse_for_state(
                                target=spectator,
                                state=spectator_state,
                            )
                            ps.add(spectator, pulse)
                    ps.barrier()

                # Ramsey sequence for the target qubit
                for target in _target_qubits:
                    ef_label = Target.ef_label(target)
                    x180 = exp.pulse.x180(target)
                    ef90 = ef_hpi_pulses[ef_label]
                    ps.add(target, x180)
                    ps.barrier()
                    ps.add(ef_label, ef90)
                    ps.barrier()
                    ps.add(ef_label, Blank(t_ns))
                    ps.barrier()
                    if second_rotation_axis == "X":
                        ps.add(ef_label, ef90.shifted(np.pi))
                    else:
                        ps.add(ef_label, ef90.shifted(-np.pi / 2))
                    ps.barrier()
                    ps.add(target, x180)
            return ps

        ef_labels = [Target.ef_label(target) for target in target_qubits]
        detuned_frequencies = {
            ef_label: exp.targets[ef_label].frequency + detuning
            for ef_label in ef_labels
        }

        sweep_result = exp.measurement_service.sweep_parameter(
            sequence=gf_ramsey_sequence,
            sweep_range=sweep_range,
            frequencies=detuned_frequencies,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
        )

        for target, sweep_data in sweep_result.data.items():
            if target in target_qubits:
                sweep_data.rabi_param = gf_rabi_params[target]
                fit_result = fitting.fit_ramsey(
                    target=target,
                    times=sweep_data.sweep_range,
                    data=sweep_data.normalized,
                    amplitude_est=1.0,
                    offset_est=0.0,
                    plot=plot,
                )
                if fit_result.status is FitStatus.SUCCESS:
                    ef_label = Target.ef_label(target)
                    f = exp.targets[ef_label].frequency
                    t2 = fit_result["tau"]
                    ramsey_freq = fit_result["f"]
                    phi = fit_result["phi"]
                    if second_rotation_axis == "Y":
                        if phi > 0:
                            bare_freq = f + detuning + ramsey_freq
                        else:
                            bare_freq = f + detuning - ramsey_freq
                    else:
                        # NOTE: For X rotation, we cannot guarantee the sign of frequency.
                        bare_freq = f + detuning - ramsey_freq
                    r2 = fit_result["r2"]
                    ramsey_data = RamseyData.new(
                        sweep_data=sweep_data,
                        t2=t2,
                        ramsey_freq=ramsey_freq,
                        bare_freq=bare_freq,
                        r2=r2,
                    )
                    data[target] = ramsey_data

                    print(f"Bare ef frequency with |{spectator_state}〉:")
                    print(f"  {target}: {ramsey_data.bare_freq:.6f}")
                    print("")
                    print(
                        f"  anharmonicity: {ramsey_data.bare_freq - exp.targets[target].frequency:.6f}"
                    )
                    print("")

                    if save_image:
                        fig = fit_result.get_figure()
                        viz.save_figure(fig, name=f"gf_ramsey_{target}")

    return ExperimentResult(data=data)
