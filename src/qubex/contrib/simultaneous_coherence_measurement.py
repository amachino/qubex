"""Contributed simultaneous coherence measurement helper function."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from qubex.analysis import fitting, visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)
from qubex.experiment.models.experiment_result import (
    ExperimentResult,
    RamseyData,
    SweepData,
    T1Data,
    T2Data,
)
from qubex.pulse import Blank, PulseSchedule


def simultaneous_coherence_measurement(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    time_range: ArrayLike | None = None,
    detuning: float | None = None,
    second_rotation_axis: Literal["X", "Y"] | None = None,
    shots: int | None = None,
    interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
) -> dict[str, ExperimentResult]:
    """
    Run simultaneous T1, T2 echo, and Ramsey measurements.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurements.
    targets
        Target qubits to characterize.
    time_range
        Sweep range for coherence waits in ns.
    detuning
        Ramsey detuning in GHz.
    second_rotation_axis
        Axis of the second Ramsey rotation.
    shots
        Number of shots per sweep point.
    interval
        Measurement interval in seconds.
    plot
        Whether to render plots.
    save_image
        Whether to save generated figures.

    Returns
    -------
    dict[str, ExperimentResult]
        Experiment results keyed by mode: `T1`, `T2`, and `Ramsey`.
    """
    if second_rotation_axis is None:
        second_rotation_axis = "Y"
    if shots is None:
        shots = DEFAULT_SHOTS
    if interval is None:
        interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False

    if targets is None:
        target_list = list(exp.ctx.qubit_labels)
    elif isinstance(targets, str):
        target_list = [targets]
    else:
        target_list = list(targets)

    if time_range is None:
        time_range = np.arange(0, 50_001, 1000)

    sampling_period = exp.ctx.measurement.sampling_period
    discretized_time_range = exp.ctx.util.discretize_time_range(
        time_range=np.asarray(time_range),
        sampling_period=2 * sampling_period,
    )
    if discretized_time_range is None:
        raise ValueError("time_range could not be discretized.")

    if detuning is None:
        detuning = 0.001

    exp.pulse.validate_rabi_params(target_list)

    modes = ("T1", "T2", "Ramsey")
    signals: dict[str, defaultdict[str, list[object]]] = {
        mode: defaultdict(list) for mode in modes
    }
    data_t1: dict[str, T1Data] = {}
    data_t2: dict[str, T2Data] = {}
    data_ramsey: dict[str, RamseyData] = {}

    x90_pulses = {target: exp.pulse.get_hpi_pulse(target) for target in target_list}

    def t1_sequence(target: str, wait_ns: int) -> PulseSchedule:
        with PulseSchedule([target]) as ps:
            ps.add(target, x90_pulses[target].repeated(2))
            ps.add(target, Blank(wait_ns))
        return ps

    def t2_sequence(target: str, wait_ns: int) -> PulseSchedule:
        half_wait_ns = wait_ns // 2
        with PulseSchedule([target]) as ps:
            ps.add(target, x90_pulses[target])
            ps.add(target, Blank(half_wait_ns))
            ps.add(target, x90_pulses[target].repeated(2).shifted(np.pi / 2))
            ps.add(target, Blank(half_wait_ns))
            ps.add(target, x90_pulses[target].scaled(-1))
        return ps

    def ramsey_sequence(target: str, wait_ns: int) -> PulseSchedule:
        with PulseSchedule([target]) as ps:
            x90 = x90_pulses[target]
            ps.add(target, x90)
            ps.add(target, Blank(wait_ns))
            if second_rotation_axis == "X":
                ps.add(target, x90.shifted(np.pi))
            else:
                ps.add(target, x90.shifted(-np.pi / 2))
        return ps

    for target in target_list:
        for wait_ns in discretized_time_range:
            t1_schedules = t1_sequence(target, wait_ns)
            t2_schedules = t2_sequence(target, wait_ns)
            ramsey_schedules = ramsey_sequence(target, wait_ns)

            detuned_frequencies = {
                target: exp.ctx.qubits[target].frequency + detuning
                for target in target_list
            }
            measurements = {
                "T1": exp.measurement_service.measure(
                    sequence=t1_schedules,
                    shots=shots,
                    interval=interval,
                    plot=False,
                ),
                "T2": exp.measurement_service.measure(
                    sequence=t2_schedules,
                    shots=shots,
                    interval=interval,
                    plot=False,
                ),
                "Ramsey": exp.measurement_service.measure(
                    sequence=ramsey_schedules,
                    frequencies=detuned_frequencies,
                    shots=shots,
                    interval=interval,
                    plot=False,
                ),
            }
            for mode in modes:
                for measured_target, data in measurements[mode].data.items():
                    signals[mode][measured_target].append(data.kerneled)

    sweep_data: dict[str, dict[str, SweepData]] = {
        mode: {
            target: SweepData(
                target=target,
                data=np.asarray(values),
                sweep_range=discretized_time_range,
                rabi_param=exp.pulse.rabi_params.get(target),
                state_centers=exp.ctx.state_centers.get(target),
                title="Sweep result",
                xlabel="Sweep value",
                ylabel="Measured value",
                xaxis_type="linear",
                yaxis_type="linear",
            )
            for target, values in signals[mode].items()
        }
        for mode in modes
    }

    for target, sweep_result in sweep_data["T1"].items():
        fit_result_t1 = fitting.fit_exp_decay(
            target=target,
            x=sweep_result.sweep_range,
            y=0.5 * (1 - sweep_result.normalized),
            plot=plot,
            title="T1",
            xlabel="Time (μs)",
            ylabel="Normalized signal",
            xaxis_type="linear",
        )
        if fit_result_t1["status"] != "success":
            continue

        t1_data = T1Data.new(
            sweep_result,
            t1=fit_result_t1["tau"],
            t1_err=fit_result_t1["tau_err"],
            r2=fit_result_t1["r2"],
        )
        data_t1[target] = t1_data

        if save_image:
            viz.save_figure_image(
                fit_result_t1["fig"],
                name=f"t1_{target}",
            )

    for target, sweep_result in sweep_data["T2"].items():
        fit_result_t2 = fitting.fit_exp_decay(
            target=target,
            x=sweep_result.sweep_range,
            y=0.5 * (1 + sweep_result.normalized),
            plot=plot,
            title="T2 echo",
            xlabel="Time (μs)",
            ylabel="Normalized signal",
            xaxis_type="linear",
        )
        if fit_result_t2["status"] != "success":
            continue

        t2_data = T2Data.new(
            sweep_result,
            t2=fit_result_t2["tau"],
            t2_err=fit_result_t2["tau_err"],
            r2=fit_result_t2["r2"],
        )
        data_t2[target] = t2_data

        if save_image:
            viz.save_figure_image(
                fit_result_t2["fig"],
                name=f"t2_echo_{target}",
            )

    for target, sweep_result in sweep_data["Ramsey"].items():
        fit_result_ramsey = fitting.fit_ramsey(
            target=target,
            times=sweep_result.sweep_range,
            data=sweep_result.normalized,
            amplitude_est=1.0,
            offset_est=0.0,
            plot=plot,
        )
        if fit_result_ramsey["status"] != "success":
            continue

        freq = exp.ctx.qubits[target].frequency
        ramsey_freq = fit_result_ramsey["f"]
        phi = fit_result_ramsey["phi"]
        if second_rotation_axis == "Y":
            if phi > 0:
                bare_freq = freq + detuning + ramsey_freq
            else:
                bare_freq = freq + detuning - ramsey_freq
        else:
            bare_freq = freq + detuning - ramsey_freq

        ramsey_data = RamseyData.new(
            sweep_result,
            t2=fit_result_ramsey["tau"],
            ramsey_freq=ramsey_freq,
            bare_freq=bare_freq,
            r2=fit_result_ramsey["r2"],
        )
        data_ramsey[target] = ramsey_data

        if save_image:
            viz.save_figure_image(
                fit_result_ramsey["fig"],
                name=f"ramsey_{target}",
            )

    return {
        "T1": ExperimentResult(data=data_t1),
        "T2": ExperimentResult(data=data_t2),
        "Ramsey": ExperimentResult(data=data_ramsey),
    }
