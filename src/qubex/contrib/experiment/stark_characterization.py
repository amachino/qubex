"""Contributed stark-driven characterization helper functions."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

import qubex.visualization as viz
from qubex.analysis import fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)
from qubex.experiment.models.experiment_result import (
    ExperimentResult,
    RamseyData,
    T1Data,
)
from qubex.pulse import FlatTop, PulseSchedule, VirtualZ

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


def _normalize_stark_param(
    *,
    targets: list[str],
    value: float | Mapping[str, float] | None,
    default: float,
    name: str,
) -> dict[str, float]:
    if value is None:
        return dict.fromkeys(targets, default)
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for target in targets:
            if target not in value:
                raise ValueError(f"`{name}` is missing target `{target}`.")
            result[target] = value[target]
        return result
    return dict.fromkeys(targets, float(value))


def _normalize_stark_detuning(
    *,
    targets: list[str],
    value: float | Mapping[str, float] | None,
) -> dict[str, float]:
    detuning_map = _normalize_stark_param(
        targets=targets,
        value=value,
        default=0.15,
        name="stark_detuning",
    )
    for detuning in detuning_map.values():
        if abs(detuning) > 0.2:
            raise ValueError(
                "Detuning of a stark tone must not exceed 0.2 GHz: the guard-banded AWG baseband limit."
            )
    return detuning_map


def stark_t1_experiment(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    stark_detuning: float | dict[str, float] | None = None,
    stark_amplitude: float | dict[str, float] | None = None,
    stark_ramptime: float | dict[str, float] | None = None,
    time_range: ArrayLike | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    **deprecated_options: Any,
) -> ExperimentResult[T1Data]:
    """
    Run a Stark-driven T1 experiment.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurements.
    targets
        Target qubits to characterize.
    stark_detuning
        Stark-tone detuning in GHz for each target.
    stark_amplitude
        Stark-tone relative drive amplitude for each target.
    stark_ramptime
        Stark-tone ramp time in ns for each target.
    time_range
        Sweep range for wait time in ns.
    n_shots
        Number of shots per sweep point.
    shot_interval
        Measurement interval in seconds.
    plot
        Whether to render plots.
    save_image
        Whether to save generated figures.
    xaxis_type
        X-axis scale for plots.

    Returns
    -------
    ExperimentResult[T1Data]
        Stark-driven T1 fitting results for each target.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_t1_experiment",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False
    if xaxis_type is None:
        xaxis_type = "log"

    target_list = _normalize_targets(exp, targets)
    detuning_map = _normalize_stark_detuning(targets=target_list, value=stark_detuning)
    amplitude_map = _normalize_stark_param(
        targets=target_list,
        value=stark_amplitude,
        default=0.1,
        name="stark_amplitude",
    )
    ramptime_map = _normalize_stark_param(
        targets=target_list,
        value=stark_ramptime,
        default=10,
        name="stark_ramptime",
    )

    exp.pulse.validate_rabi_params(target_list)

    if time_range is None:
        time_range = np.logspace(np.log10(100), np.log10(200 * 1000), 51)
    sampling_period = exp.ctx.measurement.sampling_period
    sweep_range = exp.ctx.util.discretize_time_range(
        np.asarray(time_range),
        sampling_period=sampling_period,
    )

    data: dict[str, T1Data] = {}
    for target in target_list:
        power = exp.pulse.calc_control_amplitude(
            target=target,
            rabi_rate=amplitude_map[target],
        )
        if power > 1:
            raise ValueError("Drive amplitude of a stark tone must not exceed 1")
        ramptime = ramptime_map[target]
        detuning = detuning_map[target]

        def stark_t1_sequence(
            t_ns: int,
            target: str = target,
            ramptime: float = ramptime,
            power: float = power,
            detuning: float = detuning,
        ) -> PulseSchedule:
            with PulseSchedule([target]) as ps:
                ps.add(target, exp.pulse.get_hpi_pulse(target).repeated(2))
                ps.add(
                    target,
                    FlatTop(
                        duration=t_ns + ramptime * 2,
                        amplitude=power,
                        tau=ramptime,
                    ).detuned(detuning=detuning),
                )
            return ps

        sweep_result = exp.measurement_service.sweep_parameter(
            sequence=stark_t1_sequence,
            sweep_range=sweep_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
            title="Stark-driven T1 decay",
            xlabel="Time (μs)",
            ylabel="Measured value",
            xaxis_type=xaxis_type,
        )

        for qubit, sweep_data in sweep_result.data.items():
            fit_result = fitting.fit_exp_decay(
                target=qubit,
                x=sweep_data.sweep_range,
                y=0.5 * (1 - sweep_data.normalized),
                plot=plot,
                title="Stark-driven T1",
                xlabel="Time (μs)",
                ylabel="Normalized signal",
                xaxis_type=xaxis_type,
                yaxis_type="linear",
            )
            if fit_result["status"] != "success":
                continue
            t1_data = T1Data.new(
                sweep_data,
                t1=fit_result["tau"],
                t1_err=fit_result["tau_err"],
                r2=fit_result["r2"],
            )
            data[qubit] = t1_data
            if save_image:
                viz.save_figure(fit_result["fig"], name=f"t1_{qubit}")

    return ExperimentResult(data=data)


def stark_ramsey_experiment(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    stark_detuning: float | dict[str, float] | None = None,
    stark_amplitude: float | dict[str, float] | None = None,
    stark_ramptime: float | dict[str, float] | None = None,
    time_range: ArrayLike | None = None,
    second_rotation_axis: Literal["X", "Y"] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    envelope_region: Literal["full", "flat"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> ExperimentResult[RamseyData]:
    """
    Run a Stark-driven Ramsey experiment.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurements.
    targets
        Target qubits to characterize.
    stark_detuning
        Stark-tone detuning in GHz for each target.
    stark_amplitude
        Stark-tone relative drive amplitude for each target.
    stark_ramptime
        Stark-tone ramp time in ns for each target.
    time_range
        Sweep range for wait time in ns.
    second_rotation_axis
        Axis of the second Ramsey rotation.
    n_shots
        Number of shots per sweep point.
    shot_interval
        Measurement interval in seconds.
    envelope_region
        Stark envelope region mode.
    plot
        Whether to render plots.
    save_image
        Whether to save generated figures.

    Returns
    -------
    ExperimentResult[RamseyData]
        Stark-driven Ramsey fitting results for each target.
    """
    if second_rotation_axis is None:
        second_rotation_axis = "Y"
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_ramsey_experiment",
    )
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if envelope_region is None:
        envelope_region = "full"
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False

    target_list = _normalize_targets(exp, targets)
    detuning_map = _normalize_stark_detuning(targets=target_list, value=stark_detuning)
    amplitude_map = _normalize_stark_param(
        targets=target_list,
        value=stark_amplitude,
        default=0.1,
        name="stark_amplitude",
    )
    ramptime_map = _normalize_stark_param(
        targets=target_list,
        value=stark_ramptime,
        default=10,
        name="stark_ramptime",
    )

    if time_range is None:
        time_range = np.arange(0, 401, 4)
    sampling_period = exp.ctx.measurement.sampling_period
    sweep_range = exp.ctx.util.discretize_time_range(
        np.asarray(time_range),
        sampling_period=sampling_period,
    )

    exp.pulse.validate_rabi_params(target_list)

    data: dict[str, RamseyData] = {}
    for target in target_list:
        power = exp.pulse.calc_control_amplitude(
            target=target,
            rabi_rate=amplitude_map[target],
        )
        if power > 1:
            raise ValueError("Drive amplitude of a stark tone must not exceed 1")
        ramptime = ramptime_map[target]
        detuning = detuning_map[target]

        def stark_ramsey_sequence(
            t_ns: int,
            target: str = target,
            ramptime: float = ramptime,
            power: float = power,
            detuning: float = detuning,
        ) -> PulseSchedule:
            x90 = exp.pulse.get_hpi_pulse(target=target)
            with PulseSchedule([target]) as ps:
                ps.add(target, x90)
                if envelope_region == "full":
                    ps.add(
                        target,
                        FlatTop(
                            duration=t_ns + ramptime * 2,
                            amplitude=power,
                            tau=ramptime,
                        ).detuned(detuning=detuning),
                    )
                    if second_rotation_axis == "X":
                        ps.add(target, x90.shifted(np.pi))
                    else:
                        ps.add(target, x90.shifted(-np.pi / 2))
                else:
                    ps.add(
                        target,
                        FlatTop(
                            duration=ramptime * 2,
                            amplitude=power,
                            tau=ramptime,
                        ).detuned(detuning=detuning),
                    )
                    ps.add(target, x90.repeated(2))
                    ps.add(
                        target,
                        FlatTop(
                            duration=t_ns + ramptime * 2,
                            amplitude=power,
                            tau=ramptime,
                        ).detuned(detuning=detuning),
                    )
                    if second_rotation_axis == "X":
                        ps.add(target, VirtualZ(theta=-np.pi))
                        ps.add(target, x90)
                    else:
                        ps.add(target, VirtualZ(theta=np.pi / 2))
                        ps.add(target, x90)
            return ps

        sweep_result = exp.measurement_service.sweep_parameter(
            sequence=stark_ramsey_sequence,
            sweep_range=sweep_range,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
        )

        for qubit, sweep_data in sweep_result.data.items():
            fit_result = fitting.fit_ramsey(
                target=qubit,
                times=sweep_data.sweep_range,
                data=sweep_data.normalized,
                title="Stark-driven Ramsey fringe",
                amplitude_est=1.0,
                offset_est=0.0,
                plot=plot,
            )
            if fit_result["status"] != "success":
                continue

            freq = exp.ctx.qubits[qubit].frequency
            ramsey_freq = fit_result["f"]
            if detuning_map[qubit] > 0:
                dressed_freq = freq - ramsey_freq
            else:
                dressed_freq = freq + ramsey_freq

            ramsey_data = RamseyData.new(
                sweep_data=sweep_data,
                t2=fit_result["tau"],
                ramsey_freq=ramsey_freq,
                bare_freq=dressed_freq,
                r2=fit_result["r2"],
            )
            data[qubit] = ramsey_data

            sign = 1 if detuning_map[qubit] > 0 else -1
            ac_stark_shift = sign * ramsey_data.ramsey_freq
            print("AC stark shift :")
            print(f"{qubit}: {ac_stark_shift:.6f}")
            print("")

            if save_image:
                viz.save_figure(fit_result["fig"], name=f"stark_ramsey_{qubit}")

    return ExperimentResult(data=data)
