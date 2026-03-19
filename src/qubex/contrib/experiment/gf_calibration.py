"""Contributed simultaneous coherence measurement helper function."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tqdm import tqdm
import plotly.graph_objects as go
import qubex.visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
    DEFAULT_RABI_TIME_RANGE,
    CALIBRATION_SHOTS,
)
from qubex.system import Target

from qubex.experiment.models.experiment_result import (
    ExperimentResult,
    RamseyData,
    SweepData,
    T1Data,
    T2Data,
    RabiData,
    RabiParam,
)
from qubex.experiment.result import Result
from qubex.pulse import Blank, PulseSchedule, FlatTop
from ._deprecated_options import resolve_shot_options

def gf_rabi_experiment(
    exp: Experiment,
    *,
    amplitudes: dict[str, float],
    time_range: ArrayLike,
    ramptime: float | None = None,
    frequencies: dict[str, float] | None = None,
    detuning: float | None = None,
    is_damped: bool = True,
    fit_threshold: float = 0.5,
    shots: int = DEFAULT_SHOTS,
    interval: float = DEFAULT_INTERVAL,
    plot: bool = True,
    store_params: bool = False,
) -> ExperimentResult[RabiData]:
    # TODO: Integrate with rabi_experiment

    amplitudes = {
        Target.ef_label(label): amplitude for label, amplitude in amplitudes.items()
    }
    ge_labels = [Target.ge_label(label) for label in amplitudes]
    ef_labels = [Target.ef_label(label) for label in amplitudes]

    # drive time range
    time_range = np.array(time_range, dtype=np.float64)

    if ramptime is None:
        ramptime = 0.0

    effective_time_range = time_range + ramptime

    # target frequencies
    if frequencies is None:
        frequencies = {
            target: exp.targets[target].frequency for target in amplitudes
        }

    # ef rabi sequence with rect pulses of duration T
    def gf_rabi_sequence(T: int) -> PulseSchedule:
        with PulseSchedule() as ps:
            # prepare qubits to the excited state
            for ge in ge_labels:
                ps.add(ge, exp.x180(ge))
            ps.barrier()
            # apply the ef drive to induce the ef Rabi oscillation
            for ef in ef_labels:
                ps.add(
                    ef,
                    FlatTop(
                        duration=T + 2 * ramptime,
                        amplitude=amplitudes[ef],
                        tau=ramptime,
                    ),
                )
            ps.barrier()
            for ge in ge_labels:
                ps.add(ge, exp.x180(ge))
        return ps

    # detune target frequencies if necessary
    if detuning is not None:
        frequencies = {
            target: frequencies[target] + detuning for target in amplitudes
        }

    # run the Rabi experiment by sweeping the drive time
    sweep_result = exp.sweep_parameter(
        sequence=gf_rabi_sequence,
        sweep_range=time_range,
        frequencies=frequencies,
        shots=shots,
        interval=interval,
        plot=plot,
    )

    # fit the Rabi oscillation
    gf_rabi_params = {}
    gf_rabi_data = {}
    for qubit, data in sweep_result.data.items():
        ef_label = Target.ef_label(qubit)
        ge_label = Target.ge_label(qubit)
        gf_label = f"{ge_label}_{ef_label}"
        ge_rabi_param = exp.ge_rabi_params[qubit]
        iq_e = ge_rabi_param.endpoints[0]
        fit_result = fitting.fit_rabi(
            target=qubit,
            times=effective_time_range,
            data=data.data,
            reference_point=iq_e,
            plot=plot,
            is_damped=is_damped,
        )

        if fit_result["status"] == "error" or fit_result["r2"] < fit_threshold:
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
            data=data.data,
            time_range=effective_time_range,
            rabi_param=gf_rabi_params[gf_label],
        )

    # store the Rabi parameters if necessary
    if store_params:
        exp.store_rabi_params(gf_rabi_params)

    # create the experiment result
    result = ExperimentResult(
        data=gf_rabi_data,
        rabi_params=gf_rabi_params,
    )

    # return the result
    return result

def obtain_gf_rabi_params(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
    ramptime: float | None = None,
    frequencies: dict[str, float] | None = None,
    is_damped: bool = True,
    fit_threshold: float = 0.5,
    shots: int = CALIBRATION_SHOTS,
    interval: float = DEFAULT_INTERVAL,
    plot: bool = True,
    store_params: bool = False,
) -> ExperimentResult[RabiData]:
    # TODO: Integrate with obtain_rabi_params

    if targets is None:
        targets = exp.qubit_labels
    elif isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    time_range = np.asarray(time_range)

    if ramptime is None:
        ramptime = 32 - 12

    amplitudes = {
        target: exp.params.get_ef_control_amplitude(target) for target in targets
    }

    rabi_data = {}
    rabi_params = {}
    for target in targets:
        ge_label = Target.ge_label(target)
        ef_label = Target.ef_label(target)
        gf_label = f"{ge_label}_{ef_label}"
        data = gf_rabi_experiment(
            exp=exp,
            amplitudes={target: amplitudes[target]},
            time_range=time_range,
            ramptime=ramptime,
            frequencies=frequencies,
            is_damped=is_damped,
            fit_threshold=fit_threshold,
            shots=shots,
            interval=interval,
            store_params=store_params,
            plot=plot,
        ).data[gf_label]
        rabi_data[gf_label] = data
        rabi_params[gf_label] = data.rabi_param

    result = ExperimentResult(
        data=rabi_data,
        rabi_params=rabi_params,
    )
    return result

def gf_chevron_pattern(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    detuning_range: ArrayLike = np.linspace(-0.05, 0.05, 51),
    time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
    frequencies: dict[str, float] | None = None,
    amplitudes: dict[str, float] | None = None,
    rabi_params: dict[str, RabiParam] | None = None,
    shots: int = DEFAULT_SHOTS,
    interval: float = DEFAULT_INTERVAL,
    plot: bool = True,
    save_image: bool = True,
) -> Result:
    if targets is None:
        targets = exp.qubit_labels
    elif isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    if frequencies is None:
        ef_targets = [Target.ef_label(target) for target in targets]
        frequencies = {
            target: exp.targets[target].frequency for target in ef_targets
        }

    detuning_range = np.array(detuning_range, dtype=np.float64)
    time_range = np.array(time_range, dtype=np.float64)

    if amplitudes is None:
        ef_labels = [Target.ef_label(target) for target in targets]
        ef_targets = [exp.targets[ef] for ef in ef_labels]

        amplitudes = {
            ef.label: exp.params.get_ef_control_amplitude(ef.qubit)
            for ef in ef_targets
        }

    ramptime = 32 - 12

    shared_rabi_params: dict[str, RabiParam]
    if rabi_params is None:
        print("Obtaining Rabi parameters between g and f...")
        shared_rabi_params = obtain_gf_rabi_params(
            exp=exp,
            targets=targets,
            time_range=time_range,
            fit_threshold=0.0,
            shots=shots,
            interval=interval,
            plot=False,
            store_params=False,
        ).rabi_params  # type: ignore
    else:
        shared_rabi_params = rabi_params

    rabi_rates: dict[str, NDArray] = {}
    chevron_data: dict[str, NDArray] = {}
    resonant_frequencies: dict[str, float] = {}

    print(f"Targets : {targets}")
    subgroups = exp.util.create_qubit_subgroups(targets)
    figs = {}

    for idx, subgroup in enumerate(subgroups):
        if len(subgroup) == 0:
            continue

        print(f"Subgroup ({idx + 1}/{len(subgroups)}) : {subgroup}")

        rabi_rates_buffer: dict[str, list[float]] = defaultdict(list)
        chevron_data_buffer: dict[str, list[NDArray]] = defaultdict(list)

        ef_subgroup = [Target.ef_label(target) for target in subgroup]

        for detuning in tqdm(detuning_range):
            with exp.util.no_output():

                def gf_rabi_sequence(
                    T: int,
                ) -> PulseSchedule:
                    with PulseSchedule() as ps:
                        # prepare qubits to the excited state
                        for ge in subgroup:
                            ps.add(ge, exp.x180(ge))
                        ps.barrier()
                        # apply the ef drive to induce the ef Rabi oscillation
                        for ef in ef_subgroup:
                            ps.add(
                                ef,
                                FlatTop(
                                    duration=T + 2 * ramptime,
                                    amplitude=amplitudes[ef],
                                    tau=ramptime,
                                ),
                            )
                    return ps

                sweep_result = exp.sweep_parameter(
                    sequence=gf_rabi_sequence,
                    sweep_range=time_range,
                    frequencies={
                        label: frequencies[label] + detuning
                        for label in ef_subgroup
                    },
                    shots=shots,
                    interval=interval,
                    plot=False,
                )
                sweep_data = sweep_result.data

                for target, data in sweep_data.items():
                    fit_result = fitting.fit_rabi(
                        target=data.target,
                        times=data.sweep_range,
                        data=data.data,
                        plot=False,
                    )
                    rabi_rates_buffer[target].append(
                        fit_result.get("frequency", np.nan)
                    )
                    ge_label = Target.ge_label(target)
                    ef_label = Target.ef_label(target)
                    gf_label = f"{ge_label}_{ef_label}"
                    data.rabi_param = shared_rabi_params[gf_label]
                    chevron_data_buffer[target].append(data.normalized)

        for target in subgroup:
            rabi_rates[target] = np.array(rabi_rates_buffer[target])
            chevron_data[target] = np.array(chevron_data_buffer[target]).T
            ef = Target.ef_label(target)
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=detuning_range + frequencies[ef],
                    y=time_range,
                    z=chevron_data[target],
                    colorscale="Viridis",
                )
            )
            fig.update_layout(
                title=dict(
                    text=f"Chevron pattern : {target}",
                    subtitle=dict(
                        text=f"control_amplitude={amplitudes[ef]:.6g}",
                        font=dict(
                            size=13,
                            family="monospace",
                        ),
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
                control_frequencies=detuning_range + frequencies[ef],
                rabi_frequencies=rabi_rates[target],
                plot=plot,
            )
            resonant_frequencies[target] = fit_result["f_resonance"]

            if save_image:
                viz.save_figure_image(
                    fig,
                    name=f"chevron_pattern_{target}",
                    width=600,
                    height=400,
                )
                fig_fit = fit_result["fig"]
                if fig_fit is not None:
                    viz.save_figure_image(
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
            "time_range": time_range,
            "detuning_range": detuning_range,
            "frequencies": frequencies,
            "chevron_data": chevron_data,
            "rabi_rates": rabi_rates,
            "resonant_frequencies": resonant_frequencies,
            "fig": figs,
        }
    )