from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from qxpulse import (
    PulseSchedule,
    Rect,
)
from tqdm import tqdm

import qubex.visualization as viz
from qubex.analysis import fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_RABI_TIME_RANGE,
    DEFAULT_SHOTS,
)
from qubex.experiment.models.result import Result
from qubex.system.target import Target


def ef_chevron_pattern(
    exp: Experiment,
    *,
    targets: Collection[str] | str | None = None,
    detuning_range: ArrayLike = np.linspace(-0.05, 0.05, 51),
    time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
    frequencies: dict[str, float] | None = None,
    amplitudes: dict[str, float] | None = None,
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
        frequencies = {
            Target.ef_label(target): exp.targets[Target.ef_label(target)].frequency
            for target in targets
        }

    detuning_range = np.array(detuning_range, dtype=np.float64)
    time_range = np.array(time_range, dtype=np.float64)

    if amplitudes is None:
        amplitudes = {
            Target.ef_label(target): exp.params.control_amplitude.get(
                Target.ef_label(target),
                exp.params.control_amplitude[target] / np.sqrt(2),
            )
            for target in targets
        }

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

        def ef_rabi_sequence(T: int) -> PulseSchedule:
            with PulseSchedule(subgroup) as ps:
                for target in subgroup:
                    ps.add(target, exp.x180(target))
                    ps.barrier()
                    ps.add(
                        Target.ef_label(target),
                        Rect(
                            duration=T,
                            amplitude=amplitudes[Target.ef_label(target)],
                        ),
                    )
                    ps.barrier()
                    ps.add(target, exp.x180(target))
            return ps

        for detuning in tqdm(detuning_range):
            with exp.util.no_output():
                mod_freqs = {
                    Target.ef_label(target): frequencies[Target.ef_label(target)]
                    + detuning
                    for target in subgroup
                }
                with exp.modified_frequencies(mod_freqs):
                    sweep_result = exp.sweep_parameter(
                        sequence=ef_rabi_sequence,
                        sweep_range=time_range,
                        shots=shots,
                        interval=interval,
                        plot=False,
                    )
                    sweep_data = sweep_result.data

                    for target, data in sweep_data.items():
                        ef_label = Target.ef_label(target)
                        ge_rabi_param = exp.ge_rabi_params[target]
                        iq_g = ge_rabi_param.endpoints[0]
                        fit_result = fitting.fit_rabi(
                            target=data.target,
                            times=data.sweep_range,
                            data=data.data,
                            reference_point=iq_g,
                            plot=False,
                        )
                        rabi_rates_buffer[ef_label].append(
                            fit_result.get("frequency", np.nan)
                        )
                        chevron_data_buffer[ef_label].append(data.normalized)

        for target in subgroup:
            ef_label = Target.ef_label(target)
            rabi_rates[target] = np.array(rabi_rates_buffer[ef_label])
            chevron_data[target] = np.array(chevron_data_buffer[ef_label]).T

            fig = viz.make_figure()
            fig.add_trace(
                go.Heatmap(
                    x=detuning_range + frequencies[ef_label],
                    y=time_range,
                    z=chevron_data[target],
                    colorscale="Viridis",
                )
            )
            fig.update_layout(
                title=dict(
                    text=f"Chevron pattern : {ef_label}",
                    subtitle=dict(
                        text=f"control_amplitude={amplitudes[ef_label]:.6g}, f_ge = {exp.targets[target].frequency:.4f} GHz",
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

            try:
                fit_result = fitting.fit_detuned_rabi(
                    target=target,
                    control_frequencies=detuning_range + frequencies[ef_label],
                    rabi_frequencies=rabi_rates[target],
                    plot=plot,
                )
                resonant_frequencies[target] = fit_result["f_resonance"]
            except Exception:
                resonant_frequencies[target] = np.nan

            if save_image:
                viz.save_figure(
                    fig,
                    name=f"ef_chevron_pattern_{target}",
                    width=600,
                    height=400,
                )
                fig_fit = fit_result["fig"]
                if fig_fit is not None:
                    viz.save_figure(
                        fig_fit,
                        name=f"ef_chevron_pattern_fit_{target}",
                        width=600,
                        height=300,
                    )

    rabi_rates = dict(sorted(rabi_rates.items()))
    chevron_data = dict(sorted(chevron_data.items()))
    resonant_frequencies = dict(sorted(resonant_frequencies.items()))

    print("")
    print("anharmonicity (GHz):")
    for target in sorted(resonant_frequencies.keys()):
        print(
            f"    {target}: {resonant_frequencies[target] - exp.targets[target].frequency:.6g}"
        )
    print("")

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
