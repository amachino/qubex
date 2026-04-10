"""Readout power optimization based on GMM readout fidelity."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import CALIBRATION_SHOTS, DEFAULT_INTERVAL
from qubex.experiment.models.result import Result
from qubex.measurement.classifiers.state_classifier_gmm import StateClassifierGMM

from ._deprecated_options import resolve_shot_options


def _compute_readout_fidelity(
    iq_0: NDArray,
    iq_1: NDArray,
    phase: float = 0.0,
) -> float:
    """Fit a spherical GMM and return the average readout fidelity."""
    classifier = StateClassifierGMM.fit({0: iq_0, 1: iq_1}, phase=phase)
    fidelities = []
    for state, iq in enumerate([iq_0, iq_1]):
        predicted = classifier.predict(iq)
        correct = int(np.sum(predicted == state))
        fidelities.append(correct / len(iq))
    return float(np.mean(fidelities))


def find_optimal_readout_power(
    exp: Experiment,
    target: str,
    *,
    amplitude_range: ArrayLike | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    fidelity_ratio: float = 0.99,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Find readout amplitude maximizing readout fidelity.

    Parameters
    ----------
    target
        Target qubit label.
    amplitude_range
        Readout amplitude sweep range.
    fidelity_ratio
        Fraction of peak fidelity used as the acceptance threshold (0--1).
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="find_optimal_readout_power",
    )
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if amplitude_range is None:
        amplitude_range = np.arange(0.01, 0.26, 0.01)
    else:
        amplitude_range = np.asarray(amplitude_range)

    phase = exp.measurement_service.ctx.reference_phases.get(target, 0.0)

    fidelities: list[float] = []
    for amplitude in tqdm(amplitude_range):
        readout_amp = {target: float(amplitude)}
        result_0 = exp.measurement_service.measure_state(
            {target: "0"},
            mode="single",
            readout_amplitudes=readout_amp,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
        result_1 = exp.measurement_service.measure_state(
            {target: "1"},
            mode="single",
            readout_amplitudes=readout_amp,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
        iq_0 = result_0.data[target].kerneled
        iq_1 = result_1.data[target].kerneled
        fidelities.append(_compute_readout_fidelity(iq_0, iq_1, phase=phase))

    fid_arr = np.array(fidelities)
    fid_peak = float(fid_arr.max())
    plateau_mask = fid_arr >= fid_peak * fidelity_ratio
    best_idx = int(np.where(plateau_mask)[0][0])
    optimal_amplitude = float(amplitude_range[best_idx])

    fig_fid = viz.make_figure(width=600, height=300)
    fig_fid.add_scatter(
        x=list(amplitude_range),
        y=list(fid_arr),
        name="Readout fidelity",
        mode="lines+markers",
    )
    fig_fid.add_vline(
        x=optimal_amplitude,
        line_width=2,
        line_color="red",
        opacity=0.6,
    )
    fig_fid.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=(
            f"amp_opt: {optimal_amplitude:.4f}<br>"
            f"fidelity: {fid_arr[best_idx]:.4f}<br>"
            f"peak: {fid_peak:.4f}"
        ),
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
    )
    fig_fid.update_layout(
        title=f"Readout fidelity : {target}",
        xaxis_title="Amplitude (arb. units)",
        yaxis_title="Readout fidelity",
    )

    classifier_result = exp.build_classifier(
        targets=target,
        readout_amplitudes={target: optimal_amplitude},
        plot=plot,
    )

    figures: dict[str, Any] = {"readout_fidelity": fig_fid}
    if classifier_result.figures is not None:
        figures.update(classifier_result.figures)

    if plot:
        fig_fid.show()
        print(
            f"amp_opt: {optimal_amplitude:.4f}  "
            f"readout_fidelity: {fid_arr[best_idx]:.4f}  "
            f"peak: {fid_peak:.4f}"
        )

    if save_image:
        viz.save_figure(
            fig_fid,
            name=f"optimal_readout_power_fidelity_{target}",
            width=600,
            height=300,
        )

    return Result(
        data={
            "optimal_amplitude": optimal_amplitude,
            "amplitude_range": amplitude_range,
            "readout_fidelity": fid_arr,
            "classifier_result": classifier_result,
        },
        figure=fig_fid,
        figures=figures,
    )
