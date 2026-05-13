"""Contributed helpers for measurement-induced decay characterization."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

import qubex.visualization as viz
from qubex.analysis import FitStatus, IQPlotter, fitting
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
)
from qubex.experiment.models.experiment_result import (
    ExperimentResult,
    SweepData,
    T1Data,
)
from qubex.measurement.models.measure_result import MultipleMeasureResult
from qubex.pulse import Blank, PulseSchedule

from ._deprecated_options import resolve_shot_options

READOUT_DURATION_GRANULARITY_NS = 32.0
REFERENCE_KEY = "reference"


@dataclass(frozen=True)
class _StimulateCondition:
    target: str
    amplitude: float


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    if targets is None:
        return list(exp.ctx.qubit_labels)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _normalize_optional_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str] | None:
    if targets is None:
        return None
    if isinstance(targets, str):
        return [exp.ctx.resolve_qubit_label(targets)]
    return [exp.ctx.resolve_qubit_label(target) for target in targets]


def _unique_in_order(values: Collection[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _resolve_stimulate_conditions(
    exp: Experiment,
    *,
    measurement_targets: list[str],
    stimulate_amplitude: float | Mapping[str, float] | None,
    stimulate_targets: list[str] | None,
) -> list[_StimulateCondition]:
    if isinstance(stimulate_amplitude, Mapping):
        amplitude_map = {
            exp.ctx.resolve_qubit_label(target): float(amplitude)
            for target, amplitude in stimulate_amplitude.items()
        }
        condition_targets = _unique_in_order(
            [*(stimulate_targets or []), *amplitude_map]
        )
        if len(condition_targets) == 0:
            raise ValueError(
                "`stimulate_targets` or `stimulate_amplitude` mapping must not be empty."
            )
        return [
            _StimulateCondition(
                target=target,
                amplitude=amplitude_map.get(
                    target,
                    exp.params.get_readout_amplitude(target),
                ),
            )
            for target in condition_targets
        ]

    default_targets = (
        measurement_targets if stimulate_targets is None else stimulate_targets
    )
    if stimulate_amplitude is None:
        return [
            _StimulateCondition(
                target=target,
                amplitude=exp.params.get_readout_amplitude(target),
            )
            for target in _unique_in_order(default_targets)
        ]

    return [
        _StimulateCondition(target=target, amplitude=float(stimulate_amplitude))
        for target in _unique_in_order(default_targets)
    ]


def _is_multiple(value: float, unit: float) -> bool:
    if unit <= 0:
        return False
    return bool(np.isclose(value / unit, round(value / unit)))


def _discretize_readout_durations(
    exp: Experiment,
    time_range: ArrayLike,
) -> NDArray[np.float64]:
    sampling_period = float(exp.ctx.measurement.sampling_period)
    if not _is_multiple(READOUT_DURATION_GRANULARITY_NS, sampling_period):
        raise ValueError(
            "READOUT_DURATION_GRANULARITY_NS must be a multiple of the pulse "
            f"sampling period: {READOUT_DURATION_GRANULARITY_NS} ns vs "
            f"{sampling_period} ns."
        )

    sweep_range = exp.ctx.util.discretize_time_range(
        np.asarray(time_range, dtype=np.float64),
        sampling_period=READOUT_DURATION_GRANULARITY_NS,
    )
    if np.any(sweep_range <= 0):
        raise ValueError("time_range must contain positive readout durations.")
    return sweep_range


def _build_measurement_induced_decay_sequence(
    exp: Experiment,
    measurement_targets: list[str],
    stimulate_target: str,
    stimulate_amplitude: float,
    wait_ns: float,
) -> PulseSchedule:
    with PulseSchedule() as ps:
        for target in measurement_targets:
            ps.add(target, exp.pulse.get_hpi_pulse(target).repeated(2))
        ps.barrier()

        readout_target = exp.experiment_system.resolve_read_label(stimulate_target)
        ps.add(
            readout_target,
            exp.pulse.readout(
                stimulate_target,
                duration=wait_ns,
                amplitude=stimulate_amplitude,
            ),
        )
        ps.barrier()

        for target in measurement_targets:
            readout_target = exp.experiment_system.resolve_read_label(target)
            ps.add(readout_target, exp.pulse.readout(target))
    return ps


def _build_reference_t1_sequence(
    exp: Experiment,
    measurement_target: str,
    wait_ns: float,
) -> PulseSchedule:
    with PulseSchedule() as ps:
        ps.add(measurement_target, exp.pulse.get_hpi_pulse(measurement_target).repeated(2))
        ps.add(measurement_target, Blank(wait_ns))
        ps.barrier()

        readout_target = exp.experiment_system.resolve_read_label(measurement_target)
        ps.add(readout_target, exp.pulse.readout(measurement_target))
    return ps


def _extract_iq_value(
    measure_result: MultipleMeasureResult,
    exp: Experiment,
    *,
    target: str,
    capture_index: int,
) -> complex:
    readout_target = exp.experiment_system.resolve_read_label(target)
    first_targets = list(measure_result.data)
    if readout_target in first_targets:
        data_target = readout_target
    elif target in first_targets:
        data_target = target
    else:
        raise KeyError(
            f"Neither {readout_target=} nor {target=} exists. "
            f"available targets: {first_targets}"
        )

    captures = measure_result.data[data_target]
    try:
        capture = captures[capture_index]
    except IndexError as exc:
        raise IndexError(
            f"capture_index={capture_index} is out of range for {data_target}."
        ) from exc
    return complex(np.mean(np.asarray(capture.kerneled)))


def _get_rabi_param(exp: Experiment, target: str) -> Any:
    rabi_param = exp.pulse.ge_rabi_params.get(target)
    if rabi_param is None:
        rabi_param = exp.pulse.rabi_params.get(target)
    if rabi_param is None:
        raise ValueError(f"`{target}` does not have a RabiParam.")
    return rabi_param


def _format_stimulate_label(
    *,
    target: str,
    condition: _StimulateCondition | None,
) -> str:
    if condition is None:
        return REFERENCE_KEY
    prefix = "self" if condition.target == target else condition.target
    return f"stimulate={prefix}, stimulate_amplitude={condition.amplitude:g}"


def measurement_induced_decay_experiment(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    stimulate_amplitude: float | Mapping[str, float] | None = None,
    stimulate_targets: Collection[str] | str | None = None,
    time_range: ArrayLike | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    capture_index: int = -1,
    reference: bool = False,
    plot: bool | None = None,
    save_image: bool | None = None,
    enable_tqdm: bool | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    **deprecated_options: Any,
) -> dict[str, dict[str, ExperimentResult[T1Data]]]:
    """
    Run a T1 decay experiment while applying a readout pulse during the wait.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurements.
    targets
        Target qubits to prepare, finally read out, and fit. Each target is
        measured in a separate experiment sequence.
    stimulate_amplitude
        Readout-pulse amplitude applied during the T1 wait. Provide a float for
        all stimulated targets, or a mapping keyed by qubit/resonator to
        stimulate. Mapping keys may differ from `targets`. Defaults to each
        stimulated target's configured readout amplitude.
    stimulate_targets
        Qubits whose readout resonators receive the stimulating readout pulse,
        one condition at a time. If omitted, measured targets are stimulated.
        Targets included here but not in `stimulate_amplitude` use configured
        readout amplitudes from params.
    time_range
        Stimulating readout durations in ns. Values are rounded to the 32 ns
        readout-duration grid after confirming compatibility with the pulse
        sampling period.
    n_shots
        Number of shots per sweep point.
    shot_interval
        Measurement interval in ns.
    capture_index
        Capture index to use for fitting. The default `-1` selects the final
        measurement readout after the stimulating readout pulse.
    reference
        Whether to run a no-stimulus T1 measurement before stimulated
        conditions for each measured target. Reference results are stored under
        the `"reference"` key.
    plot
        Whether to render IQ-plane sweep and fitted decay plots.
    save_image
        Whether to save generated fit figures.
    enable_tqdm
        Whether to show a tqdm progress bar during sweep execution.
    xaxis_type
        X-axis scale for plots.

    Returns
    -------
    dict[str, dict[str, ExperimentResult[T1Data]]]
        Measurement-induced T1 fitting results keyed first by measured target
        and then by stimulated target. Each leaf value keeps the standard T1
        `ExperimentResult[T1Data]` shape.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="measurement_induced_decay_experiment",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False
    if enable_tqdm is None:
        enable_tqdm = False
    if xaxis_type is None:
        xaxis_type = "log"

    target_list = _normalize_targets(exp, targets)
    normalized_stimulate_targets = _normalize_optional_targets(exp, stimulate_targets)
    stimulate_conditions = _resolve_stimulate_conditions(
        exp,
        measurement_targets=target_list,
        stimulate_amplitude=stimulate_amplitude,
        stimulate_targets=normalized_stimulate_targets,
    )

    exp.pulse.validate_rabi_params(target_list)

    if time_range is None:
        time_range = np.logspace(np.log10(100), np.log10(200 * 1000), 51)
    sweep_range = _discretize_readout_durations(exp, time_range)

    results: dict[str, dict[str, ExperimentResult[T1Data]]] = {}
    print(f"Target qubits: {target_list}")
    print(
        "Stimulate qubits: "
        f"{[condition.target for condition in stimulate_conditions]}"
    )
    if reference:
        print("Reference: enabled")

    for measurement_target in target_list:
        target_results: dict[str, ExperimentResult[T1Data]] = {}
        conditions: list[_StimulateCondition | None] = [
            *([None] if reference else []),
            *stimulate_conditions,
        ]
        for condition in conditions:
            if condition is None:
                result_key = REFERENCE_KEY
                print(
                    "Conducting reference T1 with "
                    f"target={measurement_target}...\n"
                )
            else:
                result_key = condition.target
                print(
                    "Conducting measurement-induced decay with "
                    f"target={measurement_target}, "
                    f"stimulate_target={condition.target}, "
                    f"stimulate_amplitude={condition.amplitude:g}...\n"
                )

            signals: list[complex] = []
            plotter = IQPlotter(
                {measurement_target: exp.ctx.state_centers[measurement_target]}
                if measurement_target in exp.ctx.state_centers
                else {}
            )
            reset_qubits = {
                exp.ctx.resolve_qubit_label(target)
                for target in (
                    [measurement_target]
                    if condition is None
                    else [measurement_target, condition.target]
                )
            }
            exp.ctx.reset_awg_and_capunits(qubits=reset_qubits)

            for wait_ns in tqdm(
                sweep_range,
                desc="Sweeping parameters",
                disable=not enable_tqdm,
            ):
                if condition is None:
                    sequence = _build_reference_t1_sequence(
                        exp=exp,
                        measurement_target=measurement_target,
                        wait_ns=float(wait_ns),
                    )
                else:
                    sequence = _build_measurement_induced_decay_sequence(
                        exp=exp,
                        measurement_targets=[measurement_target],
                        stimulate_target=condition.target,
                        stimulate_amplitude=condition.amplitude,
                        wait_ns=float(wait_ns),
                    )
                measure_result = exp.measurement_service.execute(
                    sequence,
                    mode="avg",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    final_measurement=False,
                    reset_awg_and_capunits=False,
                    plot=False,
                )

                signals.append(
                    _extract_iq_value(
                        measure_result,
                        exp,
                        target=measurement_target,
                        capture_index=capture_index,
                    )
                )
                if plot:
                    plotter.update({measurement_target: np.asarray(signals)})

            if plot:
                plotter.clear()
                plotter.to_figure().show()

            iq_values = np.asarray(signals, dtype=np.complex128)
            sweep_data = SweepData(
                target=measurement_target,
                data=iq_values,
                sweep_range=sweep_range,
                rabi_param=_get_rabi_param(exp, measurement_target),
                state_centers=exp.ctx.state_centers.get(measurement_target),
                title=(
                    "T1 reference"
                    if condition is None
                    else (
                        "Measurement-induced decay "
                        f"(stimulate: {condition.target})"
                    )
                ),
                xlabel="Time (μs)",
                ylabel="Measured value",
                xaxis_type=xaxis_type,
                yaxis_type="linear",
            )
            fit_result = fitting.fit_exp_decay(
                target=f"{measurement_target}, "
                f"{_format_stimulate_label(target=measurement_target, condition=condition)}",
                x=sweep_data.sweep_range,
                y=0.5 * (1 - sweep_data.normalized),
                plot=plot,
                title="Measurement-induced T1",
                xlabel="Time (μs)",
                ylabel="Normalized signal",
                xaxis_type=xaxis_type,
                yaxis_type="linear",
            )
            data: dict[str, T1Data] = {}
            if fit_result.status is FitStatus.SUCCESS:
                data[measurement_target] = T1Data.new(
                    sweep_data,
                    t1=fit_result["tau"],
                    t1_err=fit_result["tau_err"],
                    r2=fit_result["r2"],
                )
                if save_image:
                    fig = fit_result.get_figure()
                    suffix = (
                        REFERENCE_KEY
                        if condition is None
                        else f"stim_{condition.target}"
                    )
                    viz.save_figure(
                        fig,
                        name=(
                            f"measurement_induced_t1_{measurement_target}"
                            f"_{suffix}"
                        ),
                    )

            target_results[result_key] = ExperimentResult(
                data=data,
                rabi_params=exp.pulse.rabi_params,
            )

        results[measurement_target] = target_results

    return results
