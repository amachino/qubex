"""Contributed simultaneous qubit spectroscopy helper function."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from contextlib import ExitStack
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_SHOTS
from qubex.experiment.experiment_util import ExperimentUtil
from qubex.experiment.models.result import Result
from qubex.pulse import FlatTop, Gaussian, PulseSchedule
from qubex.system import MixingUtil
from qubex.system.quel1.quel1_system_constants import CNCO_CENTER_CTRL_HZ

DEFAULT_QUBIT_FREQUENCY_SCAN_INTERVAL = 1024.0
DEFAULT_QUBIT_FREQUENCY_SCAN_SUBRANGE_WIDTH_GHZ = 0.3
DEFAULT_QUBIT_FREQUENCY_SCAN_GAUSSIAN_DURATION = 1024
DEFAULT_QUBIT_FREQUENCY_SCAN_GAUSSIAN_SIGMA = 128
DEFAULT_QUBIT_FREQUENCY_SCAN_READOUT_DURATION = 1024
DEFAULT_QUBIT_FREQUENCY_SCAN_READOUT_RAMPTIME = 128
DEFAULT_QUBIT_SPECTROSCOPY_POWER_START_DB = -60
DEFAULT_QUBIT_SPECTROSCOPY_POWER_STOP_DB = 0
DEFAULT_QUBIT_SPECTROSCOPY_POWER_STEP_DB = 5


def simultaneous_qubit_spectroscopy(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    frequency_range: ArrayLike | None = None,
    power_range: ArrayLike | None = None,
    readout_amplitude: Mapping[str, float] | float | None = None,
    control_amplitudes: Mapping[str, float] | None = None,
    readout_amplitudes: Mapping[str, float] | float | None = None,
    readout_frequency: Mapping[str, float] | float | None = None,
    readout_frequencies: Mapping[str, float] | float | None = None,
    simultaneous_drive: bool | None = None,
    validate_resources: bool | None = None,
    shots: int | None = None,
    interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
) -> Result:
    """
    Run shared qubit spectroscopy sweeps for multiple targets simultaneously.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurements.
    targets
        Target qubits to scan. If omitted, all context qubits are used.
    frequency_range
        Shared control frequency sweep points in GHz. If omitted, the first
        target's control box default range is applied to all targets.
    power_range
        Drive power sweep points in dB. If omitted, the standard qubit
        spectroscopy power range is used.
    readout_amplitude
        Readout amplitude. A scalar is applied to all targets, while a mapping
        provides per-target values.
    control_amplitudes
        Optional per-target amplitude scale factors applied to the power-derived
        drive amplitude. Missing values use a scale factor of `1.0`.
    readout_amplitudes
        Alias for `readout_amplitude`.
    readout_frequency
        Readout frequency in GHz. A scalar is applied to all targets, while a
        mapping provides per-target values.
    readout_frequencies
        Alias for `readout_frequency`.
    simultaneous_drive
        Whether readout starts with the drive pulse. If false, readout starts
        after all drive pulses in each sweep point.
    validate_resources
        Whether to reject obvious target and readout label conflicts before
        execution. This shallow validation is enabled by default.
    shots
        Number of shots per sweep point.
    interval
        Measurement interval.
    plot
        Whether to render per-target plots.
    save_image
        Whether to save generated figures.

    Returns
    -------
    Result
        Qubit spectroscopy result. For one target, the payload mirrors
        `qubit_spectroscopy`: `frequency_range`, `power_range`, and `data`.
        For multiple targets, the payload is keyed by target label and each
        value uses that same structure.

    Notes
    -----
    This helper intentionally does not manage LO/CNCO retuning, resource
    allocation, or crosstalk mitigation. It retunes control LO/CNCO settings
    using the same subrange strategy as `scan_qubit_frequencies`. The built-in
    validation only rejects duplicate target labels, duplicate readout labels,
    and control/readout label collisions. Callers must provide shared sweep
    points that are valid for the active hardware configuration.
    """
    if simultaneous_drive is None:
        simultaneous_drive = True
    if validate_resources is None:
        validate_resources = True
    if shots is None:
        shots = DEFAULT_SHOTS
    if interval is None:
        interval = DEFAULT_QUBIT_FREQUENCY_SCAN_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True

    qubits = _resolve_targets(exp, targets)
    resolved_readout_amplitude = _resolve_alias_value(
        canonical=readout_amplitude,
        alias=readout_amplitudes,
        canonical_name="readout_amplitude",
        alias_name="readout_amplitudes",
    )
    resolved_readout_frequency = _resolve_alias_value(
        canonical=readout_frequency,
        alias=readout_frequencies,
        canonical_name="readout_frequency",
        alias_name="readout_frequencies",
    )

    shared_frequency_range = _normalize_frequency_range(exp, qubits, frequency_range)
    frequency_arrays = {qubit: shared_frequency_range.copy() for qubit in qubits}
    coarse_frequencies = _resolve_coarse_frequency_centers(frequency_arrays)
    powers = _normalize_power_range(power_range)
    resonators = {qubit: exp.ctx.resolve_read_label(qubit) for qubit in qubits}
    if validate_resources:
        _validate_resource_labels(qubits=qubits, resonators=resonators)
    control_amplitude_map = _resolve_target_values(
        qubits,
        provided=control_amplitudes,
        defaults=dict.fromkeys(qubits, 1.0),
        value_name="control_amplitudes",
    )
    readout_amplitude_map = _resolve_target_values(
        qubits,
        provided=resolved_readout_amplitude,
        defaults=exp.ctx.params.readout_amplitude,
        value_name="readout_amplitudes",
    )
    readout_frequency_map = _resolve_readout_frequencies(
        exp,
        qubits,
        resonators=resonators,
        provided=resolved_readout_frequency,
    )

    signals: dict[str, list[list[complex]]] = {qubit: [] for qubit in qubits}

    point_count = len(next(iter(frequency_arrays.values())))
    coarse_groups = _iter_coarse_frequency_groups(
        qubits=qubits,
        coarse_frequencies=coarse_frequencies,
        point_count=point_count,
    )
    for power in powers:
        power_linear = 10 ** (float(power) / 10)
        base_control_amplitude = float(np.sqrt(power_linear))
        power_signals: dict[str, list[complex]] = {qubit: [] for qubit in qubits}
        point_control_amplitudes = {
            qubit: base_control_amplitude * control_amplitude_map[qubit]
            for qubit in qubits
        }
        for group in coarse_groups:
            with ExitStack() as stack:
                for qubit in qubits:
                    stack.enter_context(
                        _modified_control_backend_settings(
                            exp=exp,
                            qubit=qubit,
                            coarse_frequency=coarse_frequencies[qubit][group.start],
                        )
                    )
                exp.ctx.reset_awg_and_capunits(qubits=set(qubits))
                for point_index in range(group.start, group.stop):
                    frequencies = {
                        qubit: float(frequency_arrays[qubit][point_index])
                        for qubit in qubits
                    }
                    frequencies.update(
                        {
                            resonators[qubit]: readout_frequency_map[qubit]
                            for qubit in qubits
                        }
                    )
                    with exp.ctx.modified_frequencies(frequencies):
                        schedule = _build_schedule(
                            qubits=qubits,
                            resonators=resonators,
                            control_amplitudes=point_control_amplitudes,
                            readout_amplitudes=readout_amplitude_map,
                            simultaneous_drive=simultaneous_drive,
                        )
                        result = exp.measurement_service.execute(
                            schedule=schedule,
                            mode="avg",
                            shots=shots,
                            interval=interval,
                            reset_awg_and_capunits=False,
                        )
                    for qubit in qubits:
                        power_signals[qubit].append(
                            complex(result.data[qubit][-1].kerneled)
                        )
        for qubit in qubits:
            signals[qubit].append(power_signals[qubit])

    payloads: dict[str, dict[str, object]] = {}
    figures: dict[str, go.Figure] = {}
    for qubit in qubits:
        signal_array = np.asarray(signals[qubit], dtype=np.complex128)
        phase_data = _normalize_phase_rows(signal_array)
        readout_amplitude = readout_amplitude_map[qubit]
        fig = _make_target_figure(
            target=qubit,
            frequency_range=frequency_arrays[qubit],
            power_range=powers,
            data=phase_data,
            readout_amplitude=readout_amplitude,
        )
        figures[qubit] = fig
        if plot:
            fig.show()
        if save_image:
            viz.save_figure(
                fig,
                name=f"qubit_spectroscopy_{qubit}",
                width=600,
                height=300,
            )
        payloads[qubit] = {
            "frequency_range": frequency_arrays[qubit],
            "power_range": powers,
            "data": phase_data,
            "signals": signal_array,
            # TODO: Remove this legacy payload key after callers migrate to .figure/.figures.
            "fig": fig,
        }

    if len(qubits) == 1:
        target = qubits[0]
        return Result(data=payloads[target], figure=figures[target])

    return Result(
        data=payloads,
        figure=figures[qubits[0]],
        figures=figures,
    )


def _resolve_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    """Return resolved qubit labels for the requested targets."""
    if targets is None:
        target_list = list(exp.ctx.qubit_labels)
    elif isinstance(targets, str):
        target_list = [targets]
    else:
        target_list = list(targets)
    if len(target_list) == 0:
        raise ValueError("targets must contain at least one qubit.")
    return [exp.ctx.resolve_qubit_label(target) for target in target_list]


def _validate_resource_labels(
    *,
    qubits: list[str],
    resonators: Mapping[str, str],
) -> None:
    """Reject obvious duplicate or ambiguous labels in a simultaneous schedule."""
    duplicate_qubits = _find_duplicates(qubits)
    if duplicate_qubits:
        joined = ", ".join(duplicate_qubits)
        raise ValueError(
            f"simultaneous targets contain duplicate target label(s): {joined}."
        )

    readout_labels = [resonators[qubit] for qubit in qubits]
    duplicate_readouts = _find_duplicates(readout_labels)
    if duplicate_readouts:
        joined = ", ".join(duplicate_readouts)
        raise ValueError(
            f"simultaneous targets resolve to duplicate readout label(s): {joined}."
        )

    collisions = sorted(set(qubits) & set(readout_labels))
    if collisions:
        joined = ", ".join(collisions)
        raise ValueError(
            "simultaneous spectroscopy requires distinct control and readout "
            f"labels; conflicting label(s): {joined}."
        )


def _find_duplicates(values: list[str]) -> list[str]:
    """Return duplicate strings in first repeated occurrence order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _resolve_alias_value(
    *,
    canonical: object | None,
    alias: object | None,
    canonical_name: str,
    alias_name: str,
) -> Any:
    """Resolve canonical and alias keyword values."""
    if canonical is not None and alias is not None:
        raise ValueError(f"`{alias_name}` conflicts with `{canonical_name}`.")
    return canonical if canonical is not None else alias


def _normalize_frequency_range(
    exp: Experiment,
    qubits: list[str],
    frequency_range: ArrayLike | None,
) -> NDArray[np.float64]:
    """Return one shared float frequency array."""
    if frequency_range is None:
        return _default_frequency_range(exp, qubits[0])
    if isinstance(frequency_range, Mapping):
        raise TypeError("frequency_range must be shared and one-dimensional.")
    return _normalize_frequency_array(
        frequency_range,
        label="frequency_range",
    )


def _normalize_power_range(
    power_range: ArrayLike | None,
) -> NDArray[np.float64]:
    """Return one non-empty 1D power range in dB."""
    if power_range is None:
        power_range = np.arange(
            DEFAULT_QUBIT_SPECTROSCOPY_POWER_START_DB,
            DEFAULT_QUBIT_SPECTROSCOPY_POWER_STOP_DB,
            DEFAULT_QUBIT_SPECTROSCOPY_POWER_STEP_DB,
        )
    array = np.asarray(power_range, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("power_range must be one-dimensional.")
    if len(array) == 0:
        raise ValueError("power_range must not be empty.")
    return array


def _resolve_coarse_frequency_centers(
    frequency_arrays: Mapping[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    """Return per-point coarse center frequencies matching scan subranges."""
    coarse_frequencies: dict[str, NDArray[np.float64]] = {}
    for qubit, frequency_range in frequency_arrays.items():
        subranges = ExperimentUtil.split_frequency_range(
            frequency_range=frequency_range,
            subrange_width=DEFAULT_QUBIT_FREQUENCY_SCAN_SUBRANGE_WIDTH_GHZ,
        )
        centers: list[float] = []
        for subrange in subranges:
            center = float((subrange[0] + subrange[-1]) / 2)
            centers.extend([center] * len(subrange))
        coarse_frequencies[qubit] = np.asarray(centers, dtype=np.float64)
    return coarse_frequencies


def _iter_coarse_frequency_groups(
    *,
    qubits: list[str],
    coarse_frequencies: Mapping[str, NDArray[np.float64]],
    point_count: int,
) -> list[range]:
    """Return consecutive point groups sharing all control coarse settings."""
    groups: list[range] = []
    group_start = 0
    previous_key: tuple[float, ...] | None = None
    for point_index in range(point_count):
        key = tuple(float(coarse_frequencies[qubit][point_index]) for qubit in qubits)
        if previous_key is None:
            previous_key = key
            continue
        if key != previous_key:
            groups.append(range(group_start, point_index))
            group_start = point_index
            previous_key = key
    groups.append(range(group_start, point_count))
    return groups


def _modified_control_backend_settings(
    *,
    exp: Experiment,
    qubit: str,
    coarse_frequency: float,
):
    """Return a backend-settings context for one control target."""
    ctrl_box = exp.ctx.experiment_system.get_control_box_for_qubit(qubit)
    lo, cnco, _ = MixingUtil.calc_lo_cnco(
        coarse_frequency * 1e9,
        ssb=ctrl_box.traits.ctrl_ssb,
        cnco_center=CNCO_CENTER_CTRL_HZ,
    )
    return exp.ctx.system_manager.modified_backend_settings(
        label=qubit,
        lo_freq=lo,
        cnco_freq=cnco,
        fnco_freq=0,
    )


def _normalize_phase_rows(
    signals: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Return qubit-spectroscopy-compatible phases normalized per power row."""
    phases = np.angle(signals)
    phases -= np.median(phases, axis=1, keepdims=True) - np.pi
    phases %= 2 * np.pi
    phases -= np.pi
    return phases


def _default_frequency_range(
    exp: Experiment,
    qubit: str,
) -> NDArray[np.float64]:
    """Return the default control frequency range for one target."""
    ctrl_box = exp.ctx.experiment_system.get_control_box_for_qubit(qubit)
    start, stop, step = ctrl_box.traits.default_control_frequency_range
    return np.arange(start, stop, step, dtype=np.float64)


def _normalize_frequency_array(
    values: ArrayLike,
    *,
    label: str,
) -> NDArray[np.float64]:
    """Return one non-empty 1D frequency array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    if len(array) == 0:
        raise ValueError(f"{label} must not be empty.")
    return array


def _resolve_target_values(
    qubits: list[str],
    *,
    provided: Mapping[str, float] | float | None,
    defaults: Mapping[str, float],
    value_name: str,
) -> dict[str, float]:
    """Return per-target scalar values using provided overrides and defaults."""
    if provided is not None and not isinstance(provided, Mapping):
        value = float(provided)
        return dict.fromkeys(qubits, value)

    values: dict[str, float] = {}
    for qubit in qubits:
        if provided is not None and qubit in provided:
            values[qubit] = float(provided[qubit])
        elif qubit in defaults:
            values[qubit] = float(defaults[qubit])
        else:
            raise ValueError(f"{value_name} is missing target: {qubit}")
    return values


def _resolve_readout_frequencies(
    exp: Experiment,
    qubits: list[str],
    *,
    resonators: Mapping[str, str],
    provided: Mapping[str, float] | float | None,
) -> dict[str, float]:
    """Return per-target readout frequencies from overrides or context targets."""
    if provided is not None and not isinstance(provided, Mapping):
        value = float(provided)
        return dict.fromkeys(qubits, value)

    values: dict[str, float] = {}
    for qubit in qubits:
        if provided is not None and qubit in provided:
            values[qubit] = float(provided[qubit])
        else:
            values[qubit] = float(exp.ctx.targets[resonators[qubit]].frequency)
    return values


def _build_schedule(
    *,
    qubits: list[str],
    resonators: Mapping[str, str],
    control_amplitudes: Mapping[str, float],
    readout_amplitudes: Mapping[str, float],
    simultaneous_drive: bool,
) -> PulseSchedule:
    """Build one simultaneous drive/readout pulse schedule."""
    labels = [*qubits, *(resonators[qubit] for qubit in qubits)]
    with PulseSchedule(labels) as schedule:
        for qubit in qubits:
            schedule.add(
                qubit,
                Gaussian(
                    duration=DEFAULT_QUBIT_FREQUENCY_SCAN_GAUSSIAN_DURATION,
                    amplitude=control_amplitudes[qubit],
                    sigma=DEFAULT_QUBIT_FREQUENCY_SCAN_GAUSSIAN_SIGMA,
                ),
            )
        if not simultaneous_drive:
            schedule.barrier()
        for qubit in qubits:
            schedule.add(
                resonators[qubit],
                FlatTop(
                    duration=DEFAULT_QUBIT_FREQUENCY_SCAN_READOUT_DURATION,
                    amplitude=readout_amplitudes[qubit],
                    tau=DEFAULT_QUBIT_FREQUENCY_SCAN_READOUT_RAMPTIME,
                ),
            )
    return schedule


def _make_target_figure(
    *,
    target: str,
    frequency_range: NDArray[np.float64],
    power_range: NDArray[np.float64],
    data: NDArray[np.float64],
    readout_amplitude: float,
) -> go.Figure:
    """Return a spectroscopy figure for one target."""
    fig = viz.make_figure()
    fig.add_trace(
        go.Heatmap(
            x=frequency_range,
            y=power_range,
            z=data,
            colorscale="Viridis",
            colorbar=dict(
                title=dict(
                    text="Phase (rad)",
                    side="right",
                )
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Qubit spectroscopy : {target}",
            subtitle=dict(
                text=f"readout_amplitude={readout_amplitude:.6g}",
                font=dict(
                    size=13,
                    family="monospace",
                ),
            ),
        ),
        xaxis_title="Frequency (GHz)",
        yaxis_title="Power (dB)",
        width=600,
        height=300,
        margin=dict(t=80),
    )
    return fig
