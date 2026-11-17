"""Contributed single-drive qubit spectroscopy with multiplexed readout candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_SHOTS
from qubex.experiment.models.result import Result
from qubex.experiment.services.characterization_service import (
    DEFAULT_QUBIT_FREQUENCY_SCAN_GAUSSIAN_DURATION,
    DEFAULT_QUBIT_FREQUENCY_SCAN_GAUSSIAN_SIGMA,
    DEFAULT_QUBIT_FREQUENCY_SCAN_INTERVAL,
    DEFAULT_QUBIT_FREQUENCY_SCAN_READOUT_DURATION,
    DEFAULT_QUBIT_FREQUENCY_SCAN_READOUT_RAMPTIME,
    DEFAULT_QUBIT_FREQUENCY_SCAN_SUBRANGE_WIDTH_GHZ,
    DEFAULT_QUBIT_SPECTROSCOPY_POWER_START_DB,
    DEFAULT_QUBIT_SPECTROSCOPY_POWER_STEP_DB,
    DEFAULT_QUBIT_SPECTROSCOPY_POWER_STOP_DB,
)
from qubex.pulse import FlatTop, Gaussian, PulseSchedule


def multiplexed_readout_qubit_spectroscopy(
    exp: Experiment,
    target: str,
    *,
    readout_frequencies: Mapping[str, float],
    readout_amplitudes: Mapping[str, float] | None = None,
    frequency_range: ArrayLike | None = None,
    power_range: ArrayLike | None = None,
    subrange_width: float | None = None,
    simultaneous_drive: bool | None = None,
    shots: int | None = None,
    interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
) -> Result:
    """
    Sweep one drive target while reading every supplied readout candidate.

    Each sweep point drives only `target` and reads all readout candidate slots
    in the same schedule, so one call yields a separate drive-frequency by
    drive-power spectroscopy map for every candidate. No qubit-to-resonator
    mapping is assumed: candidate slots are routed by their configured readout
    resources only, and `target` need not own any of them.

    Parameters
    ----------
    exp : Experiment
        Experiment instance used for pulse execution.
    target : str
        Single qubit label to drive. Its own readout resource is not read
        unless it is also listed in `readout_frequencies`.
    readout_frequencies : Mapping[str, float]
        Readout candidate frequencies in GHz keyed by configured readout slot.
        Keys may be readout labels such as `"RQ00"` or qubit labels such as
        `"Q00"` and are normalized to readout labels. Insertion order defines
        candidate order. Values must be finite and positive.
    readout_amplitudes : Mapping[str, float] | None, optional
        Readout amplitudes keyed by readout slot, using the same key rules as
        `readout_frequencies`. Every key must be a candidate. Candidates
        without an entry use the configured readout amplitude of their slot.
        Values must be finite and non-negative.
    frequency_range : ArrayLike | None, optional
        One-dimensional drive frequency sweep points in GHz. Defaults to the
        drive target's control-box default range.
    power_range : ArrayLike | None, optional
        One-dimensional drive power sweep points in dB. The drive amplitude is
        `sqrt(10 ** (power / 10))`. Defaults to the standard qubit
        spectroscopy power range.
    subrange_width : float | None, optional
        Maximum drive frequency span in GHz per backend sweep segment. Backend
        limits may shorten segments. Defaults to the qubit frequency scan
        default.
    simultaneous_drive : bool | None, optional
        Whether readout starts with the drive pulse. False starts readout after
        the drive pulse ends. Defaults to True.
    shots : int | None, optional
        Number of averaged shots per sweep point. Defaults to `DEFAULT_SHOTS`.
    interval : float | None, optional
        Shot interval in ns. Defaults to the qubit frequency scan interval.
    plot : bool | None, optional
        Whether to show one figure per candidate. Defaults to True.
    save_image : bool | None, optional
        Whether to save one figure per candidate. Defaults to False.

    Returns
    -------
    Result
        Result whose `data` contains:

        - `target`: resolved drive qubit label.
        - `readout_labels`: normalized candidate readout labels in order.
        - `frequency_range`: drive frequency axis in GHz, shape `(F,)`.
        - `power_range`: drive power axis in dB, shape `(P,)`.
        - `simultaneous_drive`: applied timing mode.
        - `shots` and `interval`: applied averaging settings.
        - `candidates`: mapping from readout label to a payload with
          `readout_label`, `candidate_index`, `readout_frequency` (GHz),
          `readout_amplitude`, `frequency_range`, `power_range`, `signals`
          (raw averaged complex IQ, shape `(P, F)`), `phases` (row-normalized
          phase in rad, shape `(P, F)`), and `data` (the same array as
          `phases`, matching the `qubit_spectroscopy` payload key).

        `figures` maps each readout label to its phase heatmap, and `figure`
        is the first candidate's heatmap.

    Raises
    ------
    TypeError
        If `target` is not a string or a mapping or sweep argument has the
        wrong type.
    ValueError
        If labels cannot be resolved, candidates are empty or duplicated, a
        value is non-finite or out of range, or a candidate has no readout
        amplitude.

    Notes
    -----
    This helper touches hardware. Only the drive target's carrier settings are
    retuned, segment by segment, through the backend sweep planner; every
    applied configuration is restored on exit, including after failures.
    AWG and capture units of the drive target and every candidate slot are
    reset once per segment. Readout candidate frequencies are applied as
    per-point carrier overrides without retuning readout LO or NCO settings,
    so they must be realizable under the current readout configuration.
    Persistent parameters and calibration data are not modified.

    Examples
    --------
    >>> result = multiplexed_readout_qubit_spectroscopy(
    ...     exp,
    ...     "Q01",
    ...     readout_frequencies={
    ...         "RQ00": 10.21,
    ...         "RQ01": 10.29,
    ...         "RQ02": 10.36,
    ...         "RQ03": 10.43,
    ...     },
    ...     readout_amplitudes={"RQ00": 0.02, "RQ01": 0.03},
    ... )
    >>> signals = result.data["candidates"]["RQ02"]["signals"]
    """
    if simultaneous_drive is None:
        simultaneous_drive = True
    if shots is None:
        shots = DEFAULT_SHOTS
    if interval is None:
        interval = DEFAULT_QUBIT_FREQUENCY_SCAN_INTERVAL
    if subrange_width is None:
        subrange_width = DEFAULT_QUBIT_FREQUENCY_SCAN_SUBRANGE_WIDTH_GHZ
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False
    if int(shots) <= 0:
        raise ValueError("shots must be positive.")

    qubit = _resolve_drive_target(exp, target)
    candidate_frequencies = _resolve_readout_frequencies(exp, readout_frequencies)
    readout_labels = list(candidate_frequencies)
    candidate_amplitudes = _resolve_readout_amplitudes(
        exp,
        readout_labels,
        readout_amplitudes,
    )
    frequencies = _resolve_frequency_range(exp, qubit, frequency_range)
    powers = _normalize_power_range(power_range)
    data_keys = {label: exp.ctx.resolve_qubit_label(label) for label in readout_labels}
    reset_qubits = {qubit, *data_keys.values()}
    drive_amplitudes = np.sqrt(10 ** (powers / 10))

    signals = {
        label: np.zeros((len(powers), len(frequencies)), dtype=np.complex128)
        for label in readout_labels
    }
    plan = exp.ctx.system_manager.plan_frequency_sweep(
        qubit,
        frequencies,
        subrange_width=subrange_width,
    )
    for segment in plan:
        with exp.ctx.system_manager.modified_sweep_configuration(segment.configuration):
            exp.ctx.reset_awg_and_capunits(qubits=reset_qubits)
            for power_index, drive_amplitude in enumerate(drive_amplitudes):
                schedule = _build_schedule(
                    qubit=qubit,
                    readout_labels=readout_labels,
                    drive_amplitude=float(drive_amplitude),
                    readout_amplitudes=candidate_amplitudes,
                    simultaneous_drive=simultaneous_drive,
                )
                for point_index, drive_frequency in zip(
                    segment.indices, segment.values, strict=True
                ):
                    result = exp.measurement_service.execute(
                        schedule=schedule,
                        mode="avg",
                        n_shots=shots,
                        shot_interval=interval,
                        frequencies={
                            qubit: float(drive_frequency),
                            **candidate_frequencies,
                        },
                        reset_awg_and_capunits=False,
                    )
                    for label in readout_labels:
                        key = data_keys[label]
                        if key not in result.data:
                            raise RuntimeError(
                                f"Measurement result has no data for readout `{label}`."
                            )
                        signals[label][power_index, point_index] = complex(
                            result.data[key][-1].kerneled
                        )

    candidates: dict[str, dict[str, object]] = {}
    figures: dict[str, go.Figure] = {}
    for index, label in enumerate(readout_labels):
        phases = _normalize_phase_rows(signals[label])
        fig = _make_candidate_figure(
            target=qubit,
            readout_label=label,
            frequency_range=frequencies,
            power_range=powers,
            phases=phases,
            readout_frequency=candidate_frequencies[label],
            readout_amplitude=candidate_amplitudes[label],
        )
        figures[label] = fig
        if plot:
            fig.show()
        if save_image:
            viz.save_figure(
                fig,
                name=f"multiplexed_readout_qubit_spectroscopy_{qubit}_{label}",
                width=600,
                height=300,
            )
        candidates[label] = {
            "readout_label": label,
            "candidate_index": index,
            "readout_frequency": candidate_frequencies[label],
            "readout_amplitude": candidate_amplitudes[label],
            "frequency_range": frequencies,
            "power_range": powers,
            "signals": signals[label],
            "phases": phases,
            "data": phases,
        }

    return Result(
        data={
            "target": qubit,
            "readout_labels": readout_labels,
            "frequency_range": frequencies,
            "power_range": powers,
            "simultaneous_drive": simultaneous_drive,
            "shots": shots,
            "interval": interval,
            "candidates": candidates,
        },
        figure=figures[readout_labels[0]],
        figures=figures,
    )


def _resolve_drive_target(exp: Experiment, target: str) -> str:
    """Return the resolved qubit label of the single drive target."""
    if not isinstance(target, str):
        raise TypeError("target must be a single qubit label string.")
    if target == "":
        raise ValueError("target must not be empty.")
    return exp.ctx.resolve_qubit_label(target)


def _resolve_readout_frequencies(
    exp: Experiment,
    readout_frequencies: Mapping[str, float],
) -> dict[str, float]:
    """Return candidate frequencies keyed by normalized readout label."""
    if not isinstance(readout_frequencies, Mapping):
        raise TypeError("readout_frequencies must be a mapping keyed by readout slot.")
    if len(readout_frequencies) == 0:
        raise ValueError("readout_frequencies must contain at least one candidate.")
    values: dict[str, float] = {}
    for key, value in readout_frequencies.items():
        label = exp.ctx.resolve_read_label(key)
        if label in values:
            raise ValueError(
                f"readout_frequencies contains duplicate readout slot `{label}`."
            )
        frequency = float(value)
        if not math.isfinite(frequency):
            raise ValueError(f"readout_frequencies[{key!r}] must be finite.")
        if frequency <= 0:
            raise ValueError(f"readout_frequencies[{key!r}] must be positive.")
        values[label] = frequency
    return values


def _resolve_readout_amplitudes(
    exp: Experiment,
    readout_labels: list[str],
    readout_amplitudes: Mapping[str, float] | None,
) -> dict[str, float]:
    """Return per-candidate readout amplitudes from overrides or parameters."""
    if readout_amplitudes is not None and not isinstance(readout_amplitudes, Mapping):
        raise TypeError("readout_amplitudes must be a mapping keyed by readout slot.")
    overrides: dict[str, float] = {}
    for key, value in (readout_amplitudes or {}).items():
        label = exp.ctx.resolve_read_label(key)
        if label not in readout_labels:
            raise ValueError(
                f"readout_amplitudes key `{key}` is not a readout candidate."
            )
        if label in overrides:
            raise ValueError(
                f"readout_amplitudes contains duplicate readout slot `{label}`."
            )
        amplitude = float(value)
        if not math.isfinite(amplitude):
            raise ValueError(f"readout_amplitudes[{key!r}] must be finite.")
        if amplitude < 0:
            raise ValueError(f"readout_amplitudes[{key!r}] must be non-negative.")
        overrides[label] = amplitude

    defaults = exp.ctx.params.readout_amplitude
    values: dict[str, float] = {}
    for label in readout_labels:
        if label in overrides:
            values[label] = overrides[label]
            continue
        qubit = exp.ctx.resolve_qubit_label(label)
        if qubit not in defaults:
            raise ValueError(
                f"No readout amplitude is available for readout candidate `{label}`."
            )
        values[label] = float(defaults[qubit])
    return values


def _resolve_frequency_range(
    exp: Experiment,
    qubit: str,
    frequency_range: ArrayLike | None,
) -> NDArray[np.float64]:
    """Return the drive frequency sweep points in GHz."""
    if frequency_range is None:
        ctrl_box = exp.ctx.experiment_system.get_control_box_for_qubit(qubit)
        start, stop, step = ctrl_box.traits.default_control_frequency_range
        return np.arange(start, stop, step, dtype=np.float64)
    if isinstance(frequency_range, Mapping):
        raise TypeError("frequency_range must be a one-dimensional array.")
    return _normalize_axis(frequency_range, name="frequency_range")


def _normalize_power_range(power_range: ArrayLike | None) -> NDArray[np.float64]:
    """Return the drive power sweep points in dB."""
    if power_range is None:
        return np.arange(
            DEFAULT_QUBIT_SPECTROSCOPY_POWER_START_DB,
            DEFAULT_QUBIT_SPECTROSCOPY_POWER_STOP_DB,
            DEFAULT_QUBIT_SPECTROSCOPY_POWER_STEP_DB,
            dtype=np.float64,
        )
    return _normalize_axis(power_range, name="power_range")


def _normalize_axis(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Return one non-empty, finite, one-dimensional float array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _build_schedule(
    *,
    qubit: str,
    readout_labels: list[str],
    drive_amplitude: float,
    readout_amplitudes: Mapping[str, float],
    simultaneous_drive: bool,
) -> PulseSchedule:
    """Build one schedule driving `qubit` and reading every candidate slot."""
    with PulseSchedule([qubit, *readout_labels]) as schedule:
        schedule.add(
            qubit,
            Gaussian(
                duration=DEFAULT_QUBIT_FREQUENCY_SCAN_GAUSSIAN_DURATION,
                amplitude=drive_amplitude,
                sigma=DEFAULT_QUBIT_FREQUENCY_SCAN_GAUSSIAN_SIGMA,
            ),
        )
        if not simultaneous_drive:
            schedule.barrier()
        for label in readout_labels:
            schedule.add(
                label,
                FlatTop(
                    duration=DEFAULT_QUBIT_FREQUENCY_SCAN_READOUT_DURATION,
                    amplitude=readout_amplitudes[label],
                    tau=DEFAULT_QUBIT_FREQUENCY_SCAN_READOUT_RAMPTIME,
                ),
            )
    return schedule


def _normalize_phase_rows(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return phases centered on each power row's median, wrapped to [-pi, pi)."""
    phases = np.angle(signals)
    phases -= np.median(phases, axis=1, keepdims=True) - np.pi
    phases %= 2 * np.pi
    phases -= np.pi
    return phases


def _make_candidate_figure(
    *,
    target: str,
    readout_label: str,
    frequency_range: NDArray[np.float64],
    power_range: NDArray[np.float64],
    phases: NDArray[np.float64],
    readout_frequency: float,
    readout_amplitude: float,
) -> go.Figure:
    """Return the phase heatmap of one readout candidate."""
    fig = viz.make_figure()
    fig.add_trace(
        go.Heatmap(
            x=frequency_range,
            y=power_range,
            z=phases,
            colorscale="Viridis",
            colorbar=dict(title=dict(text="Phase (rad)", side="right")),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Qubit spectroscopy : {target} via {readout_label}",
            subtitle=dict(
                text=(
                    f"readout_frequency={readout_frequency:.6g} GHz, "
                    f"readout_amplitude={readout_amplitude:.6g}"
                ),
                font=dict(size=13, family="monospace"),
            ),
        ),
        xaxis_title="Drive frequency (GHz)",
        yaxis_title="Drive power (dB)",
        width=600,
        height=300,
        margin=dict(t=80),
    )
    return fig
