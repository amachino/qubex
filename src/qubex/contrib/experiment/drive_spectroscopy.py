"""Contributed drive spectroscopy helper function."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

import qubex.visualization as viz
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import DEFAULT_INTERVAL, DEFAULT_SHOTS
from qubex.experiment.models.result import Result
from qubex.pulse import Blank, PulseSchedule, Waveform
from qubex.system import MixingUtil
from qubex.system.quel1.quel1_system_constants import CNCO_CENTER_CTRL_HZ

from ._deprecated_options import resolve_shot_options

DEFAULT_DRIVE_SPECTROSCOPY_BAND_WIDTH_GHZ = 0.4
DEFAULT_DRIVE_SPECTROSCOPY_AWG_LIMIT_HZ = 200_000_000
DEFAULT_DRIVE_SPECTROSCOPY_AMPLITUDE_START = 0.0
DEFAULT_DRIVE_SPECTROSCOPY_AMPLITUDE_STOP = 1.0
DEFAULT_DRIVE_SPECTROSCOPY_AMPLITUDE_STEP = 0.02

__all__ = [
    "drive_spectroscopy",
]


@dataclass(frozen=True)
class _FrequencyBand:
    """One drive-frequency band with fixed LO/CNCO/FNCO settings."""

    indices: NDArray[np.int64]
    frequencies: NDArray[np.float64]
    center_frequency: float
    lo_freq: int | None
    cnco_freq: int
    fnco_freq: int
    mixed_frequency: float
    fnco_freqs: dict[str, int]
    mixed_frequencies: dict[str, float]
    max_awg_offset_hz: float

    def as_payload(self) -> dict[str, object]:
        """Return metadata safe to store in the result payload."""
        return {
            "indices": self.indices,
            "frequency_range": self.frequencies,
            "center_frequency": self.center_frequency,
            "lo_freq": self.lo_freq,
            "cnco_freq": self.cnco_freq,
            "fnco_freq": self.fnco_freq,
            "mixed_frequency": self.mixed_frequency,
            "fnco_freqs": self.fnco_freqs,
            "mixed_frequencies": self.mixed_frequencies,
            "max_awg_offset_hz": self.max_awg_offset_hz,
        }


def drive_spectroscopy(
    exp: Experiment,
    target: str,
    *,
    drive_pulse: Waveform,
    measure_qubits: Collection[str] | None = None,
    frequency_range: ArrayLike | None = None,
    amplitude_range: ArrayLike | None = None,
    readout_amplitude: float | Mapping[str, float] | None = None,
    readout_frequency: float | Mapping[str, float] | None = None,
    validate_resources: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Sweep one drive target while measuring one or more qubits.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurement.
    target
        Drive target label. This may be a regular qubit control target or a
        registered custom target such as a bSWAP drive.
    measure_qubits
        Qubits to read out at each sweep point. If omitted, all qubits selected
        in the experiment context are measured.
    frequency_range
        Drive frequency sweep points in GHz. If omitted, the drive target's
        associated qubit default control frequency range is used.
    amplitude_range
        Nominal drive-pulse amplitude scale values. If omitted, a linear
        0-to-1 sweep is used. These values are not Rabi-normalized when the
        drive backend settings are changed between frequency bands.
    drive_pulse
        Drive pulse template. The template is scaled by the power-derived
        amplitude at each power point.
    readout_amplitude
        Scalar readout amplitude for all measured qubits, or mapping keyed by
        measured qubit. If omitted, context readout amplitudes are used.
    readout_frequency
        Scalar readout frequency in GHz for all measured qubits, or mapping
        keyed by measured qubit. If omitted, context readout target frequencies
        are used.
    validate_resources
        Whether to reject duplicate measured qubits, duplicate readout labels,
        and drive/readout label collisions before execution.
    n_shots
        Number of shots per sweep point.
    shot_interval
        Measurement interval.
    plot
        Whether to render per-measured-qubit plots.
    save_image
        Whether to save generated figures.
    **deprecated_options
        Deprecated aliases ``shots`` and ``interval``.

    Returns
    -------
    Result
        Spectroscopy payload containing a top-level primary measurement and a
        ``measurements`` mapping keyed by measured qubit.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="drive_spectroscopy",
    )
    if deprecated_options:
        unexpected = ", ".join(sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")
    if validate_resources is None:
        validate_resources = True
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True

    drive_target = exp.ctx.experiment_system.get_target(target)
    drive_label = drive_target.label
    target_qubit = exp.ctx.resolve_qubit_label(drive_label)
    measured_qubits = _resolve_measure_qubits(exp, measure_qubits)
    readout_labels = {
        qubit: exp.ctx.resolve_read_label(qubit) for qubit in measured_qubits
    }
    if validate_resources:
        _validate_resource_labels(
            drive_label=drive_label,
            measure_qubits=measured_qubits,
            readout_labels=readout_labels,
        )

    frequencies = _normalize_frequency_range(
        exp,
        target_qubit=target_qubit,
        frequency_range=frequency_range,
    )
    amplitudes = _normalize_amplitude_range(amplitude_range)
    readout_amplitudes = _resolve_values(
        measured_qubits,
        provided=readout_amplitude,
        defaults=exp.ctx.params.readout_amplitude,
        value_name="readout_amplitude",
    )
    readout_frequencies = _resolve_readout_frequencies(
        exp,
        measured_qubits,
        readout_labels=readout_labels,
        provided=readout_frequency,
    )

    bands = _split_frequency_bands(
        exp=exp,
        frequencies=frequencies,
        drive_label=drive_label,
        drive_target=drive_target,
        target_qubit=target_qubit,
    )
    signals = {
        qubit: np.empty((len(amplitudes), len(frequencies)), dtype=np.complex128)
        for qubit in measured_qubits
    }
    reset_qubits = list(dict.fromkeys([target_qubit, *measured_qubits]))

    for band in bands:
        with _modified_drive_backend_settings(
            exp=exp,
            drive_label=drive_label,
            band=band,
        ):
            exp.ctx.reset_awg_and_capunits(qubits=reset_qubits)
            for amplitude_index, drive_amplitude in enumerate(amplitudes):
                for local_frequency_index, drive_frequency in enumerate(
                    band.frequencies
                ):
                    frequency_index = int(band.indices[local_frequency_index])
                    frequency_context = {
                        drive_label: float(drive_frequency),
                        **{
                            readout_labels[qubit]: readout_frequencies[qubit]
                            for qubit in measured_qubits
                        },
                    }
                    result = exp.measure(
                        _build_schedule(
                            drive_label=drive_label,
                            target_qubit=target_qubit,
                            measured_qubits=measured_qubits,
                            drive_amplitude=float(drive_amplitude),
                            drive_pulse=drive_pulse,
                        ),
                        frequencies=frequency_context,
                        readout_amplitudes=readout_amplitudes,
                        n_shots=n_shots,
                        shot_interval=shot_interval,
                        reset_awg_and_capunits=False,
                        plot=False,
                    )
                    for qubit in measured_qubits:
                        signals[qubit][amplitude_index, frequency_index] = (
                            _extract_kerneled(result.data[qubit])
                        )

    measurements: dict[str, dict[str, object]] = {}
    figures: dict[str, go.Figure] = {}
    for qubit in measured_qubits:
        signal_array = signals[qubit]
        magnitude_data = np.abs(signal_array)
        pca_projection, pca_metadata = _project_iq_by_band(signal_array, bands)
        pca_row_detrended = pca_projection - np.nanmedian(
            pca_projection,
            axis=1,
            keepdims=True,
        )
        fig = _make_measurement_figure(
            target=drive_label,
            measure_qubit=qubit,
            frequency_range=frequencies,
            amplitude_range=amplitudes,
            data=magnitude_data,
            readout_amplitude=readout_amplitudes[qubit],
        )
        figures[qubit] = fig
        if plot:
            fig.show()
        if save_image:
            suffix = "" if len(measured_qubits) == 1 else f"_{qubit}"
            viz.save_figure(
                fig,
                name=f"drive_spectroscopy_{drive_label}{suffix}",
                width=600,
                height=300,
            )
        measurements[qubit] = {
            "frequency_range": frequencies,
            "amplitude_range": amplitudes,
            "data": magnitude_data,
            "data_kind": "magnitude",
            "signals": signal_array,
            "pca_projection": pca_projection,
            "pca_projection_row_detrended": pca_row_detrended,
            "pca_projection_metadata": pca_metadata,
            "pca_data_kind": "per_band_iq_pca",
            "readout_amplitude": readout_amplitudes[qubit],
            "readout_frequency": readout_frequencies[qubit],
            # TODO: Remove this legacy payload key after callers migrate to .figures.
            "fig": fig,
        }

    primary_qubit = measured_qubits[0]
    primary = measurements[primary_qubit]
    payload = {
        "frequency_range": frequencies,
        "amplitude_range": amplitudes,
        "data": primary["data"],
        "data_kind": primary["data_kind"],
        "signals": primary["signals"],
        "pca_projection": primary["pca_projection"],
        "pca_projection_row_detrended": primary["pca_projection_row_detrended"],
        "pca_projection_metadata": primary["pca_projection_metadata"],
        "pca_data_kind": primary["pca_data_kind"],
        "target": drive_label,
        "target_qubit": target_qubit,
        "measure_qubits": measured_qubits,
        "measurements": measurements,
        "frequency_bands": [band.as_payload() for band in bands],
        "readout_amplitude": primary["readout_amplitude"],
        "readout_frequency": primary["readout_frequency"],
        # TODO: Remove this legacy payload key after callers migrate to .figure.
        "fig": figures[primary_qubit],
    }
    return Result(
        data=payload,
        figure=figures[primary_qubit],
        figures=figures,
    )


def _resolve_measure_qubits(
    exp: Experiment,
    measure_qubits: Collection[str] | None,
) -> list[str]:
    """Return measured qubits, defaulting to all experiment-context qubits."""
    if measure_qubits is None:
        requested = list(exp.ctx.qubit_labels)
    else:
        requested = list(measure_qubits)
    if len(requested) == 0:
        raise ValueError("measure_qubits must contain at least one qubit.")
    return [exp.ctx.resolve_qubit_label(qubit) for qubit in requested]


def _validate_resource_labels(
    *,
    drive_label: str,
    measure_qubits: list[str],
    readout_labels: Mapping[str, str],
) -> None:
    """Reject duplicate or ambiguous labels in a drive spectroscopy schedule."""
    duplicate_qubits = _find_duplicates(measure_qubits)
    if duplicate_qubits:
        joined = ", ".join(duplicate_qubits)
        raise ValueError(f"measure_qubits contains duplicate qubit label(s): {joined}.")

    readout_label_values = [readout_labels[qubit] for qubit in measure_qubits]
    duplicate_readouts = _find_duplicates(readout_label_values)
    if duplicate_readouts:
        joined = ", ".join(duplicate_readouts)
        raise ValueError(
            f"measure_qubits resolve to duplicate readout label(s): {joined}."
        )

    if drive_label in readout_label_values:
        raise ValueError(
            "drive_spectroscopy requires distinct drive and readout labels; "
            f"conflicting label: {drive_label}."
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


def _normalize_frequency_range(
    exp: Experiment,
    *,
    target_qubit: str,
    frequency_range: ArrayLike | None,
) -> NDArray[np.float64]:
    """Return one non-empty 1D drive frequency array."""
    if frequency_range is None:
        ctrl_box = exp.ctx.experiment_system.get_control_box_for_qubit(target_qubit)
        start, stop, step = ctrl_box.traits.default_control_frequency_range
        frequency_range = np.arange(start, stop, step, dtype=np.float64)
    array = np.asarray(frequency_range, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("frequency_range must be one-dimensional.")
    if len(array) == 0:
        raise ValueError("frequency_range must not be empty.")
    return array


def _normalize_amplitude_range(
    amplitude_range: ArrayLike | None,
) -> NDArray[np.float64]:
    """Return one non-empty 1D nominal drive-amplitude array."""
    if amplitude_range is None:
        amplitude_range = np.arange(
            DEFAULT_DRIVE_SPECTROSCOPY_AMPLITUDE_START,
            DEFAULT_DRIVE_SPECTROSCOPY_AMPLITUDE_STOP
            + DEFAULT_DRIVE_SPECTROSCOPY_AMPLITUDE_STEP / 2,
            DEFAULT_DRIVE_SPECTROSCOPY_AMPLITUDE_STEP,
        )
    array = np.asarray(amplitude_range, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("amplitude_range must be one-dimensional.")
    if len(array) == 0:
        raise ValueError("amplitude_range must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("amplitude_range must contain only finite values.")
    return array


def _resolve_values(
    qubits: list[str],
    *,
    provided: float | Mapping[str, float] | None,
    defaults: Mapping[str, float],
    value_name: str,
) -> dict[str, float]:
    """Return scalar, mapped, or default values keyed by measured qubit."""
    if provided is None:
        values: dict[str, float] = {}
        for qubit in qubits:
            if qubit not in defaults:
                raise ValueError(f"{value_name} is missing target: {qubit}")
            values[qubit] = float(defaults[qubit])
        return values
    if isinstance(provided, Mapping):
        values = {}
        for qubit in qubits:
            if qubit in provided:
                values[qubit] = float(provided[qubit])
            elif qubit in defaults:
                values[qubit] = float(defaults[qubit])
            else:
                raise ValueError(f"{value_name} is missing target: {qubit}")
        return values
    return {qubit: float(provided) for qubit in qubits}


def _resolve_readout_frequencies(
    exp: Experiment,
    qubits: list[str],
    *,
    readout_labels: Mapping[str, str],
    provided: float | Mapping[str, float] | None,
) -> dict[str, float]:
    """Return scalar, mapped, or default readout frequencies keyed by measured qubit."""
    defaults = {
        qubit: float(exp.ctx.targets[readout_labels[qubit]].frequency)
        for qubit in qubits
    }
    return _resolve_values(
        qubits,
        provided=provided,
        defaults=defaults,
        value_name="readout_frequency",
    )


def _modified_drive_backend_settings(
    *,
    exp: Experiment,
    drive_label: str,
    band: _FrequencyBand,
):
    """Return a backend-settings context for the drive target."""
    if len(band.fnco_freqs) > 1:
        return _modified_shared_port_backend_settings(
            exp=exp,
            band=band,
        )
    return exp.ctx.system_manager.modified_backend_settings(
        label=drive_label,
        lo_freq=band.lo_freq,
        cnco_freq=band.cnco_freq,
        fnco_freq=band.fnco_freq,
    )


@contextmanager
def _modified_shared_port_backend_settings(
    *,
    exp: Experiment,
    band: _FrequencyBand,
) -> Iterator[None]:
    """Temporarily update multiple generator channels on one shared control port."""
    sm = exp.ctx.system_manager
    targets = {
        label: exp.ctx.experiment_system.get_target(label) for label in band.fnco_freqs
    }
    ports = [target.channel.port for target in targets.values()]
    first_port = ports[0]
    if any(
        port.box_id != first_port.box_id or port.number != first_port.number
        for port in ports[1:]
    ):
        raise ValueError("shared backend settings require one common control port.")

    backend_controller = sm.backend_controller
    config_port = getattr(backend_controller, "config_port", None)
    config_channel = getattr(backend_controller, "config_channel", None)
    initialize_awg_and_capunits = getattr(
        backend_controller,
        "initialize_awg_and_capunits",
        None,
    )
    get_box_config_cache_snapshot = getattr(
        sm,
        "_get_box_config_cache_snapshot",
        None,
    )
    replace_box_config_cache = getattr(
        sm,
        "_replace_box_config_cache",
        None,
    )
    if (
        not callable(config_port)
        or not callable(config_channel)
        or not callable(initialize_awg_and_capunits)
        or not callable(get_box_config_cache_snapshot)
    ):
        raise NotImplementedError(
            "Active backend does not support shared backend-settings updates."
        )

    box_cache = get_box_config_cache_snapshot()
    original_box_cache = deepcopy(box_cache)
    original_port = {
        "lo_freq": first_port.lo_freq,
        "cnco_freq": first_port.cnco_freq,
    }
    original_fncos = {
        label: target.channel.fnco_freq for label, target in targets.items()
    }
    try:
        config_port(
            box_name=first_port.box_id,
            port=first_port.number,
            lo_freq_hz=band.lo_freq,
            cnco_freq_hz=band.cnco_freq,
        )
        port_cache = box_cache[first_port.box_id]["ports"][first_port.number]
        port_cache["lo_freq"] = band.lo_freq
        port_cache["cnco_freq"] = band.cnco_freq
        for label, target in targets.items():
            fnco_freq = band.fnco_freqs[label]
            config_channel(
                box_name=first_port.box_id,
                port=first_port.number,
                channel=target.channel.number,
                fnco_freq_hz=fnco_freq,
            )
            port_cache["channels"][target.channel.number]["fnco_freq"] = fnco_freq
            exp.ctx.experiment_system.update_port_params(
                label,
                lo_freq=band.lo_freq,
                cnco_freq=band.cnco_freq,
                fnco_freq=fnco_freq,
            )

        update_cache = getattr(backend_controller, "update_box_config_cache", None)
        if callable(update_cache):
            update_cache({first_port.box_id: box_cache[first_port.box_id]})
        initialize_awg_and_capunits([first_port.box_id])
        yield
    finally:
        for label in targets:
            exp.ctx.experiment_system.update_port_params(
                label,
                lo_freq=original_port["lo_freq"],
                cnco_freq=original_port["cnco_freq"],
                fnco_freq=original_fncos[label],
            )
        config_port(
            box_name=first_port.box_id,
            port=first_port.number,
            lo_freq_hz=original_port["lo_freq"],
            cnco_freq_hz=original_port["cnco_freq"],
        )
        for label, target in targets.items():
            config_channel(
                box_name=first_port.box_id,
                port=first_port.number,
                channel=target.channel.number,
                fnco_freq_hz=original_fncos[label],
            )
        if callable(replace_box_config_cache):
            replace_box_config_cache(original_box_cache)


def _build_schedule(
    *,
    drive_label: str,
    target_qubit: str,
    measured_qubits: list[str],
    drive_amplitude: float,
    drive_pulse: Waveform,
) -> PulseSchedule:
    """Build one drive pulse, passive blanks, and a barrier before default readout."""
    labels = [
        drive_label,
        *(qubit for qubit in measured_qubits if qubit != target_qubit),
    ]
    with PulseSchedule(labels) as schedule:
        schedule.add(
            drive_label,
            drive_pulse.scaled(drive_amplitude),
        )
        blank = Blank(float(drive_pulse.duration))
        for qubit in measured_qubits:
            if qubit != target_qubit:
                schedule.add(qubit, blank)
        schedule.barrier()
    return schedule


def _extract_kerneled(data: Any) -> complex:
    """Return the final kerneled IQ value from either result-data shape."""
    if isinstance(data, list):
        return complex(data[-1].kerneled)
    return complex(data.kerneled)


def _project_iq_by_band(
    signals: NDArray[np.complex128],
    bands: list[_FrequencyBand],
) -> tuple[NDArray[np.float64], list[dict[str, object]]]:
    """Return per-band PCA projection of row-centered complex IQ signals."""
    projected = np.full(signals.shape, np.nan, dtype=np.float64)
    metadata: list[dict[str, object]] = []
    for band_index, band in enumerate(bands):
        band_projected, band_metadata = _project_iq_band(signals[:, band.indices])
        projected[:, band.indices] = band_projected
        metadata.append({"band_index": band_index, **band_metadata})
    return projected, metadata


def _project_iq_band(
    signals: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Project one band onto its dominant IQ contrast axis."""
    centered = signals - np.nanmedian(signals, axis=1, keepdims=True)
    iq = np.column_stack([centered.real.ravel(), centered.imag.ravel()])
    valid = np.all(np.isfinite(iq), axis=1)
    if np.count_nonzero(valid) < 2:
        return np.zeros(signals.shape, dtype=np.float64), {
            "pc": [1.0, 0.0],
            "scale": 1.0,
        }

    iq_valid = iq[valid] - np.nanmedian(iq[valid], axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(iq_valid, full_matrices=False)
    pc = vh[0]
    projected = (iq @ pc).reshape(signals.shape)
    if np.nanmax(projected) > abs(np.nanmin(projected)):
        projected = -projected
        pc = -pc
    scale = _robust_scale(projected)
    return projected / scale, {"pc": pc.tolist(), "scale": scale}


def _robust_scale(values: NDArray[np.float64]) -> float:
    """Return an IQR-derived finite scale for PCA projection."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    q25, q75 = np.nanpercentile(finite, [25, 75])
    scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanstd(finite)
    if not np.isfinite(scale) or scale == 0:
        return 1.0
    return float(scale)


def _make_measurement_figure(
    *,
    target: str,
    measure_qubit: str,
    frequency_range: NDArray[np.float64],
    amplitude_range: NDArray[np.float64],
    data: NDArray[np.float64],
    readout_amplitude: float,
) -> go.Figure:
    """Return a drive spectroscopy heatmap for one measured qubit."""
    fig = viz.make_figure()
    fig.add_trace(
        go.Heatmap(
            x=frequency_range,
            y=amplitude_range,
            z=data,
            colorscale="Viridis",
            colorbar=dict(
                title=dict(
                    text="Signal (arb. units)",
                    side="right",
                )
            ),
        )
    )
    subtitle = f"readout={measure_qubit}, readout_amplitude={readout_amplitude:.6g}"
    fig.update_layout(
        title=dict(
            text=f"Drive spectroscopy : {target}",
            subtitle=dict(
                text=subtitle,
                font=dict(
                    size=13,
                    family="monospace",
                ),
            ),
        ),
        xaxis_title="Drive frequency (GHz)",
        yaxis_title="Drive amplitude scale",
        width=600,
        height=300,
        margin=dict(t=80),
    )
    return fig


def _split_frequency_bands(
    *,
    exp: Experiment,
    frequencies: NDArray[np.float64],
    drive_label: str,
    drive_target: Any,
    target_qubit: str,
) -> list[_FrequencyBand]:
    """Split drive frequencies into bands satisfying width and AWG-offset limits."""
    if len(frequencies) == 0:
        raise ValueError("frequency_range must not be empty.")
    if not np.all(np.diff(frequencies) >= 0):
        raise ValueError("frequency_range must be sorted in ascending order.")

    bands: list[_FrequencyBand] = []
    start = 0
    while start < len(frequencies):
        best_band: _FrequencyBand | None = None
        stop = start + 1
        while stop <= len(frequencies):
            candidate = frequencies[start:stop]
            if (
                float(candidate[-1] - candidate[0])
                > DEFAULT_DRIVE_SPECTROSCOPY_BAND_WIDTH_GHZ
            ):
                break
            band = _make_frequency_band(
                exp=exp,
                indices=np.arange(start, stop, dtype=np.int64),
                frequencies=candidate,
                drive_label=drive_label,
                drive_target=drive_target,
                target_qubit=target_qubit,
            )
            if band.max_awg_offset_hz > DEFAULT_DRIVE_SPECTROSCOPY_AWG_LIMIT_HZ:
                break
            best_band = band
            stop += 1
        if best_band is None:
            raise ValueError(
                "frequency_range contains a point that cannot be represented within "
                f"±{DEFAULT_DRIVE_SPECTROSCOPY_AWG_LIMIT_HZ / 1e6:.0f} MHz AWG offset."
            )
        bands.append(best_band)
        start = int(best_band.indices[-1]) + 1
    return bands


def _make_frequency_band(
    *,
    exp: Experiment,
    indices: NDArray[np.int64],
    frequencies: NDArray[np.float64],
    drive_label: str,
    drive_target: Any,
    target_qubit: str,
) -> _FrequencyBand:
    """Return fixed LO/CNCO/FNCO settings for one frequency band."""
    center_frequency = float((frequencies[0] + frequencies[-1]) / 2)
    center_frequency_hz = center_frequency * 1e9
    ssb = drive_target.channel.port.sideband
    setting_frequencies_hz = _drive_backend_setting_frequencies(
        exp=exp,
        drive_label=drive_label,
        drive_target=drive_target,
        target_qubit=target_qubit,
        center_frequency_hz=center_frequency_hz,
    )
    basis_frequency_hz = 0.5 * (
        min(setting_frequencies_hz.values()) + max(setting_frequencies_hz.values())
    )
    lo, cnco, _ = MixingUtil.calc_lo_cnco(
        basis_frequency_hz,
        ssb=ssb,
        cnco_center=CNCO_CENTER_CTRL_HZ,
    )
    fnco_freqs: dict[str, int] = {}
    mixed_frequencies: dict[str, float] = {}
    for label, frequency_hz in setting_frequencies_hz.items():
        fnco_freq, mixed_frequency_hz = MixingUtil.calc_fnco(
            f=frequency_hz,
            ssb=ssb,
            lo=lo,
            cnco=cnco,
        )
        fnco_freqs[label] = int(fnco_freq)
        mixed_frequencies[label] = float(mixed_frequency_hz / 1e9)
    drive_mixed_frequency_hz = mixed_frequencies[drive_label] * 1e9
    awg_offsets_hz = frequencies * 1e9 - drive_mixed_frequency_hz
    return _FrequencyBand(
        indices=indices,
        frequencies=frequencies.copy(),
        center_frequency=center_frequency,
        lo_freq=lo,
        cnco_freq=cnco,
        fnco_freq=fnco_freqs[drive_label],
        mixed_frequency=mixed_frequencies[drive_label],
        fnco_freqs=fnco_freqs,
        mixed_frequencies=mixed_frequencies,
        max_awg_offset_hz=float(np.max(np.abs(awg_offsets_hz))),
    )


def _drive_backend_setting_frequencies(
    *,
    exp: Experiment,
    drive_label: str,
    drive_target: Any,
    target_qubit: str,
    center_frequency_hz: float,
) -> dict[str, float]:
    """Return per-target center frequencies that must share one control port."""
    setting_frequencies = {drive_label: center_frequency_hz}
    ge_label = _resolve_ge_label(exp, target_qubit)
    if ge_label == drive_label:
        return setting_frequencies
    try:
        ge_target = exp.ctx.experiment_system.get_target(ge_label)
    except KeyError:
        return setting_frequencies
    if not _same_port(drive_target.channel.port, ge_target.channel.port):
        return setting_frequencies
    setting_frequencies[ge_label] = float(ge_target.frequency) * 1e9
    return setting_frequencies


def _resolve_ge_label(exp: Experiment, target_qubit: str) -> str:
    """Return the GE target label for a qubit, with legacy-context fallback."""
    resolver = getattr(exp.ctx, "resolve_ge_label", None)
    if callable(resolver):
        return str(resolver(target_qubit))
    resolver = getattr(exp.ctx.experiment_system, "resolve_ge_label", None)
    if callable(resolver):
        return str(resolver(target_qubit))
    return target_qubit


def _same_port(left: Any, right: Any) -> bool:
    """Return whether two channel ports identify the same hardware port."""
    return getattr(left, "box_id", None) == getattr(right, "box_id", None) and getattr(
        left, "number", None
    ) == getattr(right, "number", None)
