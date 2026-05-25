"""Contributed stark-driven characterization helper functions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from itertools import product
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

import qubex.visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.analysis.state_tomography import (
    mle_fit_density_matrix,
    plot_ghz_state_tomography,
)
from qubex.clifford import Clifford
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_CR_RAMPTIME,
    DEFAULT_CR_TIME_RANGE,
    DEFAULT_INTERVAL,
    DEFAULT_MAX_N_CLIFFORDS_1Q,
    DEFAULT_MAX_N_CLIFFORDS_2Q,
    DEFAULT_RABI_TIME_RANGE,
    DEFAULT_RB_N_TRIALS,
    DEFAULT_SHOTS,
    DRAG_COEFF,
    DRAG_HPI_DURATION,
    DRAG_PI_DURATION,
    HPI_DURATION,
    HPI_RAMPTIME,
    PI_DURATION,
    PI_RAMPTIME,
)
from qubex.experiment.models import Result
from qubex.experiment.models.calibration_note import DragParam, FlatTopParam
from qubex.experiment.models.experiment_result import (
    AmplCalibData,
    ExperimentResult,
    RabiData,
    RamseyData,
    SweepData,
    T1Data,
    T2Data,
)
from qubex.experiment.models.rabi_param import RabiParam
from qubex.measurement.models import MeasureResult
from qubex.pulse import (
    Blank,
    CrossResonance,
    Drag,
    FlatTop,
    PulseArray,
    PulseSchedule,
    RampType,
    VirtualZ,
    Waveform,
)
from qubex.system import Target, TargetType
from qubex.typing import TargetMap

from ._deprecated_options import resolve_shot_options

StarkShiftModel = Literal["ideal", "duffing", "experimental"]
StarkShiftLookup = (
    ArrayLike | Mapping[float, float] | Mapping[float, Mapping[float, float]]
)
StarkDriveQubit = Literal["control", "target"]


def _get_source_target(exp: Experiment, target: str) -> Target:
    try:
        return exp.targets[target]
    except KeyError:
        raise KeyError(f"Target `{target}` is not registered.") from None


def stark_target(exp: Experiment, target: str) -> str:
    """
    Return the custom Stark target label for a qubit-like target.

    Parameters
    ----------
    exp
        Experiment instance used to resolve the canonical qubit label.
    target
        Qubit or registered target label.

    Returns
    -------
    str
        Custom Stark target label.
    """
    qubit_label = exp.ctx.resolve_qubit_label(target)
    return f"{qubit_label}_stark"


def insitu_target(exp: Experiment, target: str) -> str:
    """
    Return the custom in-situ target label for a qubit-like target.

    Parameters
    ----------
    exp
        Experiment instance used to resolve the canonical qubit label.
    target
        Qubit or registered target label.

    Returns
    -------
    str
        Custom in-situ target label.
    """
    qubit_label = exp.ctx.resolve_qubit_label(target)
    return f"{qubit_label}_insitu"


def stark_cr_target(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_drive_qubit: StarkDriveQubit = "target",
) -> str:
    """
    Return the dressed CR target label used under a Stark tone.

    If the control qubit is Stark-dressed, the CR schedule uses
    ``control_insitu-target``. If the target qubit is Stark-dressed, it uses
    ``control-target_insitu``.
    """
    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    if stark_drive_qubit == "control":
        return f"{insitu_target(exp, control_label)}-{target_label}"
    if stark_drive_qubit == "target":
        return f"{control_label}-{insitu_target(exp, target_label)}"
    raise ValueError("`stark_drive_qubit` must be 'control' or 'target'.")


def make_stark_channel(
    exp: Experiment,
    target: str,
    detuning: float = 0.0,
    lsi: bool = False,
    channel: int = 0,
) -> None:
    """
    Register a detuned Stark-drive target on a shifted generator channel.

    Parameters
    ----------
    exp
        Experiment instance used to register the custom target.
    target
        Source qubit or registered target label.
    detuning
        Frequency offset in GHz from the source target frequency.
    lsi
        Whether to push the calculated FNCO update to LSI settings.
    channel
        Channel-number offset from the source target channel.
    """
    source_target = _get_source_target(exp, target)
    qubit_label = exp.ctx.resolve_qubit_label(target)
    port = source_target.channel.port
    exp.register_custom_target(
        label=stark_target(exp, target),
        frequency=source_target.frequency + detuning,
        box_id=port.box_id,
        port_number=port.number,
        channel_number=source_target.channel.number + channel,
        qubit_label=qubit_label,
        update_lsi=lsi,
    )


def make_insitu_channel(
    exp: Experiment,
    target: str,
    detuning: float = 0.0,
    lsi: bool = False,
    channel: int = 0,
) -> None:
    """
    Register a detuned in-situ target on a shifted generator channel.

    Parameters
    ----------
    exp
        Experiment instance used to register the custom target.
    target
        Source qubit or registered target label.
    detuning
        Frequency offset in GHz from the source target frequency.
    lsi
        Whether to push the calculated FNCO update to LSI settings.
    channel
        Channel-number offset from the source target channel.
    """
    source_target = _get_source_target(exp, target)
    qubit_label = exp.ctx.resolve_qubit_label(target)
    port = source_target.channel.port
    exp.register_custom_target(
        label=insitu_target(exp, target),
        frequency=source_target.frequency + detuning,
        box_id=port.box_id,
        port_number=port.number,
        channel_number=source_target.channel.number + channel,
        qubit_label=qubit_label,
        update_lsi=lsi,
    )


def make_stark_cr_channel(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_drive_qubit: StarkDriveQubit = "target",
    frequency: float | None = None,
    detuning: float = 0.0,
    lsi: bool = False,
    channel: int = 0,
) -> None:
    """
    Register the dressed CR target used during Stark-driven CR calibration.

    The hardware channel is copied from the bare CR target. The frequency
    defaults to the dressed target-qubit frequency, because CR is driven on the
    control line at the target transition. Pass ``frequency`` explicitly if the
    dressed CR resonance is intentionally offset from that value.
    """
    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    if stark_drive_qubit not in ("control", "target"):
        raise ValueError("`stark_drive_qubit` must be 'control' or 'target'.")

    bare_cr_label = f"{control_label}-{target_label}"
    source_cr = _get_source_target(exp, bare_cr_label)
    cr_label = stark_cr_target(
        exp,
        control_label,
        target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    if frequency is None:
        dressed_target = (
            insitu_target(exp, target_label)
            if stark_drive_qubit == "target"
            else target_label
        )
        frequency = exp.targets[dressed_target].frequency

    port = source_cr.channel.port
    exp.register_custom_target(
        label=cr_label,
        frequency=frequency + detuning,
        box_id=port.box_id,
        port_number=port.number,
        channel_number=source_cr.channel.number + channel,
        qubit_label=control_label,
        target_type=TargetType.CTRL_CR,
        update_lsi=lsi,
    )


def _plot_sequence_sample(
    sequence: PulseSchedule,
    *,
    title: str,
    plot: bool,
) -> PulseSchedule:
    if plot:
        sequence.plot(title=title)
    return sequence


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
            if fit_result.status is not FitStatus.SUCCESS:
                continue
            t1_data = T1Data.new(
                sweep_data,
                t1=fit_result["tau"],
                t1_err=fit_result["tau_err"],
                r2=fit_result["r2"],
            )
            data[qubit] = t1_data
            if save_image:
                fig = fit_result.get_figure()
                viz.save_figure(fig, name=f"t1_{qubit}")

    return ExperimentResult(data=data)


def stark_rabi_sequence(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    amplitude: float | Mapping[str, float] | None = None,
    stark_ramptime: float | None = None,
    duration: float = 100.0,
    ramptime: float | None = None,
    plot: bool = True,
) -> PulseSchedule:
    """Build and optionally plot one in-situ Rabi sequence under a Stark tone."""
    if ramptime is None:
        ramptime = 0.0
    stark_power = _stark_drive_amplitude(
        exp,
        target=target,
        stark_amplitude=stark_amplitude,
    )
    drive_amplitude = _single_target_amplitude(
        exp,
        target=target,
        amplitude=amplitude,
    )
    insitu_pulse = FlatTop(
        duration=duration + 2 * ramptime,
        amplitude=drive_amplitude,
        tau=ramptime,
        sampling_period=_measurement_sampling_period(exp),
    )
    sequence = _stark_wrapped_schedule(
        stark_label=stark_target(exp, target),
        insitu_label=insitu_target(exp, target),
        stark_amplitude=stark_power,
        stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
        insitu_sequence=insitu_pulse,
    )
    return _plot_sequence_sample(
        sequence,
        title=f"AC Stark in-situ Rabi sample: {target}",
        plot=plot,
    )


def stark_rabi_experiment(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    amplitude: float | Mapping[str, float] | None = None,
    stark_ramptime: float | None = None,
    time_range: ArrayLike | None = None,
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
    Run an in-situ Rabi experiment while the Stark tone is applied.

    The fitted Rabi data are keyed by the physical qubit label, while the drive
    pulse is applied to the in-situ custom target.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_rabi_experiment",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if time_range is None:
        time_range = DEFAULT_RABI_TIME_RANGE
    if ramptime is None:
        ramptime = 0.0
    if is_damped is None:
        is_damped = True
    if fit_threshold is None:
        fit_threshold = 0.5
    if plot is None:
        plot = True
    if store_params is None:
        store_params = False

    insitu_label = insitu_target(exp, target)
    drive_amplitude = _single_target_amplitude(
        exp,
        target=target,
        amplitude=amplitude,
    )
    sweep_range = np.asarray(time_range, dtype=float)
    effective_time_range = sweep_range + ramptime

    if frequencies is None:
        frequencies = {insitu_label: exp.targets[insitu_label].frequency}
    if detuning is not None:
        frequencies = {
            label: frequency + detuning for label, frequency in frequencies.items()
        }

    reference_points = exp.measurement_service.obtain_reference_points(
        [target],
        n_shots=DEFAULT_SHOTS,
    )["iq"]

    def sequence(duration: float) -> PulseSchedule:
        return stark_rabi_sequence(
            exp,
            target=target,
            amplitude=drive_amplitude,
            stark_amplitude=stark_amplitude,
            stark_ramptime=stark_ramptime,
            duration=duration,
            ramptime=ramptime,
            plot=False,
        )

    sweep_result = exp.measurement_service.sweep_parameter(
        sequence=sequence,
        sweep_range=sweep_range,
        frequencies=frequencies,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
    )

    rabi_params: dict[str, RabiParam] = {}
    for qubit, sweep_data in sweep_result.data.items():
        fit_result = fitting.fit_rabi(
            target=qubit,
            times=effective_time_range,
            data=sweep_data.data,
            reference_point=reference_points.get(qubit),
            plot=plot,
            is_damped=is_damped,
        )
        if fit_result.status is FitStatus.ERROR or fit_result["r2"] < fit_threshold:
            rabi_params[qubit] = RabiParam.nan(target=qubit)
        else:
            rabi_params[qubit] = RabiParam(
                target=qubit,
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
    if store_params:
        exp.ctx.store_rabi_params(rabi_params)

    return ExperimentResult(
        data={
            qubit: RabiData(
                target=qubit,
                data=sweep_data.data,
                time_range=effective_time_range,
                rabi_param=rabi_params[qubit],
                state_centers=exp.ctx.state_centers.get(qubit),
            )
            for qubit, sweep_data in sweep_result.data.items()
        },
        rabi_params=rabi_params,
    )


def stark_repeat_sequence_sample(
    exp: Experiment,
    sequence: TargetMap[Waveform],
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    count: int = 1,
    plot: bool = True,
) -> PulseSchedule:
    """Build and optionally plot one repeated in-situ sequence under Stark tones."""
    resolved_ramptime = _resolve_stark_ramptime(stark_ramptime)
    with PulseSchedule() as ps:
        for target, pulse in sequence.items():
            stark_label = stark_target(exp, target)
            insitu_label = insitu_target(exp, target)
            repeated = pulse.repeated(count)
            stark_power = _stark_drive_amplitude(
                exp,
                target=target,
                stark_amplitude=stark_amplitude,
            )
            ps.add(
                stark_label,
                FlatTop(
                    duration=repeated.duration + 2 * resolved_ramptime,
                    amplitude=stark_power,
                    tau=resolved_ramptime,
                ),
            )
            ps.add(insitu_label, Blank(resolved_ramptime))
            ps.add(insitu_label, repeated)
    return _plot_sequence_sample(
        ps,
        title=f"AC Stark repeated sequence sample: count={count}",
        plot=plot,
    )


def stark_repeat_sequence(
    exp: Experiment,
    sequence: TargetMap[Waveform],
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    initial_states: dict[str, str] | None = None,
    repetitions: int = 20,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    **deprecated_options: Any,
) -> ExperimentResult[SweepData]:
    """Sweep repeated in-situ gate sequences while the Stark tone is applied."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_repeat_sequence",
    )

    def repeated_sequence(count: int) -> PulseSchedule:
        return stark_repeat_sequence_sample(
            exp,
            sequence,
            stark_amplitude=stark_amplitude,
            stark_ramptime=stark_ramptime,
            count=count,
            plot=False,
        )

    result = exp.measurement_service.sweep_parameter(
        sequence=repeated_sequence,
        sweep_range=np.arange(repetitions + 1),
        initial_states=initial_states,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
        xlabel="Number of repetitions",
    )
    if plot:
        result.plot(normalize=True)
    return result


def stark_chevron_pattern(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    detuning_range: ArrayLike | None = None,
    time_range: ArrayLike | None = None,
    frequencies: dict[str, float] | None = None,
    amplitude: float | Mapping[str, float] | None = None,
    rabi_params: dict[str, RabiParam] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    save_image: bool = True,
    **deprecated_options: Any,
) -> Result:
    """Measure a frequency-time chevron for the in-situ target under Stark drive."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_chevron_pattern",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if detuning_range is None:
        detuning_range = np.linspace(-0.05, 0.05, 51)
    if time_range is None:
        time_range = DEFAULT_RABI_TIME_RANGE

    insitu_label = insitu_target(exp, target)
    if frequencies is None:
        frequencies = {insitu_label: exp.targets[insitu_label].frequency}

    detuning_values = np.asarray(detuning_range, dtype=float)
    time_values = np.asarray(time_range, dtype=float)
    drive_amplitude = _single_target_amplitude(
        exp,
        target=target,
        amplitude=amplitude,
    )
    stark_power = _stark_drive_amplitude(
        exp,
        target=target,
        stark_amplitude=stark_amplitude,
    )

    if rabi_params is None:
        print("Obtaining Rabi parameters...")
        shared_rabi_params = exp.measurement_service.obtain_rabi_params(
            targets=target,
            amplitudes={target: drive_amplitude},
            time_range=time_values,
            fit_threshold=0.0,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            store_params=False,
        ).rabi_params
    else:
        shared_rabi_params = rabi_params
    if shared_rabi_params is None:
        raise ValueError("Rabi parameters could not be resolved for chevron fitting.")

    rabi_rates_buffer: dict[str, list[float]] = defaultdict(list)
    chevron_buffer: dict[str, list[NDArray]] = defaultdict(list)
    for detuning in detuning_values:
        with exp.ctx.util.no_output():
            rabi_result = stark_rabi_experiment(
                exp,
                target=target,
                amplitude=drive_amplitude,
                stark_amplitude=stark_amplitude,
                stark_ramptime=stark_ramptime,
                time_range=time_values,
                frequencies={insitu_label: frequencies[insitu_label] + detuning},
                plot=False,
                n_shots=n_shots,
                shot_interval=shot_interval,
            )
        if rabi_result.rabi_params is None:
            raise ValueError("Rabi fit did not return parameters.")
        rabi_rates_buffer[target].append(rabi_result.rabi_params[target].frequency)
        rabi_result.data[target].rabi_param = shared_rabi_params[target]
        chevron_buffer[target].append(rabi_result.data[target].normalized)

    rabi_rates = {target: np.asarray(rabi_rates_buffer[target])}
    chevron_data = {target: np.asarray(chevron_buffer[target]).T}

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=detuning_values + frequencies[insitu_label],
            y=time_values,
            z=chevron_data[target],
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Stark Chevron pattern : {target}",
            subtitle=dict(
                text=f"control_amplitude={drive_amplitude:.6g}, stark_amplitude={stark_power:.6g}",
                font=dict(size=13, family="monospace"),
            ),
        ),
        xaxis_title="Drive frequency (GHz)",
        yaxis_title="Time (ns)",
        width=600,
        height=400,
        margin=dict(t=80),
    )
    if plot:
        fig.show()

    fit_result = fitting.fit_detuned_rabi(
        target=target,
        control_frequencies=detuning_values + frequencies[insitu_label],
        rabi_frequencies=rabi_rates[target],
        plot=plot,
    )
    resonant_frequencies = {target: fit_result["f_resonance"]}
    detuning = resonant_frequencies[target] - exp.targets[target].frequency
    print("Detuning frequency")
    print(f" {target}: {detuning:.6f}")
    make_insitu_channel(
        exp,
        target=target,
        detuning=detuning,
        lsi=False,
        channel=0,
    )

    figures = {"chevron": fig}
    if save_image:
        viz.save_figure(fig, name=f"stark_chevron_pattern_{target}")
        fig_fit = fit_result.get_figure()
        figures["fit"] = fig_fit
        viz.save_figure(fig_fit, name=f"stark_chevron_pattern_fit_{target}")

    return Result(
        data={
            "time_range": time_values,
            "detuning_range": detuning_values,
            "frequencies": frequencies,
            "chevron_data": chevron_data,
            "rabi_rates": rabi_rates,
            "resonant_frequencies": resonant_frequencies,
        },
        figure=fig,
        figures=figures,
    )


def stark_rb_sequence_1q(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    n: int = 8,
    seed: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    interleaved_clifford: str | Clifford | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
    plot: bool = True,
) -> PulseSchedule:
    """
    Build and optionally plot one 1Q RB schedule under a continuous Stark tone.

    This is the same schedule generator used by `stark_rb_experiment_1q`.
    """
    source_target = exp.ctx.experiment_system.get_target(target)
    if source_target.is_cr:
        raise ValueError(f"`{target}` is not a 1Q target.")

    insitu_label = insitu_target(exp, target)
    resolved_clifford = (
        None
        if interleaved_clifford is None
        else _resolve_clifford(exp, interleaved_clifford)
    )
    rb_sequence = exp.benchmarking_service.rb_sequence_1q(
        target=insitu_label,
        n=n,
        x90=_target_map_get(x90, target=target, insitu=insitu_label),
        interleaved_waveform=_target_map_get(
            interleaved_waveform,
            target=target,
            insitu=insitu_label,
        ),
        interleaved_clifford=resolved_clifford,
        seed=seed,
    )
    sequence = _stark_wrapped_schedule(
        stark_label=stark_target(exp, target),
        insitu_label=insitu_label,
        stark_amplitude=_stark_drive_amplitude(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
        ),
        stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
        insitu_sequence=rb_sequence,
    )
    return _plot_sequence_sample(
        sequence,
        title=f"AC Stark RB sample: {target}, n={n}",
        plot=plot,
    )


def stark_purity_sequence_1q(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    n: int = 8,
    seed: int | None = None,
    basis: Literal["X", "Y", "Z"] = "Z",
    x90: TargetMap[Waveform] | None = None,
    interleaved_clifford: str | Clifford | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
    plot: bool = True,
) -> PulseSchedule:
    """
    Build and optionally plot one 1Q purity benchmarking schedule under Stark.

    This is the same schedule generator used by `stark_purity_experiment_1q`.
    """
    from .purity_benchmarking import purity_sequence_1q

    source_target = exp.ctx.experiment_system.get_target(target)
    if source_target.is_cr:
        raise ValueError(f"`{target}` is not a 1Q target.")

    insitu_label = insitu_target(exp, target)
    resolved_clifford = (
        None
        if interleaved_clifford is None
        else _resolve_clifford(exp, interleaved_clifford)
    )
    purity_sequence = purity_sequence_1q(
        exp,
        insitu_label,
        n=n,
        x90=_target_map_get(x90, target=target, insitu=insitu_label),
        interleaved_waveform=_target_map_get(
            interleaved_waveform,
            target=target,
            insitu=insitu_label,
        ),
        interleaved_clifford=resolved_clifford,
        seed=seed,
        basis=basis,
    )
    sequence = _stark_wrapped_schedule(
        stark_label=stark_target(exp, target),
        insitu_label=insitu_label,
        stark_amplitude=_stark_drive_amplitude(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
        ),
        stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
        insitu_sequence=purity_sequence,
    )
    return _plot_sequence_sample(
        sequence,
        title=f"AC Stark purity sample: {target}, n={n}, basis={basis}",
        plot=plot,
    )


def stark_rb_experiment_1q(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    interleaved_clifford: Clifford | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool = True,
    save_image: bool = True,
    **deprecated_options: Any,
) -> Result:
    """Run single-qubit randomized benchmarking under a continuous Stark tone."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_rb_experiment_1q",
    )
    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS
    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_1Q
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if xaxis_type is None:
        xaxis_type = "linear"

    source_target = exp.ctx.experiment_system.get_target(target)
    if source_target.is_cr:
        raise ValueError(f"`{target}` is not a 1Q target.")

    if seeds is None:
        seed_values = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seed_values = np.asarray(seeds, dtype=int)
        if len(seed_values) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )
    sweep_range = _clifford_sweep_range(
        n_cliffords_range=n_cliffords_range,
        max_n_cliffords=max_n_cliffords,
    )

    def sequence(n_clifford: int, seed: int) -> PulseSchedule:
        return stark_rb_sequence_1q(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
            stark_ramptime=stark_ramptime,
            n=n_clifford,
            seed=seed,
            x90=x90,
            interleaved_clifford=interleaved_clifford,
            interleaved_waveform=interleaved_waveform,
            plot=False,
        )

    mean_data: dict[str, list[float]] = defaultdict(list)
    std_data: dict[str, list[float]] = defaultdict(list)
    for n_clifford in sweep_range:
        trial_data: dict[str, list[float]] = defaultdict(list)
        for seed in seed_values:
            result = exp.measurement_service.measure(
                sequence=sequence(int(n_clifford), int(seed)),
                mode="avg",
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=False,
            )
            for qubit, data in result.data.items():
                z_value = exp.pulse.rabi_params[qubit].normalize(data.kerneled)
                trial_data[qubit].append((z_value + 1) / 2)
        mean_data[target].append(float(np.mean(trial_data[target])))
        std_data[target].append(float(np.std(trial_data[target])))
        if (
            n_cliffords_range is None
            and mean_data[target][-1] - 0.5 * std_data[target][-1] < 0.5
        ):
            break

    actual_sweep_range = sweep_range[: len(mean_data[target])]
    mean = np.asarray(mean_data[target])
    std = np.asarray(std_data[target]) if n_trials > 1 else None
    fit_result = fitting.fit_rb(
        target=target,
        x=actual_sweep_range,
        y=mean,
        error_y=std,
        bounds=((0, 0, 0), (0.5, 1, 1)),
        title="Stark randomized benchmarking",
        xlabel="Number of Cliffords",
        ylabel="Normalized signal",
        xaxis_type=xaxis_type,
        yaxis_type="linear",
        plot=plot,
    )
    fig = fit_result.get_figure()
    if save_image:
        viz.save_figure(fig, name=f"stark_rb_experiment_1q_{target}")
    return Result(
        data={
            target: {
                "n_cliffords": actual_sweep_range,
                "mean": mean,
                "std": std,
                **fit_result,
            }
        },
        figures={target: fig},
    )


def stark_purity_experiment_1q(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    interleaved_clifford: Clifford | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool = True,
    save_image: bool = True,
    **deprecated_options: Any,
) -> Result:
    """Run single-qubit purity benchmarking under a continuous Stark tone."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_purity_experiment_1q",
    )
    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS
    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_1Q
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if xaxis_type is None:
        xaxis_type = "linear"

    source_target = exp.ctx.experiment_system.get_target(target)
    if source_target.is_cr:
        raise ValueError(f"`{target}` is not a 1Q target.")

    if seeds is None:
        seed_values = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seed_values = np.asarray(seeds, dtype=int)
        if len(seed_values) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )
    sweep_range = _clifford_sweep_range(
        n_cliffords_range=n_cliffords_range,
        max_n_cliffords=max_n_cliffords,
    )

    def sequence(
        n_clifford: int,
        seed: int,
        basis: Literal["X", "Y", "Z"],
    ) -> PulseSchedule:
        return stark_purity_sequence_1q(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
            stark_ramptime=stark_ramptime,
            n=n_clifford,
            seed=seed,
            basis=basis,
            x90=x90,
            interleaved_clifford=interleaved_clifford,
            interleaved_waveform=interleaved_waveform,
            plot=False,
        )

    mean_data: dict[str, list[float]] = defaultdict(list)
    std_data: dict[str, list[float]] = defaultdict(list)
    for n_clifford in sweep_range:
        trial_data: dict[str, list[float]] = defaultdict(list)
        for seed in seed_values:
            basis_values: dict[str, list[float]] = defaultdict(list)
            for basis in ("X", "Y", "Z"):
                result = exp.measurement_service.measure(
                    sequence=sequence(int(n_clifford), int(seed), basis),
                    mode="avg",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                )
                for qubit, data in result.data.items():
                    basis_values[qubit].append(
                        exp.pulse.rabi_params[qubit].normalize(data.kerneled)
                    )
            for qubit, values in basis_values.items():
                x_val, y_val, z_val = values
                trial_data[qubit].append(0.5 * (1 + x_val**2 + y_val**2 + z_val**2))
        mean_data[target].append(float(np.mean(trial_data[target])))
        std_data[target].append(float(np.std(trial_data[target])))
        if (
            n_cliffords_range is None
            and mean_data[target][-1] - 0.5 * std_data[target][-1] < 0.5
        ):
            break

    actual_sweep_range = sweep_range[: len(mean_data[target])]
    mean = np.asarray(mean_data[target])
    std = np.asarray(std_data[target]) if n_trials > 1 else None
    fit_result = fitting.fit_rb(
        target=target,
        x=actual_sweep_range,
        y=mean,
        error_y=std,
        bounds=((0, 0, 0), (0.5, 1, 1)),
        title="Stark purity benchmarking",
        xlabel="Number of Cliffords",
        ylabel="Purity",
        xaxis_type=xaxis_type,
        yaxis_type="linear",
        plot=plot,
    )
    fig = fit_result.get_figure()
    if save_image:
        viz.save_figure(fig, name=f"stark_purity_experiment_1q_{target}")
    return Result(
        data={
            target: {
                "n_cliffords": actual_sweep_range,
                "mean": mean,
                "std": std,
                **fit_result,
            }
        },
        figures={target: fig},
    )


def _interleaved_fit_result(
    *,
    target: str,
    reference_result: Result,
    interleaved_result: Result,
    clifford: Clifford,
    plot: bool,
    save_image: bool,
    image_name: str,
    title_prefix: str,
    dimension: int = 2,
) -> Result:
    reference_data = reference_result.data[target]
    interleaved_data = interleaved_result.data[target]
    rb_fit_result = fitting.fit_rb(
        target=target,
        x=reference_data["n_cliffords"],
        y=reference_data["mean"],
        error_y=reference_data["std"],
        dimension=dimension,
        plot=False,
    )
    irb_fit_result = fitting.fit_rb(
        target=target,
        x=interleaved_data["n_cliffords"],
        y=interleaved_data["mean"],
        error_y=interleaved_data["std"],
        dimension=dimension,
        plot=False,
        title=title_prefix,
    )
    p_rb = rb_fit_result["p"]
    p_irb = irb_fit_result["p"]
    p_rb_err = rb_fit_result["p_err"]
    p_irb_err = irb_fit_result["p_err"]
    gate_error = (dimension - 1) * (1 - (p_irb / p_rb)) / dimension
    gate_fidelity = 1 - gate_error
    gate_fidelity_err = (
        (dimension - 1)
        / dimension
        * np.sqrt((p_irb_err / p_rb) ** 2 + (p_rb_err * p_irb / p_rb**2) ** 2)
    )
    fig = fitting.plot_irb(
        target=target,
        x=reference_data["n_cliffords"],
        y_rb=reference_data["mean"],
        y_irb=interleaved_data["mean"],
        error_y_rb=reference_data["std"],
        error_y_irb=interleaved_data["std"],
        A_rb=rb_fit_result["A"],
        A_irb=irb_fit_result["A"],
        p_rb=p_rb,
        p_irb=p_irb,
        C_rb=rb_fit_result["C"],
        C_irb=irb_fit_result["C"],
        gate_fidelity=gate_fidelity,
        gate_fidelity_err=gate_fidelity_err,
        plot=plot,
        title=f"{title_prefix} of {clifford.name}",
        xlabel="Number of Cliffords",
        ylabel="Normalized signal",
    )
    if save_image:
        viz.save_figure(fig, name=image_name)

    avg_gate_error_rb = rb_fit_result["avg_gate_error"]
    print()
    print(
        f"Average gate fidelity (RB)  : {rb_fit_result['avg_gate_fidelity'] * 100:.3f} ± {rb_fit_result['avg_gate_fidelity_err'] * 100:.3f}%"
    )
    print(
        f"Average gate fidelity (IRB) : {irb_fit_result['avg_gate_fidelity'] * 100:.3f} ± {irb_fit_result['avg_gate_fidelity_err'] * 100:.3f}%"
    )
    print()
    print(f"Gate error    : {gate_error * 100:.3f} ± {gate_fidelity_err * 100:.3f}%")
    print(f"Gate fidelity : {gate_fidelity * 100:.3f} ± {gate_fidelity_err * 100:.3f}%")
    print()
    if gate_error < 0.1 * avg_gate_error_rb:
        print(
            f"Warning: Gate error ({gate_error * 100:.3f}%) is too low compared to the average gate error (RB) ({avg_gate_error_rb * 100:.3f}%)."
        )

    return Result(
        data={
            target: {
                "gate_error": gate_error,
                "gate_fidelity": gate_fidelity,
                "gate_fidelity_err": gate_fidelity_err,
                "rb_fit_result": rb_fit_result,
                "irb_fit_result": irb_fit_result,
            }
        },
        figure=fig,
        figures={target: fig},
    )


def stark_irb_experiment(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    interleaved_clifford: str | Clifford,
    stark_ramptime: float | None = None,
    interleaved_waveform: TargetMap[PulseSchedule] | TargetMap[Waveform] | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    save_image: bool = True,
    **deprecated_options: Any,
) -> Result:
    """Run interleaved randomized benchmarking under a Stark tone."""
    clifford = _resolve_clifford(exp, interleaved_clifford)
    rb_result = stark_rb_experiment_1q(
        exp,
        target=target,
        stark_amplitude=stark_amplitude,
        stark_ramptime=stark_ramptime,
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
        save_image=False,
        **deprecated_options,
    )
    irb_result = stark_rb_experiment_1q(
        exp,
        target=target,
        stark_amplitude=stark_amplitude,
        stark_ramptime=stark_ramptime,
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        interleaved_waveform=interleaved_waveform,  # pyright: ignore[reportArgumentType]
        interleaved_clifford=clifford,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
        save_image=False,
        **deprecated_options,
    )
    return _interleaved_fit_result(
        target=target,
        reference_result=rb_result,
        interleaved_result=irb_result,
        clifford=clifford,
        plot=plot,
        save_image=save_image,
        image_name=f"stark_interleaved_randomized_benchmarking_{target}",
        title_prefix="Stark interleaved randomized benchmarking",
    )


def stark_ipurity_experiment(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    interleaved_clifford: str | Clifford,
    stark_ramptime: float | None = None,
    interleaved_waveform: TargetMap[PulseSchedule] | TargetMap[Waveform] | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    save_image: bool = True,
    **deprecated_options: Any,
) -> Result:
    """Run interleaved purity benchmarking under a Stark tone."""
    clifford = _resolve_clifford(exp, interleaved_clifford)
    purity_result = stark_purity_experiment_1q(
        exp,
        target=target,
        stark_amplitude=stark_amplitude,
        stark_ramptime=stark_ramptime,
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
        save_image=False,
        **deprecated_options,
    )
    interleaved_result = stark_purity_experiment_1q(
        exp,
        target=target,
        stark_amplitude=stark_amplitude,
        stark_ramptime=stark_ramptime,
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        interleaved_waveform=interleaved_waveform,  # pyright: ignore[reportArgumentType]
        interleaved_clifford=clifford,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
        save_image=False,
        **deprecated_options,
    )
    return _interleaved_fit_result(
        target=target,
        reference_result=purity_result,
        interleaved_result=interleaved_result,
        clifford=clifford,
        plot=plot,
        save_image=save_image,
        image_name=f"stark_interleaved_purity_benchmarking_{target}",
        title_prefix="Stark interleaved purity benchmarking",
    )


def stark_interleaved_randomized_benchmarking(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    interleaved_clifford: str | Clifford,
    stark_ramptime: float | None = None,
    interleaved_waveform: TargetMap[PulseSchedule] | TargetMap[Waveform] | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    save_image: bool = True,
    **deprecated_options: Any,
) -> Result:
    """Alias for :func:`stark_irb_experiment`."""
    return stark_irb_experiment(
        exp,
        target=target,
        stark_amplitude=stark_amplitude,
        stark_ramptime=stark_ramptime,
        interleaved_clifford=interleaved_clifford,
        interleaved_waveform=interleaved_waveform,
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
        save_image=save_image,
        **deprecated_options,
    )


def stark_interleaved_purity_benchmarking(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    interleaved_clifford: str | Clifford,
    stark_ramptime: float | None = None,
    interleaved_waveform: TargetMap[PulseSchedule] | TargetMap[Waveform] | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    save_image: bool = True,
    **deprecated_options: Any,
) -> Result:
    """Alias for :func:`stark_ipurity_experiment`."""
    return stark_ipurity_experiment(
        exp,
        target=target,
        stark_amplitude=stark_amplitude,
        stark_ramptime=stark_ramptime,
        interleaved_clifford=interleaved_clifford,
        interleaved_waveform=interleaved_waveform,
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
        save_image=save_image,
        **deprecated_options,
    )


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
            if fit_result.status is not FitStatus.SUCCESS:
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
                fig = fit_result.get_figure()
                viz.save_figure(fig, name=f"stark_ramsey_{qubit}")

    return ExperimentResult(data=data)


def _resolve_wait_time(
    exp: Experiment,
    *,
    target: str,
    wait_time: int | None,
) -> int:
    if wait_time is not None:
        return wait_time

    t1_dict = exp.ctx.system_manager.config_loader.load_param_data("t1")
    if target not in t1_dict:
        raise ValueError(f"T1 data is not available for target `{target}`.")

    half_t1 = t1_dict[target] * np.log(2)
    sampling_period = exp.ctx.measurement.sampling_period
    return int(np.round(half_t1 / sampling_period) * sampling_period)


def stark_p1_experiment(
    exp: Experiment,
    target: str,
    *,
    stark_detuning: float | None = None,
    stark_amplitude: float | None = None,
    stark_ramptime: float | None = None,
    wait_time: int | None = None,
    mode: Literal["single", "avg"] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    **deprecated_options: Any,
) -> MeasureResult:
    """
    Measure excited-state population under a Stark drive.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurement.
    target
        Target qubit to measure.
    stark_detuning
        Stark-tone detuning in GHz.
    stark_amplitude
        Stark-tone relative drive amplitude.
    stark_ramptime
        Stark-tone ramp time in ns.
    wait_time
        Stark-tone flat-top wait time in ns.
    mode
        Measurement mode.
    n_shots
        Number of shots.
    shot_interval
        Measurement interval in seconds.
    plot
        Whether to render the measurement result.

    Returns
    -------
    MeasureResult
        Raw classified measurement result for the Stark-driven P1 experiment.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_p1_experiment",
    )
    if mode is None:
        mode = "avg"
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = False

    if stark_detuning is None:
        stark_detuning = 0.15
    elif abs(stark_detuning) > 0.2:
        raise ValueError("Detuning of a stark tone exceeds 0.2 GHz AWG limit.")

    if stark_amplitude is None:
        stark_amplitude = 0.1

    if stark_ramptime is None:
        stark_ramptime = 10

    wait_time = _resolve_wait_time(
        exp,
        target=target,
        wait_time=wait_time,
    )
    exp.pulse.validate_rabi_params([target])

    stark_power = exp.pulse.calc_control_amplitude(
        target=target,
        rabi_rate=stark_amplitude,
    )
    if stark_power > 1:
        raise ValueError("Stark drive amplitude must not exceed 1")

    def stark_p1_sequence() -> PulseSchedule:
        with PulseSchedule([target]) as ps:
            ps.add(target, exp.pulse.get_hpi_pulse(target).repeated(2))
            ps.add(
                target,
                FlatTop(
                    duration=wait_time + stark_ramptime * 2,
                    amplitude=stark_power,
                    tau=stark_ramptime,
                ).detuned(detuning=stark_detuning),
            )
        return ps

    return exp.measurement_service.measure(
        sequence=stark_p1_sequence(),
        mode=mode,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
    )


def stark_p1_spectroscopy(
    exp: Experiment,
    target: str,
    *,
    stark_detuning: float | None = None,
    stark_ramptime: float | None = None,
    stark_amplitude_range: ArrayLike | None = None,
    wait_time: int | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Sweep Stark amplitude and estimate excited-state population.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurements.
    target
        Target qubit to characterize.
    stark_detuning
        Stark-tone detuning in GHz.
    stark_ramptime
        Stark-tone ramp time in ns.
    stark_amplitude_range
        Sweep range for Stark-tone relative drive amplitude.
    wait_time
        Stark-tone flat-top wait time in ns.
    n_shots
        Number of shots per sweep point.
    shot_interval
        Measurement interval in seconds.
    plot
        Whether to render the spectroscopy figure.
    save_image
        Whether to save the generated figure.

    Returns
    -------
    Result
        Spectroscopy result containing sweep amplitudes, P1 data, and figures.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_p1_spectroscopy",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False

    if stark_detuning is None:
        stark_detuning = 0.15
    elif abs(stark_detuning) > 0.2:
        raise ValueError("Detuning of a stark tone exceeds 0.2 GHz AWG limit.")

    if stark_amplitude_range is None:
        stark_amplitude_range = np.linspace(0, 0.1, 51)
    amplitude_range = np.asarray(stark_amplitude_range, dtype=float)
    if amplitude_range.ndim != 1:
        raise ValueError("`stark_amplitude_range` must be a 1-D array.")
    if len(amplitude_range) == 0:
        raise ValueError("`stark_amplitude_range` must not be empty.")
    for stark_amplitude in amplitude_range:
        stark_power = exp.pulse.calc_control_amplitude(
            target=target,
            rabi_rate=stark_amplitude,
        )
        if stark_power > 1:
            raise ValueError("Stark drive amplitude must not exceed 1")

    if stark_ramptime is None:
        stark_ramptime = 50

    wait_time = _resolve_wait_time(
        exp,
        target=target,
        wait_time=wait_time,
    )
    exp.pulse.validate_rabi_params([target])

    results: list[MeasureResult] = []
    p1_list: list[float] = []
    for stark_amplitude in amplitude_range:
        result = stark_p1_experiment(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
            stark_detuning=stark_detuning,
            stark_ramptime=stark_ramptime,
            wait_time=wait_time,
            mode="single",
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
        )
        results.append(result)
        p1_list.append(result.probabilities.get("1", 0.0))

    fig = go.Figure()
    fig.add_scatter(name="data", x=amplitude_range, y=p1_list)
    fig.update_layout(
        title="P1 spectroscopy",
        xaxis_title="Stark Amplitude (GHz)",
        yaxis_title="Probability_1",
        showlegend=True,
    )

    if plot:
        fig.show()
    if save_image:
        viz.save_figure(fig, name=f"stark_p1_spectroscopy_{target}")

    return Result(
        data={
            "raw_result": results,
            "amplitude_range": amplitude_range,
            "p1": p1_list,
            # TODO: Remove this legacy payload key after callers migrate to .figure.
            "fig": fig,
        },
        figure=fig,
    )


def _estimate_ac_stark_shift(
    *,
    stark_amplitude: ArrayLike,
    stark_detuning: float,
    model: StarkShiftModel,
    anharmonicity: float | None,
) -> np.ndarray:
    """
    Estimate the signed AC Stark shift in GHz.

    The detuning convention is drive frequency minus qubit frequency in GHz.
    Therefore a positive detuning shifts the qubit transition downward.
    """
    if stark_detuning == 0:
        raise ValueError("Stark detuning must be non-zero to estimate AC Stark shift.")
    amplitude = np.asarray(stark_amplitude, dtype=float)
    if model == "ideal":
        return -np.sign(stark_detuning) * (
            np.sqrt(stark_detuning**2 + amplitude**2) - abs(stark_detuning)
        )

    if anharmonicity is None:
        raise ValueError("`anharmonicity` is required for the Duffing Stark model.")
    denominator = 2 * stark_detuning * (anharmonicity - stark_detuning)
    if denominator == 0:
        raise ValueError(
            "Duffing Stark model is singular when stark_detuning equals anharmonicity."
        )
    return -(anharmonicity * amplitude**2) / denominator


def _resolve_anharmonicity(
    exp: Experiment,
    *,
    target: str,
) -> float:
    qubit = exp.ctx.resolve_qubit_label(target)
    anharmonicity_dict = exp.ctx.system_manager.config_loader.load_param_data(
        "qubit_anharmonicity"
    )
    config_value = anharmonicity_dict.get(qubit)
    if config_value is not None:
        config_anharmonicity = float(config_value)
        if np.isfinite(config_anharmonicity):
            return config_anharmonicity

    value = exp.ctx.qubits[qubit].anharmonicity
    if not np.isfinite(value):
        raise ValueError(f"Anharmonicity is not available for target `{target}`.")
    return float(value)


def _interp_shift_lookup(
    *,
    amplitudes: ArrayLike,
    amplitude_to_shift: Mapping[float, float],
) -> np.ndarray:
    points = sorted(
        (float(amplitude), float(shift))
        for amplitude, shift in amplitude_to_shift.items()
    )
    if len(points) == 0:
        raise ValueError("Experimental Stark shift lookup table must not be empty.")

    lookup_amplitudes = np.asarray([point[0] for point in points], dtype=float)
    lookup_shifts = np.asarray([point[1] for point in points], dtype=float)
    requested_amplitudes = np.asarray(amplitudes, dtype=float)
    if (
        requested_amplitudes.min() < lookup_amplitudes[0]
        or requested_amplitudes.max() > lookup_amplitudes[-1]
    ):
        raise ValueError(
            "Experimental Stark shift lookup table does not cover the requested amplitude range."
        )
    return np.interp(requested_amplitudes, lookup_amplitudes, lookup_shifts)


def _select_nested_lookup(
    *,
    stark_detuning: float,
    lookup: Mapping[float, Mapping[float, float]],
) -> Mapping[float, float]:
    for detuning, amplitude_to_shift in lookup.items():
        if np.isclose(float(detuning), stark_detuning):
            return amplitude_to_shift
    raise ValueError(
        "Experimental Stark shift lookup table does not contain the requested detuning."
    )


def _experimental_shift_from_mapping(
    *,
    stark_amplitude: ArrayLike,
    stark_detuning: float,
    detuning_range: np.ndarray,
    stark_shift_lookup: Mapping[float, object],
) -> np.ndarray:
    lookup_values = list(stark_shift_lookup.values())
    if all(isinstance(value, Mapping) for value in lookup_values):
        return _interp_shift_lookup(
            amplitudes=stark_amplitude,
            amplitude_to_shift=_select_nested_lookup(
                stark_detuning=stark_detuning,
                lookup=stark_shift_lookup,  # pyright: ignore[reportArgumentType]
            ),
        )

    if len(detuning_range) != 1:
        raise ValueError(
            "Use a detuning-keyed lookup dictionary or a three-column lookup table when sweeping multiple Stark detunings."
        )
    return _interp_shift_lookup(
        amplitudes=stark_amplitude,
        amplitude_to_shift=stark_shift_lookup,  # pyright: ignore[reportArgumentType]
    )


def _experimental_shift_from_array(
    *,
    stark_amplitude: ArrayLike,
    stark_detuning: float,
    detuning_range: np.ndarray,
    stark_shift_lookup: ArrayLike,
) -> np.ndarray:
    lookup = np.asarray(stark_shift_lookup, dtype=float)
    if lookup.ndim != 2 or lookup.shape[1] not in (2, 3):
        raise ValueError(
            "Experimental Stark shift lookup table must have shape (N, 2) or (N, 3)."
        )

    if lookup.shape[1] == 2:
        if len(detuning_range) != 1:
            raise ValueError(
                "Use a three-column lookup table when sweeping multiple Stark detunings."
            )
        amplitude_to_shift = dict(zip(lookup[:, 0], lookup[:, 1], strict=True))
        return _interp_shift_lookup(
            amplitudes=stark_amplitude,
            amplitude_to_shift=amplitude_to_shift,
        )

    detuning_mask = np.isclose(lookup[:, 0], stark_detuning)
    if not np.any(detuning_mask):
        raise ValueError(
            "Experimental Stark shift lookup table does not contain the requested detuning."
        )
    detuning_lookup = lookup[detuning_mask]
    amplitude_to_shift = dict(
        zip(detuning_lookup[:, 1], detuning_lookup[:, 2], strict=True)
    )
    return _interp_shift_lookup(
        amplitudes=stark_amplitude,
        amplitude_to_shift=amplitude_to_shift,
    )


def _estimate_experimental_ac_stark_shift(
    *,
    stark_amplitude: ArrayLike,
    stark_detuning: float,
    detuning_range: np.ndarray,
    stark_shift_lookup: StarkShiftLookup | None,
) -> np.ndarray:
    if stark_shift_lookup is None:
        raise ValueError(
            "`stark_shift_lookup` is required when stark_shift_model='experimental'."
        )
    if isinstance(stark_shift_lookup, Mapping):
        return _experimental_shift_from_mapping(
            stark_amplitude=stark_amplitude,
            stark_detuning=stark_detuning,
            detuning_range=detuning_range,
            stark_shift_lookup=stark_shift_lookup,
        )
    return _experimental_shift_from_array(
        stark_amplitude=stark_amplitude,
        stark_detuning=stark_detuning,
        detuning_range=detuning_range,
        stark_shift_lookup=stark_shift_lookup,
    )


def _normalize_ac_stark_detuning_range(
    stark_detuning: float | ArrayLike | None,
) -> np.ndarray:
    if stark_detuning is None:
        stark_detuning = [-0.15, 0.15]

    detuning_range = np.atleast_1d(np.asarray(stark_detuning, dtype=float))
    if detuning_range.ndim != 1:
        raise ValueError("`stark_detuning` must be a scalar or 1-D array.")
    if np.any(detuning_range == 0):
        raise ValueError("Stark detuning must be non-zero for AC Stark spectroscopy.")
    if np.any(np.abs(detuning_range) > 0.2):
        raise ValueError("Detuning of a stark tone exceeds 0.2 GHz AWG limit.")
    return detuning_range


def ac_stark_shift_spectroscopy(
    exp: Experiment,
    target: str,
    *,
    stark_detuning: float | ArrayLike | None = None,
    stark_ramptime: float | None = None,
    stark_amplitude_range: ArrayLike | None = None,
    stark_shift_model: StarkShiftModel = "duffing",
    stark_shift_lookup: StarkShiftLookup | None = None,
    wait_time_range: ArrayLike | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Sweep Stark-induced qubit-frequency shift and wait time, then plot P1.

    This helper builds a P1(T1)-style Stark spectroscopy map by repeatedly
    calling :func:`stark_p1_spectroscopy` at each wait time. The horizontal
    axis is an AC Stark shift from either a theoretical model or an
    experimentally calibrated lookup table.

    Parameters
    ----------
    exp
        Experiment instance to use for pulse generation and measurements.
    target
        Target qubit to characterize.
    stark_detuning
        Stark-tone detuning in GHz, defined as drive frequency minus qubit
        frequency. A scalar produces one signed branch; a 1-D array such as
        ``[-0.15, 0.15]`` measures both positive and negative shift branches.
        When omitted, both branches are measured with ``[-0.15, 0.15]``.
    stark_ramptime
        Stark-tone ramp time in ns.
    stark_amplitude_range
        Sweep range for Stark-tone relative drive amplitude.
    stark_shift_model
        Theoretical model used to map Stark amplitude to frequency shift.
        ``"ideal"`` uses the exact dressed-state two-level formula.
        ``"duffing"`` uses the perturbative transmon Duffing model with the
        qubit anharmonicity. ``"experimental"`` uses ``stark_shift_lookup``.
        Defaults to ``"duffing"``.
    stark_shift_lookup
        Experimental amplitude-to-shift lookup table used only with
        ``stark_shift_model="experimental"``. For multiple detunings, pass
        either a three-column table ``[stark_detuning, stark_amplitude,
        ac_stark_shift]`` or a nested dictionary
        ``{stark_detuning: {stark_amplitude: ac_stark_shift}}``. For a single
        detuning, a two-column table ``[stark_amplitude, ac_stark_shift]`` or
        a dictionary ``{stark_amplitude: ac_stark_shift}`` is also accepted.
        All values are in GHz.
    wait_time_range
        Sweep range for Stark-tone flat-top wait time in ns.
    n_shots
        Number of shots per sweep point.
    shot_interval
        Measurement interval in seconds.
    plot
        Whether to render the spectroscopy heatmap.
    save_image
        Whether to save the generated figure.

    Returns
    -------
    Result
        Spectroscopy result containing wait times, theoretical shifts, P1 map,
        raw nested Stark P1 spectroscopy results, and the heatmap figure.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="ac_stark_shift_spectroscopy",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if save_image is None:
        save_image = False
    if stark_shift_model not in ("ideal", "duffing", "experimental"):
        raise ValueError(
            "`stark_shift_model` must be 'ideal', 'duffing', or 'experimental'."
        )
    if stark_shift_model != "experimental" and stark_shift_lookup is not None:
        raise ValueError(
            "`stark_shift_lookup` is only used when stark_shift_model='experimental'."
        )

    detuning_range = _normalize_ac_stark_detuning_range(stark_detuning)
    if stark_amplitude_range is None:
        stark_amplitude_range = np.linspace(0, 0.1, 51)
    amplitude_range = np.asarray(stark_amplitude_range, dtype=float)
    if amplitude_range.ndim != 1:
        raise ValueError("`stark_amplitude_range` must be a 1-D array.")
    if len(amplitude_range) == 0:
        raise ValueError("`stark_amplitude_range` must not be empty.")
    if np.any(amplitude_range < 0):
        raise ValueError("`stark_amplitude_range` must contain non-negative values.")

    if wait_time_range is None:
        wait_time_range = np.arange(0, 20_001, 2_000)
    sampling_period = exp.ctx.measurement.sampling_period
    wait_times = np.asarray(
        exp.ctx.util.discretize_time_range(
            np.asarray(wait_time_range),
            sampling_period=sampling_period,
        ),
        dtype=int,
    )
    if wait_times.ndim != 1:
        raise ValueError("`wait_time_range` must be a 1-D array.")
    if len(wait_times) == 0:
        raise ValueError("`wait_time_range` must not be empty.")

    resolved_anharmonicity = (
        _resolve_anharmonicity(
            exp,
            target=target,
        )
        if stark_shift_model == "duffing"
        else None
    )
    if stark_shift_model == "experimental":
        shift_chunks = [
            _estimate_experimental_ac_stark_shift(
                stark_amplitude=amplitude_range,
                stark_detuning=float(detuning),
                detuning_range=detuning_range,
                stark_shift_lookup=stark_shift_lookup,
            )
            for detuning in detuning_range
        ]
    else:
        shift_chunks = [
            _estimate_ac_stark_shift(
                stark_amplitude=amplitude_range,
                stark_detuning=float(detuning),
                model=stark_shift_model,
                anharmonicity=resolved_anharmonicity,
            )
            for detuning in detuning_range
        ]
    shift_axis = np.concatenate(shift_chunks)
    detuning_axis = np.repeat(detuning_range, len(amplitude_range))
    amplitude_axis = np.tile(amplitude_range, len(detuning_range))
    sort_order = np.argsort(shift_axis, kind="stable")
    shift_axis = shift_axis[sort_order]
    detuning_axis = detuning_axis[sort_order]
    amplitude_axis = amplitude_axis[sort_order]
    unique_shift_mask = np.ones(len(shift_axis), dtype=bool)
    unique_shift_mask[1:] = np.diff(shift_axis) != 0
    shift_axis = shift_axis[unique_shift_mask]
    detuning_axis = detuning_axis[unique_shift_mask]
    amplitude_axis = amplitude_axis[unique_shift_mask]

    p1_rows: list[np.ndarray] = []
    raw_results: list[dict[float, Result]] = []
    for wait_time in wait_times:
        row_by_detuning: dict[float, np.ndarray] = {}
        raw_row: dict[float, Result] = {}
        for detuning in detuning_range:
            result = stark_p1_spectroscopy(
                exp,
                target=target,
                stark_detuning=float(detuning),
                stark_ramptime=stark_ramptime,
                stark_amplitude_range=amplitude_range,
                wait_time=int(wait_time),
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=False,
                save_image=False,
            )
            p1 = np.asarray(result.data["p1"], dtype=float)
            row_by_detuning[float(detuning)] = p1
            raw_row[float(detuning)] = result

        row = np.concatenate(
            [row_by_detuning[float(detuning)] for detuning in detuning_range]
        )
        p1_rows.append(row[sort_order][unique_shift_mask])
        raw_results.append(raw_row)

    p1_map = np.vstack(p1_rows)
    wait_time_us = wait_times / 1000

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=shift_axis,
            y=wait_time_us,
            z=p1_map,
            zmin=0,
            zmax=1,
            colorscale="RdBu_r",
            colorbar=dict(title="P1"),
        )
    )
    fig.update_layout(
        title="Stark P1(T1) spectroscopy",
        xaxis_title="AC Stark shift from qubit frequency (GHz)",
        yaxis_title="Wait Time (μs)",
    )

    if plot:
        fig.show()
    if save_image:
        viz.save_figure(fig, name=f"ac_stark_shift_spectroscopy_{target}")

    return Result(
        data={
            "raw_result": raw_results,
            "stark_detuning_range": detuning_range,
            "stark_amplitude_range": amplitude_range,
            "stark_shift_model": stark_shift_model,
            "stark_shift_lookup": stark_shift_lookup,
            "anharmonicity": resolved_anharmonicity,
            "ac_stark_shift_range": shift_axis,
            "shift_detuning_axis": detuning_axis,
            "shift_amplitude_axis": amplitude_axis,
            "wait_time_range": wait_times,
            "wait_time_us": wait_time_us,
            "p1": p1_map,
            # TODO: Remove this legacy payload key after callers migrate to .figure.
            "fig": fig,
        },
        figure=fig,
    )


def _resolve_stark_ramptime(stark_ramptime: float | None) -> float:
    return 50.0 if stark_ramptime is None else float(stark_ramptime)


def _measurement_sampling_period(exp: Experiment) -> float:
    return float(exp.ctx.measurement.sampling_period)


def _pulse_area(pulse: Waveform, sampling_period: float) -> float:
    return float(pulse.real.sum() * sampling_period)


def _stark_drive_amplitude(
    exp: Experiment,
    *,
    target: str,
    stark_amplitude: float,
) -> float:
    amplitude = exp.pulse.calc_control_amplitude(
        target=target,
        rabi_rate=stark_amplitude,
    )
    if amplitude > 1:
        raise ValueError("Drive amplitude of a stark tone must not exceed 1.")
    return amplitude


def _bare_control_amplitude(
    exp: Experiment,
    *,
    target: str,
    rabi_rate: float,
) -> float:
    """Convert a Rabi rate using the physical qubit's stored Rabi params."""
    return exp.pulse.calc_control_amplitude(
        target=exp.ctx.resolve_qubit_label(target),
        rabi_rate=rabi_rate,
    )


def _bare_rabi_rate(
    exp: Experiment,
    *,
    target: str,
    control_amplitude: float,
) -> float:
    """Convert a control amplitude using the physical qubit's stored Rabi params."""
    return exp.pulse.calc_rabi_rate(
        exp.ctx.resolve_qubit_label(target),
        control_amplitude,
    )


def _amplitude_sweep(
    *,
    center: float,
    n_points: int,
    n_rotations: int,
    amplitude_range: ArrayLike | None,
) -> np.ndarray:
    if amplitude_range is not None:
        values = np.asarray(amplitude_range, dtype=float)
    else:
        lower = np.clip(center * (1 - 0.5 / n_rotations), 0, 1)
        upper = np.clip(center * (1 + 0.5 / n_rotations), 0, 1)
        if lower == upper:
            lower = 0
            upper = 1
        values = np.linspace(lower, upper, n_points)
    if values.ndim != 1:
        raise ValueError("Amplitude range must be a 1-D array.")
    if len(values) == 0:
        raise ValueError("Amplitude range must not be empty.")
    return values


def _target_map_get(
    values: Mapping[str, Any] | None,
    *,
    target: str,
    insitu: str,
) -> Any:
    if values is None:
        return None
    if insitu in values:
        return values[insitu]
    return values.get(target)


def _single_target_amplitude(
    exp: Experiment,
    *,
    target: str,
    amplitude: float | Mapping[str, float] | None,
) -> float:
    if amplitude is None:
        return exp.ctx.params.get_control_amplitude(target)
    if isinstance(amplitude, Mapping):
        return float(amplitude[target])
    return float(amplitude)


def _stark_wrapped_schedule(
    *,
    stark_label: str,
    insitu_label: str,
    stark_amplitude: float,
    stark_ramptime: float,
    insitu_sequence: Waveform | PulseArray | PulseSchedule,
) -> PulseSchedule:
    with PulseSchedule() as ps:
        ps.add(
            stark_label,
            FlatTop(
                duration=insitu_sequence.duration + 2 * stark_ramptime,
                amplitude=stark_amplitude,
                tau=stark_ramptime,
            ),
        )
        ps.add(insitu_label, Blank(stark_ramptime))
        if isinstance(insitu_sequence, PulseSchedule):
            ps.call(insitu_sequence)
        else:
            ps.add(insitu_label, insitu_sequence)
    return ps


def _resolve_clifford(exp: Experiment, clifford: str | Clifford) -> Clifford:
    if isinstance(clifford, Clifford):
        return clifford
    resolved = exp.clifford.get(clifford)
    if resolved is None:
        raise ValueError(f"Invalid Clifford: {clifford}")
    return resolved


def _clifford_sweep_range(
    *,
    n_cliffords_range: ArrayLike | None,
    max_n_cliffords: int,
) -> np.ndarray:
    if n_cliffords_range is not None:
        values = np.asarray(n_cliffords_range, dtype=int)
        if values.ndim != 1:  # pyright: ignore[reportAttributeAccessIssue]
            raise ValueError("`n_cliffords_range` must be a 1-D array.")
        return values
    values: list[int] = []
    idx = 0
    while True:
        n_clifford = 0 if idx == 0 else 2 ** (idx - 1)
        if n_clifford > max_n_cliffords:
            break
        values.append(n_clifford)
        idx += 1
    return np.asarray(values, dtype=int)


def _stark_drive_qubit_label(
    *,
    control_qubit: str,
    target_qubit: str,
    stark_drive_qubit: StarkDriveQubit,
) -> str:
    if stark_drive_qubit == "control":
        return control_qubit
    if stark_drive_qubit == "target":
        return target_qubit
    raise ValueError("`stark_drive_qubit` must be 'control' or 'target'.")


def _dressed_cr_labels(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_drive_qubit: StarkDriveQubit,
) -> tuple[str, str, str, str, str]:
    dressed_qubit = _stark_drive_qubit_label(
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        stark_drive_qubit=stark_drive_qubit,
    )
    dressed_label = insitu_target(exp, dressed_qubit)
    cr_control = dressed_label if stark_drive_qubit == "control" else control_qubit
    cr_target = dressed_label if stark_drive_qubit == "target" else target_qubit
    cr_label = stark_cr_target(
        exp,
        control_qubit,
        target_qubit,
        stark_drive_qubit=stark_drive_qubit,
    )
    return dressed_qubit, dressed_label, cr_control, cr_target, cr_label


def _stark_cr_frequencies(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_drive_qubit: StarkDriveQubit,
    cr_frequency: float | None,
) -> tuple[float, float, float]:
    control_frequency = (
        exp.targets[insitu_target(exp, control_qubit)].frequency
        if stark_drive_qubit == "control"
        else exp.targets[control_qubit].frequency
    )
    target_frequency = (
        exp.targets[insitu_target(exp, target_qubit)].frequency
        if stark_drive_qubit == "target"
        else exp.targets[target_qubit].frequency
    )
    if cr_frequency is None:
        cr_frequency = target_frequency
    return control_frequency, target_frequency, cr_frequency


def _ensure_stark_cr_channel(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_drive_qubit: StarkDriveQubit,
    cr_frequency: float,
    update_lsi: bool,
) -> str:
    cr_label = stark_cr_target(
        exp,
        control_qubit,
        target_qubit,
        stark_drive_qubit=stark_drive_qubit,
    )
    if cr_label not in exp.targets:
        make_stark_cr_channel(
            exp,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            stark_drive_qubit=stark_drive_qubit,
            frequency=cr_frequency,
            lsi=update_lsi,
        )
    return cr_label


def _basis_rotation(
    x90: Waveform,
    basis: Literal["X", "Y", "Z"],
) -> Waveform | None:
    if basis == "X":
        return x90.shifted(-np.pi / 2)
    if basis == "Y":
        return x90
    if basis == "Z":
        return None
    raise ValueError("`basis` must be 'X', 'Y', or 'Z'.")


def _tomography_x90(
    exp: Experiment,
    *,
    qubit: str,
    dressed_label: str,
    stark_drive_qubit: StarkDriveQubit,
    x90: TargetMap[Waveform] | None,
) -> Waveform:
    if (
        stark_drive_qubit == "control"
        and qubit == exp.ctx.resolve_qubit_label(dressed_label)
    ) or (
        stark_drive_qubit == "target"
        and qubit == exp.ctx.resolve_qubit_label(dressed_label)
    ):
        override = _target_map_get(x90, target=qubit, insitu=dressed_label)
        return override if override is not None else exp.pulse.x90(dressed_label)
    override = x90.get(qubit) if x90 is not None else None
    return override if override is not None else exp.pulse.x90(qubit)


def stark_cr_tomography_sequence(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    basis: Literal["X", "Y", "Z"] = "Z",
    control_state: Literal["0", "1"] = "0",
    cr_duration: float = 128.0,
    ramptime: float | None = None,
    stark_ramptime: float | None = None,
    cr_amplitude: float = 1.0,
    cr_phase: float = 0.0,
    cancel_amplitude: float = 0.0,
    cancel_phase: float = 0.0,
    ramp_type: RampType = "RaisedCosine",
    x90: TargetMap[Waveform] | None = None,
    plot: bool = True,
) -> PulseSchedule:
    """
    Build one Stark-driven CR tomography schedule.

    The dressed qubit receives its tomography basis rotation while the Stark
    tone is still on. The Stark tone is then turned off before readout.
    """
    if control_state not in ("0", "1"):
        raise ValueError("`control_state` must be '0' or '1'.")
    if ramptime is None:
        ramptime = DEFAULT_CR_RAMPTIME

    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    dressed_qubit, dressed_label, cr_control, cr_target, _cr_label = _dressed_cr_labels(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    stark_power = _stark_drive_amplitude(
        exp,
        target=dressed_qubit,
        stark_amplitude=stark_amplitude,
    )

    cr_sequence = CrossResonance(
        control_qubit=cr_control,
        target_qubit=cr_target,
        cr_amplitude=cr_amplitude,
        cr_duration=cr_duration,
        cr_ramptime=ramptime,
        cr_phase=cr_phase,
        cancel_amplitude=cancel_amplitude,
        cancel_phase=cancel_phase,
        echo=False,
        ramp_type=ramp_type,
    )

    x90_control = _tomography_x90(
        exp,
        qubit=control_label,
        dressed_label=dressed_label,
        stark_drive_qubit=stark_drive_qubit,
        x90=x90,
    )
    x90_target = _tomography_x90(
        exp,
        qubit=target_label,
        dressed_label=dressed_label,
        stark_drive_qubit=stark_drive_qubit,
        x90=x90,
    )
    dressed_x90 = x90_control if dressed_qubit == control_label else x90_target
    bare_x90 = x90_target if dressed_qubit == control_label else x90_control
    bare_qubit = target_label if dressed_qubit == control_label else control_label

    with PulseSchedule() as dressed_body:
        if control_state == "1" and dressed_qubit == control_label:
            dressed_body.add(dressed_label, exp.pulse.x180(dressed_label))
            dressed_body.barrier()
        dressed_body.call(cr_sequence)
        dressed_body.barrier()
        dressed_basis = _basis_rotation(dressed_x90, basis)
        if dressed_basis is not None:
            dressed_body.add(dressed_label, dressed_basis)

    stark_block = _stark_wrapped_schedule(
        stark_label=stark_target(exp, dressed_qubit),
        insitu_label=dressed_label,
        stark_amplitude=stark_power,
        stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
        insitu_sequence=dressed_body,
    )

    with PulseSchedule() as ps:
        if control_state == "1" and dressed_qubit != control_label:
            ps.add(control_label, exp.pulse.x180(control_label))
            ps.barrier()
        ps.call(stark_block)
        ps.barrier()
        bare_basis = _basis_rotation(bare_x90, basis)
        if bare_basis is not None:
            ps.add(bare_qubit, bare_basis)

    return _plot_sequence_sample(
        ps,
        title=(
            f"AC Stark CR tomography: {control_label}-{target_label}, "
            f"stark={stark_drive_qubit}, state={control_state}, basis={basis}"
        ),
        plot=plot,
    )


def _normalized_tomography_value(
    exp: Experiment,
    *,
    qubit: str,
    data: Any,
    use_zvalues: bool,
) -> float:
    rabi_param = exp.pulse.rabi_params[qubit]
    if rabi_param is None:
        raise ValueError("Rabi parameters are not stored.")
    if use_zvalues:
        p = data.kerneled
        g, e = exp.ctx.state_centers[qubit][0], exp.ctx.state_centers[qubit][1]
        v_ge = e - g
        v_gp = p - g
        v_gp_proj = np.real(v_gp * np.conj(v_ge)) / np.abs(v_ge)
        return float(1 - 2 * np.abs(v_gp_proj) / np.abs(v_ge))
    return float(rabi_param.normalize(data.kerneled))


def _stark_cr_state_tomography(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit,
    control_state: Literal["0", "1"],
    cr_duration: float,
    ramptime: float,
    stark_ramptime: float | None,
    cr_amplitude: float,
    cr_phase: float,
    cancel_amplitude: float,
    cancel_phase: float,
    cr_frequency: float,
    ramp_type: RampType,
    x90: TargetMap[Waveform] | None,
    n_shots: int,
    shot_interval: float,
    use_zvalues: bool,
    plot: bool,
) -> Result:
    buffer: dict[str, list[float]] = defaultdict(list)
    cr_label = stark_cr_target(
        exp,
        control_qubit,
        target_qubit,
        stark_drive_qubit=stark_drive_qubit,
    )
    for basis in ("X", "Y", "Z"):
        sequence = stark_cr_tomography_sequence(
            exp,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            stark_amplitude=stark_amplitude,
            stark_drive_qubit=stark_drive_qubit,
            basis=basis,
            control_state=control_state,
            cr_duration=cr_duration,
            ramptime=ramptime,
            stark_ramptime=stark_ramptime,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            ramp_type=ramp_type,
            x90=x90,
            plot=False,
        )
        measure_result = exp.measurement_service.measure(
            sequence,
            n_shots=n_shots,
            shot_interval=shot_interval,
            frequencies={cr_label: cr_frequency},
            reset_awg_and_capunits=False,
            plot=plot,
        )
        for qubit, data in measure_result.data.items():
            buffer[qubit].append(
                _normalized_tomography_value(
                    exp,
                    qubit=qubit,
                    data=data,
                    use_zvalues=use_zvalues,
                )
            )
    return Result(
        data={
            qubit: (
                values[0],
                values[1],
                values[2],
            )
            for qubit, values in buffer.items()
        }
    )


def stark_measure_cr_dynamics(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    stark_ramptime: float | None = None,
    cr_amplitude: float | None = None,
    cr_phase: float | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    cr_frequency: float | None = None,
    control_state: Literal["0", "1"] | None = None,
    x90: TargetMap[Waveform] | None = None,
    ramp_type: RampType = "RaisedCosine",
    use_zvalues: bool = False,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    auto_register_cr_channel: bool = True,
    update_lsi: bool = False,
    plot: bool = True,
) -> Result:
    """Measure CR dynamics while one qubit is Stark-dressed."""
    if control_state is None:
        control_state = "0"
    if time_range is None:
        time_values = np.asarray(DEFAULT_CR_TIME_RANGE, dtype=float)
    else:
        time_values = np.asarray(time_range, dtype=float)
    if ramptime is None:
        ramptime = DEFAULT_CR_RAMPTIME
    if cr_amplitude is None:
        cr_amplitude = 1.0
    if cr_phase is None:
        cr_phase = 0.0
    if cancel_amplitude is None:
        cancel_amplitude = 0.0
    if cancel_phase is None:
        cancel_phase = 0.0
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True

    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    _control_frequency, _target_frequency, resolved_cr_frequency = (
        _stark_cr_frequencies(
            exp,
            control_qubit=control_label,
            target_qubit=target_label,
            stark_drive_qubit=stark_drive_qubit,
            cr_frequency=cr_frequency,
        )
    )
    if auto_register_cr_channel:
        _ensure_stark_cr_channel(
            exp,
            control_qubit=control_label,
            target_qubit=target_label,
            stark_drive_qubit=stark_drive_qubit,
            cr_frequency=resolved_cr_frequency,
            update_lsi=update_lsi,
        )

    if reset_awg_and_capunits:
        exp.ctx.reset_awg_and_capunits(qubits=[control_label, target_label])

    control_states = []
    target_states = []
    for duration in time_values:
        result = _stark_cr_state_tomography(
            exp,
            control_qubit=control_label,
            target_qubit=target_label,
            stark_amplitude=stark_amplitude,
            stark_drive_qubit=stark_drive_qubit,
            control_state=control_state,
            cr_duration=float(duration) + 2 * ramptime,
            ramptime=ramptime,
            stark_ramptime=stark_ramptime,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            cr_frequency=resolved_cr_frequency,
            ramp_type=ramp_type,
            x90=x90,
            n_shots=n_shots,
            shot_interval=shot_interval,
            use_zvalues=use_zvalues,
            plot=False,
        )
        control_states.append(np.asarray(result[control_label]))
        target_states.append(np.asarray(result[target_label]))

    control_states_array = np.asarray(control_states)
    target_states_array = np.asarray(target_states)
    effective_drive_range = time_values + ramptime

    fit_result = fitting.fit_rotation(
        effective_drive_range,
        target_states_array,
        plot=False,
        title=(
            f"Stark CR target dynamics of {control_label}-{target_label} "
            f": |{control_state}>"
        ),
        xlabel="Drive time (ns)",
        ylabel=f"Target qubit : {target_label}",
    )

    if plot:
        viz.plot_bloch_vectors(
            effective_drive_range,
            control_states_array,
            title=(
                f"Stark CR control dynamics of {control_label}-{target_label} "
                f": |{control_state}>"
            ),
            xlabel="Drive time (ns)",
            ylabel=f"Control qubit : {control_label}",
        )
        fit_result.get_figure().show()
        fit_result.get_figure("fig3d").show()

    return Result(
        data={
            "time_range": time_values,
            "effective_drive_range": effective_drive_range,
            "control_states": control_states_array,
            "target_states": target_states_array,
            "fit_result": fit_result,
            "cr_amplitude": cr_amplitude,
            "ramptime": ramptime,
            "cr_frequency": resolved_cr_frequency,
        }
    )


def stark_cr_hamiltonian_tomography(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    stark_ramptime: float | None = None,
    cr_amplitude: float | None = None,
    cr_phase: float | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    cr_frequency: float | None = None,
    x90: TargetMap[Waveform] | None = None,
    use_zvalues: bool = False,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    auto_register_cr_channel: bool = True,
    update_lsi: bool = False,
    plot: bool = True,
) -> Result:
    """Run CR Hamiltonian tomography while one qubit is Stark-dressed."""
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if cr_amplitude is None:
        cr_amplitude = 1.0
    if ramptime is None:
        ramptime = DEFAULT_CR_RAMPTIME
    if cr_phase is None:
        cr_phase = 0.0
    if cancel_amplitude is None:
        cancel_amplitude = 0.0
    if cancel_phase is None:
        cancel_phase = 0.0

    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    cr_label = stark_cr_target(
        exp,
        control_label,
        target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    control_frequency, target_frequency, resolved_cr_frequency = _stark_cr_frequencies(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
        cr_frequency=cr_frequency,
    )
    if auto_register_cr_channel:
        _ensure_stark_cr_channel(
            exp,
            control_qubit=control_label,
            target_qubit=target_label,
            stark_drive_qubit=stark_drive_qubit,
            cr_frequency=resolved_cr_frequency,
            update_lsi=update_lsi,
        )

    if reset_awg_and_capunits:
        exp.ctx.reset_awg_and_capunits(qubits=[control_label, target_label])

    result_0 = stark_measure_cr_dynamics(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_amplitude=stark_amplitude,
        stark_drive_qubit=stark_drive_qubit,
        time_range=time_range,
        ramptime=ramptime,
        stark_ramptime=stark_ramptime,
        cr_amplitude=cr_amplitude,
        cr_phase=cr_phase,
        cancel_amplitude=cancel_amplitude,
        cancel_phase=cancel_phase,
        cr_frequency=resolved_cr_frequency,
        control_state="0",
        x90=x90,
        use_zvalues=use_zvalues,
        n_shots=n_shots,
        shot_interval=shot_interval,
        reset_awg_and_capunits=False,
        auto_register_cr_channel=False,
        plot=False,
    )
    result_1 = stark_measure_cr_dynamics(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_amplitude=stark_amplitude,
        stark_drive_qubit=stark_drive_qubit,
        time_range=time_range,
        ramptime=ramptime,
        stark_ramptime=stark_ramptime,
        cr_amplitude=cr_amplitude,
        cr_phase=cr_phase,
        cancel_amplitude=cancel_amplitude,
        cancel_phase=cancel_phase,
        cr_frequency=resolved_cr_frequency,
        control_state="1",
        x90=x90,
        use_zvalues=use_zvalues,
        n_shots=n_shots,
        shot_interval=shot_interval,
        reset_awg_and_capunits=False,
        auto_register_cr_channel=False,
        plot=False,
    )

    omega_0 = result_0["fit_result"]["Omega"]
    omega_1 = result_1["fit_result"]["Omega"]
    omega = np.concatenate([0.5 * (omega_0 + omega_1), 0.5 * (omega_0 - omega_1)])
    coeffs = dict(
        zip(
            ["IX", "IY", "IZ", "ZX", "ZY", "ZZ"],
            omega / (2 * np.pi),
            strict=True,
        )
    )

    xt_rotation = coeffs["IX"] + 1j * coeffs["IY"]
    xt_rotation_amplitude = np.abs(xt_rotation)
    xt_rotation_amplitude_hw = _bare_control_amplitude(
        exp,
        target=target_label,
        rabi_rate=xt_rotation_amplitude,
    )
    xt_rotation_phase = np.angle(xt_rotation)

    cr_rotation = coeffs["ZX"] + 1j * coeffs["ZY"]
    cr_rotation_amplitude = np.abs(cr_rotation)
    cr_rotation_amplitude_hw = _bare_control_amplitude(
        exp,
        target=target_label,
        rabi_rate=cr_rotation_amplitude,
    )
    cr_rotation_phase = np.angle(cr_rotation)
    zx90_duration = 1 / (4 * cr_rotation_amplitude)
    cr_rabi_rate = _bare_rabi_rate(
        exp,
        target=control_label,
        control_amplitude=cr_amplitude,
    )
    f_delta = control_frequency - target_frequency

    fig_c = viz.make_figure()
    fig_c.set_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    for row, result in enumerate((result_0, result_1), start=1):
        fig_src = viz.make_bloch_vectors_figure(
            result["effective_drive_range"],
            result["control_states"],
        )
        for trace in fig_src.data:
            data = go.Scatter(
                x=trace.x,
                y=trace.y,
                mode=trace.mode,
                line=trace.line,
                marker=trace.marker,
                name=trace.name,
                showlegend=row == 1,
            )
            fig_c.add_trace(data, row=row, col=1)
    fig_c.update_xaxes(title_text="Drive time (ns)", row=2, col=1)
    fig_c.update_yaxes(title_text="Control : |0>", range=[-1.1, 1.1], row=1, col=1)
    fig_c.update_yaxes(title_text="Control : |1>", range=[-1.1, 1.1], row=2, col=1)
    fig_c.update_layout(
        title=dict(
            text=f"Stark CR control dynamics : {cr_label}",
            subtitle=dict(
                text=(
                    f"stark={stark_drive_qubit}, "
                    f"Δ = {f_delta * 1e3:.0f} MHz, "
                    f"Ω = {cr_rabi_rate * 1e3:.1f} MHz, "
                    f"τ = {ramptime:.0f} ns"
                )
            ),
        ),
        height=400,
        width=600,
        showlegend=True,
        margin=dict(t=90),
    )

    fig_t = viz.make_figure()
    fig_t.set_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    for row, result in enumerate((result_0, result_1), start=1):
        fig_src = result["fit_result"].get_figure()
        for trace in fig_src.data:
            data = go.Scatter(
                x=trace.x,
                y=trace.y,
                mode=trace.mode,
                line=trace.line,
                marker=trace.marker,
                name=trace.name,
                showlegend=row == 1,
            )
            fig_t.add_trace(data, row=row, col=1)
    fig_t.update_xaxes(title_text="Drive time (ns)", row=2, col=1)
    fig_t.update_yaxes(title_text="Control : |0>", range=[-1.1, 1.1], row=1, col=1)
    fig_t.update_yaxes(title_text="Control : |1>", range=[-1.1, 1.1], row=2, col=1)
    fig_t.update_layout(
        title=dict(
            text=f"Stark CR target dynamics : {cr_label}",
            subtitle=dict(
                text=(
                    f"stark={stark_drive_qubit}, "
                    f"Δ = {f_delta * 1e3:.0f} MHz, "
                    f"Ω = {cr_rabi_rate * 1e3:.1f} MHz, "
                    f"τ = {ramptime:.0f} ns"
                )
            ),
        ),
        height=400,
        width=600,
        showlegend=True,
        margin=dict(t=90),
    )

    if plot:
        fig_c.show()
        fig_t.show()
        print("Dressed qubit frequencies:")
        print(f"  ω_c ({control_label}) : {control_frequency * 1e3:.3f} MHz")
        print(f"  ω_t ({target_label}) : {target_frequency * 1e3:.3f} MHz")
        print(f"  Δ ({cr_label}) : {f_delta * 1e3:.3f} MHz")
        print(f"  CR drive frequency : {resolved_cr_frequency * 1e3:.3f} MHz")
        print("Rotation rates:")
        for key, value in coeffs.items():
            print(f"  {key} : {value * 1e3:+.4f} MHz")
        print("XT (crosstalk) rotation:")
        print(
            f"  rate  : {xt_rotation_amplitude * 1e3:.4f} MHz ({xt_rotation_amplitude_hw:.6f})"
        )
        print(f"  phase : {xt_rotation_phase:.4f} rad")
        print("CR (cross-resonance) rotation:")
        print(
            f"  rate  : {cr_rotation_amplitude * 1e3:.4f} MHz ({cr_rotation_amplitude_hw:.6f})"
        )
        print(f"  phase : {cr_rotation_phase:.4f} rad")
        print(f"Estimated ZX90 gate length : {zx90_duration:.1f} ns")

    return Result(
        data={
            "cr_label": cr_label,
            "stark_drive_qubit": stark_drive_qubit,
            "cr_frequency": resolved_cr_frequency,
            "control_frequency": control_frequency,
            "target_frequency": target_frequency,
            "Omega": omega,
            "coeffs": coeffs,
            "cr_rotation_amplitude": cr_rotation_amplitude,
            "cr_rotation_amplitude_hw": cr_rotation_amplitude_hw,
            "cr_rotation_phase": cr_rotation_phase,
            "xt_rotation_amplitude": xt_rotation_amplitude,
            "xt_rotation_amplitude_hw": xt_rotation_amplitude_hw,
            "xt_rotation_phase": xt_rotation_phase,
            "cr_drive_amplitude": cr_rabi_rate,
            "cr_drive_amplitude_hw": cr_amplitude,
            "zx90_duration": zx90_duration,
            "result_0": result_0,
            "result_1": result_1,
            "fig_c": fig_c,
            "fig_t": fig_t,
        },
        figures={"control": fig_c, "target": fig_t},
    )


def stark_update_cr_params(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    stark_ramptime: float | None = None,
    cr_amplitude: float | None = None,
    cr_phase: float | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    cr_frequency: float | None = None,
    update_cr_phase: bool = True,
    update_cancel_pulse: bool = True,
    store_params: bool = True,
    x90: TargetMap[Waveform] | None = None,
    use_zvalues: bool = False,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    auto_register_cr_channel: bool = True,
    update_lsi: bool = False,
    plot: bool = True,
) -> Result:
    """Update dressed CR phase/cancellation parameters from Stark tomography."""
    if ramptime is None:
        ramptime = DEFAULT_CR_RAMPTIME
    if cr_amplitude is None:
        cr_amplitude = 1.0
    if cr_phase is None:
        cr_phase = 0.0
    if cancel_amplitude is None:
        cancel_amplitude = 0.0
    if cancel_phase is None:
        cancel_phase = 0.0

    current_cr_pulse = cr_amplitude * np.exp(1j * cr_phase)
    current_cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase)

    result = stark_cr_hamiltonian_tomography(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        stark_amplitude=stark_amplitude,
        stark_drive_qubit=stark_drive_qubit,
        time_range=time_range,
        ramptime=ramptime,
        stark_ramptime=stark_ramptime,
        cr_amplitude=cr_amplitude,
        cr_phase=cr_phase,
        cancel_amplitude=cancel_amplitude,
        cancel_phase=cancel_phase,
        cr_frequency=cr_frequency,
        x90=x90,
        use_zvalues=use_zvalues,
        n_shots=n_shots,
        shot_interval=shot_interval,
        reset_awg_and_capunits=reset_awg_and_capunits,
        auto_register_cr_channel=auto_register_cr_channel,
        update_lsi=update_lsi,
        plot=plot,
    )

    shift = -result["cr_rotation_phase"]
    cancel_pulse = -result["xt_rotation_amplitude_hw"] * np.exp(
        1j * result["xt_rotation_phase"]
    )
    new_cr_pulse = (
        current_cr_pulse * np.exp(1j * shift) if update_cr_phase else current_cr_pulse
    )
    new_cancel_pulse = (
        (current_cancel_pulse + cancel_pulse) * np.exp(1j * shift)
        if update_cancel_pulse
        else current_cancel_pulse
    )

    cr_param = {
        "target": result["cr_label"],
        "duration": 0.0,
        "ramptime": ramptime,
        "cr_amplitude": float(np.abs(new_cr_pulse)),
        "cr_phase": float(np.angle(new_cr_pulse)),
        "cr_beta": 0.0,
        "cancel_amplitude": float(np.abs(new_cancel_pulse)),
        "cancel_phase": float(np.angle(new_cancel_pulse)),
        "cancel_beta": 0.0,
        "rotary_amplitude": 0.0,
        "zx_rotation_rate": float(result["coeffs"]["ZX"] / cr_amplitude),
    }
    if store_params:
        exp.ctx.calib_note.update_cr_param(result["cr_label"], cr_param)

    if plot:
        print("Updated Stark CR params:")
        print(f"  target           : {result['cr_label']}")
        print(
            f"  CR amplitude     : {cr_amplitude:+.4f} -> {cr_param['cr_amplitude']:+.4f}"
        )
        print(f"  CR phase         : {cr_phase:+.4f} -> {cr_param['cr_phase']:+.4f}")
        print(
            f"  Cancel amplitude : {cancel_amplitude:+.4f} -> {cr_param['cancel_amplitude']:+.4f}"
        )
        print(
            f"  Cancel phase     : {cancel_phase:+.4f} -> {cr_param['cancel_phase']:+.4f}"
        )

    return Result(data={**result, "cr_param": cr_param})


def obtain_cr_params_under_stark(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    cr_frequency: float | None = None,
    time_range: ArrayLike | None = None,
    ramptime: float | None = None,
    cr_amplitude: float | None = None,
    n_iterations: int | None = None,
    n_cycles: int | None = None,
    n_points_per_cycle: int | None = None,
    use_stored_params: bool | None = None,
    tolerance: float | None = None,
    adiabatic_safe_factor: float | None = None,
    max_amplitude: float | None = None,
    max_time_range: float | None = None,
    x90: TargetMap[Waveform] | None = None,
    use_zvalues: bool = False,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    reset_awg_and_capunits: bool | None = None,
    auto_register_cr_channel: bool = True,
    update_lsi: bool = False,
    plot: bool | None = None,
) -> Result:
    """Obtain dressed CR parameters while one qubit is Stark-driven."""
    if n_iterations is None:
        n_iterations = 4
    if n_cycles is None:
        n_cycles = 2
    if n_points_per_cycle is None:
        n_points_per_cycle = 6
    if use_stored_params is None:
        use_stored_params = False
    if tolerance is None:
        tolerance = 0.005e-3
    if max_amplitude is None:
        max_amplitude = 1.0
    if max_time_range is None:
        max_time_range = 4096.0
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if plot is None:
        plot = True
    if ramptime is None:
        ramptime = DEFAULT_CR_RAMPTIME

    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    control_frequency, target_frequency, resolved_cr_frequency = _stark_cr_frequencies(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
        cr_frequency=cr_frequency,
    )
    if auto_register_cr_channel:
        _ensure_stark_cr_channel(
            exp,
            control_qubit=control_label,
            target_qubit=target_label,
            stark_drive_qubit=stark_drive_qubit,
            cr_frequency=resolved_cr_frequency,
            update_lsi=update_lsi,
        )
    cr_label = stark_cr_target(
        exp,
        control_label,
        target_label,
        stark_drive_qubit=stark_drive_qubit,
    )

    if adiabatic_safe_factor is None:
        adiabatic_safe_factor = 0.75
    sampling_period = _measurement_sampling_period(exp)

    def create_time_range(zx90_duration: float) -> NDArray:
        period = 4 * zx90_duration
        dt = (period / n_points_per_cycle) // sampling_period * sampling_period
        duration = min(period * n_cycles, max_time_range)
        return np.arange(0, duration + 1, dt)

    f_delta = abs(target_frequency - control_frequency)
    max_cr_rabi = adiabatic_safe_factor * f_delta
    max_cr_amplitude = _bare_control_amplitude(
        exp,
        target=control_label,
        rabi_rate=max_cr_rabi,
    )
    max_cr_amplitude = float(np.clip(max_cr_amplitude, 0.0, max_amplitude))

    current_cr_param = exp.ctx.calib_note.get_cr_param(cr_label)
    if use_stored_params and current_cr_param is not None:
        cr_amplitude = current_cr_param["cr_amplitude"]
        cr_phase = current_cr_param["cr_phase"]
        cancel_amplitude = current_cr_param["cancel_amplitude"]
        cancel_phase = current_cr_param["cancel_phase"]
        zx90_duration = 1 / (4 * cr_amplitude * current_cr_param["zx_rotation_rate"])
        time_values = create_time_range(zx90_duration)
    else:
        cr_amplitude = cr_amplitude if cr_amplitude is not None else max_cr_amplitude
        cr_phase = 0.0
        cancel_amplitude = 0.0
        cancel_phase = 0.0
        time_values = (
            np.asarray(DEFAULT_CR_TIME_RANGE, dtype=float)
            if time_range is None
            else np.asarray(time_range, dtype=float)
        )

    params_history = [
        {
            "time_range": time_values,
            "cr_phase": cr_phase,
            "cancel_amplitude": cancel_amplitude,
            "cancel_phase": cancel_phase,
            "cr_frequency": resolved_cr_frequency,
        }
    ]
    coeffs_history: dict[str, list[float]] = defaultdict(list)
    figs_history = []

    print(f"Conducting Stark CR Hamiltonian tomography for {cr_label}...")
    for index in range(n_iterations):
        print(f"Iteration {index + 1}/{n_iterations}")
        params = params_history[-1]
        result = stark_update_cr_params(
            exp,
            control_qubit=control_label,
            target_qubit=target_label,
            stark_amplitude=stark_amplitude,
            stark_drive_qubit=stark_drive_qubit,
            time_range=params["time_range"],
            ramptime=ramptime,
            stark_ramptime=stark_ramptime,
            cr_amplitude=cr_amplitude,
            cr_phase=float(params["cr_phase"]),
            cancel_amplitude=float(params["cancel_amplitude"]),
            cancel_phase=float(params["cancel_phase"]),
            cr_frequency=resolved_cr_frequency,
            x90=x90,
            use_zvalues=use_zvalues,
            n_shots=n_shots,
            shot_interval=shot_interval,
            reset_awg_and_capunits=reset_awg_and_capunits,
            auto_register_cr_channel=False,
            plot=plot,
        )
        next_time_range = create_time_range(result["zx90_duration"])
        params_history.append(
            {
                "time_range": next_time_range,
                "cr_phase": result["cr_param"]["cr_phase"],
                "cancel_amplitude": result["cr_param"]["cancel_amplitude"],
                "cancel_phase": result["cr_param"]["cancel_phase"],
                "cr_frequency": resolved_cr_frequency,
            }
        )
        figs_history.append({"fig_c": result["fig_c"], "fig_t": result["fig_t"]})
        for key, value in result["coeffs"].items():
            coeffs_history[key].append(value)

        if index > 0:
            ix = coeffs_history["IX"][-1]
            iy = coeffs_history["IY"][-1]
            ix_diff = coeffs_history["IX"][-2] - ix
            iy_diff = coeffs_history["IY"][-2] - iy
            if abs(ix) < tolerance and abs(iy) < tolerance:
                print("Convergence reached.")
                print(f"  IX : {ix * 1e3:.4f} MHz")
                print(f"  IY : {iy * 1e3:.4f} MHz")
                break
            if abs(ix_diff) < tolerance and abs(iy_diff) < tolerance:
                print("Convergence reached.")
                print(f"  IX_diff : {ix_diff * 1e3:.4f} MHz")
                print(f"  IY_diff : {iy_diff * 1e3:.4f} MHz")
                break

    hamiltonian_coeffs = {
        key: np.asarray(value) for key, value in coeffs_history.items()
    }
    fig = viz.make_figure()
    for key, value in hamiltonian_coeffs.items():
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(value) + 1),
                y=value * 1e3,
                mode="lines+markers",
                name=f"{key}/2",
            )
        )
    fig.update_layout(
        title=f"Stark CR Hamiltonian coefficients : {cr_label}",
        xaxis_title="Number of steps",
        yaxis_title="Coefficient (MHz)",
    )
    if plot:
        fig.show()

    return Result(
        data={
            "cr_label": cr_label,
            "stark_drive_qubit": stark_drive_qubit,
            "cr_frequency": resolved_cr_frequency,
            "control_frequency": control_frequency,
            "target_frequency": target_frequency,
            "params_history": params_history,
            "coeffs_history": hamiltonian_coeffs,
            "figs_history": figs_history,
        },
        figure=fig,
        figures={"coeffs": fig},
    )


def stark_obtain_cr_params(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    **kwargs: Any,
) -> Result:
    """Alias for :func:`obtain_cr_params_under_stark`."""
    return obtain_cr_params_under_stark(
        exp,
        control_qubit,
        target_qubit,
        **kwargs,
    )


def _dressed_gate_label(
    exp: Experiment,
    *,
    qubit: str,
    dressed_qubit: str,
) -> str:
    return insitu_target(exp, qubit) if qubit == dressed_qubit else qubit


def _stark_zx90_body(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_drive_qubit: StarkDriveQubit,
    cr_duration: float | None = None,
    cr_ramptime: float | None = None,
    cr_amplitude: float | None = None,
    cr_phase: float | None = None,
    cr_beta: float | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    cancel_beta: float | None = None,
    rotary_amplitude: float | None = None,
    echo: bool = True,
    x180: TargetMap[Waveform] | Waveform | None = None,
    x180_margin: float | None = None,
) -> PulseSchedule:
    cr_label = stark_cr_target(
        exp,
        control_qubit,
        target_qubit,
        stark_drive_qubit=stark_drive_qubit,
    )
    cr_param = exp.ctx.calib_note.get_cr_param(
        cr_label,
        valid_days=exp.ctx.calibration_valid_days,
    )
    if cr_param is None:
        raise ValueError(f"CR parameters for {cr_label} are not stored.")
    _, _, resolved_cr_frequency = _stark_cr_frequencies(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        stark_drive_qubit=stark_drive_qubit,
        cr_frequency=None,
    )
    _ensure_stark_cr_channel(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        stark_drive_qubit=stark_drive_qubit,
        cr_frequency=resolved_cr_frequency,
        update_lsi=False,
    )

    (
        dressed_qubit,
        _dressed_label,
        cr_control,
        cr_target,
        _cr_label,
    ) = _dressed_cr_labels(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        stark_drive_qubit=stark_drive_qubit,
    )

    if x180_margin is None:
        x180_margin = 0.0
    if x180 is None:
        pi_pulse = exp.pulse.x180(cr_control)
    elif isinstance(x180, Waveform):
        pi_pulse = x180
    else:
        pi_pulse = _target_map_get(
            x180,
            target=control_qubit,
            insitu=_dressed_gate_label(
                exp,
                qubit=control_qubit,
                dressed_qubit=dressed_qubit,
            ),
        )
    if pi_pulse is None:
        raise ValueError("Could not resolve the echo pi pulse.")

    if cr_amplitude is None:
        cr_amplitude = cr_param["cr_amplitude"]
    if cr_duration is None:
        cr_duration = cr_param["duration"]
    if cr_ramptime is None:
        cr_ramptime = cr_param["ramptime"]
    if cr_phase is None:
        cr_phase = cr_param["cr_phase"]
    if cr_beta is None:
        cr_beta = cr_param["cr_beta"]
    if cancel_amplitude is None:
        cancel_amplitude = cr_param["cancel_amplitude"]
    if cancel_phase is None:
        cancel_phase = cr_param["cancel_phase"]
    if cancel_beta is None:
        cancel_beta = cr_param["cancel_beta"]
    if rotary_amplitude is None:
        rotary_amplitude = cr_param["rotary_amplitude"]

    cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase) + rotary_amplitude
    return CrossResonance(
        control_qubit=cr_control,
        target_qubit=cr_target,
        cr_amplitude=cr_amplitude,
        cr_duration=cr_duration,
        cr_ramptime=cr_ramptime,
        cr_phase=cr_phase,
        cr_beta=cr_beta,
        cancel_amplitude=np.abs(cancel_pulse),
        cancel_phase=np.angle(cancel_pulse),
        cancel_beta=cancel_beta,
        echo=echo,
        pi_pulse=pi_pulse,
        pi_margin=x180_margin,
    )


def stark_zx90(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    cr_duration: float | None = None,
    cr_ramptime: float | None = None,
    cr_amplitude: float | None = None,
    cr_phase: float | None = None,
    cr_beta: float | None = None,
    cancel_amplitude: float | None = None,
    cancel_phase: float | None = None,
    cancel_beta: float | None = None,
    rotary_amplitude: float | None = None,
    echo: bool = True,
    x180: TargetMap[Waveform] | Waveform | None = None,
    x180_margin: float | None = None,
    plot: bool = False,
) -> PulseSchedule:
    """Build a ZX90 schedule using dressed CR parameters under a Stark tone."""
    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    dressed_qubit = _stark_drive_qubit_label(
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    body = _stark_zx90_body(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
        cr_duration=cr_duration,
        cr_ramptime=cr_ramptime,
        cr_amplitude=cr_amplitude,
        cr_phase=cr_phase,
        cr_beta=cr_beta,
        cancel_amplitude=cancel_amplitude,
        cancel_phase=cancel_phase,
        cancel_beta=cancel_beta,
        rotary_amplitude=rotary_amplitude,
        echo=echo,
        x180=x180,
        x180_margin=x180_margin,
    )
    schedule = _stark_wrapped_schedule(
        stark_label=stark_target(exp, dressed_qubit),
        insitu_label=insitu_target(exp, dressed_qubit),
        stark_amplitude=_stark_drive_amplitude(
            exp,
            target=dressed_qubit,
            stark_amplitude=stark_amplitude,
        ),
        stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
        insitu_sequence=body,
    )
    return _plot_sequence_sample(
        schedule,
        title=(
            f"AC Stark ZX90: {control_label}-{target_label}, stark={stark_drive_qubit}"
        ),
        plot=plot,
    )


def _stark_cnot_body(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_drive_qubit: StarkDriveQubit,
    zx90: PulseSchedule | None = None,
    x90: TargetMap[Waveform] | None = None,
) -> PulseSchedule:
    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    dressed_qubit = _stark_drive_qubit_label(
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    _, _, resolved_cr_frequency = _stark_cr_frequencies(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
        cr_frequency=None,
    )
    _ensure_stark_cr_channel(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
        cr_frequency=resolved_cr_frequency,
        update_lsi=False,
    )
    control_gate = _dressed_gate_label(
        exp,
        qubit=control_label,
        dressed_qubit=dressed_qubit,
    )
    target_gate = _dressed_gate_label(
        exp,
        qubit=target_label,
        dressed_qubit=dressed_qubit,
    )
    target_x90 = _target_map_get(x90, target=target_label, insitu=target_gate)
    if target_x90 is None:
        target_x90 = exp.pulse.x90(target_gate)
    if zx90 is None:
        zx90 = _stark_zx90_body(
            exp,
            control_qubit=control_label,
            target_qubit=target_label,
            stark_drive_qubit=stark_drive_qubit,
        )
    with PulseSchedule() as ps:
        ps.call(zx90)
        ps.add(control_gate, VirtualZ(-np.pi / 2))
        ps.add(target_gate, target_x90.scaled(-1))
    return ps


def stark_cnot(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    zx90: PulseSchedule | None = None,
    x90: TargetMap[Waveform] | None = None,
    plot: bool = False,
) -> PulseSchedule:
    """Build a CNOT schedule using a dressed ZX90 under a Stark tone."""
    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    dressed_qubit = _stark_drive_qubit_label(
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    body = _stark_cnot_body(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
        zx90=zx90,
        x90=x90,
    )
    schedule = _stark_wrapped_schedule(
        stark_label=stark_target(exp, dressed_qubit),
        insitu_label=insitu_target(exp, dressed_qubit),
        stark_amplitude=_stark_drive_amplitude(
            exp,
            target=dressed_qubit,
            stark_amplitude=stark_amplitude,
        ),
        stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
        insitu_sequence=body,
    )
    return _plot_sequence_sample(
        schedule,
        title=(
            f"AC Stark CNOT: {control_label}-{target_label}, stark={stark_drive_qubit}"
        ),
        plot=plot,
    )


def calibrate_stark_zx90(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    ramptime: float | None = None,
    duration: float | None = None,
    amplitude_range: ArrayLike | None = None,
    initial_state: str | None = None,
    degree: int | None = None,
    adiabatic_safe_factor: float | None = None,
    max_amplitude: float | None = None,
    rotary_multiple: float | None = None,
    use_drag: bool | None = None,
    duration_unit: float | None = None,
    duration_buffer: float | None = None,
    n_repetitions: int | None = None,
    x180: TargetMap[Waveform] | Waveform | None = None,
    x180_margin: float | None = None,
    use_zvalues: bool | None = None,
    store_params: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
) -> Result:
    """Calibrate a dressed ZX90 amplitude under a Stark tone."""
    if initial_state is None:
        initial_state = "0"
    if degree is None:
        degree = 3
    if max_amplitude is None:
        max_amplitude = 1.0
    if rotary_multiple is None:
        rotary_multiple = 9.0
    if use_drag is None:
        use_drag = True
    if duration_unit is None:
        duration_unit = 16.0
    if duration_buffer is None:
        duration_buffer = 1.05
    if n_repetitions is None:
        n_repetitions = 1
    if x180_margin is None:
        x180_margin = 0.0
    if use_zvalues is None:
        use_zvalues = False
    if store_params is None:
        store_params = True
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if plot is None:
        plot = True
    if ramptime is None:
        ramptime = DEFAULT_CR_RAMPTIME
    if adiabatic_safe_factor is None:
        adiabatic_safe_factor = 0.75

    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    dressed_qubit = _stark_drive_qubit_label(
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    dressed_label = insitu_target(exp, dressed_qubit)
    cr_label = stark_cr_target(
        exp,
        control_label,
        target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    cr_param = exp.ctx.calib_note.get_cr_param(cr_label)
    if cr_param is None:
        raise ValueError(f"CR parameters for {cr_label} are not stored.")

    cr_amplitude = cr_param["cr_amplitude"]
    cr_phase = cr_param["cr_phase"]
    cancel_amplitude = cr_param["cancel_amplitude"]
    cancel_phase = cr_param["cancel_phase"]
    zx_rotation_rate = cr_param["zx_rotation_rate"]
    zx_frequency = zx_rotation_rate * cr_amplitude
    rotary_amplitude = _bare_control_amplitude(
        exp,
        target=target_label,
        rabi_rate=zx_frequency * rotary_multiple,
    )
    cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase) + rotary_amplitude

    control_frequency, target_frequency, _cr_frequency = _stark_cr_frequencies(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
        cr_frequency=None,
    )
    _ensure_stark_cr_channel(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
        cr_frequency=_cr_frequency,
        update_lsi=False,
    )
    f_delta = abs(target_frequency - control_frequency)
    max_cr_rabi = adiabatic_safe_factor * f_delta
    max_cr_amplitude = _bare_control_amplitude(
        exp,
        target=control_label,
        rabi_rate=max_cr_rabi,
    )
    max_cr_amplitude = float(np.clip(max_cr_amplitude, 0.0, max_amplitude))

    if duration is None:
        if cr_param["duration"] == 0.0:
            duration = duration_buffer / (8 * zx_frequency) + ramptime
            if duration % duration_unit != 0:
                duration = (duration // duration_unit + 1) * duration_unit
        else:
            duration = cr_param["duration"]
    if duration % duration_unit != 0:
        print(
            f"Warning: Duration {duration} ns is not a multiple of duration_unit {duration_unit} ns."
        )

    def ecr_sequence(
        amplitude: float,
        duration: float,
        n_repetitions: int,
    ) -> PulseSchedule:
        scaled_cancel_pulse = amplitude / cr_amplitude * cancel_pulse
        ecr = _stark_zx90_body(
            exp,
            control_qubit=control_label,
            target_qubit=target_label,
            stark_drive_qubit=stark_drive_qubit,
            cr_duration=duration,
            cr_ramptime=ramptime,
            cr_amplitude=amplitude,
            cr_phase=cr_phase,
            cancel_amplitude=np.abs(scaled_cancel_pulse),
            cancel_phase=np.angle(scaled_cancel_pulse),
            rotary_amplitude=0.0,
            echo=True,
            x180=x180,
            x180_margin=x180_margin,
        ).repeated(n_repetitions)
        with PulseSchedule() as body:
            if initial_state != "0":
                init_target = (
                    dressed_label if dressed_qubit == control_label else control_label
                )
                body.add(
                    init_target,
                    exp.pulse.get_pulse_for_state(init_target, initial_state),
                )
                body.barrier()
            body.call(ecr)
        return _stark_wrapped_schedule(
            stark_label=stark_target(exp, dressed_qubit),
            insitu_label=dressed_label,
            stark_amplitude=_stark_drive_amplitude(
                exp,
                target=dressed_qubit,
                stark_amplitude=stark_amplitude,
            ),
            stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
            insitu_sequence=body,
        )

    def calibrate(
        amplitude_values: ArrayLike,
        duration: float,
        n_repetitions: int,
    ) -> dict[str, Any]:
        amplitude_array = np.asarray(amplitude_values, dtype=float)
        min_amplitude = np.clip(amplitude_array[0], 0.0, max_cr_amplitude)
        max_ampl = np.clip(amplitude_array[-1], 0.0, max_cr_amplitude)
        swept_amplitudes = np.linspace(
            min_amplitude,
            max_ampl,
            len(amplitude_array),
        )
        sweep_result = exp.measurement_service.sweep_parameter(
            lambda amplitude: ecr_sequence(
                amplitude=amplitude,
                duration=duration,
                n_repetitions=n_repetitions,
            ),
            sweep_range=swept_amplitudes,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
        )
        sweep_data = sweep_result.data[target_label]
        signal = sweep_data.zvalues if use_zvalues else sweep_data.normalized
        fit_result = fitting.fit_polynomial(
            target=cr_label,
            x=swept_amplitudes,
            y=signal,
            degree=degree,
            title=f"Stark ZX90 calibration (n = {n_repetitions})",
            xlabel="Amplitude (arb. units)",
            ylabel="Signal",
        )
        root = fit_result["root"]
        if np.isnan(root):
            root = None
        return {
            "amplitude_range": swept_amplitudes,
            "signal": signal,
            "root": root,
            "fit_result": fit_result,
        }

    rough_result_n1 = None
    rough_result_n3 = None
    if amplitude_range is None:
        print(
            f"Estimating CR amplitude of {cr_label} (n_repetitions = {n_repetitions})"
        )
        rough_result_n1 = calibrate(
            amplitude_values=np.linspace(0.0, cr_amplitude * 2, 20),
            duration=duration,
            n_repetitions=n_repetitions,
        )
        rough_amplitude = rough_result_n1["root"]
        if rough_amplitude is None:
            duration = (duration * duration_buffer // duration_unit + 1) * duration_unit
            print(f"Retrying with duration = {duration} ns")
            rough_result_n1 = calibrate(
                amplitude_values=np.linspace(0.0, cr_amplitude * 2, 20),
                duration=duration,
                n_repetitions=n_repetitions,
            )
            rough_amplitude = rough_result_n1["root"]
            if rough_amplitude is None:
                raise ValueError(
                    "Could not find a root for the CR amplitude calibration."
                )
        print(
            f"Estimating CR amplitude of {cr_label} (n_repetitions = {n_repetitions + 2})"
        )
        rough_result_n3 = calibrate(
            amplitude_values=np.linspace(0.0, cr_amplitude * 2, 20),
            duration=duration,
            n_repetitions=n_repetitions + 2,
        )
        rough_amplitude_n3 = rough_result_n3["root"]
        rough_amplitudes = [float(rough_amplitude)]
        if rough_amplitude_n3 is None:
            print(
                "Warning: Could not find a rough root for "
                f"n_repetitions = {n_repetitions + 2}; "
                f"using the n_repetitions = {n_repetitions} fine range."
            )
        else:
            rough_amplitudes.append(float(rough_amplitude_n3))
        min_amplitude = min(rough_amplitudes) * 0.8
        max_amplitude = max(rough_amplitudes) * 1.2
        amplitude_range = np.linspace(
            min_amplitude,
            max_amplitude,
            50,
        )
    else:
        amplitude_range = np.asarray(amplitude_range, dtype=float)

    print(f"Calibrating CR amplitude of {cr_label} (n_repetitions = {n_repetitions})")
    result_n1 = calibrate(
        amplitude_values=amplitude_range,
        duration=duration,
        n_repetitions=n_repetitions,
    )
    amplitude_range = np.asarray(result_n1["amplitude_range"])
    signal_n1 = result_n1["signal"]
    fit_result_n1 = result_n1["fit_result"]

    print(
        f"Calibrating CR amplitude of {cr_label} (n_repetitions = {n_repetitions + 2})"
    )
    result_n3 = calibrate(
        amplitude_values=amplitude_range,
        duration=duration,
        n_repetitions=n_repetitions + 2,
    )
    signal_n3 = result_n3["signal"]
    fit_result_n3 = result_n3["fit_result"]

    signal = signal_n1 - signal_n3
    fit_result = fitting.fit_polynomial(
        target=cr_label,
        x=amplitude_range,
        y=signal,
        degree=degree,
        title="Stark ZX90 calibration",
        xlabel="Amplitude (arb. units)",
        ylabel="Signal",
    )
    calibrated_cr_amplitude = fit_result["root"]
    if np.isnan(calibrated_cr_amplitude):
        print("Could not find a root for the CR amplitude calibration.")
        calibrated_cr_amplitude = 1.0

    calibrated_cancel_amplitude = (
        calibrated_cr_amplitude / cr_amplitude * cancel_amplitude
    )
    calibrated_rotary_amplitude = (
        calibrated_cr_amplitude / cr_amplitude * rotary_amplitude
    )

    if use_drag:
        delta_ct = 2 * np.pi * (control_frequency - target_frequency)
        cr_beta = -1 / delta_ct
        cancel_beta = 0.0
    else:
        cr_beta = 0.0
        cancel_beta = 0.0

    if store_params:
        exp.ctx.calib_note.update_cr_param(
            cr_label,
            {
                "target": cr_label,
                "duration": duration,
                "ramptime": ramptime,
                "cr_amplitude": calibrated_cr_amplitude,
                "cr_phase": cr_phase,
                "cr_beta": cr_beta,
                "cancel_amplitude": calibrated_cancel_amplitude,
                "cancel_phase": cancel_phase,
                "cancel_beta": cancel_beta,
                "rotary_amplitude": calibrated_rotary_amplitude,
                "zx_rotation_rate": zx_rotation_rate,
            },
        )

    print()
    print("Calibrated Stark CR parameters:")
    print(f"  target           : {cr_label}")
    print(f"  CR duration      : {duration:.1f} ns")
    print(f"  CR ramptime      : {ramptime:.1f} ns")
    print(f"  CR amplitude     : {calibrated_cr_amplitude:.6f}")
    print(f"  CR phase         : {cr_phase:.6f}")
    print(f"  CR beta          : {cr_beta:.6f}")
    print(f"  Cancel amplitude : {calibrated_cancel_amplitude:.6f}")
    print(f"  Cancel phase     : {cancel_phase:.6f}")
    print(f"  Cancel beta      : {cancel_beta:.6f}")
    print(f"  Rotary amplitude : {calibrated_rotary_amplitude:.6f}")
    print()

    if plot:
        stark_zx90(
            exp,
            control_label,
            target_label,
            stark_amplitude=stark_amplitude,
            stark_drive_qubit=stark_drive_qubit,
            stark_ramptime=stark_ramptime,
            plot=True,
        )

    return Result(
        data={
            "amplitude_range": amplitude_range,
            "signal": signal,
            **fit_result,
            "n1": {"signal": signal_n1, **fit_result_n1},
            "n3": {"signal": signal_n3, **fit_result_n3},
            "rough_n1": rough_result_n1,
            "rough_n3": rough_result_n3,
        }
    )


def stark_bell_state_sequence(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    control_basis: Literal["X", "Y", "Z"] = "Z",
    target_basis: Literal["X", "Y", "Z"] = "Z",
    zx90: PulseSchedule | None = None,
    x90: TargetMap[Waveform] | None = None,
    plot: bool = False,
) -> PulseSchedule:
    """Build a Bell-state preparation and basis-rotation schedule under Stark."""
    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    dressed_qubit = _stark_drive_qubit_label(
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    control_gate = _dressed_gate_label(
        exp,
        qubit=control_label,
        dressed_qubit=dressed_qubit,
    )
    target_gate = _dressed_gate_label(
        exp,
        qubit=target_label,
        dressed_qubit=dressed_qubit,
    )
    control_x90 = _target_map_get(x90, target=control_label, insitu=control_gate)
    target_x90 = _target_map_get(x90, target=target_label, insitu=target_gate)
    if control_x90 is None:
        control_x90 = exp.pulse.x90(control_gate)
    if target_x90 is None:
        target_x90 = exp.pulse.x90(target_gate)

    with PulseSchedule() as body:
        body.add(control_gate, control_x90.shifted(np.pi / 2))
        body.call(
            _stark_cnot_body(
                exp,
                control_qubit=control_label,
                target_qubit=target_label,
                stark_drive_qubit=stark_drive_qubit,
                zx90=zx90,
                x90=x90,
            )
        )
        control_rotation = _basis_rotation(control_x90, control_basis)
        target_rotation = _basis_rotation(target_x90, target_basis)
        if control_rotation is not None:
            body.add(control_gate, control_rotation)
        if target_rotation is not None:
            body.add(target_gate, target_rotation)

    schedule = _stark_wrapped_schedule(
        stark_label=stark_target(exp, dressed_qubit),
        insitu_label=insitu_target(exp, dressed_qubit),
        stark_amplitude=_stark_drive_amplitude(
            exp,
            target=dressed_qubit,
            stark_amplitude=stark_amplitude,
        ),
        stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
        insitu_sequence=body,
    )
    return _plot_sequence_sample(
        schedule,
        title=(
            f"AC Stark Bell sequence: {control_label}-{target_label}, "
            f"basis={control_basis}{target_basis}, stark={stark_drive_qubit}"
        ),
        plot=plot,
    )


def stark_measure_bell_state(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    control_basis: Literal["X", "Y", "Z"] = "Z",
    target_basis: Literal["X", "Y", "Z"] = "Z",
    zx90: PulseSchedule | None = None,
    x90: TargetMap[Waveform] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    plot_sequence: bool = False,
    plot_raw: bool = True,
    plot_mitigated: bool = True,
    save_image: bool = True,
    reset_awg_and_capunits: bool = True,
) -> Result:
    """Measure Bell-state probabilities under a Stark tone."""
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if exp.ctx.state_centers is None:
        exp.build_classifier(plot=False)

    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    pair = [control_label, target_label]
    sequence = stark_bell_state_sequence(
        exp,
        control_label,
        target_label,
        stark_amplitude=stark_amplitude,
        stark_drive_qubit=stark_drive_qubit,
        stark_ramptime=stark_ramptime,
        control_basis=control_basis,
        target_basis=target_basis,
        zx90=zx90,
        x90=x90,
        plot=False,
    )
    result = exp.measurement_service.measure(
        sequence,
        mode="single",
        n_shots=n_shots,
        shot_interval=shot_interval,
        reset_awg_and_capunits=reset_awg_and_capunits,
    )
    basis_labels = result.get_basis_labels(pair)
    prob_dict_raw = result.get_probabilities(pair)
    prob_dict_raw = {label: prob_dict_raw.get(label, 0) for label in basis_labels}
    prob_dict_mitigated = result.get_mitigated_probabilities(pair)

    labels = [f"|{label}>" for label in prob_dict_raw]
    prob_arr_raw = np.asarray(list(prob_dict_raw.values()))
    prob_arr_mitigated = np.asarray(list(prob_dict_mitigated.values()))

    fig = viz.make_figure()
    if plot_raw:
        fig.add_trace(go.Bar(x=labels, y=prob_arr_raw, name="Raw"))
    if plot_mitigated:
        fig.add_trace(go.Bar(x=labels, y=prob_arr_mitigated, name="Mitigated"))
    fig.update_layout(
        title=f"Stark Bell state measurement: {control_label}-{target_label}",
        xaxis_title=f"State ({control_basis}{target_basis} basis)",
        yaxis_title="Probability",
        barmode="group",
        yaxis_range=[0, 1],
    )
    if plot:
        if plot_sequence:
            sequence.plot(
                title=(f"Stark Bell measurement: {control_basis}{target_basis} basis")
            )
        fig.show(config={"toImageButtonOptions": {"format": "png", "scale": 3}})
    if save_image:
        viz.save_figure(
            fig, f"stark_bell_state_measurement_{control_label}-{target_label}"
        )

    return Result(
        data={
            "raw": prob_arr_raw,
            "mitigated": prob_arr_mitigated,
            "result": result,
        },
        figure=fig,
    )


def stark_bell_state_tomography(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    readout_mitigation: bool = True,
    zx90: PulseSchedule | None = None,
    x90: TargetMap[Waveform] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    save_image: bool = True,
    mle_fit: bool = True,
) -> Result:
    """Run Bell-state tomography for a dressed 2Q gate under a Stark tone."""
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    dim = 4
    probabilities = {}
    for control_basis, target_basis in tqdm(
        product(["X", "Y", "Z"], repeat=2),
        desc="Measuring Stark Bell state",
    ):
        result = stark_measure_bell_state(
            exp,
            control_label,
            target_label,
            stark_amplitude=stark_amplitude,
            stark_drive_qubit=stark_drive_qubit,
            stark_ramptime=stark_ramptime,
            control_basis=control_basis,
            target_basis=target_basis,
            zx90=zx90,
            x90=x90,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            save_image=False,
        )
        basis = f"{control_basis}{target_basis}"
        probabilities[basis] = (
            result["mitigated"] if readout_mitigation else result["raw"]
        )

    expected_values = {}
    paulis = {
        "I": np.array([[1, 0], [0, 1]]),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for control_basis, control_pauli in paulis.items():
        for target_basis, target_pauli in paulis.items():
            basis = f"{control_basis}{target_basis}"
            if basis == "II":
                p = probabilities["ZZ"]
                e = p[0b00] + p[0b01] + p[0b10] + p[0b11]
            elif basis in ["IX", "IY", "IZ"]:
                p = probabilities[f"Z{target_basis}"]
                e = p[0b00] - p[0b01] + p[0b10] - p[0b11]
            elif basis in ["XI", "YI", "ZI"]:
                p = probabilities[f"{control_basis}Z"]
                e = p[0b00] + p[0b01] - p[0b10] - p[0b11]
            else:
                p = probabilities[basis]
                e = p[0b00] - p[0b01] - p[0b10] + p[0b11]
            rho += e * np.kron(control_pauli, target_pauli)
            expected_values[basis] = e

    rho = mle_fit_density_matrix(expected_values) if mle_fit else rho / dim
    bell_state = np.zeros((dim, 1), dtype=np.complex128)
    bell_state[0, 0] = 1 / np.sqrt(2)
    bell_state[-1, 0] = 1 / np.sqrt(2)
    fidelity = float(np.real(bell_state.T.conj() @ rho @ bell_state))

    fig = plot_ghz_state_tomography(
        rho=rho,
        qubits=[control_label, target_label],
        fidelity=fidelity,
        plot=plot,
        save_image=save_image,
        width=600,
        height=366,
        file_name=f"stark_bell_state_tomography_{control_label}-{target_label}",
    )["figure"]
    return Result(
        data={
            "probabilities": probabilities,
            "expected_values": expected_values,
            "density_matrix": rho,
            "fidelity": fidelity,
        },
        figure=fig,
    )


def stark_rb_sequence_2q(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    n: int = 8,
    seed: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    zx90: PulseSchedule | None = None,
    interleaved_clifford: str | Clifford | None = None,
    interleaved_waveform: PulseSchedule | None = None,
    plot: bool = True,
) -> PulseSchedule:
    """Build and optionally plot one dressed 2Q RB schedule under Stark."""
    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    dressed_qubit = _stark_drive_qubit_label(
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    control_gate = _dressed_gate_label(
        exp,
        qubit=control_label,
        dressed_qubit=dressed_qubit,
    )
    target_gate = _dressed_gate_label(
        exp,
        qubit=target_label,
        dressed_qubit=dressed_qubit,
    )
    cr_label = stark_cr_target(
        exp,
        control_label,
        target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    _, _, resolved_cr_frequency = _stark_cr_frequencies(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
        cr_frequency=None,
    )
    _ensure_stark_cr_channel(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
        cr_frequency=resolved_cr_frequency,
        update_lsi=False,
    )
    xi90 = _target_map_get(x90, target=control_label, insitu=control_gate)
    ix90 = _target_map_get(x90, target=target_label, insitu=target_gate)
    xi90 = xi90 or exp.pulse.x90(control_gate)
    ix90 = ix90 or exp.pulse.x90(target_gate)
    z90 = VirtualZ(np.pi / 2)
    zx90 = zx90 or _stark_zx90_body(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
    )

    resolved_clifford = (
        None
        if interleaved_clifford is None
        else _resolve_clifford(exp, interleaved_clifford)
    )
    if resolved_clifford is None:
        cliffords, inverse = (
            exp.benchmarking_service.clifford_generator.create_rb_sequences(
                n=n,
                type="2Q",
                seed=seed,
            )
        )
    else:
        if interleaved_waveform is None:
            if resolved_clifford.name == "ZX90":
                interleaved_waveform = zx90
            else:
                raise ValueError("interleaved_waveform must be provided.")
        cliffords, inverse = (
            exp.benchmarking_service.clifford_generator.create_irb_sequences(
                n=n,
                interleave=resolved_clifford,
                type="2Q",
                seed=seed,
            )
        )

    with PulseSchedule([control_gate, cr_label, target_gate]) as body:

        def add_gate(gate: str) -> None:
            if gate == "XI90":
                body.add(control_gate, xi90)
            elif gate == "IX90":
                body.add(target_gate, ix90)
            elif gate == "ZI90":
                body.add(control_gate, z90)
            elif gate == "IZ90":
                body.add(target_gate, z90)
                body.add(cr_label, z90)
            elif gate == "ZX90":
                body.barrier()
                body.call(zx90)
                body.barrier()
            else:
                raise ValueError("Invalid gate.")

        for clifford in cliffords:
            for gate in clifford:
                add_gate(gate)
            if interleaved_waveform is not None:
                body.barrier()
                body.call(interleaved_waveform)
                body.barrier()
        for gate in inverse:
            add_gate(gate)

    schedule = _stark_wrapped_schedule(
        stark_label=stark_target(exp, dressed_qubit),
        insitu_label=insitu_target(exp, dressed_qubit),
        stark_amplitude=_stark_drive_amplitude(
            exp,
            target=dressed_qubit,
            stark_amplitude=stark_amplitude,
        ),
        stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
        insitu_sequence=body,
    )
    return _plot_sequence_sample(
        schedule,
        title=(
            f"AC Stark 2Q RB sample: {control_label}-{target_label}, "
            f"n={n}, stark={stark_drive_qubit}"
        ),
        plot=plot,
    )


def stark_rb_experiment_2q(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    zx90: PulseSchedule | None = None,
    interleaved_clifford: str | Clifford | None = None,
    interleaved_waveform: PulseSchedule | None = None,
    mitigate_readout: bool = True,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] = "linear",
    plot: bool = True,
    save_image: bool = True,
    reset_awg_and_capunits: bool = True,
) -> Result:
    """Run dressed two-qubit RB under a Stark tone."""
    if exp.ctx.state_centers is None:
        raise ValueError("State classifiers are not built.")
    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS
    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_2Q
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    target = stark_cr_target(
        exp,
        control_label,
        target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    if seeds is None:
        seed_values = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seed_values = np.asarray(seeds, dtype=int)
        if len(seed_values) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )
    sweep_range = _clifford_sweep_range(
        n_cliffords_range=n_cliffords_range,
        max_n_cliffords=max_n_cliffords,
    )
    if reset_awg_and_capunits:
        exp.ctx.reset_awg_and_capunits(qubits=[control_label, target_label])

    mean_data: list[float] = []
    std_data: list[float] = []
    actual_sweep_range = []
    for n_clifford in sweep_range:
        trial_data = []
        for seed in seed_values:
            sequence = stark_rb_sequence_2q(
                exp,
                control_label,
                target_label,
                stark_amplitude=stark_amplitude,
                stark_drive_qubit=stark_drive_qubit,
                stark_ramptime=stark_ramptime,
                n=int(n_clifford),
                seed=int(seed),
                x90=x90,
                zx90=zx90,
                interleaved_clifford=interleaved_clifford,
                interleaved_waveform=interleaved_waveform,
                plot=False,
            )
            result = exp.measurement_service.measure(
                sequence,
                mode="single",
                n_shots=n_shots,
                shot_interval=shot_interval,
                reset_awg_and_capunits=False,
                plot=False,
            )
            if mitigate_readout:
                prob = result.get_mitigated_probabilities([control_label, target_label])
            else:
                prob = result.get_probabilities([control_label, target_label])
            trial_data.append(prob["00"])
        mean = float(np.mean(trial_data))
        std = float(np.std(trial_data))
        mean_data.append(mean)
        std_data.append(std)
        actual_sweep_range.append(n_clifford)
        if n_cliffords_range is None and mean - 0.5 * std < 0.25:
            break

    actual_sweep_array = np.asarray(actual_sweep_range, dtype=int)
    mean_array = np.asarray(mean_data)
    std_array = np.asarray(std_data) if n_trials > 1 else None
    fit_result = fitting.fit_rb(
        target=target,
        x=actual_sweep_array,
        y=mean_array,
        error_y=std_array,
        dimension=4,
        title="Stark two-qubit randomized benchmarking",
        xlabel="Number of Cliffords",
        ylabel="Normalized signal",
        xaxis_type=xaxis_type,
        yaxis_type="linear",
        plot=plot,
    )
    fig = fit_result.get_figure()
    if save_image:
        viz.save_figure(fig, name=f"stark_rb_experiment_2q_{target}")
    return Result(
        data={
            target: {
                "n_cliffords": actual_sweep_array,
                "mean": mean_array,
                "std": std_array,
                **fit_result,
            }
        },
        figures={target: fig},
    )


def stark_interleaved_randomized_benchmarking_2q(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    interleaved_clifford: str | Clifford,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    interleaved_waveform: PulseSchedule | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    zx90: PulseSchedule | None = None,
    mitigate_readout: bool = True,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    save_image: bool = True,
) -> Result:
    """Run dressed 2Q interleaved randomized benchmarking under Stark."""
    clifford = _resolve_clifford(exp, interleaved_clifford)
    target = stark_cr_target(
        exp,
        control_qubit,
        target_qubit,
        stark_drive_qubit=stark_drive_qubit,
    )
    rb_result = stark_rb_experiment_2q(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        stark_amplitude=stark_amplitude,
        stark_drive_qubit=stark_drive_qubit,
        stark_ramptime=stark_ramptime,
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        zx90=zx90,
        mitigate_readout=mitigate_readout,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
        save_image=False,
    )
    irb_result = stark_rb_experiment_2q(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        stark_amplitude=stark_amplitude,
        stark_drive_qubit=stark_drive_qubit,
        stark_ramptime=stark_ramptime,
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        zx90=zx90,
        interleaved_clifford=clifford,
        interleaved_waveform=interleaved_waveform,
        mitigate_readout=mitigate_readout,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
        save_image=False,
    )
    return _interleaved_fit_result(
        target=target,
        reference_result=rb_result,
        interleaved_result=irb_result,
        clifford=clifford,
        plot=plot,
        save_image=save_image,
        image_name=f"stark_interleaved_randomized_benchmarking_2q_{target}",
        title_prefix="Stark 2Q interleaved randomized benchmarking",
        dimension=4,
    )


def calibrate_stark_default_pulse(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    pulse_type: Literal["pi", "hpi"],
    stark_ramptime: float | None = None,
    duration: float | None = None,
    ramptime: float | None = None,
    n_points: int = 20,
    n_rotations: int = 1,
    amplitude_range: ArrayLike | None = None,
    r2_threshold: float = 0.5,
    update_params: bool = True,
    plot: bool = True,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> ExperimentResult[AmplCalibData]:
    """
    Calibrate an in-situ pi or half-pi pulse while the Stark tone is applied.

    The Stark tone is emitted on ``stark_target(exp, target)`` and the pulse
    being calibrated is stored under ``insitu_target(exp, target)``.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="calibrate_stark_default_pulse",
    )
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    exp.pulse.validate_rabi_params([target])
    stark_label = stark_target(exp, target)
    insitu_label = insitu_target(exp, target)
    stark_ramptime = _resolve_stark_ramptime(stark_ramptime)
    stark_power = _stark_drive_amplitude(
        exp,
        target=target,
        stark_amplitude=stark_amplitude,
    )
    sampling_period = _measurement_sampling_period(exp)

    if pulse_type == "hpi":
        pulse = FlatTop(
            duration=duration if duration is not None else HPI_DURATION,
            amplitude=1,
            tau=ramptime if ramptime is not None else HPI_RAMPTIME,
        )
        rabi_rate = 0.25 / _pulse_area(pulse, sampling_period)
    elif pulse_type == "pi":
        pulse = FlatTop(
            duration=duration if duration is not None else PI_DURATION,
            amplitude=1,
            tau=ramptime if ramptime is not None else PI_RAMPTIME,
        )
        rabi_rate = 0.5 / _pulse_area(pulse, sampling_period)
    else:
        raise ValueError("`pulse_type` must be 'pi' or 'hpi'.")

    n_per_rotation = 2 if pulse_type == "pi" else 4
    estimated_amplitude = exp.pulse.calc_control_amplitude(target, rabi_rate)
    sweep_range = _amplitude_sweep(
        center=estimated_amplitude,
        n_points=n_points,
        n_rotations=n_rotations,
        amplitude_range=amplitude_range,
    )

    def sequence(amplitude: float) -> PulseSchedule:
        insitu_sequence = pulse.scaled(amplitude).repeated(n_per_rotation * n_rotations)
        return _stark_wrapped_schedule(
            stark_label=stark_label,
            insitu_label=insitu_label,
            stark_amplitude=stark_power,
            stark_ramptime=stark_ramptime,
            insitu_sequence=insitu_sequence,
        )

    sweep_data = exp.measurement_service.sweep_parameter(
        sequence=sequence,
        sweep_range=sweep_range,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
    ).data[target]

    fit_result = fitting.fit_ampl_calib_data(
        target=insitu_label,
        amplitude_range=sweep_range,
        data=sweep_data.normalized,
        plot=plot,
        title=f"Stark {pulse_type} pulse calibration",
        ylabel="Normalized signal",
    )

    r2 = fit_result["r2"]
    if r2 > r2_threshold and update_params:
        params = FlatTopParam(
            target=insitu_label,
            duration=pulse.duration,
            amplitude=fit_result["amplitude"],
            tau=pulse.tau,
        )
        if pulse_type == "hpi":
            exp.ctx.calib_note.update_hpi_param(insitu_label, params)
        else:
            exp.ctx.calib_note.update_pi_param(insitu_label, params)
    elif r2 <= r2_threshold:
        print(f"Error: R² value is too low ({r2:.3f})")
        print(f"Calibration data not stored for {insitu_label}.")

    return ExperimentResult(
        data={
            target: AmplCalibData.new(
                sweep_data=sweep_data,
                calib_value=fit_result["amplitude"],
                r2=r2,
            )
        }
    )


def calibrate_stark_hpi_pulse(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    duration: float | None = None,
    ramptime: float | None = None,
    amplitude_range: ArrayLike | None = None,
    n_points: int = 20,
    n_rotations: int = 1,
    r2_threshold: float = 0.5,
    plot: bool = True,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> ExperimentResult[AmplCalibData]:
    """Calibrate an in-situ half-pi pulse while the Stark tone is applied."""
    return calibrate_stark_default_pulse(
        exp,
        target=target,
        pulse_type="hpi",
        stark_amplitude=stark_amplitude,
        stark_ramptime=stark_ramptime,
        duration=duration,
        ramptime=ramptime,
        amplitude_range=amplitude_range,
        n_points=n_points,
        n_rotations=n_rotations,
        r2_threshold=r2_threshold,
        plot=plot,
        n_shots=n_shots,
        shot_interval=shot_interval,
        **deprecated_options,
    )


def calibrate_stark_pi_pulse(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    duration: float | None = None,
    ramptime: float | None = None,
    amplitude_range: ArrayLike | None = None,
    n_points: int = 20,
    n_rotations: int = 1,
    r2_threshold: float = 0.5,
    plot: bool = True,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> ExperimentResult[AmplCalibData]:
    """Calibrate an in-situ pi pulse while the Stark tone is applied."""
    return calibrate_stark_default_pulse(
        exp,
        target=target,
        pulse_type="pi",
        stark_amplitude=stark_amplitude,
        stark_ramptime=stark_ramptime,
        duration=duration,
        ramptime=ramptime,
        amplitude_range=amplitude_range,
        n_points=n_points,
        n_rotations=n_rotations,
        r2_threshold=r2_threshold,
        plot=plot,
        n_shots=n_shots,
        shot_interval=shot_interval,
        **deprecated_options,
    )


def _build_drag_pulse_for_calibration(
    exp: Experiment,
    *,
    target: str,
    insitu_label: str,
    pulse_type: Literal["pi", "hpi"],
    duration: float | None,
    drag_coeff: float,
    use_stored_amplitude: bool,
    use_stored_beta: bool,
) -> tuple[Drag, float]:
    sampling_period = _measurement_sampling_period(exp)
    if pulse_type == "hpi":
        stored = exp.ctx.calib_note.get_drag_hpi_param(insitu_label)
        beta = stored["beta"] if stored is not None and use_stored_beta else None
        if beta is None:
            beta = -drag_coeff / exp.ctx.qubits[target].alpha
        pulse = Drag(
            duration=duration if duration is not None else DRAG_HPI_DURATION,
            amplitude=1,
            beta=beta,
        )
        rabi_rate = 0.25 / _pulse_area(pulse, sampling_period)
    elif pulse_type == "pi":
        stored = exp.ctx.calib_note.get_drag_pi_param(insitu_label)
        beta = stored["beta"] if stored is not None and use_stored_beta else None
        if beta is None:
            beta = -drag_coeff / exp.ctx.qubits[target].alpha
        pulse = Drag(
            duration=duration if duration is not None else DRAG_PI_DURATION,
            amplitude=1,
            beta=beta,
        )
        rabi_rate = 0.5 / _pulse_area(pulse, sampling_period)
    else:
        raise ValueError("`pulse_type` must be 'pi' or 'hpi'.")

    if stored is not None and use_stored_amplitude:
        amplitude = float(stored["amplitude"])
    else:
        amplitude = exp.pulse.calc_control_amplitude(target, rabi_rate)
    return pulse, amplitude


def calibrate_stark_drag_amplitude(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    pulse_type: Literal["pi", "hpi"],
    stark_ramptime: float | None = None,
    duration: float | None = None,
    n_points: int = 20,
    n_rotations: int = 4,
    amplitude_range: ArrayLike | None = None,
    r2_threshold: float = 0.5,
    drag_coeff: float = DRAG_COEFF,
    use_stored_amplitude: bool = False,
    use_stored_beta: bool = False,
    plot: bool = True,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> Result:
    """Calibrate in-situ DRAG pulse amplitude while the Stark tone is applied."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="calibrate_stark_drag_amplitude",
    )
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    exp.pulse.validate_rabi_params([target])
    stark_label = stark_target(exp, target)
    insitu_label = insitu_target(exp, target)
    stark_ramptime = _resolve_stark_ramptime(stark_ramptime)
    stark_power = _stark_drive_amplitude(
        exp,
        target=target,
        stark_amplitude=stark_amplitude,
    )
    pulse, estimated_amplitude = _build_drag_pulse_for_calibration(
        exp,
        target=target,
        insitu_label=insitu_label,
        pulse_type=pulse_type,
        duration=duration,
        drag_coeff=drag_coeff,
        use_stored_amplitude=use_stored_amplitude,
        use_stored_beta=use_stored_beta,
    )
    sweep_range = _amplitude_sweep(
        center=estimated_amplitude,
        n_points=n_points,
        n_rotations=n_rotations,
        amplitude_range=amplitude_range,
    )
    n_per_rotation = 2 if pulse_type == "pi" else 4

    def sequence(amplitude: float) -> PulseSchedule:
        insitu_sequence = pulse.scaled(amplitude).repeated(n_per_rotation * n_rotations)
        return _stark_wrapped_schedule(
            stark_label=stark_label,
            insitu_label=insitu_label,
            stark_amplitude=stark_power,
            stark_ramptime=stark_ramptime,
            insitu_sequence=insitu_sequence,
        )

    sweep_data = exp.measurement_service.sweep_parameter(
        sequence=sequence,
        sweep_range=sweep_range,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
    ).data[target]

    fit_result = fitting.fit_ampl_calib_data(
        target=insitu_label,
        amplitude_range=sweep_range,
        data=sweep_data.normalized,
        plot=plot,
        title=f"Stark DRAG {pulse_type} amplitude calibration",
        ylabel="Normalized signal",
    )

    r2 = fit_result["r2"]
    if r2 > r2_threshold:
        params = DragParam(
            target=insitu_label,
            duration=pulse.duration,
            amplitude=fit_result["amplitude"],
            beta=pulse.beta,
        )
        if pulse_type == "hpi":
            exp.ctx.calib_note.update_drag_hpi_param(insitu_label, params)
        else:
            exp.ctx.calib_note.update_drag_pi_param(insitu_label, params)
    else:
        print(f"Error: R² value is too low ({r2:.3f})")
        print(f"Calibration data not stored for {insitu_label}.")

    return Result(data={target: fit_result})


def calibrate_stark_drag_beta(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    pulse_type: Literal["pi", "hpi"] = "hpi",
    stark_ramptime: float | None = None,
    beta_range: ArrayLike | None = None,
    duration: float | None = None,
    n_turns: int = 1,
    degree: int = 3,
    plot: bool = True,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> Result:
    """Calibrate in-situ DRAG beta while the Stark tone is applied."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="calibrate_stark_drag_beta",
    )
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if beta_range is None:
        beta_range = np.linspace(-2.0, 2.0, 20)

    exp.pulse.validate_rabi_params([target])
    stark_label = stark_target(exp, target)
    insitu_label = insitu_target(exp, target)
    stark_ramptime = _resolve_stark_ramptime(stark_ramptime)
    stark_power = _stark_drive_amplitude(
        exp,
        target=target,
        stark_amplitude=stark_amplitude,
    )

    if pulse_type == "hpi":
        param = exp.ctx.calib_note.get_drag_hpi_param(insitu_label)
    elif pulse_type == "pi":
        param = exp.ctx.calib_note.get_drag_pi_param(insitu_label)
    else:
        raise ValueError("`pulse_type` must be 'pi' or 'hpi'.")
    if param is None:
        raise ValueError(f"DRAG parameters are not stored for `{insitu_label}`.")

    drag_duration = duration if duration is not None else param["duration"]
    drag_amplitude = param["amplitude"]
    sweep_range = np.asarray(beta_range, dtype=float) + param["beta"]

    def drag_sequence(beta: float) -> PulseArray:
        if pulse_type == "hpi":
            x90p = Drag(duration=drag_duration, amplitude=drag_amplitude, beta=beta)
            x90m = x90p.scaled(-1)
            y90m = exp.pulse.get_hpi_pulse(insitu_label).shifted(-np.pi / 2)
            return PulseArray([x90p, PulseArray([x90m, x90p] * n_turns), y90m])
        x180p = Drag(duration=drag_duration, amplitude=drag_amplitude, beta=beta)
        x180m = x180p.scaled(-1)
        y90m = exp.pulse.get_hpi_pulse(insitu_label).shifted(-np.pi / 2)
        return PulseArray([PulseArray([x180p, x180m] * n_turns), y90m])

    def sequence(beta: float) -> PulseSchedule:
        return _stark_wrapped_schedule(
            stark_label=stark_label,
            insitu_label=insitu_label,
            stark_amplitude=stark_power,
            stark_ramptime=stark_ramptime,
            insitu_sequence=drag_sequence(beta),
        )

    sweep_data = exp.measurement_service.sweep_parameter(
        sequence=sequence,
        sweep_range=sweep_range,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=False,
    ).data[target]

    fit_result = fitting.fit_polynomial(
        target=insitu_label,
        x=sweep_range,
        y=sweep_data.normalized,
        degree=degree,
        plot=plot,
        title=f"Stark DRAG {pulse_type} beta calibration",
        xlabel="Beta",
        ylabel="Normalized signal",
    )
    beta = fit_result["root"]
    if np.isnan(beta):
        beta = 0.0

    params = DragParam(
        target=insitu_label,
        duration=drag_duration,
        amplitude=drag_amplitude,
        beta=beta,
    )
    if pulse_type == "hpi":
        exp.ctx.calib_note.update_drag_hpi_param(insitu_label, params)
    else:
        exp.ctx.calib_note.update_drag_pi_param(insitu_label, params)

    return Result(data={target: beta})


def calibrate_stark_drag_hpi_pulse(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    n_points: int = 20,
    n_rotations: int = 4,
    n_turns: int = 1,
    n_iterations: int = 2,
    amplitude_range: ArrayLike | None = None,
    degree: int = 3,
    r2_threshold: float = 0.5,
    calibrate_beta: bool = True,
    beta_range: ArrayLike | None = None,
    duration: float | None = None,
    drag_coeff: float = DRAG_COEFF,
    plot: bool = True,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> Result:
    """Run iterative in-situ DRAG half-pi amplitude and beta calibration."""
    amplitude: Result | None = None
    beta: Result | dict[str, float] | None = None
    for index in range(n_iterations):
        print(f"\nIteration {index + 1}/{n_iterations}")
        amplitude = calibrate_stark_drag_amplitude(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
            stark_ramptime=stark_ramptime,
            pulse_type="hpi",
            n_points=n_points,
            n_rotations=1 if index == 0 else n_rotations,
            amplitude_range=None if index == 0 else amplitude_range,
            r2_threshold=r2_threshold,
            duration=duration,
            drag_coeff=drag_coeff,
            use_stored_amplitude=index > 0,
            use_stored_beta=index > 0,
            plot=plot,
            n_shots=n_shots,
            shot_interval=shot_interval,
            **deprecated_options,
        )
        if calibrate_beta:
            beta = calibrate_stark_drag_beta(
                exp,
                target=target,
                stark_amplitude=stark_amplitude,
                stark_ramptime=stark_ramptime,
                pulse_type="hpi",
                beta_range=beta_range,
                n_turns=n_turns,
                duration=duration,
                degree=degree,
                plot=plot,
                n_shots=n_shots,
                shot_interval=shot_interval,
                **deprecated_options,
            )
        else:
            beta = {target: -drag_coeff / exp.ctx.qubits[target].alpha}
    return Result(data={"amplitude": amplitude, "beta": beta})


def calibrate_stark_drag_pi_pulse(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    n_points: int = 20,
    n_rotations: int = 4,
    n_turns: int = 1,
    n_iterations: int = 2,
    amplitude_range: ArrayLike | None = None,
    degree: int = 3,
    r2_threshold: float = 0.5,
    calibrate_beta: bool = True,
    beta_range: ArrayLike | None = None,
    duration: float | None = None,
    drag_coeff: float = DRAG_COEFF,
    plot: bool = True,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    **deprecated_options: Any,
) -> Result:
    """Run iterative in-situ DRAG pi amplitude and beta calibration."""
    amplitude: Result | None = None
    beta: Result | dict[str, float] | None = None
    for index in range(n_iterations):
        print(f"\nIteration {index + 1}/{n_iterations}")
        amplitude = calibrate_stark_drag_amplitude(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
            stark_ramptime=stark_ramptime,
            pulse_type="pi",
            n_points=n_points,
            n_rotations=1 if index == 0 else n_rotations,
            amplitude_range=amplitude_range,
            r2_threshold=r2_threshold,
            duration=duration,
            drag_coeff=drag_coeff,
            use_stored_amplitude=index > 0,
            use_stored_beta=index > 0,
            plot=plot,
            n_shots=n_shots,
            shot_interval=shot_interval,
            **deprecated_options,
        )
        if calibrate_beta:
            beta = calibrate_stark_drag_beta(
                exp,
                target=target,
                stark_amplitude=stark_amplitude,
                stark_ramptime=stark_ramptime,
                pulse_type="pi",
                beta_range=beta_range,
                n_turns=n_turns,
                duration=duration,
                degree=degree,
                plot=plot,
                n_shots=n_shots,
                shot_interval=shot_interval,
                **deprecated_options,
            )
        else:
            beta = {target: -drag_coeff / exp.ctx.qubits[target].alpha}
    return Result(data={"amplitude": amplitude, "beta": beta})


def stark_t1_sequence_under_stark(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    wait_time: int = 1000,
    stark_ramptime: float | None = None,
    plot: bool = True,
) -> PulseSchedule:
    """Build and optionally plot one in-situ T1 sequence under a Stark tone."""
    insitu_label = insitu_target(exp, target)
    x180 = exp.pulse.get_hpi_pulse(insitu_label).repeated(2)
    sequence = _stark_wrapped_schedule(
        stark_label=stark_target(exp, target),
        insitu_label=insitu_label,
        stark_amplitude=_stark_drive_amplitude(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
        ),
        stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
        insitu_sequence=PulseArray([x180, Blank(wait_time)]),
    )
    return _plot_sequence_sample(
        sequence,
        title=f"AC Stark in-situ T1 sample: {target}, wait={wait_time} ns",
        plot=plot,
    )


def stark_t2_sequence_under_stark(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    wait_time: int = 1000,
    stark_ramptime: float | None = None,
    plot: bool = True,
) -> PulseSchedule:
    """Build and optionally plot one in-situ echo T2 sequence under a Stark tone."""
    insitu_label = insitu_target(exp, target)
    x90 = exp.pulse.get_hpi_pulse(insitu_label)
    x180 = x90.shifted(np.pi / 2).repeated(2)
    half_wait = wait_time // 2
    sequence = _stark_wrapped_schedule(
        stark_label=stark_target(exp, target),
        insitu_label=insitu_label,
        stark_amplitude=_stark_drive_amplitude(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
        ),
        stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
        insitu_sequence=PulseArray(
            [x90, Blank(half_wait), x180, Blank(half_wait), x90.scaled(-1)]
        ),
    )
    return _plot_sequence_sample(
        sequence,
        title=f"AC Stark in-situ T2 echo sample: {target}, wait={wait_time} ns",
        plot=plot,
    )


def stark_ramsey_sequence_under_stark(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    wait_time: int = 1000,
    stark_ramptime: float | None = None,
    second_rotation_axis: Literal["X", "Y"] = "Y",
    plot: bool = True,
) -> PulseSchedule:
    """Build and optionally plot one in-situ Ramsey sequence under a Stark tone."""
    insitu_label = insitu_target(exp, target)
    x90 = exp.pulse.get_hpi_pulse(insitu_label)
    second_pulse = (
        x90.shifted(np.pi) if second_rotation_axis == "X" else x90.shifted(-np.pi / 2)
    )
    sequence = _stark_wrapped_schedule(
        stark_label=stark_target(exp, target),
        insitu_label=insitu_label,
        stark_amplitude=_stark_drive_amplitude(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
        ),
        stark_ramptime=_resolve_stark_ramptime(stark_ramptime),
        insitu_sequence=PulseArray([x90, Blank(wait_time), second_pulse]),
    )
    return _plot_sequence_sample(
        sequence,
        title=(
            f"AC Stark in-situ Ramsey sample: {target}, "
            f"wait={wait_time} ns, axis={second_rotation_axis}"
        ),
        plot=plot,
    )


def t1_experiment_under_stark(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    time_range: ArrayLike | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    save_image: bool = False,
    xaxis_type: Literal["linear", "log"] = "log",
    **deprecated_options: Any,
) -> ExperimentResult[T1Data]:
    """Measure in-situ T1 while the Stark tone is applied."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="t1_experiment_under_stark",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    if time_range is None:
        time_range = np.logspace(np.log10(100), np.log10(200 * 1000), 51)
    sweep_range = exp.ctx.util.discretize_time_range(np.asarray(time_range))

    def sequence(wait_time: int) -> PulseSchedule:
        return stark_t1_sequence_under_stark(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
            wait_time=wait_time,
            stark_ramptime=stark_ramptime,
            plot=False,
        )

    sweep_result = exp.measurement_service.sweep_parameter(
        sequence=sequence,
        sweep_range=sweep_range,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
        title="AC Stark in-situ T1",
        xlabel="Time (μs)",
        ylabel="Measured value",
        xaxis_type=xaxis_type,
    )

    data: dict[str, T1Data] = {}
    for qubit, sweep_data in sweep_result.data.items():
        fit_result = fitting.fit_exp_decay(
            target=qubit,
            x=sweep_data.sweep_range,
            y=0.5 * (1 - sweep_data.normalized),
            plot=plot,
            title="AC Stark in-situ T1",
            xlabel="Time (μs)",
            ylabel="Normalized signal",
            xaxis_type=xaxis_type,
            yaxis_type="linear",
        )
        if fit_result.status is not FitStatus.SUCCESS:
            continue
        data[qubit] = T1Data.new(
            sweep_data,
            t1=fit_result["tau"],
            t1_err=fit_result["tau_err"],
            r2=fit_result["r2"],
        )
        if save_image:
            viz.save_figure(fit_result.get_figure(), name=f"stark_insitu_t1_{qubit}")
    return ExperimentResult(data=data)


def t2_experiment_under_stark(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    time_range: ArrayLike | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] = "log",
    plot: bool = True,
    save_image: bool = False,
    **deprecated_options: Any,
) -> ExperimentResult[T2Data]:
    """Measure in-situ echo T2 while the Stark tone is applied."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="t2_experiment_under_stark",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    if time_range is None:
        time_range = np.logspace(np.log10(300), np.log10(200 * 1000), 51)
    sweep_range = exp.ctx.util.discretize_time_range(
        np.asarray(time_range),
        sampling_period=2 * _measurement_sampling_period(exp),
    )

    def sequence(wait_time: int) -> PulseSchedule:
        return stark_t2_sequence_under_stark(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
            wait_time=wait_time,
            stark_ramptime=stark_ramptime,
            plot=False,
        )

    sweep_result = exp.measurement_service.sweep_parameter(
        sequence=sequence,
        sweep_range=sweep_range,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
        title="AC Stark in-situ T2 echo",
        xlabel="Time (μs)",
        ylabel="Measured value",
        xaxis_type=xaxis_type,
    )

    data: dict[str, T2Data] = {}
    for qubit, sweep_data in sweep_result.data.items():
        fit_result = fitting.fit_exp_decay(
            target=qubit,
            x=sweep_data.sweep_range,
            y=0.5 * (1 + sweep_data.normalized),
            plot=plot,
            title="AC Stark in-situ T2 echo",
            xlabel="Time (μs)",
            ylabel="Normalized signal",
            xaxis_type=xaxis_type,
            yaxis_type="linear",
        )
        if fit_result.status is not FitStatus.SUCCESS:
            continue
        data[qubit] = T2Data.new(
            sweep_data,
            t2=fit_result["tau"],
            t2_err=fit_result["tau_err"],
            r2=fit_result["r2"],
        )
        if save_image:
            viz.save_figure(fit_result.get_figure(), name=f"stark_insitu_t2_{qubit}")
    return ExperimentResult(data=data)


def ramsey_experiment_under_stark(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    time_range: ArrayLike | None = None,
    detuning: float | None = None,
    second_rotation_axis: Literal["X", "Y"] = "Y",
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool = True,
    save_image: bool = False,
    **deprecated_options: Any,
) -> ExperimentResult[RamseyData]:
    """Measure the in-situ dressed control frequency under a Stark tone."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="ramsey_experiment_under_stark",
    )
    if n_shots is None:
        n_shots = CALIBRATION_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if detuning is None:
        detuning = 0.001
    if time_range is None:
        sweep_range = np.arange(0, 10001, 100)
    else:
        sweep_range = exp.ctx.util.discretize_time_range(np.asarray(time_range))

    insitu_label = insitu_target(exp, target)

    def sequence(wait_time: int) -> PulseSchedule:
        return stark_ramsey_sequence_under_stark(
            exp,
            target=target,
            stark_amplitude=stark_amplitude,
            wait_time=wait_time,
            stark_ramptime=stark_ramptime,
            second_rotation_axis=second_rotation_axis,
            plot=False,
        )

    dressed_frequency = exp.targets[insitu_label].frequency
    sweep_result = exp.measurement_service.sweep_parameter(
        sequence=sequence,
        sweep_range=sweep_range,
        frequencies={insitu_label: dressed_frequency + detuning},
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
    )

    data: dict[str, RamseyData] = {}
    for qubit, sweep_data in sweep_result.data.items():
        fit_result = fitting.fit_ramsey(
            target=qubit,
            times=sweep_data.sweep_range,
            data=sweep_data.normalized,
            amplitude_est=1.0,
            offset_est=0.0,
            plot=plot,
        )
        if fit_result.status is not FitStatus.SUCCESS:
            continue
        ramsey_freq = fit_result["f"]
        phase = fit_result["phi"]
        if second_rotation_axis == "Y":
            bare_freq = (
                dressed_frequency + detuning + ramsey_freq
                if phase > 0
                else dressed_frequency + detuning - ramsey_freq
            )
        else:
            bare_freq = dressed_frequency + detuning - ramsey_freq
        data[qubit] = RamseyData.new(
            sweep_data=sweep_data,
            t2=fit_result["tau"],
            ramsey_freq=ramsey_freq,
            bare_freq=bare_freq,
            r2=fit_result["r2"],
        )
        print("Bare frequency under stark drive:")
        print(f"  {qubit}: {bare_freq:.6f}")
        print("")
        print("Detuning frequency from dressed frequency")
        print(f"  {qubit}: {bare_freq - dressed_frequency:.6f}")
        print("")
        print("Detuning frequency from bare frequency")
        print(f"  {qubit}: {bare_freq - exp.targets[target].frequency:.6f}")
        print("")
        if save_image:
            viz.save_figure(
                fit_result.get_figure(), name=f"stark_insitu_ramsey_{qubit}"
            )
    return ExperimentResult(data=data)
