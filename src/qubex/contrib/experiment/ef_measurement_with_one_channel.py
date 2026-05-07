from __future__ import annotations

from contextlib import ExitStack
from collections.abc import Collection
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from plotly import graph_objects as go
from tqdm import tqdm

import qubex as qx
from qxpulse import Blank, FlatTop, PulseSchedule
from qubex import visualization as viz
from qubex.analysis import FitStatus, fitting
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
    PI_RAMPTIME,
)
from qubex.experiment.models.experiment_result import (
    AmplCalibData,
    ExperimentResult,
    FreqRabiData,
    RabiData,
)
from qubex.experiment.models.rabi_param import RabiParam
from qubex.system import MixingUtil


# Public API exported from this module (contrib-style utilities)
__all__ = [
    'calibrate_cr_pi_pulse',
    'ef_rabi_experiment',
    'ef_chevron_pattern',
]


def calibrate_cr_pi_pulse(
        ex: qx.Experiment,
        control_qubit: str,
        target_qubit: str,
        duration_range: Collection[float],
        amplitude_range: Collection[float] | None = None,
        ramptime: float | None = None,
        n_points: int | None = None,
        n_rotations: int | None = None,
        n_iterations: int | None = None,
        ratio: float | None = None,
        r2_threshold: float | None = None,
        plot: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> tuple[ExperimentResult[AmplCalibData], int, qx.PulseSchedule]:
    """
    Calibrate CR pi pulse by fitting duration first, then amplitude.

    Parameters
    ----------
    ex
        Experiment context providing measurement and device access.
    control_qubit, target_qubit
        Qubit labels used for control and target channels.
    duration_range
        Sequence of durations to sweep when fitting the pi pulse duration.
    amplitude_range
        Optional amplitude sweep range for amplitude calibration.
    ramptime
        Pulse ramp time.
    n_points
        Number of points used when generating an amplitude range if not provided.
    n_rotations, n_iterations, ratio, r2_threshold
        Calibration control parameters.
    update_params
        If True, update experiment calibration parameters (no-op here).
    plot
        If True, plotting is enabled.
    shots, interval
        Measurement shot options.

    Returns
    -------
    tuple
        A tuple of (ExperimentResult[data], fixed_duration, pulse_schedule).

    Notes
    -----
    This function preserves argument list and behavior; only docstring and
    return type annotation were adjusted to match contrib guidelines.
    """
    if n_points is None:
        if amplitude_range is not None:
            n_points = len(amplitude_range)
        else:
            n_points = 20
    if n_rotations is None:
        n_rotations = 1
    if n_iterations is None:
        n_iterations = 2
    if r2_threshold is None:
        r2_threshold = 0.5
    if ratio is None:
        ratio = 0.2
    if ramptime is None:
        ramptime = PI_RAMPTIME
    if amplitude_range is None:
        amplitude_range = np.linspace(0.9, 1, n_points)
    if plot is None:
        plot = True
    if shots is None:
        shots = CALIBRATION_SHOTS
    if interval is None:
        interval = DEFAULT_INTERVAL

    if target_qubit not in ex.ctx.calib_note.rabi_params:
        raise ValueError(f"Rabi parameters are not stored for {target_qubit}.")

    def seq(duration: float, amplitude: float = 1.0) -> qx.PulseSchedule:
        with qx.PulseSchedule() as ps:
            ps.add(
                control_qubit,
                FlatTop(
                    duration=duration,
                    amplitude=amplitude,
                    tau=ramptime,
                    type='RaisedCosine',
                ),
            )
            ps.add(
                target_qubit,
                Blank(duration=duration)
            )
        return ps

    def _plot_raw_calibration(
        x_values: Collection[float],
        y_values: Collection[float],
        title: str,
        xlabel: str,
    ) -> None:
        fig = viz.make_figure()
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=-y_values,
                mode="markers",
                name="Data",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title='Normalized signal',
            xaxis_type='linear',
            yaxis_type='linear',
        )
        fig.show()

    def calibrate_duration(duration_range: Collection[float]) -> float:
        n_per_rotation = 2

        sweep_data = ex.sweep_parameter(
            sequence=lambda sweep_duration: seq(duration=sweep_duration, amplitude=1.0),
            sweep_range=duration_range,
            repetitions=n_per_rotation * n_rotations,
            shots=shots,
            interval=interval,
            plot=True,
        ).data[target_qubit]

        _plot_raw_calibration(
            x_values=duration_range,
            y_values=sweep_data.normalized,
            title=f"CR pi pulse duration calibration : {target_qubit}",
            xlabel='duration',
        )

        fit_result = fitting.fit_ampl_calib_data(
            target=target_qubit,
            amplitude_range=duration_range,
            data=sweep_data.normalized,
            plot=plot,
            title='CR pi pulse duration calibration',
            xlabel='duration',
            ylabel='Normalized signal',
        )

        duration_r2 = fit_result['r2']
        if duration_r2 < r2_threshold:
            print(f"Error: duration fit R² value is too low ({duration_r2:.3f})")

        return fit_result['amplitude']

    def calibrate_amplitude(amplitude_range: Collection[float], fixed_duration: float) -> AmplCalibData:
        n_per_rotation = 2

        sweep_data = ex.sweep_parameter(
            sequence=lambda amplitude: seq(duration=fixed_duration, amplitude=amplitude),
            sweep_range=amplitude_range,
            repetitions=n_per_rotation * n_rotations,
            shots=shots,
            interval=interval,
            plot=True,
        ).data[target_qubit]

        _plot_raw_calibration(
            x_values=amplitude_range,
            y_values=sweep_data.normalized,
            title=f"CR pi pulse amplitude calibration : {target_qubit}",
            xlabel='amplitude',
        )

        fit_result = fitting.fit_ampl_calib_data(
            target=target_qubit,
            amplitude_range=amplitude_range,
            data=sweep_data.normalized,
            plot=plot,
            title='CR pi pulse amplitude calibration',
            xlabel='amplitude',
            ylabel='Normalized signal',
        )

        r2 = fit_result['r2']
        if r2 < r2_threshold:
            print(f"Error: R² value is too low ({r2:.3f})")

        return AmplCalibData.new(
            sweep_data=sweep_data,
            calib_value=fit_result['amplitude'],
            r2=r2,
        )

    def _update_amplitude_range(center: float, ratio: float = 0.2) -> Collection[float]:
        if ratio <= 0 or ratio >= 1:
            raise ValueError("Ratio must be between 0 and 1.")
        new_range = np.linspace(
            np.max((0, center * (1 - ratio))),
            np.min((1, center * (1 + ratio))),
            n_points,
        )
        return new_range

    data: dict[str, AmplCalibData] = {}
    _amplitude_range = amplitude_range
    # print(f"Calibrating CR pi pulse for {target_qubit}...")

    opt_duration = calibrate_duration(duration_range=duration_range)
    opt_duration_fixed = int(np.ceil(opt_duration / 2.0) * 2)
    # print(f"Optimal duration (fit): {opt_duration}")
    # print(f"Duration used for amplitude sweep (rounded up to even): {opt_duration_fixed}")

    for i in range(n_iterations):
        # print(f"Amplitude iteration {i+1}/{n_iterations}")
        data[target_qubit] = calibrate_amplitude(
            amplitude_range=_amplitude_range,
            fixed_duration=opt_duration_fixed,
        )
        _amplitude_range = _update_amplitude_range(center=data[target_qubit].calib_value, ratio=ratio)

    print("")
    print(f"Calibration results for CR pi pulse:")
    print(f"  duration: {opt_duration} -> {opt_duration_fixed} [ns]")
    for target, calib_data in data.items():
        print(f"  {target}: {calib_data.calib_value} [arb. units]")

    ps_result = seq(duration=opt_duration_fixed, amplitude=data[target_qubit].calib_value)

    return ExperimentResult(data=data), opt_duration_fixed, ps_result


def _calc_fnco_settings(
        ex: qx.Experiment,
        channel_label: str,
        drive_frequency: float,
        force_retune: bool = False,
        print_info: bool = False,
    )-> tuple[bool, dict[str, float]]:
    """
    Return device settings kwargs when an FNCO retune is needed.

    Parameters
    ----------
    ex
        Experiment providing system and channel information.
    channel_label
        Channel label to inspect and (optionally) retune.
    drive_frequency
        Desired drive frequency in GHz.
    force_retune
        If True, force recalculation of FNCO even if current settings are within
        tolerance.
    print_info
        If True, print informational messages about decisions.

    Returns
    -------
    tuple
        (retune_needed, params) where `params` can be passed to backend settings.
    """

    FINE_FREQ_TOL_GHZ = 0.150 # 150 MHz
    FNCO_MAX = 600_000_000 # 600 MHz

    target = ex.ctx.experiment_system.get_target(channel_label)
    port = target.channel.port

    try:
        current_lo = target.channel.lo_freq
    except ValueError:
        current_lo = None

    current_fnco = target.channel.fnco_freq
    current_cnco = target.channel.cnco_freq

    if current_lo is not None:
        current_diff_ghz = abs(drive_frequency - (current_lo - current_cnco - current_fnco) * 1e-9)
        if print_info:
            print(f"current lo: {current_lo*1e-9} GHz, current cnco: {current_cnco*1e-9} GHz, fnco: {current_fnco*1e-9} GHz")

    else: 
        current_diff_ghz = abs(drive_frequency - (current_cnco + current_fnco) * 1e-9)
        if print_info:
            print(f"current cnco: {current_cnco*1e-9} GHz, fnco: {current_fnco*1e-9} GHz")

    current_fnco_ok = abs(current_fnco) <= FNCO_MAX
    if not force_retune and current_diff_ghz <= FINE_FREQ_TOL_GHZ and current_fnco_ok:
        if print_info:
            print("No FNCO retune needed")
        retune_needed = False
        params = {
            'label': channel_label,
            'lo_freq': current_lo,
            'cnco_freq': current_cnco,
            'fnco_freq': current_fnco,
        }
        return retune_needed, params

    new_fnco, _ = MixingUtil.calc_fnco(
        f=drive_frequency * 1e9,
        ssb=port.sideband,
        lo=current_lo,
        cnco=current_cnco,
    )

    new_fnco = int(np.min((new_fnco, FNCO_MAX)))
    new_fnco = int(np.max((-FNCO_MAX, new_fnco)))

    # Residual fine-frequency error after applying the rounded FNCO
    if current_lo is not None:
        diff_ghz = abs(drive_frequency - (current_lo - current_cnco - new_fnco) * 1e-9)
    else:
        diff_ghz = abs(drive_frequency - (current_cnco + new_fnco) * 1e-9)
    if diff_ghz > FINE_FREQ_TOL_GHZ:
        raise RuntimeError(
            f"{channel_label}: No feasible FNCO within tolerance: residual delta {diff_ghz:.6f} GHz > {FINE_FREQ_TOL_GHZ} GHz"
        )

    if print_info:
        print(f"new fnco: {new_fnco*1e-9} GHz, projected fine frequency delta: {diff_ghz:.6f} GHz")

    retune_needed = True
    params = {
        'label': channel_label,
        'lo_freq': current_lo,
        'cnco_freq': current_cnco,
        'fnco_freq': new_fnco,
    }
    return retune_needed, params


def ef_rabi_experiment(
        ex: qx.Experiment,
        target_qubit: str,
        control_qubit: str,
        cr_amplitude: float,
        cr_duration: int,
        cr_ramptime: float,
        time_range: ArrayLike,
        ef_amplitude: float | None = None,
        ef_ramptime: float | None = None,
        is_damped: bool | None = None,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        plot: bool | None = None,
        **deprecated_options: Any,
    ) -> ExperimentResult[RabiData]:
    """
    Run an EF Rabi experiment and fit parameters.

    Parameters
    ----------
    ex
        Experiment context with measurement and pulse configuration.
    target_qubit, control_qubit
        Qubit labels for the experiment.
    cr_amplitude, cr_duration, cr_ramptime
        Parameters for the CR pi pulse used to prepare the `e` state.
    time_range
        Durations swept for the EF drive.
    ef_amplitude, ef_ramptime
        EF drive amplitude and ramp time. Defaults are taken from `ex` when
        not provided.
    is_damped
        Whether to use a damped Rabi fit.
    n_shots, shot_interval
        Measurement shot options.
    plot
        If True, plotting is enabled.

    Returns
    -------
    ExperimentResult
        Result object containing fitted `RabiData` and parameters.
    """
    # TODO: Integrate with rabi_experiment
    if is_damped is None:
        is_damped = True
    if ef_amplitude is None:
        ef_amplitude = ex.params.get_ef_control_amplitude(target_qubit)
        # print(f"Using default ef_amplitude {ef_amplitude} for {target_qubit}")
    if ef_ramptime is None:
        ef_ramptime = 0.0
    
    n_shots, shot_interval, deprecated_options = ex.measurement_service.resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        n_shots_default=DEFAULT_SHOTS,
        shot_interval_default=DEFAULT_INTERVAL,
    )
    if plot is None:
        plot = True
        
    time_range = np.array(time_range, dtype=np.float64)

    effective_time_range = time_range + ef_ramptime
    control_sampling_period = ex.measurement_service.ctx.measurement.sampling_period


    def cr_pi_pulse() -> qx.PulseSchedule:
        with qx.PulseSchedule() as ps:
            ps.add(
                control_qubit,
                FlatTop(
                    duration=cr_duration,
                    amplitude=cr_amplitude,
                    tau=cr_ramptime,
                    type='RaisedCosine',
                ),
            )
            ps.add(
                target_qubit,
                Blank(duration=cr_duration)
            )
        return ps


    # ef rabi sequence with rect pulses of duration T
    def ef_rabi_sequence(T: int) -> PulseSchedule:
        with PulseSchedule() as ps:
            ps.call(cr_pi_pulse())
            ps.barrier()
            # apply the ef drive to induce the ef Rabi oscillation
            ps.add(
                target_qubit,
                FlatTop(
                    duration=T + 2 * ef_ramptime,
                    amplitude=ef_amplitude,
                    tau=ef_ramptime,
                    sampling_period=control_sampling_period,
                ),
            )
        return ps

    # run the Rabi experiment by sweeping the drive time
    sweep_result = ex.measurement_service.sweep_parameter(
        sequence=ef_rabi_sequence,
        sweep_range=time_range,
        n_shots=n_shots,
        shot_interval=shot_interval,
        plot=plot,
        **deprecated_options,
    )

    # fit the Rabi oscillation
    ef_rabi_params = {}
    ef_rabi_data = {}
    data = sweep_result.data[target_qubit]
    ef_label = ex.measurement_service.ctx.resolve_ef_label(target_qubit)
    ge_rabi_param = ex.measurement_service.pulse.ge_rabi_params[target_qubit]
    iq_e = ge_rabi_param.endpoints[1]
    # print(f"Fitting EF Rabi for {target_qubit} with iq_e={iq_e}, is_damped={is_damped}")
    fit_result = fitting.fit_rabi(
        target=target_qubit,
        times=effective_time_range,
        data=data.data,
        reference_point=iq_e,
        plot=plot,
        is_damped=is_damped,
    )
    if fit_result.status is not FitStatus.SUCCESS:
        ef_rabi_params[ef_label] = RabiParam.nan(target=ef_label)
    else:
        ef_rabi_params[ef_label] = RabiParam(
            target=ef_label,
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
    ef_rabi_data[ef_label] = RabiData(
        target=ef_label,
        data=data.data,
        time_range=effective_time_range,
        rabi_param=ef_rabi_params[ef_label],
    )

    # create the experiment result
    result = ExperimentResult(
        data=ef_rabi_data,
        rabi_params=ef_rabi_params,
    )

    return result


def ef_chevron_pattern(
        ex: qx.Experiment,
        target_qubit: str,
        control_qubit: str,
        cr_amplitude: float,
        cr_duration: int,
        cr_ramptime: float,
        time_range: ArrayLike,
        ef_frequency: float | None = None,
        ef_amplitude: float | None = None,
        ef_ramptime: float | None = None,
        detuning_range: ArrayLike | None = None,
        is_damped: bool | None = None,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        plot: bool | None = None,
        **deprecated_options: Any,
    ) -> ExperimentResult[FreqRabiData]:
    """
    Measure an EF chevron (frequency-versus-Rabi) pattern by sweeping detuning.

    Parameters
    ----------
    ex
        Experiment context used to run measurements and manage backend settings.
    target_qubit, control_qubit
        Qubit labels.
    cr_amplitude, cr_duration, cr_ramptime
        CR pi pulse parameters used to prepare the `e` state before EF drive.
    time_range
        Time sweep used for each Rabi experiment.
    ef_frequency, detuning_range
        Center EF drive frequency and detuning sweep (in GHz).
    ef_amplitude, ef_ramptime, is_damped, n_shots, shot_interval, plot
        Measurement control options.

    Returns
    -------
    ExperimentResult
        Container with `FreqRabiData` over the detuning sweep.
    """
    if ef_frequency is None:
        ef_frequency = ex.qubits[target_qubit].frequency + ex.qubits[target_qubit].anharmonicity
    if detuning_range is None:
        detuning_range = np.linspace(-0.01, 0.01, 21)

    rabi_data: list[RabiData] = []
    rabi_rates: list[float] = []

    with ExitStack() as stack:
        retune_needed, backend_settings = _calc_fnco_settings(ex, channel_label=target_qubit, drive_frequency=ef_frequency)
        if retune_needed:
            stack.enter_context(
                ex.system_manager.modified_backend_settings(
                    **backend_settings
                )
            )
        
        # ex.obtain_rabi_params()
        for detuning in tqdm(detuning_range):
            with ex.modified_frequencies({
                    target_qubit: ef_frequency+detuning,
                }):
                rabi_result = ef_rabi_experiment(
                    ex=ex,
                    target_qubit=target_qubit,
                    control_qubit=control_qubit,
                    cr_amplitude=cr_amplitude,
                    cr_duration=cr_duration,
                    cr_ramptime=cr_ramptime,
                    time_range=time_range,
                    ef_amplitude=ef_amplitude,
                    ef_ramptime=ef_ramptime,
                    is_damped=is_damped,
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=plot,
                    **deprecated_options,
                )
            
            ef_label = ex.measurement_service.ctx.resolve_ef_label(target_qubit)
            rabi_params = rabi_result.rabi_params.get(ef_label, None)
            rabi_datum  = rabi_result.data.get(ef_label, None)
            if rabi_params is None:
                raise ValueError("Rabi parameters are not stored.")
                # print("Rabi parameters are not stored.")
                # rabi_rates.append(None)
            else:
                rabi_rates.append(rabi_params.frequency)
            if rabi_datum is None:
                raise ValueError("Rabi data are not stored.")
                # print("Rabi data are not stored.")
                # rabi_data.append(None)
            else:
                rabi_data.append(rabi_datum)

    frequency_range = detuning_range + ef_frequency

    data = {
        target_qubit: FreqRabiData(
            target=target_qubit,
            data=np.array(rabi_rates, dtype=np.float64),
            sweep_range=detuning_range,
            frequency_range=frequency_range,
            rabi_data=rabi_data,
        )
    }
    result = ExperimentResult(data=data)
    if plot:
        result.fit()
    return result
