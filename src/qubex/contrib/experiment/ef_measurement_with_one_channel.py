"""ef measurement helper functions for qubits with single control channel."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from qxpulse import Blank, FlatTop, PulseSchedule
from tqdm import tqdm

import qubex as qx
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
    "calibrate_cr_pi_pulse",
    "obtain_anharmonicity_with_cr",
]


def calibrate_cr_pi_pulse(
    ex: qx.Experiment,
    control_qubit: str,
    target_qubit: str,
    duration_range: ArrayLike,
    amplitude_range: ArrayLike | None = None,
    ramptime: float | None = None,
    n_rotations: int | None = None,
    n_iterations: int | None = None,
    ratio: float | None = None,
    r2_threshold: float | None = None,
    plot: bool | None = None,
    shots: int | None = None,
    interval: float | None = None,
) -> tuple[ExperimentResult[AmplCalibData], qx.PulseSchedule]:
    """
    Calibrate CR pi pulse by fitting duration first, then amplitude.

    Parameters
    ----------
    ex : qx.Experiment
        qx.Experiment instance.
    control_qubit : str
        Label of the control qubit (CR drive channel source).
    target_qubit : str
        Label of the target qubit being driven by the CR pulse.
    duration_range : ArrayLike
        Iterable of pulse durations (same units as device timing) to sweep
        when fitting the pi-pulse duration.
    amplitude_range : ArrayLike or None, optional
        Optional amplitude sweep range for amplitude calibration. If ``None``
        a default linear range is generated using ``n_points``.
    ramptime : float or None, optional
        Pulse ramp (tau) used when constructing shaped pulses. Defaults to
        ``PI_RAMPTIME`` if ``None``.
    n_rotations, n_iterations : int or None, optional
        Control how many rotations and iterative calibration steps to perform.
    ratio : float or None, optional
        Fractional window to narrow the amplitude sweep between iterations.
    r2_threshold : float or None, optional
        Minimum acceptable fit R²; a warning is printed when the fit quality
        is below this threshold.
    plot : bool or None, optional
        If True, enable plotting of fit results.
    shots : int or None, optional
        Number of measurement shots per point.
    interval : float or None, optional
        Measurement interval between repeats.

    Returns
    -------
    tuple
        A tuple of (ExperimentResult[data], pulse_schedule).
    """
    if n_rotations is None:
        n_rotations = 1
    if n_iterations is None:
        n_iterations = 1
    if r2_threshold is None:
        r2_threshold = 0.5
    if ratio is None:
        ratio = 0.2
    if ramptime is None:
        ramptime = PI_RAMPTIME
    if amplitude_range is None:
        amplitude_range = np.linspace(0.5, 1, 20)
    if plot is None:
        plot = True
    if shots is None:
        shots = CALIBRATION_SHOTS
    if interval is None:
        interval = DEFAULT_INTERVAL

    amplitude_range = np.asarray(amplitude_range, dtype=np.float64)

    if target_qubit not in ex.ctx.calib_note.rabi_params:
        raise ValueError(f"Rabi parameters are not stored for {target_qubit}.")

    cr_label = control_qubit + "-" + target_qubit
    control_labels = [
        target.label
        for target in ex.ctx.experiment_system.targets
        if target.is_related_to_qubits(ex.qubit_labels) and target.is_cr
    ]
    if cr_label not in control_labels:
        raise ValueError(f"CR target label `{cr_label}` is not found in the system.")

    def seq(duration: float, amplitude: float = 1.0) -> qx.PulseSchedule:
        with qx.PulseSchedule() as ps:
            ps.add(
                cr_label,
                FlatTop(
                    duration=duration,
                    amplitude=amplitude,
                    tau=ramptime,
                    type="RaisedCosine",
                ),
            )
            ps.add(target_qubit, Blank(duration=duration))
        return ps

    def calibrate_duration(duration_range: ArrayLike) -> float:
        n_per_rotation = 2
        duration_range = np.asarray(duration_range, dtype=np.float64)

        sweep_data = ex.sweep_parameter(
            sequence=lambda sweep_duration: seq(duration=sweep_duration, amplitude=1.0),
            sweep_range=duration_range,
            repetitions=n_per_rotation * n_rotations,
            shots=shots,
            interval=interval,
            plot=plot,
        ).data[target_qubit]

        fit_result = fitting.fit_ampl_calib_data(
            target=target_qubit,
            amplitude_range=duration_range,
            data=sweep_data.normalized,
            plot=plot,
            title="CR pi pulse duration calibration",
            xlabel="duration",
            ylabel="Normalized signal",
        )

        duration_r2 = fit_result["r2"]
        if duration_r2 < r2_threshold:
            print(f"Error: duration fit R² value is too low ({duration_r2:.3f})")

        return fit_result["amplitude"]

    def calibrate_amplitude(
        amplitude_range: ArrayLike, fixed_duration: float
    ) -> AmplCalibData:
        n_per_rotation = 2
        amplitude_range = np.asarray(amplitude_range, dtype=np.float64)

        sweep_data = ex.sweep_parameter(
            sequence=lambda amplitude: seq(
                duration=fixed_duration, amplitude=amplitude
            ),
            sweep_range=amplitude_range,
            repetitions=n_per_rotation * n_rotations,
            shots=shots,
            interval=interval,
            plot=plot,
        ).data[target_qubit]

        fit_result = fitting.fit_ampl_calib_data(
            target=target_qubit,
            amplitude_range=amplitude_range,
            data=sweep_data.normalized,
            plot=plot,
            title="CR pi pulse amplitude calibration",
            xlabel="amplitude",
            ylabel="Normalized signal",
        )

        r2 = fit_result["r2"]
        if r2 < r2_threshold:
            print(f"Error: R² value is too low ({r2:.3f})")

        return AmplCalibData.new(
            sweep_data=sweep_data,
            calib_value=fit_result["amplitude"],
            r2=r2,
        )

    def _update_amplitude_range(center: float, ratio: float = 0.4) -> ArrayLike:
        if ratio <= 0 or ratio >= 1:
            raise ValueError("Ratio must be between 0 and 1.")
        new_range = np.linspace(
            np.max((0, center * (1 - ratio))),
            np.min((1, center * (1 + ratio))),
            len(amplitude_range),
        )
        return new_range

    data: dict[str, AmplCalibData] = {}
    _amplitude_range = amplitude_range

    opt_duration = calibrate_duration(duration_range=duration_range)
    opt_duration_fixed = int(np.ceil(opt_duration / 2.0) * 2)

    for _ in range(n_iterations):
        data[target_qubit] = calibrate_amplitude(
            amplitude_range=_amplitude_range,
            fixed_duration=opt_duration_fixed,
        )
        _amplitude_range = _update_amplitude_range(
            center=data[target_qubit].calib_value, ratio=ratio
        )

    print("")
    print("Calibration results for CR pi pulse:")
    print(f"  duration: {opt_duration} -> {opt_duration_fixed} [ns]")
    for target, calib_data in data.items():
        print(f"  {target}: {calib_data.calib_value} [arb. units]")

    ps_result = seq(
        duration=opt_duration_fixed, amplitude=data[target_qubit].calib_value
    )

    return ExperimentResult(data=data), ps_result


def _calc_fnco_settings(
    ex: qx.Experiment,
    channel_label: str,
    drive_frequency: float,
    force_retune: bool = False,
) -> tuple[bool, dict]:
    """
    Return device settings kwargs when an FNCO retune is needed.

    Parameters
    ----------
    ex : qx.Experiment
        qx.Experiment instance.
    channel_label : str
        Channel/target label to evaluate.
    drive_frequency : float
        Desired drive frequency in GHz.
    force_retune : bool, optional
        If True, force recomputation of FNCO regardless of current state.

    Returns
    -------
    tuple
        (retune_needed, params) where `params` can be passed to backend settings.
    """
    # Normally FINE_FREQ_TOL_GHZ could be 0.250 (250 MHz) and FNCO_MAX
    # could be as high as 750 MHz, but we choose smaller values here to
    # avoid spurious signals.
    FINE_FREQ_TOL_GHZ = 0.150  # 150 MHz
    FNCO_MAX = 600_000_000  # 600 MHz

    target = ex.ctx.experiment_system.get_target(channel_label)
    port = target.channel.port

    try:
        current_lo = target.channel.lo_freq
    except ValueError:
        current_lo = None

    current_fnco = target.channel.fnco_freq
    current_cnco = target.channel.cnco_freq

    if current_fnco is None:
        raise ValueError(
            f"Current FNCO frequency for channel {channel_label} is not available."
        )

    current_diff_ghz = np.abs(drive_frequency - target.fine_frequency)
    current_fnco_ok = np.abs(current_fnco) <= FNCO_MAX

    if not force_retune and current_diff_ghz <= FINE_FREQ_TOL_GHZ and current_fnco_ok:
        retune_needed = False
        params = {
            "label": channel_label,
            "lo_freq": current_lo,
            "cnco_freq": current_cnco,
            "fnco_freq": current_fnco,
        }
        return retune_needed, params

    new_fnco, _ = MixingUtil.calc_fnco(
        f=drive_frequency * 1e9,
        ssb=port.sideband,
        lo=current_lo,
        cnco=current_cnco,
    )

    # Compare the desired FNCO magnitude with the FNCO limit (upper/lower bound)
    diff = np.abs(new_fnco) - FNCO_MAX
    if diff > FINE_FREQ_TOL_GHZ * 1e9:
        # The required FNCO is outside the allowable range and cannot be compensated by the AWG
        raise RuntimeError(
            f"{channel_label}: Calculated FNCO {new_fnco} Hz is below min {-FNCO_MAX} Hz by {diff} Hz, which exceeds tolerance {FINE_FREQ_TOL_GHZ * 1e9:.2f} Hz"
        )
    else:
        # The difference can be compensated by the AWG, so clip FNCO to the allowable range.
        new_fnco = np.clip(new_fnco, -FNCO_MAX, FNCO_MAX)

    retune_needed = True
    params = {
        "label": channel_label,
        "lo_freq": current_lo,
        "cnco_freq": current_cnco,
        "fnco_freq": int(new_fnco),
    }
    return retune_needed, params


def _ef_rabi_experiment(
    ex: qx.Experiment,
    target_qubit: str,
    cr_x180: PulseSchedule,
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
    ex : qx.Experiment
        qx.Experiment instance.
    target_qubit : str
        Target qubit label for the EF drive.
    cr_x180 : PulseSchedule
        Pulse schedule performing a CR X180; this is invoked before the EF
        drive in each sequence.
    time_range : ArrayLike
        Array-like durations to sweep for the EF drive. Units match the
        device timing (usually ns).
    ef_amplitude : float or None, optional
        EF drive amplitude. If ``None``, retrieved from ``ex.params``.
    ef_ramptime : float or None, optional
        Ramp time for the EF drive pulses.
    is_damped : bool or None, optional
        If True, use a damped Rabi model when fitting.
    n_shots : int or None, optional
        Shots per point.
    shot_interval : float or None, optional
        Interval between shots.
    plot : bool or None, optional
        Enable plotting of fit diagnostics.
    **deprecated_options : Any
        Additional deprecated keyword options forwarded to the sweep call.

    Returns
    -------
    ExperimentResult
        Result object containing fitted `RabiData` and parameters.
    """
    if is_damped is None:
        is_damped = True
    if ef_amplitude is None:
        ef_amplitude = ex.params.get_ef_control_amplitude(target_qubit)
    if ef_ramptime is None:
        ef_ramptime = 0.0

    n_shots, shot_interval, deprecated_options = (
        ex.measurement_service.resolve_shot_options(
            n_shots=n_shots,
            shot_interval=shot_interval,
            deprecated_options=deprecated_options,
            n_shots_default=DEFAULT_SHOTS,
            shot_interval_default=DEFAULT_INTERVAL,
        )
    )
    if plot is None:
        plot = True

    time_range = np.array(time_range, dtype=np.float64)

    effective_time_range = time_range + ef_ramptime
    control_sampling_period = ex.measurement_service.ctx.measurement.sampling_period

    # ef rabi sequence with rect pulses of duration T
    def ef_rabi_sequence(T: int) -> PulseSchedule:
        with PulseSchedule() as ps:
            ps.call(cr_x180)
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


def obtain_anharmonicity_with_cr(
    ex: qx.Experiment,
    target_qubit: str,
    cr_x180: PulseSchedule,
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
) -> tuple[ExperimentResult[FreqRabiData], float]:
    """
    Measure EF chevron (frequency vs Rabi rate) and estimate anharmonicity.

    Parameters
    ----------
    ex : qx.Experiment
        Experiment instance providing measurement and system management.
    target_qubit : str
        Target qubit label to probe.
    cr_x180 : PulseSchedule
        CR X180 schedule used to prepare the excited state prior to EF drive.
    time_range : ArrayLike
        Time sweep for each EF Rabi run.
    ef_frequency : float or None, optional
        Central EF drive frequency in GHz. If ``None``, computed from the
        qubit frequency plus its known anharmonicity.
    ef_amplitude, ef_ramptime : float or None, optional
        EF drive amplitude and ramp time.
    detuning_range : ArrayLike or None, optional
        Array-like detunings (in GHz) to sweep around ``ef_frequency``. A
        default symmetric range is used when ``None``.
    is_damped, n_shots, shot_interval, plot : optional
        Measurement and fitting options forwarded to the underlying Rabi
        experiment.

    Returns
    -------
    tuple
        ``(result, anharmonicity)`` where ``result`` is an
        :class:`ExperimentResult` containing a :class:`FreqRabiData` entry for
        ``target_qubit``, and ``anharmonicity`` is a float (GHz) estimated
        from the fitted resonance frequency minus the qubit fundamental
        frequency.
    """
    if ef_frequency is None:
        ef_frequency = (
            ex.qubits[target_qubit].frequency + ex.qubits[target_qubit].anharmonicity
        )
    if detuning_range is None:
        detuning_range = np.linspace(-0.01, 0.01, 21)
    if plot is None:
        plot = True

    detuning_range = np.asarray(detuning_range, dtype=np.float64)
    rabi_data: list[RabiData] = []
    rabi_rates: list[float] = []

    with ExitStack() as stack:
        retune_needed, backend_settings = _calc_fnco_settings(
            ex, channel_label=target_qubit, drive_frequency=ef_frequency
        )
        if retune_needed:
            stack.enter_context(
                ex.system_manager.modified_backend_settings(**backend_settings)
            )

        for detuning in tqdm(detuning_range):
            with ex.modified_frequencies(
                {
                    target_qubit: ef_frequency + detuning,
                }
            ):
                rabi_result = _ef_rabi_experiment(
                    ex=ex,
                    target_qubit=target_qubit,
                    cr_x180=cr_x180,
                    time_range=time_range,
                    ef_amplitude=ef_amplitude,
                    ef_ramptime=ef_ramptime,
                    is_damped=is_damped,
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                    **deprecated_options,
                )

            ef_label = ex.measurement_service.ctx.resolve_ef_label(target_qubit)
            if rabi_result.rabi_params is None:
                raise ValueError("Rabi parameters are not stored.")
            rabi_params = rabi_result.rabi_params.get(ef_label, None)
            rabi_datum = rabi_result.data.get(ef_label, None)
            if rabi_params is None:
                raise ValueError("Rabi parameters are not stored.")
            else:
                rabi_rates.append(rabi_params.frequency)
            if rabi_datum is None:
                raise ValueError("Rabi data are not stored.")
            else:
                rabi_data.append(rabi_datum)
    detuning_range = np.asarray(detuning_range, dtype=np.float64)
    frequency_range = detuning_range + ef_frequency

    data = FreqRabiData(
        target=target_qubit,
        data=np.array(rabi_rates, dtype=np.float64),
        sweep_range=detuning_range,
        frequency_range=frequency_range,
        rabi_data=rabi_data,
    )
    result = ExperimentResult(data={target_qubit: data})
    fit_result = data.fit(plot=plot)
    if fit_result.status is FitStatus.SUCCESS:
        ef_frequency = fit_result.data.get("f_resonance", None)
        if ef_frequency is None:
            raise ValueError("Resonance frequency is not available in fit result.")
        anharmonicity = ef_frequency - ex.qubits[target_qubit].frequency
        print(f"Estimated EF resonance frequency: {ef_frequency:.6f} GHz")
        print(f"Estimated anharmonicity: {anharmonicity:.6f} GHz")
        return result, anharmonicity
    else:
        raise RuntimeError(
            "Failed to fit EF chevron pattern, cannot estimate resonance frequency and anharmonicity."
        )
