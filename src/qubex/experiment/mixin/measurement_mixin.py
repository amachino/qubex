from __future__ import annotations

import logging
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Collection, Literal, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from rich.console import Console
from tqdm import tqdm

from ...analysis import IQPlotter, fitting
from ...analysis import visualization as viz
from ...backend import Target
from ...measurement import (
    MeasureResult,
    MultipleMeasureResult,
    StateClassifier,
    StateClassifierGMM,
    StateClassifierKMeans,
)
from ...measurement.measurement import (
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
    SAMPLING_PERIOD,
)
from ...pulse import (
    Blank,
    FlatTop,
    PhaseShift,
    Pulse,
    PulseArray,
    PulseSchedule,
    RampType,
    Rect,
    Waveform,
)
from ...typing import (
    IQArray,
    ParametricPulseSchedule,
    ParametricWaveformDict,
    TargetMap,
)
from ..experiment_constants import (
    CALIBRATION_SHOTS,
    CLASSIFIER_DIR,
    DEFAULT_RABI_TIME_RANGE,
    HPI_DURATION,
    HPI_RAMPTIME,
)
from ..experiment_result import ExperimentResult, RabiData, SweepData
from ..protocol import BaseProtocol, MeasurementProtocol
from ..rabi_param import RabiParam

logger = logging.getLogger(__name__)

console = Console()


class MeasurementMixin(
    BaseProtocol,
    MeasurementProtocol,
):
    def execute(
        self,
        schedule: PulseSchedule,
        *,
        frequencies: Optional[dict[str, float]] = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        add_last_measurement: bool = False,
        add_pump_pulses: bool = False,
        enable_dsp_sum: bool | None = None,
        reset_awg_and_capunits: bool = True,
        plot: bool = False,
    ) -> MultipleMeasureResult:
        if readout_duration is None:
            readout_duration = self.readout_duration
        if readout_pre_margin is None:
            readout_pre_margin = self.readout_pre_margin
        if readout_post_margin is None:
            readout_post_margin = self.readout_post_margin

        if enable_dsp_sum is None:
            enable_dsp_sum = True if mode == "single" else False

        if reset_awg_and_capunits:
            self.device_controller.initialize_awg_and_capunits(self.box_ids)

        with self.modified_frequencies(frequencies):
            result = self.measurement.execute(
                schedule=schedule,
                mode=mode,
                shots=shots,
                interval=interval,
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                readout_ramptime=readout_ramptime,
                readout_drag_coeff=readout_drag_coeff,
                readout_ramp_type=readout_ramp_type,
                add_last_measurement=add_last_measurement,
                add_pump_pulses=add_pump_pulses,
                enable_dsp_sum=enable_dsp_sum,
                plot=plot,
            )

        if plot:
            result.plot()
        return result

    def measure(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        frequencies: Optional[dict[str, float]] = None,
        initial_states: dict[str, str] | None = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        add_pump_pulses: bool = False,
        enable_dsp_sum: bool | None = None,
        reset_awg_and_capunits: bool = True,
        plot: bool = False,
    ) -> MeasureResult:
        if readout_duration is None:
            readout_duration = self.readout_duration
        if readout_pre_margin is None:
            readout_pre_margin = self.readout_pre_margin
        if readout_post_margin is None:
            readout_post_margin = self.readout_post_margin

        waveforms: dict[str, NDArray[np.complex128]] = {}

        if enable_dsp_sum is None:
            enable_dsp_sum = True if mode == "single" else False

        if isinstance(sequence, PulseSchedule):
            if not sequence.is_valid():
                raise ValueError("Invalid pulse schedule.")

            if initial_states is not None:
                labels = list(set(sequence.labels) | set(initial_states.keys()))
                with PulseSchedule(labels) as ps:
                    for target, state in initial_states.items():
                        if target in self.qubit_labels:
                            ps.add(target, self.get_pulse_for_state(target, state))
                        else:
                            raise ValueError(f"Invalid init target: {target}")
                    ps.barrier()
                    ps.call(sequence)
                waveforms = ps.get_sampled_sequences()
            else:
                waveforms = sequence.get_sampled_sequences()
        else:
            if initial_states is not None:
                labels = list(set(sequence.keys()) | set(initial_states.keys()))
                with PulseSchedule(labels) as ps:
                    for target, state in initial_states.items():
                        if target in self.qubit_labels:
                            ps.add(target, self.get_pulse_for_state(target, state))
                        else:
                            raise ValueError(f"Invalid init target: {target}")
                    ps.barrier()
                    for target, waveform in sequence.items():
                        if isinstance(waveform, Waveform):
                            ps.add(target, waveform)
                        else:
                            ps.add(target, Pulse(waveform))
                waveforms = ps.get_sampled_sequences()
            else:
                for target, waveform in sequence.items():
                    if isinstance(waveform, Waveform):
                        waveforms[target] = waveform.values
                    else:
                        waveforms[target] = np.array(waveform, dtype=np.complex128)

        if reset_awg_and_capunits:
            self.device_controller.initialize_awg_and_capunits(self.box_ids)

        with self.modified_frequencies(frequencies):
            result = self.measurement.measure(
                waveforms=waveforms,
                mode=mode,
                shots=shots,
                interval=interval,
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                readout_ramptime=readout_ramptime,
                readout_drag_coeff=readout_drag_coeff,
                readout_ramp_type=readout_ramp_type,
                add_pump_pulses=add_pump_pulses,
                enable_dsp_sum=enable_dsp_sum,
            )
        if plot:
            result.plot()
        return result

    def measure_state(
        self,
        states: dict[
            str, Literal["0", "1", "+", "-", "+i", "-i"] | Literal["g", "e", "f"]
        ],
        *,
        mode: Literal["single", "avg"] = "single",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool = False,
        plot: bool = False,
    ) -> MeasureResult:
        targets = []

        for target, state in states.items():
            targets.append(target)
            if state == "f":
                targets.append(Target.ef_label(target))

        with PulseSchedule(targets) as ps:
            for target, state in states.items():
                if state in ["0", "1", "+", "-", "+i", "-i"]:
                    ps.add(target, self.get_pulse_for_state(target, state))  # type: ignore
                elif state == "g":
                    ps.add(target, Blank(0))
                elif state == "e":
                    ps.add(target, self.get_hpi_pulse(target).repeated(2))
                elif state == "f":
                    ps.add(target, self.get_hpi_pulse(target).repeated(2))
                    ps.barrier()
                    ef_label = Target.ef_label(target)
                    ps.add(ef_label, self.get_hpi_pulse(ef_label).repeated(2))

        return self.measure(
            sequence=ps,
            mode=mode,
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )

    def measure_idle_states(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool = False,
        plot: bool = True,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        result = self.measure_state(
            states={target: "g" for target in targets},
            mode="single",
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            add_pump_pulses=add_pump_pulses,
            plot=False,
        )
        data = {target: result.data[target].kerneled for target in targets}
        counts = {
            target: self.classifiers[target].classify(
                target,
                data[target],
                plot=plot,
            )
            for target in targets
        }

        return {
            "data": data,
            "counts": counts,
        }

    def obtain_reference_points(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int | None = None,
        interval: float | None = None,
        store_reference_points: bool = True,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if shots is None:
            shots = 10000

        result = self.measure_state(
            {target: "g" for target in targets},
            mode="avg",
            shots=shots,
            interval=interval,
        )

        iq = {
            target: complex(measure_data.kerneled)
            for target, measure_data in result.data.items()
        }
        phase = {target: float(np.angle(v)) for target, v in iq.items()}
        amplitude = {target: float(np.abs(v)) for target, v in iq.items()}

        if store_reference_points:
            self.calib_note._reference_phases.update(phase)

        return {
            "iq": iq,
            "phase": phase,
            "amplitude": amplitude,
        }

    def sweep_parameter(
        self,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        *,
        sweep_range: ArrayLike,
        repetitions: int = 1,
        frequencies: dict[str, float] | None = None,
        initial_states: dict[str, str] | None = None,
        rabi_level: Literal["ge", "ef"] = "ge",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        plot: bool = True,
        title: str = "Sweep result",
        xlabel: str = "Sweep value",
        ylabel: str = "Measured value",
        xaxis_type: Literal["linear", "log"] = "linear",
        yaxis_type: Literal["linear", "log"] = "linear",
    ) -> ExperimentResult[SweepData]:
        sweep_range = np.array(sweep_range)

        if rabi_level == "ge":
            rabi_params = self.ge_rabi_params
        elif rabi_level == "ef":
            rabi_params = self.ef_rabi_params
        else:
            raise ValueError("Invalid Rabi level.")

        if callable(sequence):
            initial_sequence = sequence(sweep_range[0])
            if isinstance(initial_sequence, PulseSchedule):
                sequences = [
                    sequence(param)
                    .repeated(repetitions)  # type: ignore
                    .get_sampled_sequences(copy=False)
                    for param in sweep_range
                ]
            elif isinstance(initial_sequence, dict):
                sequences = [
                    {
                        target: waveform.repeated(repetitions).values
                        for target, waveform in sequence(param).items()  # type: ignore
                    }
                    for param in sweep_range
                ]
        else:
            raise ValueError("Invalid sequence.")

        signals = defaultdict(list)
        plotter = IQPlotter(self.state_centers)

        # initialize awgs and capture units
        self.device_controller.initialize_awg_and_capunits(self.box_ids)

        with self.modified_frequencies(frequencies):
            for seq in sequences:
                result = self.measure(
                    seq,
                    initial_states=initial_states,
                    mode="avg",
                    shots=shots,
                    interval=interval,
                    readout_amplitudes=readout_amplitudes,
                    readout_duration=readout_duration,
                    readout_pre_margin=readout_pre_margin,
                    readout_post_margin=readout_post_margin,
                    reset_awg_and_capunits=False,
                )
                for target, data in result.data.items():
                    signals[target].append(data.kerneled)
                if plot:
                    plotter.update(signals)

        if plot:
            plotter.show()

        sweep_data = {
            target: SweepData(
                target=target,
                data=np.array(values),
                sweep_range=sweep_range,
                rabi_param=rabi_params.get(target),
                state_centers=self.state_centers.get(target),
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                xaxis_type=xaxis_type,
                yaxis_type=yaxis_type,
            )
            for target, values in signals.items()
        }
        result = ExperimentResult(data=sweep_data, rabi_params=self.rabi_params)
        return result

    def sweep_measurement(
        self,
        sequence: ParametricPulseSchedule,
        *,
        sweep_range: ArrayLike,
        frequencies: dict[str, float] | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_last_measurement: bool = True,
        plot: bool = True,
        title: str = "Sweep result",
        xlabel: str = "Sweep value",
        ylabel: str = "Measured value",
        xaxis_type: Literal["linear", "log"] = "linear",
        yaxis_type: Literal["linear", "log"] = "linear",
    ) -> ExperimentResult[SweepData]:
        # TODO: Support ParametricWaveformDict and replace the sweep_parameter method

        sweep_range = np.array(sweep_range)

        rabi_params = self.ge_rabi_params

        signals = defaultdict(list)
        plotter = IQPlotter(self.state_centers)

        # initialize awgs and capture units
        self.device_controller.initialize_awg_and_capunits(self.box_ids)

        with self.modified_frequencies(frequencies):
            for param in sweep_range:
                result = self.execute(
                    sequence(param),
                    mode="avg",
                    shots=shots,
                    interval=interval,
                    readout_amplitudes=readout_amplitudes,
                    readout_duration=readout_duration,
                    readout_pre_margin=readout_pre_margin,
                    readout_post_margin=readout_post_margin,
                    reset_awg_and_capunits=False,
                    add_last_measurement=add_last_measurement,
                )
                for target, data in result.data.items():
                    signals[target].append(data[-1].kerneled)
                if plot:
                    plotter.update(signals)

        if plot:
            plotter.show()

        sweep_data = {
            target: SweepData(
                target=target,
                data=np.array(values),
                sweep_range=sweep_range,
                rabi_param=rabi_params.get(target),
                state_centers=self.state_centers.get(target),
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                xaxis_type=xaxis_type,
                yaxis_type=yaxis_type,
            )
            for target, values in signals.items()
        }
        result = ExperimentResult(data=sweep_data, rabi_params=self.rabi_params)
        return result

    def repeat_sequence(
        self,
        sequence: TargetMap[Waveform] | PulseSchedule,
        *,
        initial_states: dict[str, str] | None = None,
        repetitions: int = 20,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
    ) -> ExperimentResult[SweepData]:
        def repeated_sequence(N: int) -> PulseSchedule:
            if isinstance(sequence, dict):
                with PulseSchedule() as ps:
                    for target, pulse in sequence.items():
                        ps.add(target, pulse.repeated(N))
            elif isinstance(sequence, PulseSchedule):
                ps = sequence.repeated(N)
            else:
                raise ValueError("Invalid sequence.")
            return ps

        result = self.sweep_parameter(
            sequence=repeated_sequence,
            sweep_range=np.arange(repetitions + 1),
            initial_states=initial_states,
            shots=shots,
            interval=interval,
            plot=plot,
            xlabel="Number of repetitions",
        )

        if plot:
            result.plot(normalize=True)

        return result

    def obtain_rabi_params(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
        ramptime: float | None = None,
        amplitudes: dict[str, float] | None = None,
        frequencies: dict[str, float] | None = None,
        is_damped: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = True,
        simultaneous: bool = False,
    ) -> ExperimentResult[RabiData]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        time_range = np.asarray(time_range)

        if ramptime is None:
            ramptime = HPI_DURATION - HPI_RAMPTIME

        if amplitudes is None:
            ampl = self.params.control_amplitude
            amplitudes = {target: ampl[target] for target in targets}

        if simultaneous:
            result = self.rabi_experiment(
                amplitudes=amplitudes,
                time_range=time_range,
                ramptime=ramptime,
                frequencies=frequencies,
                is_damped=is_damped,
                shots=shots,
                interval=interval,
                plot=plot,
                store_params=store_params,
            )
        else:
            rabi_data = {}
            rabi_params = {}
            for target in targets:
                data = self.rabi_experiment(
                    amplitudes={target: amplitudes[target]},
                    time_range=time_range,
                    ramptime=ramptime,
                    frequencies=frequencies,
                    is_damped=is_damped,
                    shots=shots,
                    interval=interval,
                    store_params=store_params,
                    plot=plot,
                ).data[target]
                rabi_data[target] = data
                rabi_params[target] = data.rabi_param
            result = ExperimentResult(
                data=rabi_data,
                rabi_params=rabi_params,
            )
        return result

    def obtain_ef_rabi_params(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
        ramptime: float | None = None,
        is_damped: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RabiData]:
        # TODO: Integrate with obtain_rabi_params

        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        time_range = np.asarray(time_range)

        if ramptime is None:
            ramptime = HPI_DURATION - HPI_RAMPTIME

        ef_labels = [Target.ef_label(target) for target in targets]
        ef_targets = [self.targets[ef] for ef in ef_labels]

        amplitudes = {
            ef.label: self.params.get_control_amplitude(ef.qubit) for ef in ef_targets
        }

        rabi_data = {}
        rabi_params = {}
        for label in ef_labels:
            data = self.ef_rabi_experiment(
                amplitudes={label: amplitudes[label]},
                time_range=time_range,
                ramptime=ramptime,
                is_damped=is_damped,
                shots=shots,
                interval=interval,
                store_params=True,
                plot=plot,
            ).data[label]
            rabi_data[label] = data
            rabi_params[label] = data.rabi_param

        result = ExperimentResult(
            data=rabi_data,
            rabi_params=rabi_params,
        )
        return result

    def rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
        ramptime: float | None = None,
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[RabiData]:
        # target labels
        targets = list(amplitudes.keys())

        # drive time range
        time_range = np.array(time_range, dtype=np.float64)

        if ramptime is None:
            ramptime = 0.0

        effective_time_range = time_range + ramptime

        # measure ground states as reference points
        reference_points = self.obtain_reference_points(targets)["iq"]

        # target frequencies
        if frequencies is None:
            frequencies = {
                target: self.targets[target].frequency for target in amplitudes
            }

        # rabi sequence with rect pulses of duration T
        def rabi_sequence(T: float) -> PulseSchedule:
            with PulseSchedule(targets) as ps:
                for target in targets:
                    ps.add(
                        target,
                        FlatTop(
                            duration=T + 2 * ramptime,
                            amplitude=amplitudes[target],
                            tau=ramptime,
                        ),
                    )
            return ps

        # detune target frequencies if necessary
        if detuning is not None:
            frequencies = {
                target: frequencies[target] + detuning for target in amplitudes
            }

        # run the Rabi experiment by sweeping the drive time
        sweep_result = self.sweep_parameter(
            sequence=rabi_sequence,
            sweep_range=time_range,
            frequencies=frequencies,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        # sweep data with the target labels
        sweep_data = sweep_result.data

        # fit the Rabi oscillation
        rabi_params = {}
        for target, data in sweep_data.items():
            fit_result = fitting.fit_rabi(
                target=data.target,
                times=effective_time_range,
                data=data.data,
                reference_point=reference_points.get(target),
                plot=plot,
                is_damped=is_damped,
            )
            if fit_result["status"] != "success":
                rabi_params[target] = RabiParam.nan(target=target)
            else:
                rabi_params[target] = RabiParam(
                    target=target,
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

        # store the Rabi parameters if necessary
        if store_params:
            self.store_rabi_params(rabi_params)

        # create the Rabi data for each target
        rabi_data = {
            target: RabiData(
                target=target,
                data=data.data,
                time_range=effective_time_range,
                rabi_param=rabi_params[target],
                state_centers=self.state_centers.get(target),
            )
            for target, data in sweep_data.items()
        }

        # create the experiment result
        result = ExperimentResult(
            data=rabi_data,
            rabi_params=rabi_params,
        )

        # return the result
        return result

    def ef_rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike,
        ramptime: float | None = None,
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool = True,
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
                target: self.targets[target].frequency for target in amplitudes
            }

        # ef rabi sequence with rect pulses of duration T
        def ef_rabi_sequence(T: int) -> PulseSchedule:
            with PulseSchedule() as ps:
                # prepare qubits to the excited state
                for ge in ge_labels:
                    ps.add(ge, self.x180(ge))
                ps.barrier()
                # apply the ef drive to induce the ef Rabi oscillation
                for ef in ef_labels:
                    ps.add(ef, Rect(duration=T, amplitude=amplitudes[ef]))
            return ps

        # detune target frequencies if necessary
        if detuning is not None:
            frequencies = {
                target: frequencies[target] + detuning for target in amplitudes
            }

        # run the Rabi experiment by sweeping the drive time
        sweep_result = self.sweep_parameter(
            sequence=ef_rabi_sequence,
            sweep_range=time_range,
            frequencies=frequencies,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        # fit the Rabi oscillation
        ef_rabi_params = {}
        ef_rabi_data = {}
        for qubit, data in sweep_result.data.items():
            ef_label = Target.ef_label(qubit)
            ge_rabi_param = self.ge_rabi_params[qubit]
            iq_e = ge_rabi_param.endpoints[1]
            fit_result = fitting.fit_rabi(
                target=qubit,
                times=effective_time_range,
                data=data.data,
                reference_point=iq_e,
                plot=plot,
                is_damped=is_damped,
            )
            if fit_result["status"] != "success":
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

        # store the Rabi parameters if necessary
        if store_params:
            self.store_rabi_params(ef_rabi_params)

        # create the experiment result
        result = ExperimentResult(
            data=ef_rabi_data,
            rabi_params=ef_rabi_params,
        )

        # return the result
        return result

    def measure_state_distribution(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_states: Literal[2, 3] = 2,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool = False,
        plot: bool = True,
    ) -> list[MeasureResult]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        states = ["g", "e", "f"][:n_states]
        result = {
            state: self.measure_state(
                {target: state for target in targets},  # type: ignore
                shots=shots,
                interval=interval,
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                add_pump_pulses=add_pump_pulses,
            )
            for state in states
        }
        for target in targets:
            data = {
                f"|{state}⟩": result[state].data[target].kerneled for state in states
            }
            if plot:
                viz.scatter_iq_data(
                    data=data,
                    title=f"State distribution : {target}",
                )
        return list(result.values())

    def build_classifier(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_states: Literal[2, 3] | None = None,
        save_classifier: bool = True,
        save_dir: Path | str | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool = False,
        simultaneous: bool = False,
        plot: bool = True,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if simultaneous:
            return self._build_classifier(
                targets=targets,
                n_states=n_states,
                save_classifier=save_classifier,
                save_dir=save_dir,
                shots=shots,
                interval=interval,
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                add_pump_pulses=add_pump_pulses,
                plot=plot,
            )
        else:
            fidelities = {}
            average_fidelities = {}
            data = {}
            classifiers = {}
            for target in targets:
                result = self._build_classifier(
                    targets=target,
                    n_states=n_states,
                    save_classifier=save_classifier,
                    save_dir=save_dir,
                    shots=shots,
                    interval=interval,
                    readout_amplitudes=readout_amplitudes,
                    readout_duration=readout_duration,
                    readout_pre_margin=readout_pre_margin,
                    readout_post_margin=readout_post_margin,
                    add_pump_pulses=add_pump_pulses,
                    plot=plot,
                )
                fidelities[target] = result["readout_fidelties"][target]
                average_fidelities[target] = result["average_readout_fidelity"][target]
                data[target] = result["data"]
                classifiers[target] = result["classifiers"][target]
            return {
                "readout_fidelties": fidelities,
                "average_readout_fidelity": average_fidelities,
                "data": data,
                "classifiers": classifiers,
            }

    def _build_classifier(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_states: Literal[2, 3] | None = None,
        save_classifier: bool = True,
        save_dir: Path | str | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool = False,
        plot: bool = True,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)
        if n_states is None:
            n_states = 2
        if shots is None:
            shots = 10000

        self.obtain_reference_points(targets)

        results = self.measure_state_distribution(
            targets=targets,
            n_states=n_states,
            shots=shots,
            interval=interval,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_duration=readout_duration,
            readout_amplitudes=readout_amplitudes,
            add_pump_pulses=add_pump_pulses,
            plot=False,
        )

        data = {
            target: {
                state: result.data[target].kerneled
                for state, result in enumerate(results)
            }
            for target in targets
        }

        classifiers: TargetMap[StateClassifier]
        if self.classifier_type == "kmeans":
            classifiers = {
                target: StateClassifierKMeans.fit(data[target]) for target in targets
            }
        elif self.classifier_type == "gmm":
            classifiers = {
                target: StateClassifierGMM.fit(
                    data[target],
                    phase=self.reference_phases[target],
                )
                for target in targets
            }
        else:
            raise ValueError("Invalid classifier type.")
        self.measurement.update_classifiers(classifiers)

        if save_classifier:
            for label, classifier in classifiers.items():
                if save_dir is not None:
                    path = Path(save_dir) / self.chip_id / f"{label}.pkl"
                else:
                    path = Path(CLASSIFIER_DIR) / self.chip_id / f"{label}.pkl"
                if not path.parent.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                classifier.save(path)

        fidelities = {}
        average_fidelities = {}
        for target in targets:
            clf = classifiers[target]
            classified = []
            for state in range(n_states):
                if plot:
                    print(f"{target} prepared as |{state}⟩:")
                result = clf.classify(
                    target,
                    data[target][state],
                    plot=plot,
                )
                classified.append(result)
            fidelity = [
                classified[state][state] / sum(classified[state].values())
                for state in range(n_states)
            ]
            fidelities[target] = fidelity
            average_fidelities[target] = np.mean(fidelity)

            if plot:
                print(f"{target}:")
                print(f"  Total shots: {shots}")
                for state in range(n_states):
                    print(
                        f"  |{state}⟩ → {classified[state]}, f_{state}: {fidelity[state] * 100:.2f}%"
                    )
                print(
                    f"  Average readout fidelity : {average_fidelities[target] * 100:.2f}%\n\n"
                )

        self.calib_note.state_params = {
            target: {
                "target": target,
                "centers": {
                    str(state): [center.real, center.imag]
                    for state, center in classifiers[target].centers.items()
                },
                "reference_phase": self.calib_note._reference_phases[target],
            }
            for target in targets
        }

        return {
            "readout_fidelties": fidelities,
            "average_readout_fidelity": average_fidelities,
            "data": data,
            "classifiers": classifiers,
        }

    def state_tomography(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        method: Literal["measure", "execute"] = "measure",
        plot: bool = False,
    ) -> dict[str, tuple[float, float, float]]:
        if isinstance(sequence, PulseSchedule):
            sequence = sequence.get_sequences()
        else:
            sequence = {
                target: (
                    Pulse(waveform) if not isinstance(waveform, Waveform) else waveform
                )
                for target, waveform in sequence.items()
            }

        x90 = x90 or self.hpi_pulse

        buffer: dict[str, list[float]] = defaultdict(list)

        qubits = set(Target.qubit_label(target) for target in sequence)
        targets = list(qubits | sequence.keys())

        if reset_awg_and_capunits:
            self.reset_awg_and_capunits()

        for basis in ["X", "Y", "Z"]:
            with PulseSchedule(targets) as ps:
                # Initialization pulses
                if initial_state is not None:
                    for qubit in qubits:
                        if qubit in initial_state:
                            init_pulse = self.get_pulse_for_state(
                                target=qubit,
                                state=initial_state[qubit],
                            )
                            ps.add(qubit, init_pulse)
                    ps.barrier()

                # Pulse sequences
                for target, waveform in sequence.items():
                    ps.add(target, waveform)
                ps.barrier()

                # Basis transformation pulses
                for qubit in qubits:
                    x90p = x90[qubit]
                    y90m = x90p.shifted(-np.pi / 2)
                    if basis == "X":
                        ps.add(qubit, y90m)
                    elif basis == "Y":
                        ps.add(qubit, x90p)

            if method == "execute":
                measure_result = self.execute(
                    ps,
                    shots=shots,
                    interval=interval,
                    reset_awg_and_capunits=False,
                    add_last_measurement=True,
                    plot=plot,
                )
                for qubit, data in measure_result.data.items():
                    rabi_param = self.rabi_params[qubit]
                    if rabi_param is None:
                        raise ValueError("Rabi parameters are not stored.")
                    values = data[-1].kerneled
                    values_rotated = values * np.exp(-1j * rabi_param.angle)
                    values_normalized = (
                        np.imag(values_rotated) - rabi_param.offset
                    ) / rabi_param.amplitude
                    buffer[qubit] += [values_normalized]
            else:
                measure_result = self.measure(
                    ps,
                    shots=shots,
                    interval=interval,
                    reset_awg_and_capunits=False,
                    plot=plot,
                )
                for qubit, data in measure_result.data.items():
                    rabi_param = self.rabi_params[qubit]
                    if rabi_param is None:
                        raise ValueError("Rabi parameters are not stored.")
                    values = data.kerneled
                    values_rotated = values * np.exp(-1j * rabi_param.angle)
                    values_normalized = (
                        np.imag(values_rotated) - rabi_param.offset
                    ) / rabi_param.amplitude
                    buffer[qubit] += [values_normalized]

        result = {
            qubit: (
                values[0],  # X
                values[1],  # Y
                values[2],  # Z
            )
            for qubit, values in buffer.items()
        }
        return result

    def state_evolution_tomography(
        self,
        *,
        sequences: (
            Sequence[PulseSchedule]
            | Sequence[TargetMap[Waveform]]
            | Sequence[TargetMap[IQArray]]
        ),
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        method: Literal["measure", "execute"] = "measure",
        plot: bool = True,
    ) -> dict[str, NDArray[np.float64]]:
        buffer: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

        if reset_awg_and_capunits:
            self.reset_awg_and_capunits()

        for sequence in tqdm(sequences):
            state_vectors = self.state_tomography(
                sequence=sequence,
                x90=x90,
                initial_state=initial_state,
                shots=shots,
                interval=interval,
                reset_awg_and_capunits=False,
                method=method,
                plot=False,
            )
            for target, state_vector in state_vectors.items():
                buffer[target].append(state_vector)

        result = {target: np.array(states) for target, states in buffer.items()}

        if plot:
            for target, states in result.items():
                print(f"State evolution : {target}")
                viz.display_bloch_sphere(states)

        return result

    def partial_waveform(self, waveform: Waveform, index: int) -> Waveform:
        """Returns a partial waveform up to the given index."""

        # If the index is 0, return an empty Pulse as the initial state.
        if index == 0:
            return Pulse([])

        elif isinstance(waveform, Pulse):
            # If the index is greater than the waveform length, return the waveform itself.
            if index >= waveform.length:
                return waveform
            # If the index is less than the waveform length, return a partial waveform.
            else:
                return Pulse(waveform.values[0 : index - 1])

        # If the waveform is a PulseArray, we need to extract the partial sequence.
        elif isinstance(waveform, PulseArray):
            offset = 0
            pulse_array = PulseArray([])

            # Iterate over the objects in the PulseArray.
            for obj in waveform.elements:
                # If the object is a PhaseShift gate, we can simply add it to the array.
                if isinstance(obj, PhaseShift):
                    pulse_array.add(obj)
                    continue
                elif isinstance(obj, Pulse):
                    # If the index become equal to the offset, we can stop iterating.
                    if index == offset:
                        break
                    # If the endpoint of obj is greater than the index, add the partial Pulse and break.
                    elif obj.length > index - offset:
                        pulse = Pulse(obj.values[0 : index - offset])
                        pulse_array.add(pulse)
                        break
                    # If the endpoint of obj is less than or equal to the index, add the whole Pulse.
                    else:
                        pulse_array.add(obj)
                        offset += obj.length
                        # NOTE: Don't break here even offset become equal to index,
                        # because we may have PhaseShift gates after the Pulse.
                else:
                    # NOTE: PulseArray should be flattened before calling this function.
                    logger.error(f"Invalid type: {type(obj)}")
            return pulse_array
        else:
            logger.error(f"Invalid type: {type(waveform)}")
            return waveform

    def pulse_tomography(
        self,
        sequence: PulseSchedule | TargetMap[Waveform] | TargetMap[IQArray],
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        n_samples: int | None = 100,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        method: Literal["measure", "execute"] = "measure",
        plot: bool = True,
    ) -> TargetMap[NDArray[np.float64]]:
        self.validate_rabi_params()

        if isinstance(sequence, PulseSchedule):
            pulses = sequence.get_sequences()
        else:
            pulses = {}
            pulse_length_set = set()
            for target, waveform in sequence.items():
                if isinstance(waveform, Waveform):
                    pulse = waveform
                elif isinstance(waveform, Sequence):
                    pulse = Pulse(waveform)
                else:
                    raise ValueError("Invalid waveform.")
                pulses[target] = pulse
                pulse_length_set.add(pulse.length)
            if len(pulse_length_set) != 1:
                raise ValueError("The lengths of the waveforms must be the same.")

        pulse_length = next(iter(pulses.values())).length

        if plot:
            if isinstance(sequence, PulseSchedule):
                sequence.plot(title="Pulse sequence")
            else:
                for target in pulses:
                    pulses[target].plot(title=f"Waveform : {target}")

        if n_samples is None or pulse_length < n_samples:
            indices = np.arange(pulse_length + 1)
        else:
            indices = np.linspace(0, pulse_length, n_samples).astype(int)

        flattened_pulses = {
            target: pulse.flattened() if isinstance(pulse, PulseArray) else pulse
            for target, pulse in pulses.items()
        }

        sequences = [
            {
                target: self.partial_waveform(pulse, i)
                for target, pulse in flattened_pulses.items()
            }
            for i in indices
        ]

        result = self.state_evolution_tomography(
            sequences=sequences,
            x90=x90,
            initial_state=initial_state,
            shots=shots,
            interval=interval,
            method=method,
            plot=plot,
        )

        if plot:
            times = indices * SAMPLING_PERIOD
            for target, states in result.items():
                viz.plot_bloch_vectors(
                    times=times,
                    bloch_vectors=states,
                    title=f"State evolution : {target}",
                )

        return result

    def measure_population(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        fit_gmm: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        if self.classifiers is None:
            raise ValueError("Classifiers are not built. Run `build_classifier` first.")

        result = self.measure(
            sequence,
            mode="single",
            shots=shots,
            interval=interval,
        )
        if fit_gmm:
            probabilities = {
                target: self.classifiers[target].estimate_weights(
                    result.data[target].kerneled
                )
                for target in result.data
            }
            standard_deviations = {
                target: np.zeros_like(probabilities[target]) for target in probabilities
            }
        else:
            probabilities = {
                target: data.probabilities for target, data in result.data.items()
            }
            standard_deviations = {
                target: data.standard_deviations for target, data in result.data.items()
            }
        return probabilities, standard_deviations

    def measure_population_dynamics(
        self,
        *,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        params_list: Sequence | NDArray,
        fit_gmm: bool = False,
        xlabel: str = "Index",
        scatter_mode: str = "lines+markers",
        show_error: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        if isinstance(params_list[0], int):
            x = params_list
        else:
            try:
                float(params_list[0])
                x = params_list
            except ValueError:
                x = np.arange(len(params_list))

        buffer_pops = defaultdict(list)
        buffer_errs = defaultdict(list)

        for params in tqdm(params_list):
            prob_dict, err_dict = self.measure_population(
                sequence=sequence(params),
                fit_gmm=fit_gmm,
                shots=shots,
                interval=interval,
            )
            for target, probs in prob_dict.items():
                buffer_pops[target].append(probs)
            for target, errors in err_dict.items():
                buffer_errs[target].append(errors)

        result_pops = {
            target: np.array(buffer_pops[target]).T for target in buffer_pops
        }
        result_errs = {
            target: np.array(buffer_errs[target]).T for target in buffer_errs
        }

        for target in result_pops:
            fig = go.Figure()
            for state, probs in enumerate(result_pops[target]):
                fig.add_scatter(
                    name=f"|{state}⟩",
                    mode=scatter_mode,
                    x=x,
                    y=probs,
                    error_y=(
                        dict(
                            type="data",
                            array=result_errs[target][state],
                            visible=True,
                            thickness=1.5,
                            width=3,
                        )
                        if show_error
                        else None
                    ),
                    marker=dict(size=5),
                )
            fig.update_layout(
                title=f"Population dynamics : {target}",
                xaxis_title=xlabel,
                yaxis_title="Probability",
                yaxis_range=[-0.1, 1.1],
            )
            fig.show()

        return result_pops, result_errs

    def mle_fit_density_matrix(
        self,
        expected_values: dict[str, float],
    ) -> NDArray:
        import cvxpy as cp

        """
        Fit a physical density matrix (Hermitian, PSD, trace=1) using
        maximum likelihood estimation (MLE) from expectation values.

        This implementation is inspired by Qiskit-Ignis's MLE tomography fitter:
        https://github.com/qiskit-community/qiskit-ignis/blob/master/qiskit/ignis/verification/tomography/fitters/base_fitter.py

        While independently implemented for the Qubex project, it follows the same
        core methodology (e.g., CVXPY-based convex optimization with physical constraints).

        Args:
            expected_values: A dictionary mapping Pauli label strings like 'XX', 'XY', etc.
                            to their corresponding measured expectation values.

        Returns:
            A 4x4 complex numpy array representing the fitted physical density matrix.
        """
        paulis = {
            "I": np.array([[1, 0], [0, 1]], dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }

        A_list: list[NDArray] = []
        b_list: list[float] = []
        for basis, val in expected_values.items():
            p1, p2 = basis[0], basis[1]
            pauli_op = np.kron(paulis[p1], paulis[p2])
            A_list.append(pauli_op.reshape(1, -1).conj())
            b_list.append(val)
        A = np.vstack(A_list)
        b = np.array(b_list)

        rho = cp.Variable((4, 4), hermitian=True)
        constraints = [rho >> 0, cp.trace(rho) == 1]
        objective = cp.Minimize(cp.sum_squares(A @ cp.vec(rho, order="F") - b))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if rho.value is None:
            raise RuntimeError("CVXPY failed to solve the MLE problem.")

        # Post-process: clip tiny negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(rho.value)
        eigvals_clipped = np.clip(eigvals, 0, None)
        rho_fixed = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.conj().T
        rho_fixed /= np.trace(rho_fixed)

        return rho_fixed

    def measure_bell_state(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        control_basis: str = "Z",
        target_basis: str = "Z",
        zx90: PulseSchedule | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        if self.state_centers is None:
            self.build_classifier(plot=False)

        pair = [control_qubit, target_qubit]

        with PulseSchedule(pair) as ps:
            # prepare |+⟩|0⟩
            ps.add(control_qubit, self.y90(control_qubit))

            # create |0⟩|0⟩ + |1⟩|1⟩
            ps.call(
                self.cnot(
                    control_qubit,
                    target_qubit,
                    zx90=zx90,
                )
            )

            # apply the control basis transformation
            if control_basis == "X":
                ps.add(control_qubit, self.y90m(control_qubit))
            elif control_basis == "Y":
                ps.add(control_qubit, self.x90(control_qubit))

            # apply the target basis transformation
            if target_basis == "X":
                ps.add(target_qubit, self.y90m(target_qubit))
            elif target_basis == "Y":
                ps.add(target_qubit, self.x90(target_qubit))

        result = self.measure(
            ps,
            mode="single",
            shots=shots,
            interval=interval,
        )

        basis_labels = result.get_basis_labels(pair)
        prob_dict_raw = result.get_probabilities(pair)
        # Ensure all basis labels are present in the raw probabilities
        prob_dict_raw = {label: prob_dict_raw.get(label, 0) for label in basis_labels}
        prob_dict_mitigated = result.get_mitigated_probabilities(pair)

        labels = [f"|{i}⟩" for i in prob_dict_raw.keys()]
        prob_arr_raw = np.array(list(prob_dict_raw.values()))
        prob_arr_mitigated = np.array(list(prob_dict_mitigated.values()))

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels,
                y=prob_arr_raw,
                name="Raw",
            )
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=prob_arr_mitigated,
                name="Mitigated",
            )
        )
        fig.update_layout(
            title=f"Bell state measurement: {control_qubit}-{target_qubit}",
            xaxis_title=f"State ({control_basis}{target_basis} basis)",
            yaxis_title="Probability",
            barmode="group",
            yaxis_range=[0, 1],
        )
        if plot:
            fig.show()

            for label, p, mp in zip(labels, prob_arr_raw, prob_arr_mitigated):
                print(f"{label} : {p:.2%} -> {mp:.2%}")

        if save_image:
            viz.save_figure_image(
                fig,
                f"bell_state_measurement_{control_qubit}-{target_qubit}",
            )

        return {
            "raw": prob_arr_raw,
            "mitigated": prob_arr_mitigated,
            "result": result,
            "figure": fig,
        }

    def bell_state_tomography(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        readout_mitigation: bool = True,
        zx90: PulseSchedule | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
        mle_fit: bool = True,
    ) -> dict:
        probabilities = {}
        for control_basis, target_basis in tqdm(
            product(["X", "Y", "Z"], repeat=2),
            desc="Measuring Bell state",
        ):
            result = self.measure_bell_state(
                control_qubit,
                target_qubit,
                control_basis=control_basis,
                target_basis=target_basis,
                zx90=zx90,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )
            basis = f"{control_basis}{target_basis}"
            if readout_mitigation:
                probabilities[basis] = result["mitigated"]
            else:
                probabilities[basis] = result["raw"]

        expected_values = {}
        paulis = {
            "I": np.array([[1, 0], [0, 1]]),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
        }
        rho = np.zeros((4, 4), dtype=np.complex128)
        for control_basis, control_pauli in paulis.items():
            for target_basis, target_pauli in paulis.items():
                basis = f"{control_basis}{target_basis}"
                # calculate the expectation values
                if basis in ["IX", "IY", "IZ"]:
                    p = probabilities[f"Z{target_basis}"]
                    e = p[0] - p[1] + p[2] - p[3]
                elif basis in ["XI", "YI", "ZI"]:
                    p = probabilities[f"{control_basis}Z"]
                    e = p[0] + p[1] - p[2] - p[3]
                elif basis == "II":
                    p = probabilities["ZZ"]
                    e = p[0] - p[1] - p[2] + p[3]
                else:
                    p = probabilities[basis]
                    e = p[0] - p[1] - p[2] + p[3]
                pauli = np.kron(control_pauli, target_pauli)
                rho += e * pauli
                expected_values[basis] = e

        if mle_fit:
            rho = self.mle_fit_density_matrix(expected_values)
        else:
            rho = rho / 4
        phi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        fidelity = np.real(phi @ rho @ phi.T.conj())

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Re", "Im"),
        )

        # Real part
        fig.add_trace(
            go.Heatmap(
                z=rho.real,
                zmin=-1,
                zmax=1,
                colorscale="RdBu_r",
            ),
            row=1,
            col=1,
        )

        # Imaginary part
        fig.add_trace(
            go.Heatmap(
                z=rho.imag,
                zmin=-1,
                zmax=1,
                colorscale="RdBu_r",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title=f"Bell state tomography: {control_qubit}-{target_qubit}",
            xaxis1=dict(
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["00", "01", "10", "11"],
                scaleanchor="y1",
                tickangle=0,
            ),
            yaxis1=dict(
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["00", "01", "10", "11"],
                scaleanchor="x1",
                autorange="reversed",
            ),
            xaxis2=dict(
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["00", "01", "10", "11"],
                scaleanchor="y2",
                tickangle=0,
            ),
            yaxis2=dict(
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["00", "01", "10", "11"],
                scaleanchor="x2",
                autorange="reversed",
            ),
            width=600,
            height=356,
            margin=dict(l=70, r=70, t=90, b=70),
        )
        if plot:
            fig.show()
            print(f"State fidelity: {fidelity * 100:.3f}%")
        if save_image:
            viz.save_figure_image(
                fig,
                f"bell_state_tomography_{control_qubit}-{target_qubit}",
                width=600,
                height=356,
            )

        return {
            "probabilities": probabilities,
            "expected_values": expected_values,
            "density_matrix": rho,
            "fidelity": fidelity,
        }
