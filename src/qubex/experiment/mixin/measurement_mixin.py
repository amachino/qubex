from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Collection, Literal, Optional, Sequence

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from rich.console import Console
from tqdm import tqdm

from qubex.experiment.library.graph import (
    find_longest_1d_chain,
    get_max_undirected_weight,
    strong_edge_coloring,
    tree_center,
)

from ...analysis import IQPlotter, fitting
from ...analysis import visualization as viz
from ...analysis.state_tomography import (
    create_density_matrix,
    mle_fit_density_matrix,
    plot_ghz_state_tomography,
)
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
    CPMG,
    Blank,
    FlatTop,
    PhaseShift,
    Pulse,
    PulseArray,
    PulseSchedule,
    RampType,
    VirtualZ,
    Waveform,
)
from ...style import COLORS
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
from ..result import Result

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
        enable_dsp_demodulation: bool = True,
        enable_dsp_sum: bool | None = None,
        enable_dsp_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
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
            qubits = {Target.qubit_label(target) for target in schedule.labels}
            self.reset_awg_and_capunits(qubits=qubits)

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
                enable_dsp_demodulation=enable_dsp_demodulation,
                enable_dsp_sum=enable_dsp_sum,
                enable_dsp_classification=enable_dsp_classification,
                line_param0=line_param0,
                line_param1=line_param1,
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
        enable_dsp_demodulation: bool = True,
        enable_dsp_sum: bool | None = None,
        enable_dsp_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
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
            qubits = {Target.qubit_label(target) for target in waveforms}
            self.reset_awg_and_capunits(qubits=qubits)

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
                enable_dsp_demodulation=enable_dsp_demodulation,
                enable_dsp_sum=enable_dsp_sum,
                enable_dsp_classification=enable_dsp_classification,
                line_param0=line_param0,
                line_param1=line_param1,
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
    ) -> Result:
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

        return Result(
            data={
                "data": data,
                "counts": counts,
            }
        )

    def obtain_reference_points(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int | None = None,
        interval: float | None = None,
        store_reference_points: bool = True,
    ) -> Result:
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

        return Result(
            data={
                "iq": iq,
                "phase": phase,
                "amplitude": amplitude,
            }
        )

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
        enable_tqdm: bool = False,
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
                qubits = {
                    Target.qubit_label(target) for target in initial_sequence.labels
                }
            elif isinstance(initial_sequence, dict):
                sequences = [
                    {
                        target: waveform.repeated(repetitions).values
                        for target, waveform in sequence(param).items()  # type: ignore
                    }
                    for param in sweep_range
                ]
                qubits = {Target.qubit_label(target) for target in initial_sequence}
        else:
            raise ValueError("Invalid sequence.")

        signals = defaultdict(list)
        plotter = IQPlotter(
            {
                qubit: self.state_centers[qubit]
                for qubit in qubits
                if qubit in self.state_centers
            }
        )

        # initialize awgs and capture units
        self.reset_awg_and_capunits(qubits=qubits)

        with self.modified_frequencies(frequencies):
            for seq in tqdm(
                sequences,
                desc="Sweeping parameters",
                disable=not enable_tqdm,
            ):
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
        initial_sequence = sequence(sweep_range[0])
        qubits = {Target.qubit_label(target) for target in initial_sequence.labels}
        self.reset_awg_and_capunits(qubits=qubits)

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
        fit_threshold: float = 0.5,
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
            ramptime = HPI_DURATION - HPI_RAMPTIME  # π/2

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
                fit_threshold=fit_threshold,
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
                    fit_threshold=fit_threshold,
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
            ef.label: self.params.get_ef_control_amplitude(ef.qubit)
            for ef in ef_targets
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
        fit_threshold: float = 0.5,
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
            if fit_result["status"] == "error" or fit_result["r2"] < fit_threshold:
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
                    ps.add(
                        ef,
                        FlatTop(
                            duration=T + 2 * ramptime,
                            amplitude=amplitudes[ef],
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
    ) -> Result:
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
                fidelities[target] = result["readout_fidelities"][target]
                average_fidelities[target] = result["average_readout_fidelity"][target]
                data[target] = result["data"]
                classifiers[target] = result["classifiers"][target]

            return Result(
                data={
                    "readout_fidelities": fidelities,
                    "average_readout_fidelity": average_fidelities,
                    "data": data,
                    "classifiers": classifiers,
                }
            )

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
    ) -> Result:
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

        return Result(
            data={
                "readout_fidelities": fidelities,
                "average_readout_fidelity": average_fidelities,
                "data": data,
                "classifiers": classifiers,
            }
        )

    def measure_1q_state_fidelity(
        self,
        target: str,
        *,
        target_state: str = "+",
        waveform: Waveform | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        use_zvalues: bool = False,
        plot: bool = False,
    ) -> Result:
        if waveform is None:
            measure_result = self.state_tomography(
                sequence={target: []},
                initial_state={target: target_state},
                shots=shots,
                interval=interval,
                reset_awg_and_capunits=reset_awg_and_capunits,
                use_zvalues=use_zvalues,
                plot=plot,
            )
        else:
            measure_result = self.state_tomography(
                sequence={target: waveform},
                shots=shots,
                interval=interval,
                reset_awg_and_capunits=reset_awg_and_capunits,
                use_zvalues=use_zvalues,
                plot=plot,
            )

        state_vector = measure_result.get(target)
        if state_vector is None:
            raise ValueError(f"No measurement data for target {target}.")

        if target_state == "+":
            target_state_vector = np.array([1.0, 0.0, 0.0])
        elif target_state == "-":
            target_state_vector = np.array([-1.0, 0.0, 0.0])
        elif target_state == "+i":
            target_state_vector = np.array([0.0, 1.0, 0.0])
        elif target_state == "-i":
            target_state_vector = np.array([0.0, -1.0, 0.0])
        elif target_state == "0":
            target_state_vector = np.array([0.0, 0.0, 1.0])
        elif target_state == "1":
            target_state_vector = np.array([0.0, 0.0, -1.0])
        else:
            raise ValueError(f"Invalid target state: {target_state}")

        fidelity = np.abs(np.dot(state_vector, target_state_vector)) ** 2

        print(f"{target}: |{target_state}〉")
        print(f"  Fidelity: {fidelity:.4f}")
        print(f"  Absolute infidelity: {np.abs(1 - fidelity):.4f}")
        print(
            f"  State vector: ({state_vector[0]:.4f}, {state_vector[1]:.4f}, {state_vector[2]:.4f})"
        )
        print(f"  Target state vector: {target_state_vector}")

        return Result(
            data={
                "fidelity": fidelity,
                "absolute_infidelity": np.abs(1 - fidelity),
                "state_vector": state_vector,
                "target_state_vector": target_state_vector,
                "target_state": target_state,
                "target": target,
                "shots": shots,
                "interval": interval,
            }
        )

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
        use_zvalues: bool = False,
        plot: bool = False,
    ) -> Result:
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
            self.reset_awg_and_capunits(qubits=qubits)

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

                    if use_zvalues:
                        p = data.kerneled
                        g, e = (
                            self.state_centers[qubit][0],
                            self.state_centers[qubit][1],
                        )
                        v_ge = e - g
                        v_gp = p - g
                        v_gp_proj = np.real(v_gp * np.conj(v_ge)) / np.abs(v_ge)
                        values_normalized = 1 - 2 * np.abs(v_gp_proj) / np.abs(v_ge)
                    else:
                        values_normalized = float(rabi_param.normalize(data.kerneled))

                    buffer[qubit] += [values_normalized]

        result = {
            qubit: (
                values[0],  # X
                values[1],  # Y
                values[2],  # Z
            )
            for qubit, values in buffer.items()
        }
        return Result(data=result)

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
    ) -> Result:
        buffer: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

        if reset_awg_and_capunits:
            initial_sequence = sequences[0]
            if isinstance(initial_sequence, PulseSchedule):
                qubits = {
                    Target.qubit_label(target) for target in initial_sequence.labels
                }
            else:
                qubits = {Target.qubit_label(target) for target in initial_sequence}
            self.reset_awg_and_capunits(qubits=qubits)

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

        return Result(data=result)

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
    ) -> Result:
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
            fig.show(
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 3,
                    },
                }
            )

        return result_pops, result_errs

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
        plot_sequence: bool = False,
        plot_raw: bool = True,
        plot_mitigated: bool = True,
        save_image: bool = True,
        reset_awg_and_capunits: bool = True,
    ) -> Result:
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
                    only_low_to_high=True,
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
            reset_awg_and_capunits=reset_awg_and_capunits,
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
        if plot_raw:
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=prob_arr_raw,
                    name="Raw",
                )
            )
        if plot_mitigated:
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
            if plot_sequence:
                ps.plot(
                    title=f"Bell state measurement: {control_basis}{target_basis} basis"
                )
            fig.show(
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 3,
                    },
                }
            )

            for label, p, mp in zip(labels, prob_arr_raw, prob_arr_mitigated):
                print(f"{label} : {p:.2%} -> {mp:.2%}")

        if save_image:
            viz.save_figure_image(
                fig,
                f"bell_state_measurement_{control_qubit}-{target_qubit}",
            )

        return Result(
            data={
                "raw": prob_arr_raw,
                "mitigated": prob_arr_mitigated,
                "result": result,
                "figure": fig,
            }
        )

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
    ) -> Result:
        n_qubits = 2
        dim = 2**n_qubits
        probabilities = {}
        for control_basis, target_basis in tqdm(
            product(["X", "Y", "Z"], repeat=n_qubits),
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
        rho = np.zeros((dim, dim), dtype=np.complex128)
        for control_basis, control_pauli in paulis.items():
            for target_basis, target_pauli in paulis.items():
                basis = f"{control_basis}{target_basis}"
                # calculate the expectation values
                if basis == "II":
                    # II is always 1
                    # 00: +1, 01: +1, 10: +1, 11: +1
                    p = probabilities["ZZ"]
                    e = p[0b00] + p[0b01] + p[0b10] + p[0b11]
                elif basis in ["IX", "IY", "IZ"]:
                    # ignore the first qubit
                    # 00: +1, 01: -1, 10: +1, 11: -1
                    p = probabilities[f"Z{target_basis}"]
                    e = p[0b00] - p[0b01] + p[0b10] - p[0b11]
                elif basis in ["XI", "YI", "ZI"]:
                    # ignore the second qubit
                    # 00: +1, 01: +1, 10: -1, 11: -1
                    p = probabilities[f"{control_basis}Z"]
                    e = p[0b00] + p[0b01] - p[0b10] - p[0b11]
                else:
                    # two-qubit basis
                    # 00: +1, 01: -1, 10: -1, 11: +1
                    p = probabilities[basis]
                    e = p[0b00] - p[0b01] - p[0b10] + p[0b11]
                pauli = np.kron(control_pauli, target_pauli)
                rho += e * pauli
                expected_values[basis] = e

        if mle_fit:
            rho = mle_fit_density_matrix(expected_values)
        else:
            rho = rho / dim

        bell_state = np.zeros((dim, 1), dtype=np.complex128)
        bell_state[0, 0] = 1 / np.sqrt(2)
        bell_state[-1, 0] = 1 / np.sqrt(2)
        fidelity = float(np.real(bell_state.T.conj() @ rho @ bell_state))

        fig = plot_ghz_state_tomography(
            rho=rho,
            qubits=[control_qubit, target_qubit],
            fidelity=fidelity,
            plot=plot,
            save_image=save_image,
            width=600,
            height=366,
            file_name=f"bell_state_tomography_{control_qubit}-{target_qubit}",
        )["figure"]

        return Result(
            data={
                "probabilities": probabilities,
                "expected_values": expected_values,
                "density_matrix": rho,
                "fidelity": fidelity,
                "figure": fig,
            }
        )

    def create_entangle_sequence(
        self,
        entangle_steps: Collection[tuple[str | int, str | int]],
        *,
        initialization_pulse: str | None = None,
        optimize_sequence: bool = False,
        as_late_as_possible: bool = False,
        decouple_cr_crosstalk: bool = False,
        decouple_entangled_zz: bool = False,
        decouple_all_zz: bool = False,
        cpmg_duration_unit: float | None = None,
    ) -> PulseSchedule:
        if initialization_pulse is None:
            initialization_pulse = "Y90"

        steps: list[tuple[str, str]] = []
        qubits: list[str] = []
        G = nx.DiGraph()

        for parent, child in entangle_steps:
            if isinstance(parent, int):
                parent = self.quantum_system.get_qubit(parent).label
            if isinstance(child, int):
                child = self.quantum_system.get_qubit(child).label
            steps.append((parent, child))

            if parent not in qubits:
                qubits.append(parent)
            if child not in qubits:
                qubits.append(child)

            weight = int(self.cnot(parent, child, only_low_to_high=True).duration)
            G.add_edge(parent, child, weight=weight)

        roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
        leaf_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
        leaf_edges = [step for step in steps if step[1] in leaf_nodes]

        substeps: list[list[tuple[str, str]]] = []

        if optimize_sequence:
            path_lengths = {}
            for leaf in leaf_nodes:
                for root in roots:
                    if not nx.has_path(G, root, leaf):
                        continue
                    path = tuple(nx.shortest_path(G, source=root, target=leaf))
                    length = sum(G[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))
                    path_lengths[path] = length

            sorted_paths = sorted(
                path_lengths.items(), key=lambda x: x[1], reverse=True
            )

            all_edges = []
            for path, length in sorted_paths:
                substeps.append([])
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    if edge not in all_edges:
                        substeps[-1].append(edge)
                    all_edges.append(edge)
        else:
            substeps = [steps]

        # print(substeps)

        with PulseSchedule() as ps:
            for root in roots:
                if initialization_pulse == "Y90":
                    ps.add(root, self.y90(root))
                elif initialization_pulse == "X90":
                    ps.add(root, self.x90(root))
                elif initialization_pulse == "H":
                    ps.add(root, self.hadamard(root))
                else:
                    raise ValueError(
                        f"Invalid initialize pulse: {initialization_pulse}"
                    )
            ps.barrier()
            for steps in substeps:
                for parent, child in steps:
                    cnot = self.cnot(parent, child, only_low_to_high=True)
                    ps.call(cnot)
                    if decouple_cr_crosstalk:
                        if self.qubits[parent].index % 4 in [0, 3]:
                            control_qubit = parent
                            target_qubit = child
                        else:
                            control_qubit = child
                            target_qubit = parent
                        cr_ranges = cnot.get_pulse_ranges()[
                            f"{control_qubit}-{target_qubit}"
                        ]
                        spectators = self.get_spectators(control_qubit)
                        for spectator in spectators:
                            spec = spectator.label
                            pi = self.x180(spec)
                            if spec in ps.labels and spec != target_qubit:
                                cr_start = ps.get_offset(target_qubit) - cnot.duration
                                spec_end = ps.get_offset(spec)
                                space_before_cr = cr_start - spec_end
                                if space_before_cr >= 0:
                                    ps.add(spec, Blank(space_before_cr))
                                    blank1 = cr_ranges[0].stop * 2
                                    ps.add(
                                        spec,
                                        Blank(blank1),
                                    )
                                    ps.add(spec, pi)
                                    blank2 = (
                                        cr_ranges[1].stop - cr_ranges[0].stop
                                    ) * 2 - pi.duration
                                    ps.add(
                                        spec,
                                        Blank(blank2),
                                    )
                                    ps.add(spec, pi)

            if as_late_as_possible:
                # Put final CNOT gates as late as possible
                max_duration = ps._max_offset()
                for leaf_edge in leaf_edges:
                    if self.qubits[leaf_edge[0]].index % 4 in [0, 3]:
                        control_label = leaf_edge[0]
                        target_label = leaf_edge[1]
                    else:
                        control_label = leaf_edge[1]
                        target_label = leaf_edge[0]
                    cr_label = f"{control_label}-{target_label}"
                    if ps._offsets[control_label] == ps._offsets[target_label]:
                        offset = ps._offsets[control_label]
                        if offset < max_duration:
                            blank = max_duration - offset
                            for label in [control_label, target_label, cr_label]:
                                ps._channels[label].sequence._elements.insert(
                                    -1, Blank(duration=blank)
                                )
                                ps._offsets[label] += blank

            if decouple_entangled_zz:
                # Apply CPMG to blanks after entanglement gates
                for qubit in qubits:
                    if self.qubits[qubit].index % 4 in [0, 3]:
                        dd_duration = ps._max_offset() - ps._offsets[qubit]
                        pi = self.x180(qubit)
                        if cpmg_duration_unit is None:
                            n_pi = 2
                            duration_unit = pi.duration * 10
                        else:
                            n_pi = 2 * int(dd_duration // cpmg_duration_unit)
                            duration_unit = cpmg_duration_unit
                        if dd_duration > duration_unit:
                            tau = (dd_duration - pi.duration * n_pi) // (2 * n_pi)
                            tau = (tau // Pulse.SAMPLING_PERIOD) * Pulse.SAMPLING_PERIOD
                            ps.add(
                                qubit, CPMG(tau=tau, pi=pi, n=n_pi, alternating=False)
                            )

        if decouple_all_zz:
            # Apply CPMG to all blanks in the sequence
            with PulseSchedule() as ps_dd:
                for target, pulse_array in ps.get_sequences().items():
                    for element in pulse_array.elements:
                        if isinstance(element, Blank) and target in self.qubits:
                            if self.qubits[target].index % 4 in [0, 3]:
                                dd_duration = element.duration
                                pi = self.x180(target)
                                if cpmg_duration_unit is None:
                                    n_pi = 2
                                    duration_unit = pi.duration * 10
                                else:
                                    n_pi = 2 * int(dd_duration // cpmg_duration_unit)
                                    duration_unit = cpmg_duration_unit
                                if dd_duration > duration_unit:
                                    tau = (dd_duration - pi.duration * n_pi) // (
                                        2 * n_pi
                                    )
                                    tau = (
                                        tau // Pulse.SAMPLING_PERIOD
                                    ) * Pulse.SAMPLING_PERIOD
                                    ps_dd.add(target, CPMG(tau=tau, pi=pi, n=n_pi))
                                    continue
                        ps_dd.add(target, element)

            seq = ps_dd
        else:
            seq = ps

        return seq

    def create_ghz_sequence(
        self,
        entangle_steps: Collection[tuple[str | int, str | int]],
        *,
        initialization_pulse: str | None = None,
        optimize_sequence: bool = True,
        as_late_as_possible: bool = True,
        decouple_cr_crosstalk: bool = True,
        decouple_entangled_zz: bool = True,
        decouple_all_zz: bool = False,
        cpmg_duration_unit: float | None = None,
    ) -> PulseSchedule:
        """
        Create a GHZ state preparation sequence based on the entanglement steps.
        Returns a PulseSchedule object.
        """
        steps: list[tuple[str, str]] = []

        for parent, child in entangle_steps:
            if isinstance(parent, int):
                parent = self.quantum_system.get_qubit(parent).label
            if isinstance(child, int):
                child = self.quantum_system.get_qubit(child).label
            steps.append((parent, child))

        qubits: list[str] = [steps[0][0]]
        for parent, child in steps:
            if parent not in qubits:
                raise ValueError(
                    f"All qubits for GHZ state must branch from the first qubit: {qubits[0]}"
                )
            if child not in qubits:
                qubits.append(child)

        return self.create_entangle_sequence(
            entangle_steps=steps,
            initialization_pulse=initialization_pulse,
            optimize_sequence=optimize_sequence,
            as_late_as_possible=as_late_as_possible,
            decouple_cr_crosstalk=decouple_cr_crosstalk,
            decouple_entangled_zz=decouple_entangled_zz,
            decouple_all_zz=decouple_all_zz,
            cpmg_duration_unit=cpmg_duration_unit,
        )

    def measure_ghz_state(
        self,
        entangle_steps: Collection[tuple[str | int, str | int]],
        *,
        measurement_bases: Collection[str] | None = None,
        initialization_pulse: str | None = None,
        optimize_sequence: bool = True,
        as_late_as_possible: bool = True,
        decouple_cr_crosstalk: bool = True,
        decouple_entangled_zz: bool = True,
        decouple_all_zz: bool = False,
        cpmg_duration_unit: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        """
        Measure the n-qubit GHZ state in the specified bases.
        Returns dict with 'raw', 'mitigated', 'result', 'figure'.
        """
        if self.state_centers is None:
            self.build_classifier(plot=False)

        steps: list[tuple[str, str]] = []
        qubits: list[str] = []

        for parent, child in entangle_steps:
            if isinstance(parent, int):
                parent = self.quantum_system.get_qubit(parent).label
            if isinstance(child, int):
                child = self.quantum_system.get_qubit(child).label
            steps.append((parent, child))

            if parent not in qubits:
                qubits.append(parent)
            if child not in qubits:
                qubits.append(child)

        n_qubits = len(qubits)

        if measurement_bases is None:
            measurement_bases = ["Z"] * n_qubits
        else:
            measurement_bases = list(measurement_bases)

        seq = self.create_ghz_sequence(
            entangle_steps=steps,
            initialization_pulse=initialization_pulse,
            optimize_sequence=optimize_sequence,
            as_late_as_possible=as_late_as_possible,
            decouple_cr_crosstalk=decouple_cr_crosstalk,
            decouple_entangled_zz=decouple_entangled_zz,
            decouple_all_zz=decouple_all_zz,
            cpmg_duration_unit=cpmg_duration_unit,
        )

        with PulseSchedule() as ps:
            ps.call(seq)
            for qb, basis in zip(qubits, measurement_bases):
                if basis == "X":
                    ps.add(qb, self.y90m(qb))
                elif basis == "Y":
                    ps.add(qb, self.x90(qb))

        result = self.measure(
            ps,
            mode="single",
            shots=shots,
            interval=interval,
        )

        basis_labels = result.get_basis_labels(qubits)
        prob_dict_raw = result.get_probabilities(qubits)
        prob_dict_raw = {label: prob_dict_raw.get(label, 0) for label in basis_labels}
        prob_dict_mitigated = result.get_mitigated_probabilities(qubits)

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
            title=f"GHZ state measurement: {'-'.join(qubits)}",
            xaxis_title=f"State ({''.join(measurement_bases)} basis)",
            yaxis_title="Probability",
            barmode="group",
            yaxis_range=[0, 1],
        )
        if plot:
            ps.plot(title=f"GHZ state measurement: {''.join(measurement_bases)} basis")
            fig.show(
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 3,
                    },
                }
            )
            for label, p, mp in zip(labels, prob_arr_raw, prob_arr_mitigated):
                print(f"{label} : {p:.2%} -> {mp:.2%}")
        if save_image:
            viz.save_figure_image(
                fig,
                f"ghz_state_measurement_{'-'.join(qubits)}",
            )

        return Result(
            data={
                "raw": prob_arr_raw,
                "mitigated": prob_arr_mitigated,
                "result": result,
                "figure": fig,
            }
        )

    def ghz_state_tomography(
        self,
        entangle_steps: Collection[tuple[str | int, str | int]],
        *,
        readout_mitigation: bool = True,
        initialization_pulse: str | None = None,
        optimize_sequence: bool = True,
        as_late_as_possible: bool = True,
        decouple_cr_crosstalk: bool = True,
        decouple_entangled_zz: bool = True,
        decouple_all_zz: bool = False,
        cpmg_duration_unit: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        show_sequence: bool = True,
        save_image: bool = True,
        mle_fit: bool = True,
    ) -> Result:
        """
        Performs full state tomography on a n-qubit GHZ state.

        This involves:
        1. Measuring the GHZ state in all 3^n Pauli bases.
        2. Calculating the expectation values for all 4^n Pauli strings.
        3. Reconstructing the 2^n x 2^n density matrix using linear inversion or MLE.
        4. Calculating the fidelity with the ideal GHZ state.
        5. Plotting the resulting density matrix.

        Parameters
        ----------
        entangle_steps : list[tuple[str, str]]
            List of tuples representing the entanglement steps, e.g., [("Q00", "Q01"), ("Q01", "Q02")].
        readout_mitigation : bool
            Whether to apply readout error mitigation.
        shots : int
            Number of shots for each measurement.
        interval : float
            Time interval between measurements.
        plot : bool
            Whether to plot the resulting density matrix.
        save_image : bool
            Whether to save the plot as an image.
        mle_fit : bool
            Whether to use Maximum Likelihood Estimation (MLE) for density matrix reconstruction.

        Returns
        -------
        dict
            A dictionary containing:
            - "probabilities": Measured probabilities in all bases.
            - "expected_values": Calculated expectation values for all Pauli strings.
            - "density_matrix": Reconstructed density matrix.
            - "fidelity": Fidelity with the ideal GHZ state.
            - "figure": Plotly figure of the density matrix.
        """

        qubits: list[str] = []
        steps: list[tuple[str, str]] = []

        for parent, child in entangle_steps:
            if isinstance(parent, int):
                parent = self.quantum_system.get_qubit(parent).label
            if isinstance(child, int):
                child = self.quantum_system.get_qubit(child).label
            steps.append((parent, child))

            if parent not in qubits:
                qubits.append(parent)
            if child not in qubits:
                qubits.append(child)

        n_qubits = len(qubits)
        dim = 2**n_qubits

        if show_sequence:
            seq = self.create_ghz_sequence(
                entangle_steps=steps,
                initialization_pulse=initialization_pulse,
                optimize_sequence=optimize_sequence,
                as_late_as_possible=as_late_as_possible,
                decouple_cr_crosstalk=decouple_cr_crosstalk,
                decouple_entangled_zz=decouple_entangled_zz,
                decouple_all_zz=decouple_all_zz,
                cpmg_duration_unit=cpmg_duration_unit,
            )
            seq.plot(title=f"GHZ state preparation sequence : {'-'.join(qubits)}")

        probs_raw = {}
        probs_mit = {}
        for measurement_bases in tqdm(
            product(["X", "Y", "Z"], repeat=n_qubits),
            total=3**n_qubits,
            desc="Measuring GHZ state in all bases",
        ):
            basis_label = "".join(measurement_bases)
            result = self.measure_ghz_state(
                entangle_steps=steps,
                measurement_bases=measurement_bases,
                initialization_pulse=initialization_pulse,
                optimize_sequence=optimize_sequence,
                as_late_as_possible=as_late_as_possible,
                decouple_cr_crosstalk=decouple_cr_crosstalk,
                decouple_entangled_zz=decouple_entangled_zz,
                decouple_all_zz=decouple_all_zz,
                cpmg_duration_unit=cpmg_duration_unit,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )
            probs_raw[basis_label] = result["raw"]
            if readout_mitigation:
                probs_mit[basis_label] = result["mitigated"]

        ghz_state = np.zeros((dim, 1), dtype=np.complex128)
        ghz_state[0, 0] = 1 / np.sqrt(2)
        ghz_state[-1, 0] = 1 / np.sqrt(2)

        rho_raw = create_density_matrix(probs_raw)
        fidelity_raw = float(np.real(ghz_state.T.conj() @ rho_raw @ ghz_state))

        if readout_mitigation:
            rho_mit = create_density_matrix(probs_mit, mle_fit=False)
            fidelity_mit = float(np.real(ghz_state.T.conj() @ rho_mit @ ghz_state))

            if mle_fit:
                rho_mle = create_density_matrix(probs_mit, mle_fit=True)
                fidelity_mle = float(np.real(ghz_state.T.conj() @ rho_mle @ ghz_state))

        width, height = 800, 455

        fig_raw = plot_ghz_state_tomography(
            rho=rho_raw,
            qubits=qubits,
            fidelity=fidelity_raw,
            plot=False,
            save_image=save_image,
            width=width,
            height=height,
            file_name=f"ghz_state_tomography_raw_{'-'.join(qubits)}",
        )["figure"]

        if readout_mitigation:
            fig_mit = plot_ghz_state_tomography(
                rho=rho_mit,
                qubits=qubits,
                fidelity=fidelity_mit,
                plot=False,
                save_image=save_image,
                width=width,
                height=height,
                file_name=f"ghz_state_tomography_mit_{'-'.join(qubits)}",
            )["figure"]

            if mle_fit:
                fig_mle = plot_ghz_state_tomography(
                    rho=rho_mle,
                    qubits=qubits,
                    fidelity=fidelity_mle,
                    plot=False,
                    save_image=save_image,
                    width=width,
                    height=height,
                    file_name=f"ghz_state_tomography_mle_{'-'.join(qubits)}",
                )["figure"]

        if plot:
            if not readout_mitigation:
                fig_raw.show()
            elif mle_fit:
                fig_mle.show()
            else:
                fig_mit.show()

        result = {
            "raw": {
                "probabilities": probs_raw,
                "density_matrix": rho_raw,
                "fidelity": fidelity_raw,
                "figure": fig_raw,
            },
        }
        if readout_mitigation:
            result["mitigated"] = {
                "probabilities": probs_mit,
                "density_matrix": rho_mit,
                "fidelity": fidelity_mit,
                "figure": fig_mit,
            }
            if mle_fit:
                result["mle"] = {
                    "probabilities": probs_mit,
                    "density_matrix": rho_mle,
                    "fidelity": fidelity_mle,
                    "figure": fig_mle,
                }

        return Result(data=result)

    def create_mqc_sequence(
        self,
        entangle_steps: Collection[tuple[str | int, str | int]],
        *,
        phi: float = 0.0,
        echo: bool = True,
        initialization_pulse: str | None = None,
        optimize_sequence: bool = True,
        as_late_as_possible: bool = True,
        decouple_cr_crosstalk: bool = False,
        decouple_entangled_zz: bool = True,
        decouple_all_zz: bool = False,
        cpmg_duration_unit: float | None = None,
    ) -> PulseSchedule:
        qubits: list[str] = []
        steps: list[tuple[str, str]] = []

        for parent, child in entangle_steps:
            if isinstance(parent, int):
                parent = self.quantum_system.get_qubit(parent).label
            if isinstance(child, int):
                child = self.quantum_system.get_qubit(child).label
            steps.append((parent, child))

            if parent not in qubits:
                qubits.append(parent)
            if child not in qubits:
                qubits.append(child)

        ghz_seq = self.create_entangle_sequence(
            entangle_steps=steps,
            initialization_pulse=initialization_pulse,
            optimize_sequence=optimize_sequence,
            as_late_as_possible=as_late_as_possible,
            decouple_cr_crosstalk=decouple_cr_crosstalk,
            decouple_entangled_zz=decouple_entangled_zz,
            decouple_all_zz=decouple_all_zz,
            cpmg_duration_unit=cpmg_duration_unit,
        )

        with PulseSchedule() as seq:
            seq.call(ghz_seq)
            if echo:
                for qubit in qubits:
                    seq.add(qubit, self.x180(qubit))
            seq.barrier()
            for target in ghz_seq.get_targets():
                seq.add(target, VirtualZ(phi))
            seq.call(ghz_seq.inverted())
        return seq

    def mqc_experiment(
        self,
        entangle_steps: Collection[tuple[str | int, str | int]],
        *,
        phi_range: np.ndarray | None = None,
        n_points_per_qubit: int | None = None,
        show_sequence: bool = True,
        echo: bool = True,
        initialization_pulse: str | None = None,
        optimize_sequence: bool = True,
        as_late_as_possible: bool = True,
        decouple_cr_crosstalk: bool = False,
        decouple_entangled_zz: bool = True,
        decouple_all_zz: bool = False,
        cpmg_duration_unit: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> Result:
        qubits: list[str] = []
        source_qubits: list[str] = []
        steps: list[tuple[str, str]] = []

        for parent, child in entangle_steps:
            if isinstance(parent, int):
                parent = self.quantum_system.get_qubit(parent).label
            if isinstance(child, int):
                child = self.quantum_system.get_qubit(child).label
            steps.append((parent, child))

            if parent not in qubits:
                source_qubits.append(parent)
                qubits.append(parent)
            if child not in qubits:
                qubits.append(child)

        n_qubits = len(qubits)

        if phi_range is None:
            if n_points_per_qubit is None:
                n_points_per_qubit = 6
            phi_range = np.linspace(0, 2 * np.pi, n_points_per_qubit * n_qubits + 1)

        if show_sequence:
            seq = self.create_mqc_sequence(
                entangle_steps=steps,
                phi=0.0,
                echo=echo,
                initialization_pulse=initialization_pulse,
                optimize_sequence=optimize_sequence,
                as_late_as_possible=as_late_as_possible,
                decouple_cr_crosstalk=decouple_cr_crosstalk,
                decouple_entangled_zz=decouple_entangled_zz,
                decouple_all_zz=decouple_all_zz,
                cpmg_duration_unit=cpmg_duration_unit,
            )
            seq.plot(title=f"{n_qubits}-qubits entanglement sequence")

        result = self.sweep_parameter(
            lambda phi: self.create_mqc_sequence(
                entangle_steps=entangle_steps,
                phi=phi,
                echo=echo,
                initialization_pulse=initialization_pulse,
                optimize_sequence=optimize_sequence,
                as_late_as_possible=as_late_as_possible,
                decouple_cr_crosstalk=decouple_cr_crosstalk,
                decouple_entangled_zz=decouple_entangled_zz,
                decouple_all_zz=decouple_all_zz,
                cpmg_duration_unit=cpmg_duration_unit,
            ),
            plot=False,
            enable_tqdm=True,
            sweep_range=phi_range,
            shots=shots,
            interval=interval,
        )

        for qubit, data in result.data.items():
            title = f"Measured signal : {qubit}"
            if qubit in source_qubits:
                title += " (source qubit)"
            fig = data.plot(
                normalize=True,
                title=title,
                xlabel="Z rotation : φ (rad)",
                return_figure=True,
            )
            viz.save_figure_image(
                fig,  # type: ignore
                name=f"mqc_n{n_qubits}_{qubit}",
                format="png",
            )
            viz.save_figure_image(
                fig,  # type: ignore
                name=f"mqc_n{n_qubits}_{qubit}",
                format="svg",
            )

        coherences = {}
        for source_qubit in source_qubits:
            fourier_result = self.fourier_analysis(
                result.data[source_qubit].data,
                qubit=source_qubit,
                title=f"Fourier analysis of {n_qubits}-qubits entanglement : {source_qubit}",
            )
            coherences[source_qubit] = fourier_result["C"]

            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            os.makedirs("./data", exist_ok=True)

            np.savez(
                f"./data/{timestamp}_mqc_n{n_qubits}_raw.npz",
                result.data[source_qubit].data,
            )
            np.savez(
                f"./data/{timestamp}_mqc_n{n_qubits}_normalized.npz",
                result.data[source_qubit].normalized,
            )

        return Result(
            data={
                "phi_range": phi_range,
                "result": result,
                "coherences": coherences,
            }
        )

    @staticmethod
    def fourier_analysis(
        data: ArrayLike,
        *,
        qubit: str | None = None,
        title="Fourier analysis",
    ) -> Result:
        data = np.asarray(data)

        S = (data + 1) / 2
        N = len(S)
        F = np.fft.fft(S)

        q = np.arange(N // 2)[1:]
        I = np.abs(F[1 : N // 2]) / N
        C = 2 * np.sqrt(I)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=q,
                y=C,
                name="Amplitude",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Fourier modes",
            yaxis_title="Amplitude",
        )

        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )

        file_name = f"fourier_analysis_{qubit}" if qubit else "fourier_analysis"

        viz.save_figure_image(
            fig,
            name=file_name,
            format="png",
        )

        viz.save_figure_image(
            fig,
            name=file_name,
            format="svg",
        )

        return Result(
            data={
                "figure": fig,
                "I": I,
                "C": C,
            }
        )

    def parity_oscillation(
        self,
        entangle_steps: Collection[tuple[str | int, str | int]],
        *,
        phi_range: np.ndarray | None = None,
        n_points_per_qubit: int | None = None,
        show_sequence: bool = True,
        show_only_qubit_channels: bool = False,
        initialization_pulse: str | None = None,
        optimize_sequence: bool = True,
        as_late_as_possible: bool = True,
        decouple_cr_crosstalk: bool = False,
        decouple_entangled_zz: bool = True,
        decouple_all_zz: bool = False,
        cpmg_duration_unit: float | None = None,
        readout_mitigation: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> Result:
        if initialization_pulse is None:
            initialization_pulse = "Y90"

        qubits: list[str] = []
        source_qubits: list[str] = []
        steps: list[tuple[str, str]] = []

        for parent, child in entangle_steps:
            if isinstance(parent, int):
                parent = self.quantum_system.get_qubit(parent).label
            if isinstance(child, int):
                child = self.quantum_system.get_qubit(child).label
            steps.append((parent, child))

            if parent not in qubits:
                source_qubits.append(parent)
                qubits.append(parent)
            if child not in qubits:
                qubits.append(child)

        n_qubits = len(qubits)

        print(f"qubits: {qubits}")

        if phi_range is None:
            if n_points_per_qubit is None:
                n_points_per_qubit = 6
            phi_range = np.linspace(0, 2 * np.pi, n_points_per_qubit * n_qubits + 1)

        ghz_seq = self.create_entangle_sequence(
            entangle_steps=steps,
            initialization_pulse=initialization_pulse,
            optimize_sequence=optimize_sequence,
            as_late_as_possible=as_late_as_possible,
            decouple_cr_crosstalk=decouple_cr_crosstalk,
            decouple_entangled_zz=decouple_entangled_zz,
            decouple_all_zz=decouple_all_zz,
            cpmg_duration_unit=cpmg_duration_unit,
        )

        def sequence(phi: float) -> PulseSchedule:
            with PulseSchedule() as seq:
                seq.call(ghz_seq)
                rz = VirtualZ(phi)
                for label in seq.labels:
                    if label in self.qubit_labels:
                        seq.add(label, rz)
                        if initialization_pulse == "Y90":
                            seq.add(label, self.y90m(label))
                        elif initialization_pulse == "X90":
                            seq.add(label, self.x90m(label))
                        elif initialization_pulse == "H":
                            seq.add(label, self.hadamard(label))
                        else:
                            raise ValueError(
                                f"Invalid initialize pulse: {initialization_pulse}"
                            )
            return seq

        if show_sequence:
            seq_plot = sequence(0.0)
            if show_only_qubit_channels:
                for label in seq_plot.labels:
                    if label not in self.qubits:
                        del seq_plot._channels[label]
            seq_plot.plot(
                title=f"{n_qubits}-qubits entanglement sequence",
                show_physical_pulse=False,
            )

        parities_raw = []
        parities_mit = []
        result = []
        for phi in tqdm(phi_range):
            res = self.measure(
                sequence(phi),
                mode="single",
                shots=shots,
                interval=interval,
            )
            result.append(res)
            probs_raw = res.probabilities
            parity_raw = 0
            for label, prob in probs_raw.items():
                parity_raw += prob * (1 if label.count("1") % 2 == 0 else -1)
            parities_raw.append(parity_raw)

            if readout_mitigation:
                probs_mit = res.mitigated_probabilities
                parity_mit = 0
                for label, prob in probs_mit.items():
                    parity_mit += prob * (1 if label.count("1") % 2 == 0 else -1)
                parities_mit.append(parity_mit)

        fig = go.Figure(
            layout=go.Layout(
                title=f"Parity oscillation : {n_qubits}-qubit GHZ state",
                xaxis_title="Z rotation : φ (rad)",
                yaxis_title="Parity",
                yaxis_range=(-1.1, 1.1),
            )
        )

        fig.add_scatter(
            x=phi_range,
            y=parities_raw,
            mode="lines+markers",
            name="Raw",
            marker=dict(size=5),
        )
        if readout_mitigation:
            fig.add_scatter(
                x=phi_range,
                y=parities_mit,
                mode="lines+markers",
                name="Mitigated",
                marker=dict(size=5),
            )
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 3,
                },
            }
        )

        viz.save_figure_image(
            fig,  # type: ignore
            name=f"parity_oscillation_n{n_qubits}",
            format="png",
        )
        viz.save_figure_image(
            fig,  # type: ignore
            name=f"parity_oscillation_n{n_qubits}",
            format="svg",
        )

        self.fourier_analysis(
            parities_raw if not readout_mitigation else parities_mit,
            title=f"Fourier analysis of {n_qubits}-qubit parity oscillation",
        )

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        os.makedirs("./data", exist_ok=True)

        np.savez(
            f"./data/{timestamp}_parity_oscillation_n{n_qubits}.npz",
            parities_raw,
        )

        return Result(
            data={
                "phi_range": phi_range,
                "result": result,
                "parities_raw": parities_raw,
                "parities_mit": parities_mit,
            }
        )

    def create_1d_cluster_sequence(
        self,
        targets: Collection[str | int],
        *,
        bases: dict[int, str] | None = None,
        optimize_sequence: bool = False,
        as_late_as_possible: bool = True,
        decouple_cr_crosstalk: bool = False,
        decouple_entangled_zz: bool = False,
        decouple_all_zz: bool = False,
        cpmg_duration_unit: float | None = None,
        with_readout_pulses: bool = True,
    ):
        """
        Create a 1D cluster state preparation sequence for the given targets.
        Returns a PulseSchedule object.
        """
        targets = [
            self.quantum_system.get_qubit(target).label
            if isinstance(target, int)
            else target
            for target in targets
        ]
        if bases is None:
            bases = {}

        qubits = [
            {
                "index": i,
                "label": label,
                "type": "L" if self.qubits[label].index % 4 in [0, 3] else "H",
                "basis": bases[i] if i in bases else "Z",
            }
            for i, label in enumerate(targets)
        ]
        n_qubits = len(qubits)

        l1_edges = []
        l1_max_duration = 0
        for i in range(n_qubits - 1):
            if qubits[i]["type"] == "L":
                edge = (qubits[i]["label"], qubits[i + 1]["label"])
                l1_edges.append(edge)
                cnot = self.cnot(*edge, only_low_to_high=True)
                l1_max_duration = max(l1_max_duration, cnot.duration)

        l2_edges = []
        l2_max_duration = 0
        for i in range(n_qubits - 1):
            if qubits[i]["type"] == "H":
                edge = (qubits[i + 1]["label"], qubits[i]["label"])
                l2_edges.append(edge)
                cnot = self.cnot(*edge, only_low_to_high=True)
                l2_max_duration = max(l2_max_duration, cnot.duration)

        with PulseSchedule(targets) as ps:
            for edge in l1_edges:
                with PulseSchedule() as l1:
                    h = self.hadamard(edge[0])
                    l1.add(edge[0], h)
                    l1.call(self.cnot(*edge, only_low_to_high=True))
                l1.pad(
                    total_duration=l1_max_duration + h.duration,
                    pad_side="left",
                )
                ps.call(l1)

            for edge in l2_edges:
                with PulseSchedule() as l2:
                    l2.call(self.cnot(*edge, only_low_to_high=True))
                    h = self.hadamard(edge[1])
                    l2.add(edge[1], h)
                ps.call(l2)

            # debug: no entanglement, just Hadamard gates
            # for target in targets:
            #     ps.add(target, self.hadamard(target))

            for qubit in qubits:
                basis = qubit["basis"]
                if basis == "X":
                    ps.add(qubit["label"], self.y90m(qubit["label"]))
                elif basis == "Y":
                    ps.add(qubit["label"], self.x90(qubit["label"]))
                elif basis == "Z":
                    pass
                else:
                    raise ValueError(f"Unknown basis: {basis}")

            if with_readout_pulses:
                for qubit in qubits:
                    resonator = self.resonators[qubit["label"]].label
                    ps.add(resonator, Blank(ps.get_offset(qubit["label"])))
                    ps.add(resonator, self.readout(resonator))
        return ps

        # cluster_seq = self.create_entangle_sequence(
        #     entangle_steps=entangle_steps,
        #     initialization_pulse=initialization_pulse,
        #     optimize_sequence=optimize_sequence,
        #     as_late_as_possible=as_late_as_possible,
        #     decouple_cr_crosstalk=decouple_cr_crosstalk,
        #     decouple_entangled_zz=decouple_entangled_zz,
        #     decouple_all_zz=decouple_all_zz,
        #     cpmg_duration_unit=cpmg_duration_unit,
        # )

        # with PulseSchedule(targets) as ps:
        #     ps.call(cluster_seq)
        #     for qubit in qubits:
        #         if qubit["type"] == "H":
        #             ps.add(qubit["label"], self.hadamard(qubit["label"]))
        #     for qubit in qubits:
        #         basis = qubit["basis"]
        #         if basis == "X":
        #             ps.add(qubit["label"], self.y90m(qubit["label"]))
        #         elif basis == "Y":
        #             ps.add(qubit["label"], self.x90(qubit["label"]))
        #         elif basis == "Z":
        #             pass
        #         else:
        #             raise ValueError(f"Unknown basis: {basis}")
        # return ps

    def _measure_1d_cluster_state(
        self,
        targets: Collection[str | int],
        *,
        offset: int = 0,
        mle_fit: bool = True,
        optimize_sequence: bool = False,
        as_late_as_possible: bool = True,
        decouple_cr_crosstalk: bool = False,
        decouple_entangled_zz: bool = False,
        decouple_all_zz: bool = False,
        cpmg_duration_unit: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        method: str = "execute",
        reset_awg_and_capunits: bool = True,
    ):
        targets = [
            self.quantum_system.get_qubit(target).label
            if isinstance(target, int)
            else target
            for target in targets
        ]
        n_qubits = len(targets)

        if offset > 2:
            raise ValueError(
                "Offset must be 0, 1, or 2 for 1D cluster state measurement."
            )

        edges: dict[tuple[str, str], list[str]] = {}
        n_edges = n_qubits // 3 + 1
        for i in range(n_edges):
            if offset + i * 3 + 1 >= len(targets):
                break
            edge = (targets[offset + i * 3], targets[offset + i * 3 + 1])
            edge_spectators: list[str] = []
            for node in edge:
                node_spectators = self.get_spectators(node)
                for spectator in node_spectators:
                    node_index = targets.index(node)
                    if spectator.label in targets:
                        spectator_index = targets.index(spectator.label)
                        is_adjacent = abs(node_index - spectator_index) == 1
                        if is_adjacent and spectator.label not in edge:
                            edge_spectators.append(spectator.label)
            edges[edge] = edge_spectators
            if plot:
                print(f"Edge: {edge}, Spectators: {edge_spectators}")

        seq = self.create_1d_cluster_sequence(
            targets,
            optimize_sequence=optimize_sequence,
            as_late_as_possible=as_late_as_possible,
            decouple_cr_crosstalk=decouple_cr_crosstalk,
            decouple_entangled_zz=decouple_entangled_zz,
            decouple_all_zz=decouple_all_zz,
            cpmg_duration_unit=cpmg_duration_unit,
            with_readout_pulses=True if method == "execute" else False,
        )
        if plot:
            seq.plot()

        if reset_awg_and_capunits:
            qubits = {Target.qubit_label(target) for target in seq.labels}
            self.reset_awg_and_capunits(qubits=qubits)

        edge_sbits_result: dict[tuple[str, str], dict[str, dict]] = {
            edge: {} for edge in edges
        }
        edge_sbits_probabilities: dict[
            tuple[str, str], dict[str, dict[str, dict[str, float]]]
        ] = {
            # edge: {
            #     sbits: {
            #         pauli: {
            #             ebits: probability,
            #         }
            #     },
            # }
            edge: {}
            for edge in edges
        }

        for pauli0, pauli1 in tqdm(
            product(["X", "Y", "Z"], repeat=2),
        ):
            pauli_basis = f"{pauli0}{pauli1}"

            bases = {}
            for node0, node1 in edges:
                idx0 = targets.index(node0)
                idx1 = targets.index(node1)
                bases[idx0] = pauli0
                bases[idx1] = pauli1

            if method == "execute":
                result = self.execute(
                    self.create_1d_cluster_sequence(
                        targets,
                        bases=bases,
                        optimize_sequence=optimize_sequence,
                        as_late_as_possible=as_late_as_possible,
                        decouple_cr_crosstalk=decouple_cr_crosstalk,
                        decouple_entangled_zz=decouple_entangled_zz,
                        decouple_all_zz=decouple_all_zz,
                        cpmg_duration_unit=cpmg_duration_unit,
                        with_readout_pulses=True,
                    ),
                    mode="single",
                    shots=shots,
                    interval=interval,
                    reset_awg_and_capunits=False,
                )
            else:
                result = self.measure(
                    self.create_1d_cluster_sequence(
                        targets,
                        bases=bases,
                        optimize_sequence=optimize_sequence,
                        as_late_as_possible=as_late_as_possible,
                        decouple_cr_crosstalk=decouple_cr_crosstalk,
                        decouple_entangled_zz=decouple_entangled_zz,
                        decouple_all_zz=decouple_all_zz,
                        cpmg_duration_unit=cpmg_duration_unit,
                        with_readout_pulses=False,
                    ),
                    mode="single",
                    shots=shots,
                    interval=interval,
                    reset_awg_and_capunits=False,
                )

            for edge, spectators in edges.items():
                target_labels = list(edge) + spectators
                mitigated_counts = result.get_mitigated_counts(target_labels)
                n_spectators = len(spectators)
                spectators_bits = [
                    "".join(bits) for bits in product("01", repeat=n_spectators)
                ]
                for sbits in spectators_bits:
                    if sbits not in edge_sbits_probabilities[edge]:
                        edge_sbits_probabilities[edge][sbits] = {}

                sbits_ebits_counts: dict[str, dict[str, int]] = {
                    sbits: {} for sbits in spectators_bits
                }

                for bits, count in mitigated_counts.items():
                    ebits = bits[:2]
                    sbits = bits[2:]
                    sbits_ebits_counts[sbits][ebits] = count

                for sbits, ebits_counts in sbits_ebits_counts.items():
                    total_count = sum(ebits_counts.values())
                    edge_sbits_probabilities[edge][sbits][pauli_basis] = {
                        ebits: count / total_count if total_count > 0 else 0.0
                        for ebits, count in ebits_counts.items()
                    }

        paulis = {
            "I": np.array([[1, 0], [0, 1]]),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
        }

        for edge, sbits_probabilities in edge_sbits_probabilities.items():
            for sbits, probabilities in sbits_probabilities.items():
                if sbits not in edge_sbits_result[edge]:
                    edge_sbits_result[edge][sbits] = {}

                expected_values = {}
                rho = np.zeros((4, 4), dtype=np.complex128)
                for basis0, pauli0 in paulis.items():
                    for basis1, pauli1 in paulis.items():
                        pauli_basis = f"{basis0}{basis1}"
                        # calculate the expectation values
                        if pauli_basis == "II":
                            # II is always 1
                            # 00: +1, 01: +1, 10: +1, 11: +1
                            p = probabilities["ZZ"]
                            e = p["00"] + p["01"] + p["10"] + p["11"]
                        elif pauli_basis in ["IX", "IY", "IZ"]:
                            # ignore the first qubit
                            # 00: +1, 01: -1, 10: +1, 11: -1
                            p = probabilities[f"Z{basis1}"]
                            e = p["00"] - p["01"] + p["10"] - p["11"]
                        elif pauli_basis in ["XI", "YI", "ZI"]:
                            # ignore the second qubit
                            # 00: +1, 01: +1, 10: -1, 11: -1
                            p = probabilities[f"{basis0}Z"]
                            e = p["00"] + p["01"] - p["10"] - p["11"]
                        else:
                            # two-qubit basis
                            # 00: +1, 01: -1, 10: -1, 11: +1
                            p = probabilities[pauli_basis]
                            e = p["00"] - p["01"] - p["10"] + p["11"]
                        pauli_matrix = np.kron(pauli0, pauli1)
                        rho += e * pauli_matrix
                        expected_values[pauli_basis] = e

                if mle_fit:
                    rho = mle_fit_density_matrix(expected_values)
                else:
                    rho = rho / 4

                rho_pt = self.partial_transpose(rho)
                eigvals = np.linalg.eigvalsh(rho_pt)
                negativity = np.sum(np.abs(eigvals[eigvals < 0]))

                if plot:
                    print(f"{edge[0]}-{edge[1]} ({sbits}) : Negativity = {negativity}")

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=("Abs", "Phase"),
                    horizontal_spacing=0.26,
                )
                fig.add_trace(
                    go.Heatmap(
                        z=np.abs(rho),
                        zmin=0,
                        zmax=1,
                        colorscale="Hot_r",
                        colorbar=dict(
                            title="Abs",
                            x=0.37,
                            y=0.5,
                            thickness=15,
                            tickvals=[0, 0.5, 1],
                            ticktext=["0", "0.5", "1"],
                        ),
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Heatmap(
                        z=np.angle(rho),
                        zmin=-np.pi,
                        zmax=np.pi,
                        colorscale="Edge",
                        colorbar=dict(
                            title="Phase (rad)",
                            x=1.0,
                            y=0.5,
                            thickness=15,
                            tickvals=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                            ticktext=["-π", "-π/2", "0", "π/2", "π"],
                        ),
                    ),
                    row=1,
                    col=2,
                )

                tickvals = np.arange(4)
                ticktext = [f"{i:0{2}b}" for i in tickvals]
                tick_style = dict(
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickangle=0,
                )

                fig.update_xaxes(tick_style, row=1, col=1)
                fig.update_yaxes(
                    dict(**tick_style, autorange="reversed", scaleanchor="x1"),
                    row=1,
                    col=1,
                )
                fig.update_xaxes(tick_style, row=1, col=2)
                fig.update_yaxes(
                    dict(**tick_style, autorange="reversed", scaleanchor="x2"),
                    row=1,
                    col=2,
                )
                fig.update_layout(
                    title=dict(
                        text=f"Negativity of graph state: 𝒩 = {negativity:.3f}",
                        subtitle=dict(
                            text=f"edge: {edge}, spectators: ({', '.join(edges[edge])}) = '{sbits}'",
                        ),
                    ),
                    margin=dict(t=110),
                    width=600,
                    height=342,
                )

                if plot:
                    fig.show(
                        config={
                            "toImageButtonOptions": {
                                "format": "png",
                                "scale": 3,
                            },
                        }
                    )

                edge_sbits_result[edge][sbits]["expected_values"] = expected_values
                edge_sbits_result[edge][sbits]["density_matrix"] = rho
                edge_sbits_result[edge][sbits]["partial_transpose"] = rho_pt
                edge_sbits_result[edge][sbits]["negativity"] = negativity
                edge_sbits_result[edge][sbits]["eigenvalues"] = eigvals
                edge_sbits_result[edge][sbits]["figure"] = fig

        result = {"best": {edge: {} for edge in edges}}

        for edge, sbits_results in edge_sbits_result.items():
            best_result = max(
                sbits_results.values(),
                key=lambda x: x["negativity"] if "negativity" in x else 0,
            )
            result["best"][edge] = best_result

        result["all"] = edge_sbits_result

        return result

    def measure_1d_cluster_state(
        self,
        qubits: Collection[str | int],
        *,
        mle_fit: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        method: str = "execute",
        reset_awg_and_capunits: bool = True,
    ) -> Result:
        if plot:
            seq = self.create_1d_cluster_sequence(
                qubits,
                with_readout_pulses=True if method == "execute" else False,
            )
            seq.plot(
                title=f"1D cluster state preparation sequence for {len(qubits)} qubits",
            )

        negativities = {}
        figures = {}
        for offset in range(3):
            print(f"[{offset + 1}/3] Measuring edges with offset {offset}")
            result = self._measure_1d_cluster_state(
                qubits,
                offset=offset,
                mle_fit=mle_fit,
                shots=shots,
                interval=interval,
                plot=False,
                method=method,
                reset_awg_and_capunits=reset_awg_and_capunits,
            )
            for edge, data in result["best"].items():
                negativities[edge] = data["negativity"]
                figures[edge] = data["figure"]

        negativities = dict(
            sorted(negativities.items(), key=lambda item: item[1], reverse=False)
        )

        negativities_max = max(negativities.values())
        negativities_min = min(negativities.values())
        negativities_avg = np.mean(list(negativities.values()))
        negativities_std = np.std(list(negativities.values()))
        negativities_med = np.median(list(negativities.values()))
        if plot:
            for edge, fig in figures.items():
                fig.show(
                    config={
                        "toImageButtonOptions": {
                            "format": "png",
                            "scale": 3,
                        },
                    }
                )
            print(f"Negativities of {len(negativities)} edges:")
            print(f"  max: {negativities_max:.3f}")
            print(f"  min: {negativities_min:.3f}")
            print(f"  med: {negativities_med:.3f}")
            print(f"  avg: {negativities_avg:.3f}")
            print(f"  std: {negativities_std:.3f}")
            print("Negativities per edge:")
            for edge, negativity in negativities.items():
                print(f"  {edge[0]}-{edge[1]}: {negativity:.3f}")

            x = [f"{edge[0]}-{edge[1]}" for edge in negativities]
            y = [fidelity for fidelity in negativities.values()]
            fig = go.Figure(
                layout=go.Layout(
                    title=f"Negativities of {len(qubits)}-qubit 1D cluster state",
                    xaxis=dict(
                        title="Edges",
                        tickangle=45,
                        tickmode="array",
                        tickvals=list(range(len(x))),
                        ticktext=x,
                    ),
                    yaxis=dict(
                        title="Negativity",
                        range=[0, 0.55],
                        tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        ticktext=["0", "0.1", "0.2", "0.3", "0.4", "0.5"],
                    ),
                    width=800,
                    height=400,
                    margin=dict(l=70, r=70, t=90, b=100),
                )
            )
            fig.add_bar(x=x, y=y)
            fig.show(
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 3,
                    },
                }
            )

        return Result(
            data={
                "negativities_max": negativities_max,
                "negativities_min": negativities_min,
                "negativities_med": negativities_med,
                "negativities_avg": negativities_avg,
                "negativities_std": negativities_std,
                "negativities": negativities,
                "figures": figures,
            }
        )

    @staticmethod
    def partial_transpose(rho: NDArray, subsystem: int = 1) -> NDArray:
        """
        Perform partial transpose on a 2-qubit density matrix.

        Parameters
        ----------
        rho : NDArray
            The 2-qubit density matrix, reshaped as a 4x4 array.
        subsystem : int
            The subsystem to transpose (0 for first qubit, 1 for second qubit).

        Returns
        -------
        NDArray
            The partially transposed density matrix, reshaped as a 4x4 array.
        """
        rho_tensor = rho.reshape(2, 2, 2, 2)  # (iA, iB, jA, jB)

        if subsystem == 0:
            # (iA, iB, jA, jB) → (jA, iB, iA, jB)
            rho_pt = np.transpose(rho_tensor, (2, 1, 0, 3))
        elif subsystem == 1:
            # (iA, iB, jA, jB) → (iA, jB, jA, iB)
            rho_pt = np.transpose(rho_tensor, (0, 3, 2, 1))
        else:
            raise ValueError("subsystem must be 0 or 1")

        return rho_pt.reshape(4, 4)

    def create_connected_graphs(
        self,
        fidelities: dict[str, float] | None = None,
        *,
        t1: dict[str, float] | None = None,
        t2_echo: dict[str, float] | None = None,
        threshold: float = 0.0,
        plot: bool = False,
        show_labels: bool = False,
        show_data: bool = True,
    ) -> list[nx.DiGraph]:
        if fidelities is None:
            fidelities = self.load_property("bell_state_fidelity")

        if t1 is None:
            t1 = {}
        if t2_echo is None:
            t2_echo = {}

        G = nx.DiGraph()
        cr_labels = self.cr_labels
        for cr_label, fidelity in fidelities.items():
            if cr_label in cr_labels:
                if fidelity > threshold:
                    control, target = cr_label.split("-")

                    if not G.has_node(control):
                        G.add_node(
                            control,
                            t1=t1.get(control),
                            t2_echo=t2_echo.get(control),
                        )
                    if not G.has_node(target):
                        G.add_node(
                            target,
                            t1=t1.get(target),
                            t2_echo=t2_echo.get(target),
                        )

                    G.add_edge(
                        control,
                        target,
                        fidelity=fidelity,
                        cost=-np.log10(fidelity),
                    )

        graphs = []
        for component in nx.weakly_connected_components(G):
            graph = G.subgraph(component)
            graphs.append(graph)
        graphs.sort(key=lambda x: x.number_of_nodes(), reverse=True)

        if plot:
            max_n = max(graph.number_of_nodes() for graph in graphs)
            self.visualize_graph(
                G,
                title=f"Connected graphs : N (max) = {max_n}",
                show_labels=show_labels,
                show_data=show_data,
            )
        return graphs

    def create_maximum_graph(
        self,
        fidelities: dict[str, float] | None = None,
        *,
        threshold: float = 0.0,
        plot: bool = False,
        show_labels: bool = False,
        show_data: bool = True,
    ):
        if fidelities is None:
            fidelities = self.load_property("bell_state_fidelity")

        graphs = self.create_connected_graphs(
            fidelities=fidelities,
            threshold=threshold,
            plot=False,
        )
        if not graphs:
            raise ValueError("No connected graphs found")

        G = graphs[0]
        if plot:
            self.visualize_graph(
                G,
                title=f"Maximum graph : N = {G.number_of_nodes()}",
                show_labels=show_labels,
                show_data=show_data,
            )

        return G

    def create_maximum_1d_chain(
        self,
        fidelities: dict[str, float] | None = None,
        *,
        threshold: float = 0.0,
        plot: bool = False,
        show_labels: bool = False,
        show_data: bool = True,
    ) -> nx.Graph:
        """
        Create the maximum 1D chain in a 2D lattice graph.

        Parameters
        ----------
        fidelities : dict[str, float]
            A dictionary mapping edge labels to their fidelities.

        Returns
        -------
        nx.Graph
            A graph representing the maximum 1D chain.
        """
        if fidelities is None:
            fidelities = self.load_property("bell_state_fidelity")

        graphs = self.create_connected_graphs(
            fidelities,
            threshold=threshold,
            plot=False,
        )

        if not graphs:
            raise ValueError("No connected graphs found")

        G = graphs[0]
        path_nodes, path_edges, _ = find_longest_1d_chain(G)

        chain = nx.Graph()
        chain.add_nodes_from(path_nodes)
        for edge in path_edges:
            fidelity = get_max_undirected_weight(G, edge=edge, property="fidelity")
            chain.add_edge(*edge, fidelity=fidelity)

        if plot:
            self.visualize_graph(
                chain,
                title=f"Maximum 1D chain : N = {len(path_nodes)}",
                show_labels=show_labels,
                show_data=show_data,
            )

        return chain

    def create_maximum_spanning_tree(
        self,
        fidelities: dict[str, float] | None = None,
        *,
        threshold: float = 0.0,
        t1: dict[str, float] | None = None,
        t2_echo: dict[str, float] | None = None,
        plot: bool = False,
        show_labels: bool = False,
        show_data: bool = False,
    ):
        if fidelities is None:
            fidelities = self.load_property("bell_state_fidelity")

        graphs = self.create_connected_graphs(
            fidelities,
            threshold=threshold,
            t1=t1,
            t2_echo=t2_echo,
            plot=False,
        )

        if not graphs:
            raise ValueError("No connected graphs found")

        G = graphs[0]
        UG = G.to_undirected()
        mst = nx.minimum_spanning_tree(UG, weight="cost")

        if plot:
            self.visualize_graph(
                mst,
                title=f"Maximum spanning tree : N = {len(mst.nodes())}",
                show_labels=show_labels,
                show_data=show_data,
            )
        return mst

    def create_maximum_directed_tree(
        self,
        fidelities: dict[str, float] | None = None,
        *,
        root: str | None = None,
        max_depth: int | None = None,
        max_node: int | None = None,
        threshold: float = 0.0,
        t1: dict[str, float] | None = None,
        t2_echo: dict[str, float] | None = None,
        plot: bool = False,
        show_labels: bool = False,
        show_data: bool = True,
    ):
        if fidelities is None:
            fidelities = self.load_property("bell_state_fidelity")

        mst = self.create_maximum_spanning_tree(
            fidelities,
            threshold=threshold,
            t1=t1,
            t2_echo=t2_echo,
            plot=False,
        )
        if root is None:
            root_qubit = str(tree_center(mst)[0])
        else:
            root_qubit = root

        # BFS to determine parents and depth from the root
        parents: dict[str, str | None] = {root_qubit: None}
        depths: dict[str, int] = {root_qubit: 0}
        q = deque([root_qubit])

        n_node = 1
        while q:
            if max_node is not None and n_node >= max_node:
                break
            u = q.popleft()
            if max_depth is not None and depths[u] >= max_depth:
                continue
            for v in mst.neighbors(u):
                if v in parents:
                    continue
                parents[v] = u
                depths[v] = depths[u] + 1
                q.append(v)
                n_node += 1
                if max_node is not None and n_node >= max_node:
                    break

        DG = nx.DiGraph()
        for child, parent in parents.items():
            # Carry over existing node attributes (if any) and store depth
            node_attrs = dict(mst.nodes[child]) if child in mst.nodes else {}
            node_attrs["depth"] = depths.get(child, 0)
            DG.add_node(child, **node_attrs)
            if parent is None:
                continue
            DG.add_edge(parent, child, **mst[parent][child])

        max_depth = max(depths.values(), default=0)

        if plot:
            self.visualize_graph(
                DG,
                title=f"Maximum directed tree : N = {len(DG.nodes())}, root = {root_qubit}, depth = {max_depth}",
                show_labels=show_labels,
                show_data=show_data,
            )

        return DG

    def create_cz_rounds(
        self,
        graph: nx.Graph,
        *,
        plot: bool = False,
    ):
        edges = list(graph.edges())
        edges_remaining = edges.copy()
        rounds: list[list[tuple[str, str]]] = []
        while edges_remaining:
            used: set[str] = set()
            batch: list[tuple[str, str]] = []
            for u, v in edges_remaining:
                if u not in used and v not in used:
                    batch.append((u, v))
                    used.add(u)
                    used.add(v)
            # Remove scheduled edges and append this round
            edges_remaining = [e for e in edges_remaining if e not in batch]
            rounds.append(batch)

        chip_graph = self.quantum_system.chip_graph
        if plot:
            for round_idx, round in enumerate(rounds):
                graph = nx.Graph()
                for u, v in round:
                    graph.add_node(u)
                    graph.add_node(v)
                    graph.add_edge(u, v)

                node_values = {node: 1.0 for node in graph.nodes()}
                edge_values = {f"{u}-{v}": 1.0 for u, v in graph.edges()}
                edge_overlay_values = {f"{u}-{v}": 1.0 for u, v in graph.edges()}

                chip_graph.plot_graph_data(
                    directed=False,
                    title=f"CZ round : {round_idx}",
                    edge_values=edge_values,
                    edge_color="#eef",
                    edge_overlay=True,
                    edge_overlay_values=edge_overlay_values,
                    edge_overlay_color="turquoise",
                    node_color="white",
                    node_linecolor="ghostwhite",
                    node_textcolor="ghostwhite",
                    node_overlay=True,
                    node_overlay_values=node_values,
                    node_overlay_color="ghostwhite",
                    node_overlay_linecolor="black",
                    node_overlay_textcolor="black",
                )
        return rounds

    def create_graph_sequence(
        self,
        graph: nx.Graph,
        *,
        bases: dict[str, str] | None = None,
        with_readout_pulses: bool = True,
    ):
        nodes = list(graph.nodes())
        rounds = self.create_cz_rounds(graph, plot=False)

        with PulseSchedule(nodes) as ps:
            # Prepare qubits in |+> with Hadamards (can run in parallel)
            for node in nodes:
                ps.add(node, self.hadamard(node))

            # Apply CZ gates round by round so edges within a round run in parallel
            for batch in rounds:
                for u, v in batch:
                    ps.call(self.cz(u, v, only_low_to_high=True))

            # debug: no entanglement, just Hadamard gates
            # for node in nodes:
            #     ps.add(node, self.hadamard(node))

            # Basis rotations prior to readout
            for node in nodes:
                basis = bases[node] if bases and node in bases else "Z"
                if basis == "X":
                    ps.add(node, self.y90m(node))
                elif basis == "Y":
                    ps.add(node, self.x90(node))
                elif basis == "Z":
                    pass
                else:
                    raise ValueError(f"Unknown basis: {basis}")

            if with_readout_pulses:
                for node in nodes:
                    resonator = self.resonators[node].label
                    ps.add(resonator, Blank(ps.get_offset(node)))
                    ps.add(resonator, self.readout(resonator))
        return ps

    def create_measurement_rounds(
        self,
        G: nx.Graph,
        plot=False,
    ):
        chip_graph = self.quantum_system.chip_graph
        colored_edges = strong_edge_coloring(G)
        if plot:
            for color, edges in colored_edges.items():
                graph = nx.Graph()
                for u, v in edges:
                    graph.add_node(u)
                    graph.add_node(v)
                    graph.add_edge(u, v, color=color)

                node_values = {node: 1.0 for node in G.nodes()}
                edge_values = {f"{u}-{v}": 1.0 for u, v in G.edges()}
                edge_overlay_values = {f"{u}-{v}": 1.0 for u, v in graph.edges()}

                chip_graph.plot_graph_data(
                    directed=False,
                    title=f"Measurement round : {color}",
                    edge_values=edge_values,
                    edge_color="#eef",
                    edge_overlay=True,
                    edge_overlay_values=edge_overlay_values,
                    edge_overlay_color="turquoise",
                    node_color="white",
                    node_linecolor="ghostwhite",
                    node_textcolor="ghostwhite",
                    node_overlay=True,
                    node_overlay_values=node_values,
                    node_overlay_color="ghostwhite",
                    node_overlay_linecolor="black",
                    node_overlay_textcolor="black",
                )
        return colored_edges

    def _measure_graph_state(
        self,
        graph: nx.Graph,
        *,
        target_edges: list[tuple[str, str]],
        mle_fit: bool = True,
        use_all_spectator_pattern: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        method: str = "execute",
        reset_awg_and_capunits: bool = True,
        n_bootstrap: int | None = None,
        bootstrap_mle: bool = False,
    ):
        graph = graph.to_undirected()

        seq = self.create_graph_sequence(
            graph=graph,
            with_readout_pulses=True if method == "execute" else False,
        )
        if plot:
            seq.plot()

        if reset_awg_and_capunits:
            qubits = {Target.qubit_label(target) for target in seq.labels}
            self.reset_awg_and_capunits(qubits=qubits)

        edge_and_spectators: dict[tuple[str, str], list[str]] = {}
        for edge in target_edges:
            edge_spectators: list[str] = []
            for node in edge:
                node_spectators = graph.neighbors(node)
                for spectator in node_spectators:
                    if spectator not in edge:
                        edge_spectators.append(spectator)
            edge_and_spectators[edge] = edge_spectators
            if plot:
                print(f"Edge: {edge}, Spectators: {edge_spectators}")

        edge_sbits_result: dict[tuple[str, str], dict[str, dict]] = {
            edge: {} for edge in target_edges
        }

        edge_sbits_pauli_counts: dict[
            tuple[str, str], dict[str, dict[str, dict[str, int]]]
        ] = {
            # edge: {
            #     sbits: {
            #         pauli: {
            #             ebits: count,
            #         }
            #     },
            # }
            edge: {}
            for edge in target_edges
        }

        edge_sbits_pauli_probabilities: dict[
            tuple[str, str], dict[str, dict[str, dict[str, float]]]
        ] = {
            # edge: {
            #     sbits: {
            #         pauli: {
            #             ebits: probability,
            #         }
            #     },
            # }
            edge: {}
            for edge in target_edges
        }

        for pauli0, pauli1 in tqdm(
            product(["X", "Y", "Z"], repeat=2),
        ):
            pauli_basis = f"{pauli0}{pauli1}"

            bases = {}
            for node0, node1 in target_edges:
                bases[node0] = pauli0
                bases[node1] = pauli1

            if method == "execute":
                result = self.execute(
                    self.create_graph_sequence(
                        graph=graph,
                        bases=bases,
                        with_readout_pulses=True,
                    ),
                    mode="single",
                    shots=shots,
                    interval=interval,
                    reset_awg_and_capunits=False,
                )
            else:
                result = self.measure(
                    self.create_graph_sequence(
                        graph=graph,
                        bases=bases,
                        with_readout_pulses=False,
                    ),
                    mode="single",
                    shots=shots,
                    interval=interval,
                    reset_awg_and_capunits=False,
                )

            for edge, spectators in edge_and_spectators.items():
                target_labels = list(edge) + spectators
                mitigated_counts = result.get_mitigated_counts(target_labels)
                n_spectators = len(spectators)

                if use_all_spectator_pattern:
                    spectators_bits = [
                        "".join(bits) for bits in product("01", repeat=n_spectators)
                    ]
                else:
                    spectators_bits = ["0" * n_spectators]

                for sbits in spectators_bits:
                    if sbits not in edge_sbits_pauli_counts[edge]:
                        edge_sbits_pauli_counts[edge][sbits] = {}
                    if sbits not in edge_sbits_pauli_probabilities[edge]:
                        edge_sbits_pauli_probabilities[edge][sbits] = {}

                sbits_counts: dict[str, dict[str, int]] = {
                    sbits: {} for sbits in spectators_bits
                }

                for bits, count in mitigated_counts.items():
                    ebits = bits[:2]
                    sbits = bits[2:]
                    if use_all_spectator_pattern:
                        sbits_counts[sbits][ebits] = count
                    else:
                        if sbits == "0" * n_spectators:
                            sbits_counts[sbits][ebits] = count

                for sbits, counts in sbits_counts.items():
                    edge_sbits_pauli_counts[edge][sbits][pauli_basis] = counts

                    total_count = sum(counts.values())
                    edge_sbits_pauli_probabilities[edge][sbits][pauli_basis] = {
                        ebits: count / total_count if total_count > 0 else 0.0
                        for ebits, count in counts.items()
                    }

        paulis = {
            "I": np.array([[1, 0], [0, 1]]),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
        }

        def _compute_expected_values_and_rho_from_probs(
            probabilities: dict[str, dict[str, float]],
            use_mle: bool,
        ) -> tuple[dict[str, float], np.ndarray]:
            expected_values: dict[str, float] = {}
            rho_local = np.zeros((4, 4), dtype=np.complex128)
            for basis0, pauli0 in paulis.items():
                for basis1, pauli1 in paulis.items():
                    pauli_basis = f"{basis0}{basis1}"
                    if pauli_basis == "II":
                        p = probabilities["ZZ"]
                        e = p["00"] + p["01"] + p["10"] + p["11"]
                    elif pauli_basis in ["IX", "IY", "IZ"]:
                        p = probabilities[f"Z{basis1}"]
                        e = p["00"] - p["01"] + p["10"] - p["11"]
                    elif pauli_basis in ["XI", "YI", "ZI"]:
                        p = probabilities[f"{basis0}Z"]
                        e = p["00"] + p["01"] - p["10"] - p["11"]
                    else:
                        p = probabilities[pauli_basis]
                        e = p["00"] - p["01"] - p["10"] + p["11"]
                    pauli_matrix = np.kron(pauli0, pauli1)
                    rho_local += e * pauli_matrix
                    expected_values[pauli_basis] = e
            if use_mle:
                rho_local = mle_fit_density_matrix(expected_values)
            else:
                rho_local = rho_local / 4
            return expected_values, rho_local

        def _compute_negativity_from_probs(
            probabilities: dict[str, dict[str, float]],
            use_mle: bool,
        ) -> tuple[float, np.ndarray, np.ndarray]:
            _, rho_local = _compute_expected_values_and_rho_from_probs(
                probabilities=probabilities,
                use_mle=use_mle,
            )
            rho_pt_local = self.partial_transpose(rho_local)
            eigvals_local = np.linalg.eigvalsh(rho_pt_local)
            negativity_local = float(np.sum(np.abs(eigvals_local[eigvals_local < 0])))
            return negativity_local, rho_local, eigvals_local

        def _bootstrap_negativity(
            counts: dict[str, dict[str, int]],
            B: int,
            use_mle: bool,
        ) -> tuple[float, float, tuple[float, float]]:
            # Dirichlet bootstrap on the 4-outcome distributions for each Pauli basis
            rng = np.random.default_rng()
            samples: list[float] = []
            bases_keys = list(counts.keys())
            for _ in range(B):
                probs_b: dict[str, dict[str, float]] = {}
                for pb in bases_keys:
                    c = counts[pb]
                    # Order the 2-qubit outcome as 00,01,10,11
                    vec = np.array(
                        [
                            max(c.get("00", 0.0), 0.0),
                            max(c.get("01", 0.0), 0.0),
                            max(c.get("10", 0.0), 0.0),
                            max(c.get("11", 0.0), 0.0),
                        ],
                        dtype=float,
                    )
                    total = vec.sum()
                    if total <= 0:
                        vec = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
                    else:
                        vec = vec / total
                    # Dirichlet concentration ~ shots_eff * p
                    shots_eff = int(total)
                    alpha = vec * max(shots_eff, 1) + 1e-3
                    vec_b = rng.dirichlet(alpha)
                    probs_b[pb] = {
                        "00": float(vec_b[0]),
                        "01": float(vec_b[1]),
                        "10": float(vec_b[2]),
                        "11": float(vec_b[3]),
                    }
                neg_b, _, _ = _compute_negativity_from_probs(
                    probabilities=probs_b,
                    use_mle=use_mle,
                )
                samples.append(neg_b)
            samples_arr = np.asarray(samples)
            mean_b = float(np.mean(samples_arr))
            std_b = float(np.std(samples_arr, ddof=1))
            lo, hi = np.percentile(samples_arr, [16.0, 84.0])
            return mean_b, std_b, (float(lo), float(hi))

        for edge, sbits_pauli_probabilities in edge_sbits_pauli_probabilities.items():
            for sbits, pauli_probabilities in sbits_pauli_probabilities.items():
                if sbits not in edge_sbits_result[edge]:
                    edge_sbits_result[edge][sbits] = {}

                # Compute expected values and density matrix
                expected_values, rho = _compute_expected_values_and_rho_from_probs(
                    probabilities=pauli_probabilities,
                    use_mle=mle_fit,
                )

                rho_pt = self.partial_transpose(rho)
                eigvals = np.linalg.eigvalsh(rho_pt)
                negativity = np.sum(np.abs(eigvals[eigvals < 0]))

                # Optional bootstrap error estimation
                if n_bootstrap is not None and n_bootstrap > 0:
                    pauli_counts = edge_sbits_pauli_counts[edge][sbits]
                    _, neg_std, (neg_lo, neg_hi) = _bootstrap_negativity(
                        counts=pauli_counts,
                        B=n_bootstrap,
                        use_mle=bootstrap_mle if mle_fit else False,
                    )
                else:
                    neg_std, neg_lo, neg_hi = None, None, None

                if plot:
                    print(f"{edge[0]}-{edge[1]} ({sbits}) : Negativity = {negativity}")

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=("Abs", "Phase"),
                    horizontal_spacing=0.26,
                )
                fig.add_trace(
                    go.Heatmap(
                        z=np.abs(rho),
                        zmin=0,
                        zmax=1,
                        colorscale="Hot_r",
                        colorbar=dict(
                            title="Abs",
                            x=0.37,
                            y=0.5,
                            thickness=15,
                            tickvals=[0, 0.5, 1],
                            ticktext=["0", "0.5", "1"],
                        ),
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Heatmap(
                        z=np.angle(rho),
                        zmin=-np.pi,
                        zmax=np.pi,
                        colorscale="Edge",
                        colorbar=dict(
                            title="Phase (rad)",
                            x=1.0,
                            y=0.5,
                            thickness=15,
                            tickvals=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                            ticktext=["-π", "-π/2", "0", "π/2", "π"],
                        ),
                    ),
                    row=1,
                    col=2,
                )

                tickvals = np.arange(4)
                ticktext = [f"{i:0{2}b}" for i in tickvals]
                tick_style = dict(
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickangle=0,
                )

                fig.update_xaxes(tick_style, row=1, col=1)
                fig.update_yaxes(
                    dict(**tick_style, autorange="reversed", scaleanchor="x1"),
                    row=1,
                    col=1,
                )
                fig.update_xaxes(tick_style, row=1, col=2)
                fig.update_yaxes(
                    dict(**tick_style, autorange="reversed", scaleanchor="x2"),
                    row=1,
                    col=2,
                )
                fig.update_layout(
                    title=dict(
                        text=f"Negativity of graph state: 𝒩 = {negativity:.3f}",
                        subtitle=dict(
                            text=f"edge: {edge}, spectators: ({', '.join(edge_and_spectators[edge])}) = '{sbits}'",
                        ),
                    ),
                    margin=dict(t=110),
                    width=600,
                    height=342,
                )

                if plot:
                    fig.show(
                        config={
                            "toImageButtonOptions": {
                                "format": "png",
                                "scale": 3,
                            },
                        }
                    )

                edge_sbits_result[edge][sbits]["expected_values"] = expected_values
                edge_sbits_result[edge][sbits]["density_matrix"] = rho
                edge_sbits_result[edge][sbits]["partial_transpose"] = rho_pt
                edge_sbits_result[edge][sbits]["negativity"] = negativity
                edge_sbits_result[edge][sbits]["eigenvalues"] = eigvals
                edge_sbits_result[edge][sbits]["figure"] = fig
                edge_sbits_result[edge][sbits]["negativity_std"] = neg_std
                edge_sbits_result[edge][sbits]["negativity_ci"] = (neg_lo, neg_hi)
                # Note: CI is approximately 68% via 16–84th percentiles.

        result = {"best": {edge: {} for edge in target_edges}}

        for edge, sbits_results in edge_sbits_result.items():
            best_result = max(
                sbits_results.values(),
                key=lambda x: x["negativity"] if "negativity" in x else 0,
            )
            result["best"][edge] = best_result

        result["all"] = edge_sbits_result

        return result

    def visualize_graph(
        self,
        G: nx.Graph,
        *,
        title: str | None = None,
        property: str = "fidelity",
        show_labels: bool = False,
        show_data: bool = True,
    ) -> None:
        node_values = {node: 1 for node in G.nodes()}
        edge_values = {}
        edge_texts = {}
        if show_data:
            for u, v, data in G.edges(data=True):
                value = data.get(property)
                if property == "fidelity":
                    text = f"{value * 1e2:.1f}" if value is not None else "N/A"
                else:
                    text = f"{value:.2f}" if value is not None else "N/A"
                if value is not None:
                    edge_values[f"{u}-{v}"] = value
                    edge_texts[f"{u}-{v}"] = text
        else:
            edge_values = {f"{edge[0]}-{edge[1]}": 1 for edge in G.edges()}

        chip_graph = self.quantum_system.chip_graph
        chip_graph.plot_graph_data(
            directed=False,
            title=title or "Chip graph",
            edge_values=edge_values,
            edge_texts=edge_texts if show_labels else None,
            edge_color="turquoise" if not show_data else None,
            node_color="white",
            node_linecolor="ghostwhite",
            node_textcolor="ghostwhite",
            node_overlay=True,
            node_overlay_values=node_values,
            node_overlay_color="ghostwhite",
            node_overlay_linecolor="black",
            node_overlay_textcolor="black",
        )

    def measure_graph_state(
        self,
        graph: nx.Graph,
        *,
        mle_fit: bool = True,
        use_all_spectator_pattern: bool = True,
        shots: int = 3000,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        method: str = "execute",
        reset_awg_and_capunits: bool = True,
        n_bootstrap: int | None = 200,
        bootstrap_mle: bool = False,
    ) -> Result:
        if plot:
            seq = self.create_graph_sequence(
                graph=graph,
                with_readout_pulses=False,
            )
            seq.plot(
                title=f"Graph state preparation sequence for {len(graph.nodes())} qubits",
                n_samples=1000,
            )

        negativities = {}
        figures = {}
        negativity_errors: dict[tuple[str, str], float] = {}
        rounds = self.create_measurement_rounds(graph, plot=False)
        for round, target_edges in rounds.items():
            print(f"[{round + 1}/{len(rounds)}] Measuring edges in round #{round + 1}")

            if plot:
                G = nx.Graph()
                for u, v in target_edges:
                    G.add_node(u)
                    G.add_node(v)
                    G.add_edge(u, v)

                node_values = {node: 1.0 for node in graph.nodes()}
                edge_values = {f"{u}-{v}": 1.0 for u, v in graph.edges()}
                edge_overlay_values = {f"{u}-{v}": 1.0 for u, v in G.edges()}

                chip_graph = self.quantum_system.chip_graph
                chip_graph.plot_graph_data(
                    directed=False,
                    title=f"Measurement round : #{round + 1}",
                    edge_values=edge_values,
                    edge_color="#eef",
                    edge_overlay=True,
                    edge_overlay_values=edge_overlay_values,
                    edge_overlay_color="turquoise",
                    node_color="white",
                    node_linecolor="ghostwhite",
                    node_textcolor="ghostwhite",
                    node_overlay=True,
                    node_overlay_values=node_values,
                    node_overlay_color="ghostwhite",
                    node_overlay_linecolor="black",
                    node_overlay_textcolor="black",
                )

            result = self._measure_graph_state(
                graph=graph,
                target_edges=target_edges,
                mle_fit=mle_fit,
                use_all_spectator_pattern=use_all_spectator_pattern,
                shots=shots,
                interval=interval,
                plot=False,
                method=method,
                reset_awg_and_capunits=reset_awg_and_capunits,
                n_bootstrap=n_bootstrap,
                bootstrap_mle=bootstrap_mle,
            )
            for edge, data in result["best"].items():
                negativities[edge] = data["negativity"]
                figures[edge] = data["figure"]
                if "negativity_std" in data and data["negativity_std"] is not None:
                    negativity_errors[edge] = float(data["negativity_std"]) * 2
                data["figure"].show(
                    config={
                        "toImageButtonOptions": {
                            "format": "png",
                            "scale": 3,
                        },
                    }
                )

        negativities = dict(
            sorted(negativities.items(), key=lambda item: item[1], reverse=False)
        )

        negativities_max = max(negativities.values())
        negativities_min = min(negativities.values())
        negativities_avg = np.mean(list(negativities.values()))
        negativities_std = np.std(list(negativities.values()))
        negativities_med = np.median(list(negativities.values()))

        nonzero_edges = {
            edge: negativity
            for edge, negativity in negativities.items()
            if negativity - negativity_errors.get(edge, 0.0) > 0
        }

        if plot:
            print(f"Statistics of {len(negativities)} edges:")
            print(f"  max: {negativities_max:.3f}")
            print(f"  min: {negativities_min:.3f}")
            print(f"  med: {negativities_med:.3f}")
            print(f"  avg: {negativities_avg:.3f}")
            print(f"  std: {negativities_std:.3f}")
            print("Negativities of edges:")
            for edge, negativity in negativities.items():
                if n_bootstrap:
                    print(
                        f"  {edge[0]}-{edge[1]}: {negativity:.3f} ± {negativity_errors.get(edge, 0.0):.3f}"
                    )
                else:
                    print(f"  {edge[0]}-{edge[1]}: {negativity:.3f}")

            x = [f"{edge[0]}-{edge[1]}" for edge in negativities]
            y = [negativity for negativity in negativities.values()]
            y_err = [negativity_errors.get(edge, 0.0) for edge in negativities]

            min_y = min(
                0,
                min(
                    [
                        negativity - negativity_errors.get(edge, 0.0)
                        for edge, negativity in negativities.items()
                    ]
                ),
            )
            max_y = max(
                0.55,
                max(
                    [
                        negativity + negativity_errors.get(edge, 0.0)
                        for edge, negativity in negativities.items()
                    ]
                ),
            )

            fig = go.Figure(
                layout=go.Layout(
                    title=f"Negativities of {len(graph.nodes())}-qubit graph state",
                    xaxis=dict(
                        title="Edges",
                        title_standoff=25,
                        tickangle=90,
                        tickmode="array",
                        tickvals=list(range(len(x))),
                        ticktext=x,
                        tickfont=dict(size=10),
                    ),
                    yaxis=dict(
                        title="Negativity",
                        range=[min_y, max_y],
                        tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        ticktext=["0", "0.1", "0.2", "0.3", "0.4", "0.5"],
                    ),
                    width=800,
                    height=400,
                    margin=dict(l=70, r=70, t=90, b=100),
                )
            )
            fig.add_scatter(
                x=x,
                y=y,
                error_y=dict(
                    type="data",
                    array=y_err,
                )
                if n_bootstrap
                else None,
            )
            fig.show(
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 3,
                    },
                }
            )

            nonzero_graph = nx.Graph()
            for edge, negativity in nonzero_edges.items():
                nonzero_graph.add_edge(edge[0], edge[1], negativity=negativity)

            components = nx.connected_components(nonzero_graph)
            n_max = max(len(c) for c in components)

            self.visualize_graph(
                nonzero_graph,
                title=f"Entangled qubits : N (max) = {n_max}",
                property="negativity",
                show_labels=True,
                show_data=True,
            )

        return Result(
            data={
                "negativities": negativities,
                "negativity_errors": negativity_errors,
                "negativities_max": negativities_max,
                "negativities_min": negativities_min,
                "negativities_med": negativities_med,
                "negativities_avg": negativities_avg,
                "negativities_std": negativities_std,
                "nonzero_edges": nonzero_edges,
                "figures": figures,
            }
        )

    def _canonical_edge(
        self, edge: str | tuple[int | str, int | str]
    ) -> tuple[str, str]:
        if isinstance(edge, str):
            qubits = tuple(edge.split("-"))
            return (qubits[0], qubits[1])
        else:
            qubits = tuple(edge)
            if isinstance(qubits[0], int):
                qubit0 = self.quantum_system.get_qubit(qubits[0]).label
            else:
                qubit0 = qubits[0]
            if isinstance(qubits[1], int):
                qubit1 = self.quantum_system.get_qubit(qubits[1]).label
            else:
                qubit1 = qubits[1]
            return (qubit0, qubit1)

    def measure_bell_state_fidelities(
        self,
        targets: Collection[str | tuple[int | str, int | str]] | None = None,
        *,
        unavailable_pairs: Collection[str | tuple[int | str, int | str]] | None = None,
        readout_mitigation: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = False,
        save_data: bool = True,
        save_path: Path | str | None = None,
    ) -> Result:
        # TODO: move this to an appropriate location
        fidelities = {}

        if targets is None:
            target_pairs = self.cr_pairs
        else:
            target_pairs = [self._canonical_edge(target) for target in targets]

        if unavailable_pairs is None:
            unavailable_pairs = []
        else:
            unavailable_pairs = [
                self._canonical_edge(target) for target in unavailable_pairs
            ]

        for pair in target_pairs:
            if pair in unavailable_pairs:
                print(f"Skipping unavailable pair: {pair}")
                continue
            try:
                label = f"{pair[0]}-{pair[1]}"
                result = self.bell_state_tomography(
                    *pair,
                    readout_mitigation=readout_mitigation,
                    shots=shots,
                    interval=interval,
                )
                fidelities[label] = result["fidelity"]
            except Exception as e:
                print(f"Failed for pair {label}: {e}")

        sorted_fidelities = dict(
            sorted(fidelities.items(), key=lambda x: x[1], reverse=True)
        )

        if plot:
            n_pairs = len(sorted_fidelities)
            x = [label for label in sorted_fidelities]
            y = [fidelity for fidelity in sorted_fidelities.values()]
            fig = go.Figure(
                layout=go.Layout(
                    title="Fidelities",
                    xaxis=dict(
                        title="Edges",
                        tickangle=45,
                        tickmode="array",
                        tickvals=list(range(len(x))),
                        ticktext=x,
                        tickfont=dict(size=10),
                    ),
                    yaxis=dict(
                        title="Fidelity",
                        range=[0, 1],
                        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
                        ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"],
                    ),
                    width=n_pairs * 15 + 150,
                    height=400,
                    margin=dict(l=70, r=70, t=90, b=100),
                )
            )
            fig.add_bar(x=x, y=y)
            fig.show(
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 3,
                    },
                }
            )

        if save_data:
            self.save_property(
                "bell_state_fidelity",
                sorted_fidelities,
                save_path=save_path,
            )

        return Result(data=sorted_fidelities)

    def measure_bell_states(
        self,
        targets: Collection[str | tuple[int | str, int | str]] | None = None,
        *,
        unavailable_pairs: Collection[str | tuple[int | str, int | str]] | None = None,
        control_basis: str = "Z",
        target_basis: str = "Z",
        readout_mitigation: bool = True,
        in_parallel: bool = True,
        n_cols: int = 6,
        threshold: float = 0.0,
        title: str | None = None,
        plot: bool = True,
        plot_round: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        reset_awg_and_capunits_each_time: bool = True,
    ) -> Result:
        if targets is None:
            try:
                fidelities = self.load_property("bell_state_fidelity")
                target_pairs = [
                    self._canonical_edge(label)
                    for label, fidelity in fidelities.items()
                    if fidelity >= threshold
                ]
            except FileNotFoundError:
                target_pairs = self.cr_pairs
        else:
            target_pairs = [self._canonical_edge(target) for target in targets]

        if unavailable_pairs is None:
            unavailable_pairs = []
        else:
            unavailable_pairs = [
                self._canonical_edge(target) for target in unavailable_pairs
            ]

        all_edges = sorted(
            [
                (pair[0], pair[1])
                for pair in target_pairs
                if pair not in unavailable_pairs
                and f"{pair[0]}-{pair[1]}" in self.calib_note.cr_params
            ]
        )

        if reset_awg_and_capunits and not reset_awg_and_capunits_each_time:
            qubits = set()
            for edge in all_edges:
                qubits.add(edge[0])
                qubits.add(edge[1])
            self.reset_awg_and_capunits(qubits=qubits)

        n_edges = len(all_edges)
        n_rows = int(np.ceil(n_edges / n_cols))

        edge_indices = {
            edge: (i // n_cols + 1, i % n_cols + 1) for i, edge in enumerate(all_edges)
        }

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"{edge[0]}-{edge[1]}" for edge in all_edges],
        )

        results = {}

        if in_parallel:
            graph = self.quantum_system.chip_graph
            rounds = []
            for edges in graph.strong_edge_coloring():
                batch = []
                for edge in edges:
                    if edge[0] % 4 in [0, 3]:
                        e = self._canonical_edge((edge[0], edge[1]))
                    else:
                        e = self._canonical_edge((edge[1], edge[0]))
                    if e in all_edges:
                        batch.append(e)
                rounds.append(batch)

            for round, edges in tqdm(enumerate(rounds), desc="Measuring Bell states"):
                if plot_round:
                    G = nx.Graph()
                    for u, v in edges:
                        G.add_node(u)
                        G.add_node(v)
                        G.add_edge(u, v)

                    edge_values = {f"{u}-{v}": 1.0 for u, v in all_edges}
                    edge_overlay_values = {f"{u}-{v}": 1.0 for u, v in G.edges()}

                    chip_graph = self.quantum_system.chip_graph
                    chip_graph.plot_graph_data(
                        directed=False,
                        title=f"Measurement round : {round + 1}",
                        edge_values=edge_values,
                        edge_color="#eef",
                        edge_overlay=True,
                        edge_overlay_values=edge_overlay_values,
                        edge_overlay_color="turquoise",
                        node_color="white",
                        node_linecolor="ghostwhite",
                        node_textcolor="ghostwhite",
                        node_overlay=True,
                        # node_overlay_values=node_values,
                        node_overlay_color="ghostwhite",
                        node_overlay_linecolor="black",
                        node_overlay_textcolor="black",
                    )

                with PulseSchedule() as ps:
                    for edge in edges:
                        control_qubit, target_qubit = edge
                        # prepare |+⟩|0⟩
                        ps.add(control_qubit, self.y90(control_qubit))

                        # create |0⟩|0⟩ + |1⟩|1⟩
                        ps.call(
                            self.cnot(
                                control_qubit,
                                target_qubit,
                                only_low_to_high=True,
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
                    reset_awg_and_capunits=reset_awg_and_capunits_each_time,
                )

                for edge in edges:
                    row, col = edge_indices[edge]

                    control_qubit, target_qubit = edge
                    basis_labels = result.get_basis_labels(edge)
                    prob_dict_raw = result.get_probabilities(edge)
                    # Ensure all basis labels are present in the raw probabilities
                    prob_dict_raw = {
                        label: prob_dict_raw.get(label, 0) for label in basis_labels
                    }
                    prob_dict_mitigated = result.get_mitigated_probabilities(edge)

                    labels = [f"|{i}⟩" for i in prob_dict_raw.keys()]
                    prob_arr_raw = np.array(list(prob_dict_raw.values()))
                    prob_arr_mitigated = np.array(list(prob_dict_mitigated.values()))

                    if readout_mitigation:
                        prob_arr = prob_arr_mitigated
                    else:
                        prob_arr = prob_arr_raw

                    results[f"{edge[0]}-{edge[1]}"] = {
                        "raw_probabilities": prob_arr_raw,
                        "mitigated_probabilities": prob_arr_mitigated,
                    }

                    fig.add_bar(
                        x=labels,
                        y=prob_arr,
                        row=row,
                        col=col,
                        marker_color=COLORS[0],
                    )
        else:
            for edge in tqdm(all_edges, total=len(all_edges)):
                row, col = edge_indices[edge]
                labels = [f"|{i}⟩" for i in ["00", "01", "10", "11"]]
                result = self.measure_bell_state(
                    *edge,
                    shots=shots,
                    plot=False,
                    save_image=False,
                    reset_awg_and_capunits=reset_awg_and_capunits_each_time,
                )
                if readout_mitigation:
                    prob_arr = result["mitigated"]
                else:
                    prob_arr = result["raw"]
                results[f"{edge[0]}-{edge[1]}"] = {
                    "raw_probabilities": result["raw"],
                    "mitigated_probabilities": result["mitigated"],
                }
                fig.add_bar(
                    x=labels,
                    y=prob_arr,
                    row=row,
                    col=col,
                    marker_color=COLORS[0],
                )

        fig_subtitle = f"{n_edges} pairs, {shots} shots"
        if in_parallel:
            fig_subtitle += ", run in parallel"
        else:
            fig_subtitle += ", run sequentially"

        fig.update_layout(
            title=dict(
                text=title or "Bell state measurement",
                subtitle=dict(
                    text=fig_subtitle,
                    font_size=16,
                ),
                y=0.98,
                yanchor="top",
                font_size=22,
            ),
            height=120 * n_rows + 200,
            width=180 * n_cols + 80,
            showlegend=False,
            margin=dict(l=40, r=40, t=160, b=40),
        )
        fig.update_yaxes(
            range=[0, 0.6],
            tickvals=[0, 0.25, 0.5],
            ticktext=["0", "0.25", "0.5"],
        )
        fig.update_annotations(
            font_size=15,
            yshift=8,
        )
        if plot:
            fig.show(
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 3,
                    },
                }
            )

        return Result(
            data={
                "data": results,
                "figure": fig,
            }
        )

    def rzx_gate_property(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        angle_arr: np.ndarray = np.linspace(np.pi / 18, 4 * np.pi / 9, 8),
        measurement_times: int = 10,
    ) -> Result:
        RAD_TO_DEG = 180 / np.pi

        def cartesian_to_spherical(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x) * RAD_TO_DEG
            phi = np.arccos(z / r) * RAD_TO_DEG if r != 0 else 0
            return r, theta, phi

        result_rzx_angle = []
        for angle in tqdm(angle_arr, leave=False):
            results = []
            for _ in tqdm(range(measurement_times), leave=False):
                result = self.state_tomography(
                    self.rzx(
                        control_qubit=control_qubit,
                        target_qubit=target_qubit,
                        angle=angle,
                    )
                )
                x, y, z = result[target_qubit]
                r, theta, phi = cartesian_to_spherical(x, y, z)
                results.append([r, theta, phi])
            result_array = np.array(results)
            mean = np.mean(result_array[:, 2])
            std = np.std(result_array[:, 2])
            result_rzx_angle.append([angle, mean, std])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.array(result_rzx_angle).T[0] * RAD_TO_DEG,
                y=np.array(result_rzx_angle).T[1],
                marker=dict(color=COLORS[1]),
                error_y=dict(
                    type="data", array=np.array(result_rzx_angle).T[2], color=COLORS[1]
                ),
                name="Measured",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 90],
                y=[0, 90],
                mode="lines",
                line=dict(color=COLORS[0], dash="dash"),
                name="Ideal",
            )
        )
        fig.update_layout(
            title=f"Sweep result : {control_qubit}-{target_qubit}",
            xaxis=dict(
                title="Angle (deg)",
                range=(0, 90),
                tickvals=angle_arr * RAD_TO_DEG,
                dtick=5,
                gridcolor="gray",
                gridwidth=3,
                griddash="dot",
            ),
            yaxis=dict(
                title="Z Angle (deg)",
                range=(0, 90),
                tickvals=angle_arr * RAD_TO_DEG,
                dtick=5,
                gridcolor="gray",
                gridwidth=3,
                griddash="dot",
            ),
        )
        fig.show()
        return Result(
            data={
                "data": results,
                "figure": fig,
            }
        )
