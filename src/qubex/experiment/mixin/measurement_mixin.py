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
    MeasureData,
    MeasureResult,
    MultipleMeasureResult,
    StateClassifier,
    StateClassifierGMM,
    StateClassifierKMeans,
)
from ...measurement.measurement import (
    DEFAULT_CAPTURE_DELAY,
    DEFAULT_CAPTURE_WINDOW,
    DEFAULT_INTERVAL,
    DEFAULT_READOUT_DURATION,
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
    RABI_TIME_RANGE,
)
from ..experiment_result import ExperimentResult, RabiData, SweepData
from ..protocol import BaseProtocol, MeasurementProtocol

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
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        add_last_measurement: bool = False,
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
    ) -> MultipleMeasureResult:
        return self.measurement.execute(
            schedule=schedule,
            mode=mode,
            shots=shots,
            interval=interval,
            add_last_measurement=add_last_measurement,
            capture_window=capture_window,
            capture_margin=capture_margin,
            readout_duration=readout_duration,
            readout_amplitudes=readout_amplitudes,
            readout_ramptime=readout_ramptime,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
        )

    def measure(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        frequencies: Optional[dict[str, float]] = None,
        initial_states: dict[str, str] | None = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        control_window: float | None = None,
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        plot: bool = False,
        capture_delay_words: int | None = None,
        _use_sequencer_execute: bool = True,
    ) -> MeasureResult:
        control_window = control_window or self.control_window
        capture_window = capture_window or self.capture_window
        capture_margin = capture_margin or self.capture_margin
        readout_duration = readout_duration or self.readout_duration
        waveforms: dict[str, NDArray[np.complex128]] = {}

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

        if frequencies is None:
            result = self.measurement.measure(
                waveforms=waveforms,
                mode=mode,
                shots=shots,
                interval=interval,
                control_window=control_window,
                capture_window=capture_window,
                capture_margin=capture_margin,
                readout_duration=readout_duration,
                readout_amplitudes=readout_amplitudes,
                readout_ramptime=readout_ramptime,
                readout_drag_coeff=readout_drag_coeff,
                readout_ramp_type=readout_ramp_type,
                capture_delay_words=capture_delay_words,
                _use_sequencer_execute=_use_sequencer_execute,
            )
        else:
            with self.modified_frequencies(frequencies):
                result = self.measurement.measure(
                    waveforms=waveforms,
                    mode=mode,
                    shots=shots,
                    interval=interval,
                    control_window=control_window,
                    capture_window=capture_window,
                    capture_margin=capture_margin,
                    readout_duration=readout_duration,
                    readout_amplitudes=readout_amplitudes,
                    capture_delay_words=capture_delay_words,
                    _use_sequencer_execute=_use_sequencer_execute,
                )
        if plot:
            result.plot()
        return result

    def measure_readout_waveform(
        self,
        *,
        target: str | None = None,
        frequency: float | None = None,
        amplitude: float | None = None,
        duration: float = DEFAULT_READOUT_DURATION,
        capture_window: float = DEFAULT_CAPTURE_WINDOW,
        capture_delay: float = DEFAULT_CAPTURE_DELAY,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = False,
    ) -> MeasureData:
        if target is None:
            target = self.qubit_labels[0]
        qubit = Target.qubit_label(target)
        resonator = Target.read_label(target)
        if frequency is None:
            frequency = self.resonators[qubit].frequency
        if amplitude is None:
            amplitude = self.params.readout_amplitude[qubit]
        capture_delay_words = int(capture_delay // 8)
        with self.modified_frequencies({resonator: frequency}):
            result = self.measure(
                sequence={qubit: np.zeros(0)},
                mode=mode,
                shots=shots,
                interval=interval,
                readout_duration=duration,
                readout_amplitudes={qubit: amplitude},
                capture_window=capture_window,
                capture_delay_words=capture_delay_words,
                _use_sequencer_execute=False,
            ).data[qubit]
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
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        control_window: float | None = None,
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
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
                    ps.add(target, self.hpi_pulse[target].repeated(2))
                elif state == "f":
                    ps.add(target, self.hpi_pulse[target].repeated(2))
                    ps.barrier()
                    ef_label = Target.ef_label(target)
                    ps.add(ef_label, self.ef_hpi_pulse[ef_label].repeated(2))

        return self.measure(
            sequence=ps,
            mode=mode,
            shots=shots,
            interval=interval,
            control_window=control_window,
            capture_window=capture_window,
            capture_margin=capture_margin,
            readout_duration=readout_duration,
            readout_amplitudes=readout_amplitudes,
            plot=plot,
        )

    def sweep_parameter(
        self,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        *,
        sweep_range: ArrayLike,
        repetitions: int = 1,
        frequencies: dict[str, float] | None = None,
        rabi_level: Literal["ge", "ef"] = "ge",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        control_window: float | None = None,
        capture_window: float | None = None,
        capture_margin: float | None = None,
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
                    sequence(param).repeated(repetitions).get_sampled_sequences()  # type: ignore
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

        # TODO: workaround for the first measurement problem
        sequences = [sequences[0]] + sequences

        with self.modified_frequencies(frequencies):
            for idx, seq in enumerate(sequences):
                result = self.measure(
                    seq,
                    mode="avg",
                    shots=shots,
                    interval=interval,
                    control_window=control_window or self.control_window,
                    capture_window=capture_window or self.capture_window,
                    capture_margin=capture_margin or self.capture_margin,
                )
                if idx == 0:
                    # TODO: workaround for the first measurement problem
                    continue
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

    def repeat_sequence(
        self,
        sequence: TargetMap[Waveform] | PulseSchedule,
        *,
        repetitions: int = 20,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
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
            sweep_range=np.arange(repetitions + 1),
            sequence=repeated_sequence,
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
        time_range: ArrayLike = RABI_TIME_RANGE,
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
        time_range: ArrayLike = RABI_TIME_RANGE,
        is_damped: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RabiData]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        time_range = np.asarray(time_range)

        ef_labels = [Target.ef_label(target) for target in targets]
        ef_targets = [self.targets[ef] for ef in ef_labels]

        ampl = self.params.control_amplitude
        amplitudes = {ef.label: ampl[ef.qubit] / np.sqrt(2) for ef in ef_targets}

        rabi_data = {}
        rabi_params = {}
        for label in ef_labels:
            data = self.ef_rabi_experiment(
                amplitudes={label: amplitudes[label]},
                time_range=time_range,
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
        time_range: ArrayLike = RABI_TIME_RANGE,
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
            # TODO: Fix fit_rabi to support ramptime
            ramptime = 0.0

        effective_time_range = time_range + ramptime

        # target frequencies
        if frequencies is None:
            frequencies = {
                target: self.targets[target].frequency for target in amplitudes
            }

        # rabi sequence with rect pulses of duration T
        def rabi_sequence(T: int) -> PulseSchedule:
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
                plot=plot,
                is_damped=is_damped,
            )
            rabi_params[target] = fit_result["rabi_param"]

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
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[RabiData]:
        amplitudes = {
            Target.ef_label(label): amplitude for label, amplitude in amplitudes.items()
        }
        ge_labels = [Target.ge_label(label) for label in amplitudes]
        ef_labels = [Target.ef_label(label) for label in amplitudes]
        ef_targets = [self.targets[ef] for ef in ef_labels]

        # drive time range
        time_range = np.array(time_range, dtype=np.float64)

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
                    ps.add(ge, self.hpi_pulse[ge].repeated(2))
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

        # sweep data with the ef labels
        sweep_data = {ef.label: sweep_result.data[ef.qubit] for ef in ef_targets}

        # fit the Rabi oscillation
        rabi_params = {
            target: fitting.fit_rabi(
                target=target,
                times=data.sweep_range,
                data=data.data,
                plot=plot,
                is_damped=is_damped,
            )["rabi_param"]
            for target, data in sweep_data.items()
        }

        # store the Rabi parameters if necessary
        if store_params:
            self.store_rabi_params(rabi_params)

        # create the Rabi data for each target
        rabi_data = {
            target: RabiData(
                target=target,
                data=data.data,
                time_range=time_range,
                rabi_param=rabi_params[target],
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

    def measure_state_distribution(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_states: Literal[2, 3] = 2,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        control_window: float | None = None,
        capture_window: float | None = None,
        capture_margin: float | None = None,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
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
                control_window=control_window,
                capture_window=capture_window,
                capture_margin=capture_margin,
                readout_duration=readout_duration,
                readout_amplitudes=readout_amplitudes,
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
        n_states: Literal[2, 3] = 2,
        save_classifier: bool = True,
        save_dir: Path | str | None = None,
        shots: int = 8192,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        results = self.measure_state_distribution(
            targets=targets,
            n_states=n_states,
            shots=shots,
            interval=interval,
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
                target: StateClassifierGMM.fit(data[target]) for target in targets
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
            }
            for target in targets
        }

        return {
            "readout_fidelties": fidelities,
            "average_readout_fidelity": average_fidelities,
            "measure_results": results,
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

            measure_result = self.measure(
                ps,
                shots=shots,
                interval=interval,
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
        plot: bool = True,
    ) -> dict[str, NDArray[np.float64]]:
        buffer: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for sequence in tqdm(sequences):
            state_vectors = self.state_tomography(
                sequence=sequence,
                x90=x90,
                initial_state=initial_state,
                shots=shots,
                interval=interval,
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

    def pulse_tomography(
        self,
        sequence: PulseSchedule | TargetMap[Waveform] | TargetMap[IQArray],
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        n_samples: int | None = 100,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
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

        def partial_waveform(waveform: Waveform, index: int) -> Waveform:
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
                current_index = 0
                pulse_array = PulseArray([])

                # Iterate over the objects in the PulseArray.
                for obj in waveform.elements:
                    # If the object is a PhaseShift gate, we can simply add it to the array.
                    if isinstance(obj, PhaseShift):
                        pulse_array.add(obj)
                        continue
                    elif isinstance(obj, Pulse):
                        # If the object is a Pulse, we need to check the index.
                        if index - current_index == 0:
                            continue
                        elif index - current_index < obj.length:
                            pulse = Pulse(obj.values[0 : index - current_index - 1])
                            pulse_array.add(pulse)
                            break
                        else:
                            pulse_array.add(obj)
                            current_index += obj.length
                    else:
                        # NOTE: PulseArray should be flattened before calling this function.
                        logger.error(f"Invalid type: {type(obj)}")
                return pulse_array
            else:
                logger.error(f"Invalid type: {type(waveform)}")
                return waveform

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
                target: partial_waveform(pulse, i)
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

        prob_dict_raw = result.get_probabilities(pair)
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
