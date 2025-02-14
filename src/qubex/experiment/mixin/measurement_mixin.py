from __future__ import annotations

from collections import defaultdict
from typing import Collection, Literal, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from ...analysis import IQPlotter, fitting
from ...analysis import visualization as vis
from ...backend import Target
from ...measurement import (
    MeasureResult,
    StateClassifier,
    StateClassifierGMM,
    StateClassifierKMeans,
)
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import (
    Blank,
    PhaseShift,
    Pulse,
    PulseSchedule,
    PulseSequence,
    Rect,
    Waveform,
)
from ...typing import (
    IQArray,
    ParametricPulseSchedule,
    ParametricWaveformDict,
    TargetMap,
)
from ..experiment_constants import CALIBRATION_SHOTS, RABI_TIME_RANGE, STATE_CENTERS
from ..experiment_result import ExperimentResult, RabiData, SweepData
from ..protocol import BaseProtocol, MeasurementProtocol


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
        interval: int = DEFAULT_INTERVAL,
    ) -> MeasureResult:
        return self.measurement.execute(
            schedule=schedule,
            mode=mode,
            shots=shots,
            interval=interval,
        )

    def measure(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        frequencies: Optional[dict[str, float]] = None,
        initial_states: dict[str, str] | None = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        plot: bool = False,
    ) -> MeasureResult:
        control_window = control_window or self.control_window
        capture_window = capture_window or self.capture_window
        capture_margin = capture_margin or self.capture_margin
        readout_duration = readout_duration or self.readout_duration
        waveforms: dict[str, NDArray[np.complex128]] = {}

        if isinstance(sequence, PulseSchedule):
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
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        readout_duration: int | None = None,
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
        interval: int = DEFAULT_INTERVAL,
        control_window: int | None = None,
        capture_window: int | None = None,
        capture_margin: int | None = None,
        plot: bool = True,
        title: str = "Sweep result",
        xaxis_title: str = "Sweep value",
        yaxis_title: str = "Measured value",
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

        if isinstance(sequence, dict):
            # TODO: this parameter type (dict[str, Callable[..., Waveform]]) will be deprecated
            targets = list(sequence.keys())
            sequences = [
                {
                    target: sequence[target](param).repeated(repetitions).values
                    for target in targets
                }
                for param in sweep_range
            ]

        if callable(sequence):
            if isinstance(sequence(0), PulseSchedule):
                sequences = [
                    sequence(param).repeated(repetitions).get_sampled_sequences()  # type: ignore
                    for param in sweep_range
                ]
            elif isinstance(sequence(0), dict):
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

        generator = self.measurement.measure_batch(
            waveforms_list=sequences,
            mode="avg",
            shots=shots,
            interval=interval,
            control_window=control_window or self.control_window,
            capture_window=capture_window or self.capture_window,
            capture_margin=capture_margin or self.capture_margin,
        )

        with self.modified_frequencies(frequencies):
            for result in generator:
                for target, data in result.data.items():
                    signals[target].append(data.kerneled)
                if plot:
                    plotter.update(signals)

        if plot:
            plotter.show()

        # with self.modified_frequencies(frequencies):
        #     for seq in sequences:
        #         measure_result = self.measure(
        #             sequence=seq,
        #             mode="avg",
        #             shots=shots,
        #             interval=interval,
        #             control_window=control_window,
        #             capture_window=capture_window,
        #             capture_margin=capture_margin,
        #         )
        #         for target, data in measure_result.data.items():
        #             signals[target].append(complex(data.kerneled))
        #         if plot:
        #             plotter.update(signals)

        sweep_data = {
            target: SweepData(
                target=target,
                data=np.array(values),
                sweep_range=sweep_range,
                rabi_param=rabi_params.get(target),
                state_centers=self.state_centers.get(target),
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
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
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[SweepData]:
        def repeated_sequence(N: int) -> PulseSchedule:
            if isinstance(sequence, dict):
                targets = list(sequence.keys())
                with PulseSchedule(targets) as ps:
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
            repetitions=1,
            shots=shots,
            interval=interval,
            plot=plot,
            xaxis_title="Number of repetitions",
        )

        if plot:
            result.plot(normalize=True)

        return result

    def measure_state_distribution(
        self,
        targets: Collection[str] | None = None,
        *,
        n_states: Literal[2, 3] = 2,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> list[MeasureResult]:
        if targets is None:
            targets = self.qubit_labels
        else:
            targets = list(targets)

        states = ["g", "e", "f"][:n_states]
        result = {
            state: self.measure_state(
                {target: state for target in targets},  # type: ignore
                shots=shots,
                interval=interval,
            )
            for state in states
        }
        for target in targets:
            data = {
                f"|{state}⟩": result[state].data[target].kerneled for state in states
            }
            if plot:
                vis.plot_state_distribution(
                    data=data,
                    title=f"State distribution : {target}",
                )
        return list(result.values())

    def build_classifier(
        self,
        targets: str | Collection[str] | None = None,
        *,
        n_states: Literal[2, 3] = 2,
        shots: int = 10000,
        interval: int = DEFAULT_INTERVAL,
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

        self.system_note.put(
            STATE_CENTERS,
            {
                target: {
                    str(state): (center.real, center.imag)
                    for state, center in classifiers[target].centers.items()
                }
                for target in targets
            },
        )

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
        interval: int = DEFAULT_INTERVAL,
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
            Sequence[TargetMap[IQArray]]
            | Sequence[TargetMap[Waveform]]
            | Sequence[PulseSchedule]
        ),
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
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
                vis.display_bloch_sphere(states)

        return result

    def pulse_tomography(
        self,
        waveforms: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        n_samples: int = 100,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> TargetMap[NDArray[np.float64]]:
        self.validate_rabi_params()

        if isinstance(waveforms, PulseSchedule):
            sequences = waveforms.get_sequences()
        else:
            sequences = waveforms

        pulses: dict[str, Waveform] = {}
        pulse_length_set = set()
        for target, waveform in sequences.items():
            if isinstance(waveform, Waveform):
                pulse = waveform
            elif isinstance(waveform, list) or isinstance(waveform, np.ndarray):
                pulse = Pulse(waveform)
            else:
                raise ValueError("Invalid waveform.")
            pulses[target] = pulse
            pulse_length_set.add(pulse.length)
        if len(pulse_length_set) != 1:
            raise ValueError("The lengths of the waveforms must be the same.")

        pulse_length = pulse_length_set.pop()

        if plot:
            if isinstance(waveforms, PulseSchedule):
                waveforms.plot(title="Pulse sequence")
            else:
                for target in pulses:
                    pulses[target].plot(title=f"Waveform : {target}")

        def partial_waveform(waveform: Waveform, index: int) -> Waveform:
            """Returns a partial waveform up to the given index."""

            # If the waveform is a PulseSequence, we need to handle the PhaseShift gate.
            if isinstance(waveform, PulseSequence):
                current_index = 0
                pulse_sequence = PulseSequence([])
                for pulse in waveform._sequence:
                    # If the pulse is a PhaseShift gate, we can simply add it to the sequence.
                    if isinstance(pulse, PhaseShift):
                        pulse_sequence = pulse_sequence.added(pulse)
                        continue
                    # If the pulse is a Pulse and the length is greater than the index, we need to create a partial pulse.
                    elif current_index + pulse.length > index:
                        pulse = Pulse(pulse.values[0 : index - current_index])
                        pulse_sequence = pulse_sequence.added(pulse)
                        break
                    # If the pulse is a Pulse and the length is less than the index, we can add the pulse to the sequence.
                    else:
                        pulse_sequence = pulse_sequence.added(pulse)
                        current_index += pulse.length
                return pulse_sequence
            # If the waveform is a Pulse, we can simply return the partial waveform.
            else:
                return Pulse(waveform.values[0:index])

        if n_samples < pulse_length:
            indices = np.linspace(0, pulse_length - 1, n_samples).astype(int)
        else:
            indices = np.arange(pulse_length)

        sequences = [
            {target: partial_waveform(pulse, i) for target, pulse in pulses.items()}
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
            times = pulses.popitem()[1].times[indices]
            for target, states in result.items():
                vis.plot_bloch_vectors(
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
        interval: int = DEFAULT_INTERVAL,
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
        interval: int = DEFAULT_INTERVAL,
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

    def obtain_rabi_params(
        self,
        targets: str | Collection[str] | None = None,
        *,
        time_range: ArrayLike = RABI_TIME_RANGE,
        amplitudes: dict[str, float] | None = None,
        frequencies: dict[str, float] | None = None,
        is_damped: bool = False,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
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
        targets: Collection[str] | None = None,
        *,
        time_range: ArrayLike = RABI_TIME_RANGE,
        is_damped: bool = False,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RabiData]:
        if targets is None:
            targets = self.qubit_labels
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
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[RabiData]:
        # target labels
        targets = list(amplitudes.keys())

        # drive time range
        time_range = np.array(time_range, dtype=np.float64)

        # target frequencies
        if frequencies is None:
            frequencies = {
                target: self.targets[target].frequency for target in amplitudes
            }

        # rabi sequence with rect pulses of duration T
        def rabi_sequence(T: int) -> PulseSchedule:
            with PulseSchedule(targets) as ps:
                for target in targets:
                    ps.add(target, Rect(duration=T, amplitude=amplitudes[target]))
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
        rabi_params = {
            target: fitting.fit_rabi(
                target=data.target,
                times=data.sweep_range,
                data=data.data,
                plot=plot,
                is_damped=is_damped,
            )
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
        is_damped: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
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
            with PulseSchedule(ge_labels + ef_labels) as ps:
                # prepare qubits to the excited state
                for ge in ge_labels:
                    ps.add(ge, self.pi_pulse[ge])
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
            )
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

    def measure_bell_state(
        self,
        control_qubit: str,
        target_qubit: str,
        zx90: PulseSchedule | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> dict:
        if self.state_centers is None:
            self.build_classifier(plot=False)

        cnot = self.cnot(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            zx90=zx90,
        )
        result = self.measure(
            cnot,
            initial_states={control_qubit: "+"},
            mode="single",
            shots=shots,
            interval=interval,
        )

        labels = [f"|{i}⟩" for i in result.probabilities.keys()]
        prob = np.array(list(result.probabilities.values()))
        cm_inv = self.get_inverse_confusion_matrix([control_qubit, target_qubit])

        mitigated_prob = prob @ cm_inv

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels,
                y=prob,
                name="Raw",
            )
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=mitigated_prob,
                name="Mitigated",
            )
        )
        fig.update_layout(
            title=f"Bell state measurement: {control_qubit}-{target_qubit}",
            xaxis_title="State label",
            yaxis_title="Probability",
            barmode="group",
            yaxis_range=[0, 1],
        )
        if plot:
            fig.show()

        if save_image:
            vis.save_figure_image(
                fig,
                f"bell_state_measurement_{control_qubit}-{target_qubit}",
            )

        for label, p, mp in zip(labels, prob, mitigated_prob):
            print(f"{label} : {p:.2%} -> {mp:.2%}")

        return {
            "raw": prob,
            "mitigated": mitigated_prob,
            "figure": fig,
        }
