from __future__ import annotations

from typing import Collection, Mapping

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from ...analysis import fitting
from ...analysis import visualization as viz
from ...backend import Target
from ...clifford import Clifford
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import PulseArray, PulseSchedule, VirtualZ, Waveform
from ...typing import TargetMap
from ..experiment_result import ExperimentResult, RBData
from ..protocol import BaseProtocol, BenchmarkingProtocol, MeasurementProtocol

DEFAULT_CLIFFORD_RANGE_1Q = np.arange(0, 1001, 100)
DEFAULT_CLIFFORD_RANGE_2Q = np.arange(0, 41, 4)
DEFAULT_RB_N_TRIALS = 30


class BenchmarkingMixin(
    BaseProtocol,
    MeasurementProtocol,
    BenchmarkingProtocol,
):
    def rb_sequence(
        self,
        target: str,
        *,
        n: int,
        x90: Waveform | TargetMap[Waveform] | None = None,
        zx90: PulseSchedule | None = None,
        interleaved_waveform: Waveform | PulseSchedule | None = None,
        interleaved_clifford: Clifford | None = None,
        seed: int | None = None,
    ) -> PulseSchedule:
        target_object = self.experiment_system.get_target(target)
        if target_object.is_cr:
            if isinstance(x90, Waveform):
                raise ValueError("x90 must be a dict for 2Q gates.")
            if isinstance(interleaved_waveform, Waveform):
                raise ValueError(
                    "interleaved_waveform must be a PulseSchedule for 2Q gates."
                )
            sched = self.rb_sequence_2q(
                target=target,
                n=n,
                x90=x90,
                zx90=zx90,
                interleaved_waveform=interleaved_waveform,
                interleaved_clifford=interleaved_clifford,
                seed=seed,
            )
            return sched
        else:
            if isinstance(x90, Mapping):
                x90 = x90.get(target)
            if isinstance(interleaved_waveform, PulseSchedule):
                interleaved_waveform = interleaved_waveform.get_sequence(target)
            seq = self.rb_sequence_1q(
                target,
                n=n,
                x90=x90,
                interleaved_waveform=interleaved_waveform,
                interleaved_clifford=interleaved_clifford,
                seed=seed,
            )
            with PulseSchedule([target]) as ps:
                ps.add(target, seq)
            return ps

    def rb_sequence_1q(
        self,
        target: str,
        *,
        n: int,
        x90: Waveform | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: Waveform | None = None,
        seed: int | None = None,
    ) -> PulseArray:
        x90 = x90 or self.x90(target)
        z90 = VirtualZ(np.pi / 2)

        sequence: list[Waveform | VirtualZ] = []

        if interleaved_clifford is None:
            cliffords, inverse = self.clifford_generator.create_rb_sequences(
                n=n,
                type="1Q",
                seed=seed,
            )
        else:
            if interleaved_waveform is None:
                if interleaved_clifford.name == "X90":
                    interleaved_waveform = self.x90(target)
                elif interleaved_clifford.name == "X180":
                    interleaved_waveform = self.x180(target)
                else:
                    raise ValueError("interleaved_waveform must be provided.")
            cliffords, inverse = self.clifford_generator.create_irb_sequences(
                n=n,
                interleave=interleaved_clifford,
                type="1Q",
                seed=seed,
            )

        def add_gate(gate: str):
            if gate == "X90":
                sequence.append(x90)
            elif gate == "Z90":
                sequence.append(z90)
            else:
                raise ValueError("Invalid gate.")

        for clifford in cliffords:
            for gate in clifford:
                add_gate(gate)
            if interleaved_waveform is not None:
                sequence.append(interleaved_waveform)

        for gate in inverse:
            add_gate(gate)

        return PulseArray(sequence)

    def rb_sequence_2q(
        self,
        target: str,
        *,
        n: int,
        x90: TargetMap[Waveform] | None = None,
        zx90: PulseSchedule | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: PulseSchedule | None = None,
        seed: int | None = None,
    ) -> PulseSchedule:
        target_object = self.experiment_system.get_target(target)
        if not target_object.is_cr:
            raise ValueError(f"`{target}` is not a 2Q target.")

        control_qubit, target_qubit = Target.cr_qubit_pair(target)
        cr_label = target

        xi90 = x90.get(control_qubit) if x90 is not None else None
        ix90 = x90.get(target_qubit) if x90 is not None else None
        xi90 = xi90 or self.x90(control_qubit)
        ix90 = ix90 or self.x90(target_qubit)
        z90 = VirtualZ(np.pi / 2)

        if zx90 is None:
            zx90 = self.zx90(control_qubit, target_qubit)

        if interleaved_clifford is None:
            cliffords, inverse = self.clifford_generator.create_rb_sequences(
                n=n,
                type="2Q",
                seed=seed,
            )
        else:
            if interleaved_waveform is None:
                if interleaved_clifford.name == "ZX90":
                    interleaved_waveform = self.zx90(control_qubit, target_qubit)
                else:
                    raise ValueError("interleaved_waveform must be provided.")
            cliffords, inverse = self.clifford_generator.create_irb_sequences(
                n=n,
                interleave=interleaved_clifford,
                type="2Q",
                seed=seed,
            )

        with PulseSchedule([control_qubit, cr_label, target_qubit]) as ps:

            def add_gate(gate: str):
                if gate == "XI90":
                    ps.add(control_qubit, xi90)
                elif gate == "IX90":
                    ps.add(target_qubit, ix90)
                elif gate == "ZI90":
                    ps.add(control_qubit, z90)
                elif gate == "IZ90":
                    ps.add(target_qubit, z90)
                    ps.add(cr_label, z90)
                elif gate == "ZX90":
                    ps.barrier()
                    ps.call(zx90)
                    ps.barrier()
                else:
                    raise ValueError("Invalid gate.")

            for clifford in cliffords:
                for gate in clifford:
                    add_gate(gate)
                if interleaved_waveform is not None:
                    ps.barrier()
                    ps.call(interleaved_waveform)
                    ps.barrier()

            for gate in inverse:
                add_gate(gate)
        return ps

    def rb_experiment_1q(
        self,
        targets: Collection[str] | str,
        *,
        in_parallel: bool = False,
        n_cliffords_range: ArrayLike | None = None,
        x90: TargetMap[Waveform] | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: TargetMap[Waveform] | None = None,
        seed: int | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> ExperimentResult[RBData]:
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if n_cliffords_range is None:
            n_cliffords_range = DEFAULT_CLIFFORD_RANGE_1Q

        if shots is None:
            shots = DEFAULT_SHOTS

        if interval is None:
            interval = DEFAULT_INTERVAL

        if in_parallel:
            execution_groups = [targets]
        else:
            execution_groups = [[target] for target in targets]

        for target in targets:
            target_object = self.experiment_system.get_target(target)
            if target_object.is_cr:
                raise ValueError(f"`{target}` is not a 1Q target.")

        intermediate_results = {}
        for target_group in execution_groups:

            def rb_sequence(N: int) -> PulseSchedule:
                with PulseSchedule(target_group) as ps:
                    for target in target_group:
                        rb_sequence = self.rb_sequence_1q(
                            target,
                            n=N,
                            x90=x90.get(target) if x90 else None,
                            interleaved_waveform=interleaved_waveform.get(target)
                            if interleaved_waveform
                            else None,
                            interleaved_clifford=interleaved_clifford,
                            seed=seed,
                        )
                        ps.add(target, rb_sequence)
                return ps

            sweep_result = self.sweep_parameter(
                rb_sequence,
                sweep_range=n_cliffords_range,
                shots=shots,
                interval=interval,
                plot=False,
            )

            for target, sweep_data in sweep_result.data.items():
                fit_result = fitting.fit_rb(
                    target=target,
                    x=sweep_data.sweep_range,
                    y=(sweep_data.normalized + 1) / 2,
                    bounds=((0, 0, 0), (0.5, 1, 1)),
                    title="Randomized benchmarking",
                    xlabel="Number of Cliffords",
                    ylabel="Normalized signal",
                    xaxis_type="linear",
                    yaxis_type="linear",
                    plot=plot,
                )

                if save_image:
                    viz.save_figure_image(
                        fit_result["fig"],
                        name=f"rb_{target}",
                    )

                intermediate_results[target] = {
                    "sweep_data": sweep_data,
                    "fit_result": fit_result,
                }

        data = {
            qubit: RBData.new(
                results["sweep_data"],
                depolarizing_rate=results["fit_result"]["depolarizing_rate"],
                avg_gate_error=results["fit_result"]["avg_gate_error"],
                avg_gate_fidelity=results["fit_result"]["avg_gate_fidelity"],
            )
            for qubit, results in intermediate_results.items()
        }

        return ExperimentResult(data=data)

    def rb_experiment_2q(
        self,
        targets: Collection[str] | str,
        *,
        in_parallel: bool = False,
        n_cliffords_range: ArrayLike | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: TargetMap[PulseSchedule] | None = None,
        seed: int | None = None,
        mitigate_readout: bool = True,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
    ):
        if self.state_centers is None:
            raise ValueError("State classifiers are not built.")

        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if n_cliffords_range is None:
            n_cliffords_range = DEFAULT_CLIFFORD_RANGE_2Q

        if shots is None:
            shots = DEFAULT_SHOTS

        if interval is None:
            interval = DEFAULT_INTERVAL

        n_cliffords_range = np.array(n_cliffords_range, dtype=int)

        if in_parallel:
            execution_groups = [targets]
        else:
            execution_groups = [[target] for target in targets]

        for target in targets:
            target_object = self.experiment_system.get_target(target)
            if not target_object.is_cr:
                raise ValueError(f"`{target}` is not a 2Q target.")

        final_data = {}
        for target_group in execution_groups:

            def rb_sequence(N: int) -> PulseSchedule:
                with PulseSchedule() as ps:
                    seq: dict[str, PulseSchedule] = {}
                    for target in target_group:
                        seq[target] = self.rb_sequence_2q(
                            target=target,
                            n=N,
                            x90=x90,
                            zx90=zx90.get(target) if zx90 else None,
                            interleaved_waveform=interleaved_waveform.get(target)
                            if interleaved_waveform
                            else None,
                            interleaved_clifford=interleaved_clifford,
                            seed=seed,
                        )
                    max_duration = max([seq.duration for seq in seq.values()])

                    for target in target_group:
                        ps.call(
                            seq[target].padded(
                                total_duration=max_duration,
                                pad_side="left",
                            )
                        )
                return ps

            fidelities = {target: [] for target in target_group}
            for n_clifford in tqdm(n_cliffords_range):
                result = self.measure(
                    sequence=rb_sequence(N=n_clifford),
                    mode="single",
                    shots=shots,
                    interval=interval,
                    plot=False,
                )

                for target in target_group:
                    control_qubit, target_qubit = Target.cr_qubit_pair(target)
                    if mitigate_readout:
                        prob = result.get_mitigated_probabilities(
                            [control_qubit, target_qubit]
                        )
                    else:
                        prob = result.get_probabilities([control_qubit, target_qubit])

                    fidelities[target].append(prob["00"])

            for target in target_group:
                fit_result = fitting.fit_rb(
                    target=target,
                    x=n_cliffords_range,
                    y=np.array(fidelities[target]),
                    dimension=4,
                    title="Randomized benchmarking",
                    xlabel="Number of Cliffords",
                    ylabel="Normalized signal",
                    xaxis_type="linear",
                    yaxis_type="linear",
                    plot=plot,
                )

                final_data[target] = {
                    "n_cliffords": n_cliffords_range,
                    "fidelities": fidelities[target],
                    "depolarizing_rate": fit_result["depolarizing_rate"],
                    "avg_gate_error": fit_result["avg_gate_error"],
                    "avg_gate_fidelity": fit_result["avg_gate_fidelity"],
                }

        return final_data

    def randomized_benchmarking(
        self,
        targets: Collection[str] | str,
        *,
        in_parallel: bool = False,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        seeds: ArrayLike | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if n_trials is None:
            n_trials = DEFAULT_RB_N_TRIALS

        if seeds is None:
            seeds = np.random.randint(0, 2**32, n_trials)
        else:
            seeds = np.array(seeds, dtype=int)
            if len(seeds) != n_trials:
                raise ValueError(
                    "The number of seeds must be equal to the number of trials."
                )

        target_object = self.experiment_system.get_target(targets[0])
        is_2q = target_object.is_cr

        if n_cliffords_range is None:
            if is_2q:
                n_cliffords_range = DEFAULT_CLIFFORD_RANGE_2Q
            else:
                n_cliffords_range = DEFAULT_CLIFFORD_RANGE_1Q
        else:
            n_cliffords_range = np.array(n_cliffords_range, dtype=int)

        rb_signals = {target: [] for target in targets}

        for seed in tqdm(seeds):
            seed = int(seed)
            with self.util.no_output():
                if is_2q:
                    rb_results = self.rb_experiment_2q(
                        targets,
                        in_parallel=in_parallel,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,
                        zx90=zx90,
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                    )
                    for target, rb_result in rb_results.items():
                        rb_signals[target].append(rb_result["fidelities"])
                else:
                    rb_results = self.rb_experiment_1q(
                        targets,
                        in_parallel=in_parallel,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                        save_image=False,
                    )
                    for target, rb_result in rb_results.data.items():
                        rb_signals[target].append((rb_result.normalized + 1) / 2)

        result = {}
        for target in targets:
            mean = np.mean(rb_signals[target], axis=0)
            std = np.std(rb_signals[target], axis=0)

            fit_result = fitting.fit_rb(
                target=target,
                x=n_cliffords_range,
                y=mean,
                error_y=std,
                dimension=4 if is_2q else 2,
                plot=plot,
                title="Randomized benchmarking",
                xlabel="Number of Cliffords",
                ylabel="Normalized signal",
                xaxis_type="linear",
                yaxis_type="linear",
            )

            if save_image:
                viz.save_figure_image(
                    fit_result["fig"],
                    name=f"randomized_benchmarking_{target}",
                )

            result[target] = {
                "n_cliffords": n_cliffords_range,
                "mean": mean,
                "std": std,
                **fit_result,
            }
        return result

    def interleaved_randomized_benchmarking(
        self,
        targets: Collection[str] | str,
        *,
        in_parallel: bool = False,
        interleaved_clifford: str | Clifford,
        interleaved_waveform: TargetMap[PulseSchedule]
        | TargetMap[Waveform]
        | None = None,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        seeds: ArrayLike | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        target_object = self.experiment_system.get_target(targets[0])
        is_2q = target_object.is_cr

        if n_cliffords_range is None:
            if is_2q:
                n_cliffords_range = DEFAULT_CLIFFORD_RANGE_2Q
            else:
                n_cliffords_range = DEFAULT_CLIFFORD_RANGE_1Q
        else:
            n_cliffords_range = np.array(n_cliffords_range, dtype=int)

        if n_trials is None:
            n_trials = DEFAULT_RB_N_TRIALS

        if seeds is None:
            seeds = np.random.randint(0, 2**32, n_trials)
        else:
            seeds = np.array(seeds, dtype=int)
            if len(seeds) != n_trials:
                raise ValueError(
                    "The number of seeds must be equal to the number of trials."
                )

        if isinstance(interleaved_clifford, str):
            clifford = self.clifford.get(interleaved_clifford)
            if clifford is None:
                raise ValueError(f"Invalid Clifford: {interleaved_clifford}")
            interleaved_clifford = clifford

        rb_results = {target: [] for target in targets}
        irb_results = {target: [] for target in targets}

        for seed in tqdm(seeds):
            seed = int(seed)
            with self.util.no_output():
                if is_2q:
                    rb_result = self.rb_experiment_2q(
                        targets,
                        in_parallel=in_parallel,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,
                        zx90=zx90,
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                    )
                    for target in targets:
                        rb_signal = rb_result[target]["fidelities"]
                        rb_results[target].append(rb_signal)

                    irb_result = self.rb_experiment_2q(
                        targets=targets,
                        in_parallel=False,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,
                        zx90=zx90,
                        interleaved_waveform=interleaved_waveform,  # type: ignore
                        interleaved_clifford=interleaved_clifford,
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                    )
                    for target in targets:
                        irb_signal = irb_result[target]["fidelities"]
                        irb_results[target].append(irb_signal)
                else:
                    rb_result = self.rb_experiment_1q(
                        targets,
                        in_parallel=in_parallel,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                        save_image=False,
                    )
                    for target in targets:
                        rb_signal = (rb_result.data[target].normalized + 1) / 2
                        rb_results[target].append(rb_signal)

                    irb_result = self.rb_experiment_1q(
                        targets=targets,
                        in_parallel=in_parallel,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,
                        interleaved_waveform=interleaved_waveform,  # type: ignore
                        interleaved_clifford=interleaved_clifford,
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                        save_image=False,
                    )
                    for target in targets:
                        irb_signal = (irb_result.data[target].normalized + 1) / 2
                        irb_results[target].append(irb_signal)

        results = {}
        for target in targets:
            rb_mean = np.mean(rb_results[target], axis=0)
            rb_std = np.std(rb_results[target], axis=0)
            rb_fit_result = fitting.fit_rb(
                target=target,
                x=n_cliffords_range,
                y=rb_mean,
                error_y=rb_std,
                dimension=4 if is_2q else 2,
                plot=False,
            )
            A_rb = rb_fit_result["A"]
            p_rb = rb_fit_result["p"]
            p_rb_err = rb_fit_result["p_err"]
            C_rb = rb_fit_result["C"]
            avg_gate_fidelity_rb = rb_fit_result["avg_gate_fidelity"]
            avg_gate_fidelity_err_rb = rb_fit_result["avg_gate_fidelity_err"]

            irb_mean = np.mean(irb_results[target], axis=0)
            irb_std = np.std(irb_results[target], axis=0)
            irb_fit_result = fitting.fit_rb(
                target=target,
                x=n_cliffords_range,
                y=irb_mean,
                error_y=irb_std,
                dimension=4 if is_2q else 2,
                plot=False,
                title="Interleaved randomized benchmarking",
            )
            A_irb = irb_fit_result["A"]
            p_irb = irb_fit_result["p"]
            p_irb_err = irb_fit_result["p_err"]
            C_irb = irb_fit_result["C"]
            avg_gate_fidelity_irb = irb_fit_result["avg_gate_fidelity"]
            avg_gate_fidelity_err_irb = irb_fit_result["avg_gate_fidelity_err"]

            dimension = 4 if is_2q else 2
            gate_error = (dimension - 1) * (1 - (p_irb / p_rb)) / dimension
            gate_fidelity = 1 - gate_error

            gate_fidelity_err = (
                (dimension - 1)
                / dimension
                * np.sqrt((p_irb_err / p_rb) ** 2 + (p_rb_err * p_irb / p_rb**2) ** 2)
            )

            fig = fitting.plot_irb(
                target=target,
                x=n_cliffords_range,
                y_rb=rb_mean,
                y_irb=irb_mean,
                error_y_rb=rb_std,
                error_y_irb=irb_std,
                A_rb=A_rb,
                A_irb=A_irb,
                p_rb=p_rb,
                p_irb=p_irb,
                C_rb=C_rb,
                C_irb=C_irb,
                gate_fidelity=gate_fidelity,
                gate_fidelity_err=gate_fidelity_err,
                plot=plot,
                title=f"Interleaved randomized benchmarking of {clifford.name}",
                xlabel="Number of Cliffords",
                ylabel="Normalized signal",
            )
            if save_image:
                viz.save_figure_image(
                    fig,
                    name=f"interleaved_randomized_benchmarking_{target}",
                )

            print()
            print(
                f"Average gate fidelity (RB)  : {avg_gate_fidelity_rb * 100:.3f} ± {avg_gate_fidelity_err_rb * 100:.3f}%"
            )
            print(
                f"Average gate fidelity (IRB) : {avg_gate_fidelity_irb * 100:.3f} ± {avg_gate_fidelity_err_irb * 100:.3f}%"
            )
            print()
            print(
                f"Gate error    : {gate_error * 100:.3f} ± {gate_fidelity_err * 100:.3f}%"
            )
            print(
                f"Gate fidelity : {gate_fidelity * 100:.3f} ± {gate_fidelity_err * 100:.3f}%"
            )
            print()

            results[target] = {
                "gate_error": gate_error,
                "gate_fidelity": gate_fidelity,
                "gate_fidelity_err": gate_fidelity_err,
                "rb_fit_result": rb_fit_result,
                "irb_fit_result": irb_fit_result,
                "fig": fig,
            }
        return results

    def benchmark_1q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        in_parallel: bool = False,
        plot: bool = True,
        save_image: bool = True,
    ):
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if in_parallel:
            self.interleaved_randomized_benchmarking(
                targets,
                in_parallel=in_parallel,
                interleaved_clifford="X90",
                plot=plot,
                save_image=save_image,
            )
            self.interleaved_randomized_benchmarking(
                targets,
                in_parallel=in_parallel,
                interleaved_clifford="X180",
                plot=plot,
                save_image=save_image,
            )
        else:
            for target in targets:
                self.interleaved_randomized_benchmarking(
                    target,
                    in_parallel=in_parallel,
                    interleaved_clifford="X90",
                    plot=plot,
                    save_image=save_image,
                )
                self.interleaved_randomized_benchmarking(
                    target,
                    in_parallel=in_parallel,
                    interleaved_clifford="X180",
                    plot=plot,
                    save_image=save_image,
                )

    def benchmark_2q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        in_parallel: bool = False,
        plot: bool = True,
        save_image: bool = True,
    ):
        if targets is None:
            targets = self.cr_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if in_parallel:
            self.interleaved_randomized_benchmarking(
                targets,
                in_parallel=in_parallel,
                interleaved_clifford="ZX90",
                plot=plot,
                save_image=save_image,
            )
        else:
            for target in targets:
                self.interleaved_randomized_benchmarking(
                    target,
                    in_parallel=in_parallel,
                    interleaved_clifford="ZX90",
                    plot=plot,
                    save_image=save_image,
                )
