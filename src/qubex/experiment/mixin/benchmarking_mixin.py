from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm
from typing import Collection

from ...analysis import fitting
from ...analysis import visualization as viz
from ...backend import Target
from ...clifford import Clifford
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import PulseArray, PulseSchedule, VirtualZ, Waveform, Blank
from ...typing import TargetMap
from ..experiment_result import ExperimentResult, RBData
from ..protocol import BaseProtocol, BenchmarkingProtocol, MeasurementProtocol


class BenchmarkingMixin(
    BaseProtocol,
    MeasurementProtocol,
    BenchmarkingProtocol,
):
    def rb_sequence(
        self,
        *,
        target: str,
        n: int,
        x90: dict[str, Waveform] | None = None,
        zx90: PulseSchedule | dict[str, Waveform] | None = None,
        interleaved_waveform: (
            PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_clifford: Clifford | None = None,
        seed: int | None = None,
    ) -> PulseSchedule:
        target_object = self.experiment_system.get_target(target)
        if target_object.is_cr:
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
            if isinstance(interleaved_waveform, PulseSchedule):
                interleaved_waveform = interleaved_waveform.get_sequences()
            seq = self.rb_sequence_1q(
                target=target,
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
        *,
        target: str,
        n: int,
        x90: Waveform | dict[str, Waveform] | None = None,
        interleaved_waveform: (
            Waveform | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_clifford: Clifford | dict[str, tuple[complex, str]] | None = None,
        seed: int | None = None,
    ) -> PulseArray:
        if isinstance(x90, dict):
            x90 = x90.get(target)
        x90 = x90 or self.x90(target)
        z90 = VirtualZ(np.pi / 2)

        sequence: list[Waveform | VirtualZ | PulseArray] = []

        if interleaved_waveform is None:
            cliffords, inverse = self.clifford_generator.create_rb_sequences(
                n=n,
                type="1Q",
                seed=seed,
            )
        else:
            if interleaved_clifford is None:
                raise ValueError("`interleaved_clifford` must be provided.")
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
            if isinstance(interleaved_waveform, dict):
                interleaved_waveform = interleaved_waveform.get(target)
            if interleaved_waveform is not None:
                sequence.append(interleaved_waveform)

        for gate in inverse:
            add_gate(gate)

        return PulseArray(sequence)

    def rb_sequence_2q(
        self,
        *,
        target: str,
        n: int,
        x90: TargetMap[Waveform] | None = None,
        zx90: (
            dict[str, PulseSchedule] | PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_waveform: (
            dict[str, PulseSchedule] | PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_clifford: Clifford | dict[str, tuple[complex, str]] | None = None,
        seed: int | None = None,
    ) -> PulseSchedule:
        target_object = self.experiment_system.get_target(target)
        if not target_object.is_cr:
            raise ValueError(f"`{target}` is not a 2Q target.")
        if isinstance(zx90, dict):
            if isinstance(zx90[target], PulseSchedule):
                zx90 = zx90.get(target)
        if isinstance(interleaved_waveform, dict):
            if isinstance(interleaved_waveform[target], PulseSchedule):
                interleaved_waveform = interleaved_waveform.get(target)
        control_qubit, target_qubit = Target.cr_qubit_pair(target)
        cr_label = target
        xi90, ix90 = None, None
        if isinstance(x90, dict):
            xi90 = x90.get(control_qubit)
            ix90 = x90.get(target_qubit)
        xi90 = xi90 or self.x90(control_qubit)
        ix90 = ix90 or self.x90(target_qubit)
        z90 = VirtualZ(np.pi / 2)

        if zx90 is None:
            zx90 = self.zx90(control_qubit, target_qubit)

        if interleaved_waveform is None:
            cliffords, inverse = self.clifford_generator.create_rb_sequences(
                n=n,
                type="2Q",
                seed=seed,
            )
        else:
            if interleaved_clifford is None:
                raise ValueError("Interleave map must be provided.")
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
                    if isinstance(zx90, dict):
                        ps.add(control_qubit, zx90[control_qubit])
                        ps.add(target_qubit, zx90[target_qubit])
                    elif isinstance(zx90, PulseSchedule):
                        ps.call(zx90)
                    ps.barrier()
                else:
                    raise ValueError("Invalid gate.")

            for clifford in cliffords:
                for gate in clifford:
                    add_gate(gate)
                if interleaved_waveform is not None:
                    ps.barrier()
                    if isinstance(interleaved_waveform, dict):
                        ps.add(control_qubit, interleaved_waveform[control_qubit])
                        ps.add(target_qubit, interleaved_waveform[target_qubit])
                    elif isinstance(interleaved_waveform, PulseSchedule):
                        ps.call(interleaved_waveform)
                    ps.barrier()

            for gate in inverse:
                add_gate(gate)
        return ps

    def rb_experiment_1q(
        self,
        *,
        target: str | None = None,
        targets: Collection[str] | None = None,
        in_parallel: bool = False,
        n_cliffords_range: ArrayLike | None = None,
        x90: Waveform | dict[str, Waveform] | None = None,
        interleaved_waveform: Waveform | None = None,
        interleaved_clifford: Clifford | dict[str, tuple[complex, str]] | None = None,
        seed: int | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> ExperimentResult[RBData]:
        if (target is None and targets is None) or (target is not None and targets is not None):
            raise ValueError("Either `target` or `targets` must be provided, but not both.")

        if target is not None and in_parallel:
            raise ValueError("Cannot run a single target experiment in parallel. ")

        if n_cliffords_range is None:
            n_cliffords_range = np.arange(0, 1001, 50)

        if target:
            execution_groups = [[target]]
        else:
            _targets = list(targets)
            if in_parallel:
                execution_groups = [_targets]
            else:
                execution_groups = [[t] for t in _targets]

        all_unique_targets = list(targets) if targets is not None else [target]
        for t in all_unique_targets:
            target_object = self.experiment_system.get_target(t)
            if target_object.is_cr:
                raise ValueError(f"`{t}` is not a 1Q target.")

        intermediate_results = {}
        for target_group in execution_groups:
            def rb_sequence(N: int) -> PulseSchedule:
                with PulseSchedule(target_group) as ps:
                    for target in target_group:
                        rb_sequence = self.rb_sequence_1q(
                            target=target,
                            n=N,
                            x90=x90,
                            interleaved_waveform=interleaved_waveform,
                            interleaved_clifford=interleaved_clifford,
                            seed=seed,
                        )
                        ps.add(
                            target,
                            rb_sequence,
                        )
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
        *,
        target: str | None = None,
        targets: Collection[str] | None = None,
        in_parallel: bool = False,
        n_cliffords_range: ArrayLike = np.arange(0, 31, 3),
        x90: TargetMap[Waveform] | None = None,
        zx90: (
            dict[str, PulseSchedule] | PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_waveform: (
            dict[str, PulseSchedule] | PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        interleaved_clifford: Clifford | dict[str, tuple[complex, str]] | None = None,
        seed: int | None = None,
        mitigate_readout: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ):
        if (target is None and targets is None) or (target is not None and targets is not None):
            raise ValueError("Either `target` or `targets` must be provided, but not both.")

        if target is not None and in_parallel:
            raise ValueError("Cannot run a single target experiment in parallel. ")

        if target:
            execution_groups = [[target]]
        else:
            _targets = list(targets)
            if in_parallel:
                execution_groups = [_targets]
            else:
                execution_groups = [[t] for t in _targets]

        if self.state_centers is None:
            raise ValueError("State classifiers are not built.")

        n_cliffords_range = np.array(n_cliffords_range, dtype=int)
        
        all_unique_targets = list(targets) if targets is not None else [target]
        for t in all_unique_targets:
            target_object = self.experiment_system.get_target(t)
            if not target_object.is_cr:
                raise ValueError(f"`{t}` is not a 2Q target.")
            
        final_data = {}
        for target_group in execution_groups:
            def rb_sequence(N: int) -> PulseSchedule:
                pulse_list = []
                for target in target_group:
                    control_qubit, target_qubit = Target.cr_qubit_pair(target)
                    cr_label = target
                    pulse_list.append(control_qubit)
                    pulse_list.append(cr_label)
                    pulse_list.append(target_qubit)
                with PulseSchedule(pulse_list) as ps:
                    seq = {}
                    for target in target_group:
                        seq[target] = self.rb_sequence_2q(
                            target=target,
                            n=N,
                            x90=x90,
                            zx90=zx90,
                            interleaved_waveform=interleaved_waveform,
                            interleaved_clifford=interleaved_clifford,
                            seed=seed,
                        )
                    for target in target_group:
                        ps.call(seq[target])
                    diff = {}
                    for label in pulse_list:
                        diff[label] = ps._max_offset(pulse_list) - ps._offsets[label]

                    ps.__init__()
                    
                    for label in pulse_list:
                        duration = diff[label]
                        if duration > 0:
                            ps. add(label, Blank(duration = duration))
                    for target in target_group:
                        seq[target] = self.rb_sequence_2q(
                            target=target,
                            n=N,
                            x90=x90,
                            zx90=zx90,
                            interleaved_waveform=interleaved_waveform,
                            interleaved_clifford=interleaved_clifford,
                            seed=seed,
                        )
                    for target in target_group:
                        ps.call(seq[target])
                return ps
            
            fidelities = {cr:[] for cr in target_group}
            for n_clifford in tqdm(n_cliffords_range):
                result = self.measure(
                    sequence=rb_sequence(N=n_clifford),
                    mode="single",
                    shots=shots,
                    interval=interval,
                    plot=False,
                )

                for cr_label in target_group:
                    control_qubit, target_qubit = Target.cr_qubit_pair(cr_label)
                    if mitigate_readout:
                        prob = result.get_mitigated_probabilities([control_qubit,target_qubit])
                    else :
                        prob = result.get_probabilities([control_qubit, target_qubit])

                    fidelities[cr_label].append(prob["00"])

            for cr_label in target_group:
                fit_result = fitting.fit_rb(
                    target=cr_label,
                    x=n_cliffords_range,
                    y=np.array(fidelities[cr_label]),
                    dimension=4,
                    title="Randomized benchmarking",
                    xlabel="Number of Cliffords",
                    ylabel="Normalized signal",
                    xaxis_type="linear",
                    yaxis_type="linear",
                    plot=plot,
                )

                final_data[cr_label] = {
                "n_cliffords": n_cliffords_range,
                "fidelities": fidelities[cr_label],
                "depolarizing_rate": fit_result["depolarizing_rate"],
                "avg_gate_error": fit_result["avg_gate_error"],
                "avg_gate_fidelity": fit_result["avg_gate_fidelity"],
                }

        return final_data

    def randomized_benchmarking(
        self,
        *,
        target: str | None = None,
        targets: Collection[str] | None = None,
        in_parallel : bool = False, 
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int = 30,
        x90: Waveform | dict[str, Waveform] | None = None,
        zx90: (
            dict[str, PulseSchedule] | PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        seeds: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        if (target is None and targets is None) or (target is not None and targets is not None):
            raise ValueError("Either `target` or `targets` must be provided, but not both.")

        if target is not None and in_parallel:
            raise ValueError("Cannot run a single target experiment in parallel. ")
        
        _targets = list(targets) if targets is not None else [target]
        
        if seeds is None:
            seeds = np.random.randint(0, 2**32, n_trials)
        else:
            seeds = np.array(seeds, dtype=int)
            if len(seeds) != n_trials:
                raise ValueError(
                    "The number of seeds must be equal to the number of trials."
                )

        target_object = self.experiment_system.get_target(_targets[0])
        is_2q = target_object.is_cr

        if is_2q:
            if n_cliffords_range is None:
                n_cliffords_range = np.arange(0, 21, 2)
        else:
            self.validate_rabi_params(_targets)
            if n_cliffords_range is None:
                n_cliffords_range = np.arange(0, 1001, 100)

        n_cliffords_range = np.array(n_cliffords_range, dtype=int)

        rb_results = {t_val: [] for t_val in _targets}
        for seed in tqdm(seeds):
            seed = int(seed)
            with self.util.no_output():
                if is_2q:
                    if isinstance(x90, Waveform):
                        raise ValueError("x90 must be a dict for 2Q gates.")
                    rb_result = self.rb_experiment_2q(
                        target=target,
                        targets=targets,
                        in_parallel=in_parallel,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,
                        zx90=zx90,
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                    )
                    for t_val in _targets:
                        rb_signal = rb_result[t_val]["fidelities"]
                        rb_results[t_val].append(rb_signal)
                else:
                    rb_result = self.rb_experiment_1q(
                        target=target,
                        targets=targets,
                        in_parallel=in_parallel,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                        save_image=False,
                    )
                    for t_val in _targets:
                        rb_signal = (rb_result.data[t_val].normalized + 1) / 2
                        rb_results[t_val].append(rb_signal)

        results = {t_val : [] for t_val in _targets}
        for t_val in _targets:
            mean = np.mean(rb_results[t_val], axis=0)
            std = np.std(rb_results[t_val], axis=0)

            fit_result = fitting.fit_rb(
                target=t_val,
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
                    name=f"randomized_benchmarking_{t_val}",
                )

            results[t_val] = {
                "n_cliffords": n_cliffords_range,
                "mean": mean,
                "std": std,
                **fit_result,
            }
        return results

    def interleaved_randomized_benchmarking(
        self,
        *,
        target: str | None = None,
        targets : Collection[str] | None = None,
        in_parallel : bool = False,
        interleaved_waveform: Waveform | PulseSchedule | dict[str, PulseSchedule],
        interleaved_clifford: str | Clifford | dict[str, tuple[complex, str]],
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int = 30,
        x90: TargetMap[Waveform] | Waveform | None = None,
        zx90: (
            dict[str, PulseSchedule] | PulseSchedule | dict[str, PulseArray] | dict[str, Waveform] | None
        ) = None,
        seeds: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> dict:
        if (target is None and targets is None) or (target is not None and targets is not None):
            raise ValueError("Either `target` or `targets` must be provided, but not both.")

        if target is not None and in_parallel:
            raise ValueError("Cannot run a single target experiment in parallel. ")

        _targets = list(targets) if targets is not None else [target]

        if seeds is None:
            seeds = np.random.randint(0, 2**32, n_trials)
        else:
            seeds = np.array(seeds, dtype=int)
            if len(seeds) != n_trials:
                raise ValueError(
                    "The number of seeds must be equal to the number of trials."
                )

        target_object = self.experiment_system.get_target(_targets[0])
        is_2q = target_object.is_cr

        if is_2q:
            if n_cliffords_range is None:
                n_cliffords_range = np.arange(0, 61, 6)
        else:
            self.validate_rabi_params(_targets)
            if n_cliffords_range is None:
                n_cliffords_range = np.arange(0, 1001, 100)

        n_cliffords_range = np.array(n_cliffords_range, dtype=int)

        if isinstance(interleaved_clifford, str):
            clifford = self.clifford.get(interleaved_clifford)
            if clifford is None:
                raise ValueError(f"Invalid Clifford: {interleaved_clifford}")
            else:
                interleaved_clifford = clifford

        rb_results = {t_val : [] for t_val in _targets}
        irb_results = {t_val : [] for t_val in _targets}

        for seed in tqdm(seeds):
            seed = int(seed)
            with self.util.no_output():
                if is_2q:
                    if isinstance(x90, Waveform):
                        raise ValueError("x90 must be a dict for 2Q gates.")
                    if isinstance(zx90, Waveform):
                        raise ValueError("zx90 must be a dict for 2Q gates.")
                    if isinstance(interleaved_waveform, Waveform):
                        raise ValueError(
                            "interleaved_waveform must be a dict for 2Q gates."
                        )
                    rb_result = self.rb_experiment_2q(
                        target=target,
                        targets=targets,
                        in_parallel=in_parallel,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,
                        zx90=zx90,
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                    )
                    for t_val in _targets:
                        rb_signal = rb_result[t_val]["fidelities"]
                        rb_results[t_val].append(rb_signal)

                    irb_result = self.rb_experiment_2q(
                        target=target,
                        targets=targets,
                        in_parallel=in_parallel,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,
                        zx90=zx90,
                        interleaved_waveform=interleaved_waveform,
                        interleaved_clifford=interleaved_clifford,
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                    )
                    for t_val in _targets:
                        irb_signal = irb_result[t_val]["fidelities"]
                        irb_results[t_val].append(irb_signal)
                else:
                    rb_result = self.rb_experiment_1q(
                        target=target,
                        targets=targets,
                        in_parallel=in_parallel,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,  # type: ignore
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                        save_image=False,
                    )
                    for t_val in _targets:
                        rb_signal = (rb_result.data[t_val].normalized + 1) / 2
                        rb_results[t_val].append(rb_signal)

                    irb_result = self.rb_experiment_1q(
                        target=target,
                        targets=targets,
                        in_parallel=in_parallel,
                        n_cliffords_range=n_cliffords_range,
                        x90=x90,  # type: ignore
                        interleaved_waveform=interleaved_waveform,  # type: ignore
                        interleaved_clifford=interleaved_clifford,
                        seed=seed,
                        shots=shots,
                        interval=interval,
                        plot=False,
                        save_image=False,
                    )
                    for t in _targets:
                        irb_signal = (irb_result.data[t_val].normalized + 1) / 2
                        irb_results[t_val].append(irb_signal)

        results = {t_val : [] for t_val in _targets}
        for t_val in _targets:
            rb_mean = np.mean(rb_results[t_val], axis=0)
            rb_std = np.std(rb_results[t_val], axis=0)
            rb_fit_result = fitting.fit_rb(
                target=t_val,
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

            irb_mean = np.mean(irb_results[t_val], axis=0)
            irb_std = np.std(irb_results[t_val], axis=0)
            irb_fit_result = fitting.fit_rb(
                target=t_val,
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
                target=t_val,
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
                title="Interleaved randomized benchmarking",
                xlabel="Number of Cliffords",
                ylabel="Normalized signal",
            )
            if save_image:
                viz.save_figure_image(
                    fig,
                    name=f"interleaved_randomized_benchmarking_{t_val}",
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

            results[t_val] = {
            "gate_error": gate_error,
            "gate_fidelity": gate_fidelity,
            "gate_fidelity_err": gate_fidelity_err,
            "rb_fit_result": rb_fit_result,
            "irb_fit_result": irb_fit_result,
            "fig": fig,
            }
        return results