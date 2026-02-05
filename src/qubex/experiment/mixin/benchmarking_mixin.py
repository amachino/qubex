from __future__ import annotations

from collections import defaultdict
from typing import Collection, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike

from ...analysis import fitting
from ...analysis import visualization as viz
from ...backend import Target
from ...clifford import Clifford
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import PulseArray, PulseSchedule, VirtualZ, Waveform
from ...typing import TargetMap
from ..protocol import BaseProtocol, BenchmarkingProtocol, MeasurementProtocol
from ..result import Result

DEFAULT_RB_N_TRIALS = 30
DEFAULT_MAX_N_CLIFFORDS_1Q = 2048
DEFAULT_MAX_N_CLIFFORDS_2Q = 128


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

    def purity_sequence_1q(
        self,
        target: str,
        *,
        n: int,
        x90: Waveform | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: Waveform | None = None,
        seed: int | None = None,
        basis: Literal["X", "Y", "Z"] = "Z",
    ) -> PulseArray:
        x90 = x90 or self.x90(target)
        z90 = VirtualZ(np.pi / 2)
        y90m = x90.shifted(-np.pi / 2)
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

        if basis == "X":
            sequence.append(y90m)
        elif basis == "Y":
            sequence.append(x90)
        elif basis == "Z":
            pass

        return PulseArray(sequence)

    def purity_sequence_2q(
        self,
        target: str,
        *,
        n: int,
        x90: TargetMap[Waveform] | None = None,
        zx90: PulseSchedule | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: PulseSchedule | None = None,
        seed: int | None = None,
        basis: Literal[
            "IX",
            "IY",
            "IZ",
            "XI",
            "XX",
            "XY",
            "XZ",
            "YI",
            "YX",
            "YY",
            "YZ",
            "ZI",
            "ZX",
            "ZY",
            "ZZ",
        ] = "ZZ",
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

            if basis == "IX":
                ps.add(target_qubit, ix90.shifted(-np.pi / 2))
            elif basis == "IY":
                ps.add(target_qubit, ix90)
            elif basis == "IZ":
                pass
            elif basis == "XI":
                ps.add(control_qubit, xi90.shifted(-np.pi / 2))
            elif basis == "XX":
                ps.add(control_qubit, xi90.shifted(-np.pi / 2))
                ps.add(target_qubit, ix90.shifted(-np.pi / 2))
            elif basis == "XY":
                ps.add(control_qubit, xi90.shifted(-np.pi / 2))
                ps.add(target_qubit, ix90)
            elif basis == "XZ":
                ps.add(control_qubit, xi90.shifted(-np.pi / 2))
            elif basis == "YI":
                ps.add(control_qubit, xi90)
            elif basis == "YX":
                ps.add(control_qubit, xi90)
                ps.add(target_qubit, ix90.shifted(-np.pi / 2))
            elif basis == "YY":
                ps.add(control_qubit, xi90)
                ps.add(target_qubit, ix90)
            elif basis == "YZ":
                ps.add(control_qubit, xi90)
            elif basis == "ZI":
                pass
            elif basis == "ZX":
                ps.add(target_qubit, ix90.shifted(-np.pi / 2))
            elif basis == "ZY":
                ps.add(target_qubit, ix90)
            elif basis == "ZZ":
                pass

        return ps

    def rb_experiment_1q(
        self,
        targets: Collection[str] | str,
        *,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: TargetMap[Waveform] | None = None,
        in_parallel: bool = False,
        shots: int | None = None,
        interval: float | None = None,
        xaxis_type: Literal["linear", "log"] | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if n_cliffords_range is not None:
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

        if max_n_cliffords is None:
            max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_1Q

        if shots is None:
            shots = DEFAULT_SHOTS

        if interval is None:
            interval = DEFAULT_INTERVAL

        if xaxis_type is None:
            xaxis_type = "linear"

        for target in targets:
            target_object = self.experiment_system.get_target(target)
            if target_object.is_cr:
                raise ValueError(f"`{target}` is not a 1Q target.")

        if in_parallel:
            target_groups = [targets]
        else:
            target_groups = [[target] for target in targets]

        def rb_sequence(
            targets: list[str],
            n_clifford: int,
            seed: int,
        ) -> PulseSchedule:
            with PulseSchedule(targets) as ps:
                for target in targets:
                    rb_sequence = self.rb_sequence_1q(
                        target,
                        n=n_clifford,
                        x90=x90.get(target) if x90 else None,
                        interleaved_waveform=interleaved_waveform.get(target)
                        if interleaved_waveform
                        else None,
                        interleaved_clifford=interleaved_clifford,
                        seed=seed,
                    )
                    ps.add(target, rb_sequence)
            return ps

        return_data = {}

        for target_group in target_groups:
            idx = 0
            sweep_range = []
            mean_data = defaultdict(list)
            std_data = defaultdict(list)
            while True:
                if n_cliffords_range is None:
                    n_clifford = 0 if idx == 0 else 2 ** (idx - 1)
                    if n_clifford > max_n_cliffords:
                        break
                else:
                    if idx >= len(n_cliffords_range):
                        break
                    n_clifford = n_cliffords_range[idx]

                idx += 1
                sweep_range.append(n_clifford)

                trial_data = defaultdict(list)
                for seed in seeds:
                    seed = int(seed)  # Ensure seed is an integer
                    result = self.measure(
                        sequence=rb_sequence(
                            n_clifford=n_clifford,
                            targets=target_group,
                            seed=seed,
                        ),
                        mode="avg",
                        shots=shots,
                        interval=interval,
                        plot=False,
                    )
                    for target, data in result.data.items():
                        iq = data.kerneled
                        z = self.rabi_params[target].normalize(iq)
                        trial_data[target].append((z + 1) / 2)

                check_vals = {}

                for target in target_group:
                    mean = np.mean(trial_data[target])
                    std = np.std(trial_data[target])
                    mean_data[target].append(mean)
                    std_data[target].append(std)
                    check_vals[target] = mean - std * 0.5

                max_check_val = np.max(list(check_vals.values()))
                if n_cliffords_range is None and max_check_val < 0.5:
                    break

            sweep_range = np.array(sweep_range, dtype=int)

            mean_data = {target: np.array(data) for target, data in mean_data.items()}
            std_data = {target: np.array(data) for target, data in std_data.items()}

            for target in target_group:
                mean = mean_data[target]
                std = std_data[target] if n_trials > 1 else None

                fit_result = fitting.fit_rb(
                    target=target,
                    x=sweep_range,
                    y=mean,
                    error_y=std,
                    bounds=((0, 0, 0), (0.5, 1, 1)),
                    title="Randomized benchmarking",
                    xlabel="Number of Cliffords",
                    ylabel="Normalized signal",
                    xaxis_type=xaxis_type,
                    yaxis_type="linear",
                    plot=plot,
                )

                if save_image:
                    viz.save_figure_image(
                        fit_result["fig"],
                        name=f"rb_experiment_1q_{target}",
                    )

                return_data[target] = {
                    "n_cliffords": sweep_range,
                    "mean": mean,
                    "std": std,
                    **fit_result,
                }

        return Result(data=return_data)

    def rb_experiment_2q(
        self,
        targets: Collection[str] | str,
        *,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
        mitigate_readout: bool = True,
        shots: int | None = None,
        interval: float | None = None,
        xaxis_type: Literal["linear", "log"] | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if self.state_centers is None:
            raise ValueError("State classifiers are not built.")

        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        targets = [
            target
            for target in targets
            if self.experiment_system.get_target(target).is_cr
            and target in self.calib_note.cr_params
        ]

        if n_cliffords_range is not None:
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

        if max_n_cliffords is None:
            max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_2Q

        if shots is None:
            shots = DEFAULT_SHOTS

        if interval is None:
            interval = DEFAULT_INTERVAL

        if shots is None:
            shots = DEFAULT_SHOTS

        if interval is None:
            interval = DEFAULT_INTERVAL

        if xaxis_type is None:
            xaxis_type = "linear"

        if in_parallel:
            target_groups = [targets]
        else:
            target_groups = [[target] for target in targets]

        for target in targets:
            target_object = self.experiment_system.get_target(target)
            if not target_object.is_cr:
                raise ValueError(f"`{target}` is not a 2Q target.")

        def rb_sequence(
            targets: list[str],
            n_clifford: int,
            seed: int,
        ) -> PulseSchedule:
            with PulseSchedule() as ps:
                seq: dict[str, PulseSchedule] = {}
                for target in targets:
                    seq[target] = self.rb_sequence_2q(
                        target=target,
                        n=n_clifford,
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

        return_data = {}
        for target_group in target_groups:
            idx = 0
            sweep_range = []
            mean_data = defaultdict(list)
            std_data = defaultdict(list)
            while True:
                if n_cliffords_range is None:
                    n_clifford = 0 if idx == 0 else 2 ** (idx - 1)
                    if n_clifford > max_n_cliffords:
                        break
                else:
                    if idx >= len(n_cliffords_range):
                        break
                    n_clifford = n_cliffords_range[idx]

                idx += 1
                sweep_range.append(n_clifford)

                trial_data = defaultdict(list)
                for seed in seeds:
                    seed = int(seed)  # Ensure seed is an integer
                    result = self.measure(
                        sequence=rb_sequence(
                            n_clifford=n_clifford,
                            targets=target_group,
                            seed=seed,
                        ),
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
                            prob = result.get_probabilities(
                                [control_qubit, target_qubit]
                            )
                        trial_data[target].append(prob["00"])

                check_vals = {}

                for target in target_group:
                    mean = np.mean(trial_data[target])
                    std = np.std(trial_data[target])
                    mean_data[target].append(mean)
                    std_data[target].append(std)
                    check_vals[target] = mean - std * 0.5

                max_check_val = np.max(list(check_vals.values()))
                if n_cliffords_range is None and max_check_val < 0.25:
                    break

            sweep_range = np.array(sweep_range, dtype=int)

            mean_data = {target: np.array(data) for target, data in mean_data.items()}
            std_data = {target: np.array(data) for target, data in std_data.items()}

            for target in target_group:
                mean = mean_data[target]
                std = std_data[target] if n_trials > 1 else None

                fit_result = fitting.fit_rb(
                    target=target,
                    x=sweep_range,
                    y=mean,
                    error_y=std,
                    dimension=4,
                    title="Randomized benchmarking",
                    xlabel="Number of Cliffords",
                    ylabel="Normalized signal",
                    xaxis_type=xaxis_type,
                    yaxis_type="linear",
                    plot=plot,
                )

                if save_image:
                    viz.save_figure_image(
                        fit_result["fig"],
                        name=f"rb_experiment_1q_{target}",
                    )

                return_data[target] = {
                    "n_cliffords": sweep_range,
                    "mean": mean,
                    "std": std,
                    **fit_result,
                }

        return Result(data=return_data)

    def pb_experiment_1q(
        self,
        targets: Collection[str] | str,
        *,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: TargetMap[Waveform] | None = None,
        in_parallel: bool = False,
        shots: int | None = None,
        interval: float | None = None,
        xaxis_type: Literal["linear", "log"] | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if n_cliffords_range is not None:
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

        if max_n_cliffords is None:
            max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_1Q

        if shots is None:
            shots = DEFAULT_SHOTS

        if interval is None:
            interval = DEFAULT_INTERVAL

        if xaxis_type is None:
            xaxis_type = "linear"

        for target in targets:
            target_object = self.experiment_system.get_target(target)
            if target_object.is_cr:
                raise ValueError(f"`{target}` is not a 1Q target.")

        if in_parallel:
            target_groups = [targets]
        else:
            target_groups = [[target] for target in targets]

        def pb_sequence(
            targets: list[str],
            n_clifford: int,
            seed: int,
        ) -> PulseSchedule:
            with PulseSchedule(targets) as ps:
                for target in targets:
                    rb_sequence = self.purity_sequence_1q(
                        target,
                        n=n_clifford,
                        x90=x90.get(target) if x90 else None,
                        interleaved_waveform=interleaved_waveform.get(target)
                        if interleaved_waveform
                        else None,
                        interleaved_clifford=interleaved_clifford,
                        seed=seed,
                    )
                    ps.add(target, rb_sequence)
            return ps

        return_data = {}

        for target_group in target_groups:
            idx = 0
            sweep_range = []
            mean_data = defaultdict(list)
            std_data = defaultdict(list)
            while True:
                if n_cliffords_range is None:
                    n_clifford = 0 if idx == 0 else 2 ** (idx - 1)
                    if n_clifford > max_n_cliffords:
                        break
                else:
                    if idx >= len(n_cliffords_range):
                        break
                    n_clifford = n_cliffords_range[idx]

                idx += 1
                sweep_range.append(n_clifford)

                trial_data = defaultdict(list)
                for seed in seeds:
                    seed = int(seed)  # Ensure seed is an integer
                    result = self.measure(
                        sequence=pb_sequence(
                            n_clifford=n_clifford,
                            targets=target_group,
                            seed=seed,
                        ),
                        mode="avg",
                        shots=shots,
                        interval=interval,
                        plot=False,
                    )
                    for target, data in result.data.items():
                        iq = data.kerneled
                        z = self.rabi_params[target].normalize(iq)
                        trial_data[target].append((z + 1) / 2)

                check_vals = {}

                for target in target_group:
                    mean = np.mean(trial_data[target])
                    std = np.std(trial_data[target])
                    mean_data[target].append(mean)
                    std_data[target].append(std)
                    check_vals[target] = mean - std * 0.5

                max_check_val = np.max(list(check_vals.values()))
                if n_cliffords_range is None and max_check_val < 0.5:
                    break

            sweep_range = np.array(sweep_range, dtype=int)

            mean_data = {target: np.array(data) for target, data in mean_data.items()}
            std_data = {target: np.array(data) for target, data in std_data.items()}

            for target in target_group:
                mean = mean_data[target]
                std = std_data[target] if n_trials > 1 else None

                fit_result = fitting.fit_rb(
                    target=target,
                    x=sweep_range,
                    y=mean,
                    error_y=std,
                    bounds=((0, 0, 0), (0.5, 1, 1)),
                    title="Purity benchmarking",
                    xlabel="Number of Cliffords",
                    ylabel="Normalized signal",
                    xaxis_type=xaxis_type,
                    yaxis_type="linear",
                    plot=plot,
                )

                if save_image:
                    viz.save_figure_image(
                        fit_result["fig"],
                        name=f"pb_experiment_1q_{target}",
                    )

                return_data[target] = {
                    "n_cliffords": sweep_range,
                    "mean": mean,
                    "std": std,
                    **fit_result,
                }

        return Result(data=return_data)

    def pb_experiment_2q(
        self,
        targets: Collection[str] | str,
        *,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
        mitigate_readout: bool = True,
        shots: int | None = None,
        interval: float | None = None,
        xaxis_type: Literal["linear", "log"] | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if self.state_centers is None:
            raise ValueError("State classifiers are not built.")

        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        targets = [
            target
            for target in targets
            if self.experiment_system.get_target(target).is_cr
            and target in self.calib_note.cr_params
        ]

        if n_cliffords_range is not None:
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

        if max_n_cliffords is None:
            max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_2Q

        if shots is None:
            shots = DEFAULT_SHOTS

        if interval is None:
            interval = DEFAULT_INTERVAL

        if shots is None:
            shots = DEFAULT_SHOTS

        if interval is None:
            interval = DEFAULT_INTERVAL

        if xaxis_type is None:
            xaxis_type = "linear"

        if in_parallel:
            target_groups = [targets]
        else:
            target_groups = [[target] for target in targets]

        for target in targets:
            target_object = self.experiment_system.get_target(target)
            if not target_object.is_cr:
                raise ValueError(f"`{target}` is not a 2Q target.")

        def pb_sequence(
            targets: list[str],
            n_clifford: int,
            seed: int,
        ) -> PulseSchedule:
            with PulseSchedule() as ps:
                seq: dict[str, PulseSchedule] = {}
                for target in targets:
                    seq[target] = self.purity_sequence_2q(
                        target=target,
                        n=n_clifford,
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

        return_data = {}
        for target_group in target_groups:
            idx = 0
            sweep_range = []
            mean_data = defaultdict(list)
            std_data = defaultdict(list)
            while True:
                if n_cliffords_range is None:
                    n_clifford = 0 if idx == 0 else 2 ** (idx - 1)
                    if n_clifford > max_n_cliffords:
                        break
                else:
                    if idx >= len(n_cliffords_range):
                        break
                    n_clifford = n_cliffords_range[idx]

                idx += 1
                sweep_range.append(n_clifford)

                trial_data = defaultdict(list)
                for seed in seeds:
                    seed = int(seed)  # Ensure seed is an integer
                    result = self.measure(
                        sequence=pb_sequence(
                            n_clifford=n_clifford,
                            targets=target_group,
                            seed=seed,
                        ),
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
                            prob = result.get_probabilities(
                                [control_qubit, target_qubit]
                            )
                        trial_data[target].append(prob["00"])

                check_vals = {}

                for target in target_group:
                    mean = np.mean(trial_data[target])
                    std = np.std(trial_data[target])
                    mean_data[target].append(mean)
                    std_data[target].append(std)
                    check_vals[target] = mean - std * 0.5

                max_check_val = np.max(list(check_vals.values()))
                if n_cliffords_range is None and max_check_val < 0.25:
                    break

            sweep_range = np.array(sweep_range, dtype=int)

            mean_data = {target: np.array(data) for target, data in mean_data.items()}
            std_data = {target: np.array(data) for target, data in std_data.items()}

            for target in target_group:
                mean = mean_data[target]
                std = std_data[target] if n_trials > 1 else None

                fit_result = fitting.fit_rb(
                    target=target,
                    x=sweep_range,
                    y=mean,
                    error_y=std,
                    dimension=4,
                    title="Purity benchmarking",
                    xlabel="Number of Cliffords",
                    ylabel="Normalized signal",
                    xaxis_type=xaxis_type,
                    yaxis_type="linear",
                    plot=plot,
                )

                if save_image:
                    viz.save_figure_image(
                        fit_result["fig"],
                        name=f"pb_experiment_2q_{target}",
                    )

                return_data[target] = {
                    "n_cliffords": sweep_range,
                    "mean": mean,
                    "std": std,
                    **fit_result,
                }

        return Result(data=return_data)

    def irb_experiment(
        self,
        targets: Collection[str] | str,
        *,
        interleaved_clifford: str | Clifford,
        interleaved_waveform: TargetMap[PulseSchedule]
        | TargetMap[Waveform]
        | None = None,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if isinstance(interleaved_clifford, str):
            clifford = self.clifford.get(interleaved_clifford)
            if clifford is None:
                raise ValueError(f"Invalid Clifford: {interleaved_clifford}")
            interleaved_clifford = clifford

        is_2q = self.experiment_system.get_target(targets[0]).is_cr

        if is_2q:
            dimension = 4
            rb_result = self.rb_experiment_2q(
                targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                zx90=zx90,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )
            irb_result = self.rb_experiment_2q(
                targets=targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                zx90=zx90,
                interleaved_waveform=interleaved_waveform,  # type: ignore
                interleaved_clifford=interleaved_clifford,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )
        else:
            dimension = 2
            rb_result = self.rb_experiment_1q(
                targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )
            irb_result = self.rb_experiment_1q(
                targets=targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                interleaved_waveform=interleaved_waveform,  # type: ignore
                interleaved_clifford=interleaved_clifford,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )

        results = {}
        for target in targets:
            rb_n_cliffords = rb_result[target]["n_cliffords"]
            rb_mean = rb_result[target]["mean"]
            rb_std = rb_result[target]["std"]
            rb_fit_result = fitting.fit_rb(
                target=target,
                x=rb_n_cliffords,
                y=rb_mean,
                error_y=rb_std,
                dimension=dimension,
                plot=False,
            )
            A_rb = rb_fit_result["A"]
            p_rb = rb_fit_result["p"]
            p_rb_err = rb_fit_result["p_err"]
            C_rb = rb_fit_result["C"]
            avg_gate_error_rb = rb_fit_result["avg_gate_error"]
            avg_gate_fidelity_rb = rb_fit_result["avg_gate_fidelity"]
            avg_gate_fidelity_err_rb = rb_fit_result["avg_gate_fidelity_err"]

            irb_n_cliffords = irb_result[target]["n_cliffords"]
            irb_mean = irb_result[target]["mean"]
            irb_std = irb_result[target]["std"]
            irb_fit_result = fitting.fit_rb(
                target=target,
                x=irb_n_cliffords,
                y=irb_mean,
                error_y=irb_std,
                dimension=dimension,
                plot=False,
                title="Interleaved randomized benchmarking",
            )
            A_irb = irb_fit_result["A"]
            p_irb = irb_fit_result["p"]
            p_irb_err = irb_fit_result["p_err"]
            C_irb = irb_fit_result["C"]
            avg_gate_fidelity_irb = irb_fit_result["avg_gate_fidelity"]
            avg_gate_fidelity_err_irb = irb_fit_result["avg_gate_fidelity_err"]

            gate_error = (dimension - 1) * (1 - (p_irb / p_rb)) / dimension
            gate_fidelity = 1 - gate_error

            gate_fidelity_err = (
                (dimension - 1)
                / dimension
                * np.sqrt((p_irb_err / p_rb) ** 2 + (p_rb_err * p_irb / p_rb**2) ** 2)
            )

            fig = fitting.plot_irb(
                target=target,
                x=rb_n_cliffords,
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

            if gate_error < 0.1 * avg_gate_error_rb:
                # TODO: use a more appropriate threshold based on the system.
                # NOTE: average number of gates per 2Q Clifford: 1Q=2.589, 2Q=1.5
                print(
                    f"Warning: Gate error ({gate_error * 100:.3f}%) is too low compared to the average gate error (RB) ({avg_gate_error_rb * 100:.3f}%)."
                )

            results[target] = {
                "gate_error": gate_error,
                "gate_fidelity": gate_fidelity,
                "gate_fidelity_err": gate_fidelity_err,
                "rb_fit_result": rb_fit_result,
                "irb_fit_result": irb_fit_result,
                "fig": fig,
            }
        return Result(data=results)

    def ipb_experiment(
        self,
        targets: Collection[str] | str,
        *,
        interleaved_clifford: str | Clifford,
        interleaved_waveform: TargetMap[PulseSchedule]
        | TargetMap[Waveform]
        | None = None,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if isinstance(interleaved_clifford, str):
            clifford = self.clifford.get(interleaved_clifford)
            if clifford is None:
                raise ValueError(f"Invalid Clifford: {interleaved_clifford}")
            interleaved_clifford = clifford

        is_2q = self.experiment_system.get_target(targets[0]).is_cr

        if is_2q:
            dimension = 4
            rb_result = self.pb_experiment_2q(
                targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                zx90=zx90,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )
            irb_result = self.pb_experiment_2q(
                targets=targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                zx90=zx90,
                interleaved_waveform=interleaved_waveform,  # type: ignore
                interleaved_clifford=interleaved_clifford,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )
        else:
            dimension = 2
            rb_result = self.pb_experiment_1q(
                targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )
            irb_result = self.pb_experiment_1q(
                targets=targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                interleaved_waveform=interleaved_waveform,  # type: ignore
                interleaved_clifford=interleaved_clifford,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=False,
                save_image=False,
            )

        results = {}
        for target in targets:
            rb_n_cliffords = rb_result[target]["n_cliffords"]
            rb_mean = rb_result[target]["mean"]
            rb_std = rb_result[target]["std"]
            rb_fit_result = fitting.fit_rb(
                target=target,
                x=rb_n_cliffords,
                y=rb_mean,
                error_y=rb_std,
                dimension=dimension,
                plot=False,
            )
            A_rb = rb_fit_result["A"]
            p_rb = rb_fit_result["p"]
            p_rb_err = rb_fit_result["p_err"]
            C_rb = rb_fit_result["C"]
            avg_gate_error_rb = rb_fit_result["avg_gate_error"]
            avg_gate_fidelity_rb = rb_fit_result["avg_gate_fidelity"]
            avg_gate_fidelity_err_rb = rb_fit_result["avg_gate_fidelity_err"]

            irb_n_cliffords = irb_result[target]["n_cliffords"]
            irb_mean = irb_result[target]["mean"]
            irb_std = irb_result[target]["std"]
            irb_fit_result = fitting.fit_rb(
                target=target,
                x=irb_n_cliffords,
                y=irb_mean,
                error_y=irb_std,
                dimension=dimension,
                plot=False,
                title="Interleaved Purity benchmarking",
            )
            A_irb = irb_fit_result["A"]
            p_irb = irb_fit_result["p"]
            p_irb_err = irb_fit_result["p_err"]
            C_irb = irb_fit_result["C"]
            avg_gate_fidelity_irb = irb_fit_result["avg_gate_fidelity"]
            avg_gate_fidelity_err_irb = irb_fit_result["avg_gate_fidelity_err"]

            gate_error = (dimension - 1) * (1 - (p_irb / p_rb)) / dimension
            gate_fidelity = 1 - gate_error

            gate_fidelity_err = (
                (dimension - 1)
                / dimension
                * np.sqrt((p_irb_err / p_rb) ** 2 + (p_rb_err * p_irb / p_rb**2) ** 2)
            )

            fig = fitting.plot_irb(
                target=target,
                x=rb_n_cliffords,
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
                title=f"Interleaved Purity benchmarking of {clifford.name}",
                xlabel="Number of Cliffords",
                ylabel="Normalized signal",
            )
            if save_image:
                viz.save_figure_image(
                    fig,
                    name=f"interleaved_purity_benchmarking_{target}",
                )

            print()
            print(
                f"Average gate purity (PB)  : {avg_gate_fidelity_rb * 100:.3f} ± {avg_gate_fidelity_err_rb * 100:.3f}%"
            )
            print(
                f"Average gate purity (IPB) : {avg_gate_fidelity_irb * 100:.3f} ± {avg_gate_fidelity_err_irb * 100:.3f}%"
            )
            print()
            print(
                f"Gate error    : {gate_error * 100:.3f} ± {gate_fidelity_err * 100:.3f}%"
            )
            print(
                f"Gate fidelity : {gate_fidelity * 100:.3f} ± {gate_fidelity_err * 100:.3f}%"
            )
            print()

            if gate_error < 0.1 * avg_gate_error_rb:
                # TODO: use a more appropriate threshold based on the system.
                # NOTE: average number of gates per 2Q Clifford: 1Q=2.589, 2Q=1.5
                print(
                    f"Warning: Gate error ({gate_error * 100:.3f}%) is too low compared to the average gate error (PB) ({avg_gate_error_rb * 100:.3f}%)."
                )

            results[target] = {
                "gate_error": gate_error,
                "gate_fidelity": gate_fidelity,
                "gate_fidelity_err": gate_fidelity_err,
                "rb_fit_result": rb_fit_result,
                "irb_fit_result": irb_fit_result,
                "fig": fig,
            }
        return Result(data=results)

    def randomized_benchmarking(
        self,
        targets: Collection[str] | str,
        *,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
        xaxis_type: Literal["linear", "log"] | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        target_object = self.experiment_system.get_target(targets[0])
        is_2q = target_object.is_cr

        if is_2q:
            return self.rb_experiment_2q(
                targets=targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                zx90=zx90,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                xaxis_type=xaxis_type,
                plot=plot,
                save_image=save_image,
            )
        else:
            return self.rb_experiment_1q(
                targets=targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                xaxis_type=xaxis_type,
                plot=plot,
                save_image=save_image,
            )

    def interleaved_randomized_benchmarking(
        self,
        targets: Collection[str] | str,
        *,
        interleaved_clifford: str | Clifford,
        interleaved_waveform: TargetMap[PulseSchedule]
        | TargetMap[Waveform]
        | None = None,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if in_parallel:
            result = self.irb_experiment(
                targets=targets,
                interleaved_clifford=interleaved_clifford,
                interleaved_waveform=interleaved_waveform,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                zx90=zx90,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=plot,
                save_image=save_image,
            )
        else:
            results = {}
            for target in targets:
                result = self.irb_experiment(
                    targets=target,
                    interleaved_clifford=interleaved_clifford,
                    interleaved_waveform=interleaved_waveform,
                    n_cliffords_range=n_cliffords_range,
                    n_trials=n_trials,
                    seeds=seeds,
                    max_n_cliffords=max_n_cliffords,
                    x90=x90,
                    zx90=zx90,
                    in_parallel=in_parallel,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                    save_image=save_image,
                )
                results[target] = result[target]
            result = Result(data=results)

        return Result(data=result.data)

    def purity_benchmarking(
        self,
        targets: Collection[str] | str,
        *,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
        xaxis_type: Literal["linear", "log"] | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        target_object = self.experiment_system.get_target(targets[0])
        is_2q = target_object.is_cr

        if is_2q:
            return self.pb_experiment_2q(
                targets=targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                zx90=zx90,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                xaxis_type=xaxis_type,
                plot=plot,
                save_image=save_image,
            )
        else:
            return self.pb_experiment_1q(
                targets=targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                xaxis_type=xaxis_type,
                plot=plot,
                save_image=save_image,
            )

    def interleaved_purity_benchmarking(
        self,
        targets: Collection[str] | str,
        *,
        interleaved_clifford: str | Clifford,
        interleaved_waveform: TargetMap[PulseSchedule]
        | TargetMap[Waveform]
        | None = None,
        n_cliffords_range: ArrayLike | None = None,
        n_trials: int | None = None,
        seeds: ArrayLike | None = None,
        max_n_cliffords: int | None = None,
        x90: TargetMap[Waveform] | None = None,
        zx90: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool = False,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if in_parallel:
            result = self.ipb_experiment(
                targets=targets,
                interleaved_clifford=interleaved_clifford,
                interleaved_waveform=interleaved_waveform,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                zx90=zx90,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=plot,
                save_image=save_image,
            )
        else:
            results = {}
            for target in targets:
                result = self.ipb_experiment(
                    targets=target,
                    interleaved_clifford=interleaved_clifford,
                    interleaved_waveform=interleaved_waveform,
                    n_cliffords_range=n_cliffords_range,
                    n_trials=n_trials,
                    seeds=seeds,
                    max_n_cliffords=max_n_cliffords,
                    x90=x90,
                    zx90=zx90,
                    in_parallel=in_parallel,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                    save_image=save_image,
                )
                results[target] = result[target]
            result = Result(data=results)

        return Result(data=result.data)

    def benchmark_1q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_trials: int | None = None,
        in_parallel: bool = False,
        shots: int | None = None,
        interval: float | None = None,
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
                interleaved_clifford="X90",
                n_trials=n_trials,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=plot,
                save_image=save_image,
            )
            self.interleaved_randomized_benchmarking(
                targets,
                interleaved_clifford="X180",
                n_trials=n_trials,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=plot,
                save_image=save_image,
            )
        else:
            for target in targets:
                try:
                    self.interleaved_randomized_benchmarking(
                        target,
                        interleaved_clifford="X90",
                        n_trials=n_trials,
                        in_parallel=in_parallel,
                        shots=shots,
                        interval=interval,
                        plot=plot,
                        save_image=save_image,
                    )
                    self.interleaved_randomized_benchmarking(
                        target,
                        interleaved_clifford="X180",
                        n_trials=n_trials,
                        in_parallel=in_parallel,
                        shots=shots,
                        interval=interval,
                        plot=plot,
                        save_image=save_image,
                    )
                except Exception as e:
                    print(f"Failed to benchmark {target}: {e}")
                    continue

    def benchmark_2q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_trials: int | None = None,
        in_parallel: bool = False,
        shots: int | None = None,
        interval: float | None = None,
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
                interleaved_clifford="ZX90",
                n_trials=n_trials,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                plot=plot,
                save_image=save_image,
            )
        else:
            for target in targets:
                try:
                    self.interleaved_randomized_benchmarking(
                        target,
                        interleaved_clifford="ZX90",
                        n_trials=n_trials,
                        in_parallel=in_parallel,
                        shots=shots,
                        interval=interval,
                        plot=plot,
                        save_image=save_image,
                    )
                except Exception as e:
                    print(f"Failed to benchmark {target}: {e}")
                    continue
