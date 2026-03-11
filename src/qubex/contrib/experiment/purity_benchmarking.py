"""Contributed purity benchmarking helper functions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

import qubex.visualization as viz
from qubex.analysis import fitting
from qubex.clifford import Clifford
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_MAX_N_CLIFFORDS_1Q,
    DEFAULT_MAX_N_CLIFFORDS_2Q,
    DEFAULT_RB_N_TRIALS,
    DEFAULT_SHOTS,
)
from qubex.experiment.models import Result
from qubex.pulse import PulseArray, PulseSchedule, VirtualZ, Waveform
from qubex.typing import TargetMap

from ._deprecated_options import resolve_shot_options


def purity_sequence_1q(
    exp: Experiment,
    target: str,
    *,
    n: int,
    x90: Waveform | None = None,
    interleaved_clifford: Clifford | None = None,
    interleaved_waveform: Waveform | None = None,
    seed: int | None = None,
    basis: Literal["X", "Y", "Z"] | None = None,
) -> PulseArray:
    """
    Build a single-qubit purity benchmarking sequence.

    Parameters
    ----------
    exp
        Experiment instance that provides pulse and Clifford services.
    target
        Target qubit label.
    n
        Number of random Cliffords.
    x90
        Optional `X90` waveform override.
    interleaved_clifford
        Optional interleaved Clifford.
    interleaved_waveform
        Optional waveform for the interleaved Clifford.
    seed
        Seed for random Clifford generation.
    basis
        Measurement basis for the final rotation.

    Returns
    -------
    PulseArray
        Purity benchmarking pulse sequence.
    """
    if basis is None:
        basis = "Z"

    x90_waveform = x90 or exp.pulse.x90(target)
    z90 = VirtualZ(np.pi / 2)
    y90m = x90_waveform.shifted(-np.pi / 2)
    sequence: list[Waveform | VirtualZ] = []

    clifford_generator = exp.benchmarking_service.clifford_generator

    if interleaved_clifford is None:
        cliffords, _inverse = clifford_generator.create_rb_sequences(
            n=n,
            type="1Q",
            seed=seed,
        )
    else:
        if interleaved_waveform is None:
            if interleaved_clifford.name == "X90":
                interleaved_waveform = exp.pulse.x90(target)
            elif interleaved_clifford.name == "X180":
                interleaved_waveform = exp.pulse.x180(target)
            else:
                raise ValueError("interleaved_waveform must be provided.")
        cliffords, _inverse = clifford_generator.create_irb_sequences(
            n=n,
            interleave=interleaved_clifford,
            type="1Q",
            seed=seed,
        )

    def add_gate(gate: str) -> None:
        if gate == "X90":
            sequence.append(x90_waveform)
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
        sequence.append(x90_waveform)
    elif basis == "Z":
        pass

    return PulseArray(sequence)


def purity_sequence_2q(
    exp: Experiment,
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
    ]
    | None = None,
) -> PulseSchedule:
    """Build a two-qubit purity benchmarking sequence."""
    if basis is None:
        basis = "ZZ"

    target_object = exp.ctx.experiment_system.get_target(target)
    if not target_object.is_cr:
        raise ValueError(f"`{target}` is not a 2Q target.")

    control_qubit, target_qubit = exp.cr_pair(target)
    cr_label = target

    xi90 = x90.get(control_qubit) if x90 is not None else None
    ix90 = x90.get(target_qubit) if x90 is not None else None
    xi90 = xi90 or exp.pulse.x90(control_qubit)
    ix90 = ix90 or exp.pulse.x90(target_qubit)
    z90 = VirtualZ(np.pi / 2)

    if zx90 is None:
        zx90 = exp.pulse.zx90(control_qubit, target_qubit)

    clifford_generator = exp.benchmarking_service.clifford_generator

    if interleaved_clifford is None:
        cliffords, _inverse = clifford_generator.create_rb_sequences(
            n=n,
            type="2Q",
            seed=seed,
        )
    else:
        if interleaved_waveform is None:
            if interleaved_clifford.name == "ZX90":
                interleaved_waveform = exp.pulse.zx90(control_qubit, target_qubit)
            else:
                raise ValueError("interleaved_waveform must be provided.")
        cliffords, _inverse = clifford_generator.create_irb_sequences(
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


def pb_experiment_1q(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    interleaved_clifford: Clifford | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
    in_parallel: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Run single-qubit purity benchmarking.

    Parameters
    ----------
    targets
        Target qubits to benchmark.
    n_cliffords_range
        Cliffords count sweep range.
    n_trials
        Number of random trials.
    """
    if isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    if in_parallel is None:
        in_parallel = False
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True

    if n_cliffords_range is not None:
        n_cliffords_range = np.array(n_cliffords_range, dtype=int)

    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS

    if seeds is None:
        seeds = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seeds = np.array(seeds, dtype=int)
        if len(seeds) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )

    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_1Q

    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="pb_experiment_1q",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS

    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    if xaxis_type is None:
        xaxis_type = "linear"

    for target in targets:
        target_object = exp.ctx.experiment_system.get_target(target)
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
                rb_sequence = purity_sequence_1q(
                    exp,
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
                seed = int(seed)
                result = exp.measurement_service.measure(
                    sequence=pb_sequence(
                        n_clifford=n_clifford,
                        targets=target_group,
                        seed=seed,
                    ),
                    mode="avg",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                )
                for target, data in result.data.items():
                    iq = data.kerneled
                    z = exp.pulse.rabi_params[target].normalize(iq)
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
                viz.save_figure(
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
    exp: Experiment,
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
    in_parallel: bool | None = None,
    mitigate_readout: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Run two-qubit purity benchmarking.

    Parameters
    ----------
    targets
        Target CR labels to benchmark.
    n_cliffords_range
        Cliffords count sweep range.
    n_trials
        Number of random trials.
    """
    if in_parallel is None:
        in_parallel = False
    if mitigate_readout is None:
        mitigate_readout = True
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True

    if exp.ctx.state_centers is None:
        raise ValueError("State classifiers are not built.")

    if isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    targets = [
        target
        for target in targets
        if exp.ctx.experiment_system.get_target(target).is_cr
        and target in exp.ctx.calib_note.cr_params
    ]

    if n_cliffords_range is not None:
        n_cliffords_range = np.array(n_cliffords_range, dtype=int)

    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS

    if seeds is None:
        seeds = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seeds = np.array(seeds, dtype=int)
        if len(seeds) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )

    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_2Q

    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="pb_experiment_2q",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS

    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    if xaxis_type is None:
        xaxis_type = "linear"

    if in_parallel:
        target_groups = [targets]
    else:
        target_groups = [[target] for target in targets]

    for target in targets:
        target_object = exp.ctx.experiment_system.get_target(target)
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
                seq[target] = purity_sequence_2q(
                    exp,
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

            for target in targets:
                ps.call(
                    seq[target].padded(
                        total_duration=max_duration,
                        pad_side="left",
                        deepcopy=False,
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
                seed = int(seed)
                result = exp.measurement_service.measure(
                    sequence=pb_sequence(
                        n_clifford=n_clifford,
                        targets=target_group,
                        seed=seed,
                    ),
                    mode="single",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                )

                for target in target_group:
                    control_qubit, target_qubit = exp.cr_pair(target)
                    if mitigate_readout:
                        prob = result.get_mitigated_probabilities(
                            [control_qubit, target_qubit]
                        )
                    else:
                        prob = result.get_probabilities([control_qubit, target_qubit])
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
                viz.save_figure(
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


def ipb_experiment(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    interleaved_clifford: str | Clifford,
    interleaved_waveform: TargetMap[PulseSchedule] | TargetMap[Waveform] | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    zx90: TargetMap[PulseSchedule] | None = None,
    in_parallel: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """Run interleaved purity benchmarking."""
    if isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    if in_parallel is None:
        in_parallel = False
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="ipb_experiment",
    )

    clifford_obj: Clifford
    if isinstance(interleaved_clifford, str):
        clifford_found = exp.benchmarking_service.clifford.get(interleaved_clifford)
        if clifford_found is None:
            raise ValueError(f"Invalid Clifford: {interleaved_clifford}")
        clifford_obj = clifford_found
    else:
        clifford_obj = interleaved_clifford

    is_2q = exp.ctx.experiment_system.get_target(targets[0]).is_cr

    if is_2q:
        dimension = 4
        rb_result = pb_experiment_2q(
            exp,
            targets,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            zx90=zx90,
            in_parallel=in_parallel,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            save_image=False,
        )
        irb_result = pb_experiment_2q(
            exp,
            targets=targets,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            zx90=zx90,
            interleaved_waveform=interleaved_waveform,  # type: ignore[arg-type]
            interleaved_clifford=clifford_obj,
            in_parallel=in_parallel,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            save_image=False,
        )
    else:
        dimension = 2
        rb_result = pb_experiment_1q(
            exp,
            targets,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            in_parallel=in_parallel,
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=False,
            save_image=False,
        )
        irb_result = pb_experiment_1q(
            exp,
            targets=targets,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            interleaved_waveform=interleaved_waveform,  # type: ignore[arg-type]
            interleaved_clifford=clifford_obj,
            in_parallel=in_parallel,
            n_shots=n_shots,
            shot_interval=shot_interval,
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
            title=f"Interleaved Purity benchmarking of {clifford_obj.name}",
            xlabel="Number of Cliffords",
            ylabel="Normalized signal",
        )
        if save_image:
            viz.save_figure(
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


def purity_benchmarking(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    zx90: TargetMap[PulseSchedule] | None = None,
    in_parallel: bool | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """Dispatch purity benchmarking based on target type."""
    if isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    target_object = exp.ctx.experiment_system.get_target(targets[0])
    is_2q = target_object.is_cr
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="purity_benchmarking",
    )

    if is_2q:
        return pb_experiment_2q(
            exp,
            targets=targets,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            zx90=zx90,
            in_parallel=in_parallel,
            n_shots=n_shots,
            shot_interval=shot_interval,
            xaxis_type=xaxis_type,
            plot=plot,
            save_image=save_image,
        )

    return pb_experiment_1q(
        exp,
        targets=targets,
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        x90=x90,
        in_parallel=in_parallel,
        n_shots=n_shots,
        shot_interval=shot_interval,
        xaxis_type=xaxis_type,
        plot=plot,
        save_image=save_image,
    )


def interleaved_purity_benchmarking(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    interleaved_clifford: str | Clifford,
    interleaved_waveform: TargetMap[PulseSchedule] | TargetMap[Waveform] | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    zx90: TargetMap[PulseSchedule] | None = None,
    in_parallel: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """Dispatch interleaved purity benchmarking."""
    if isinstance(targets, str):
        targets = [targets]
    else:
        targets = list(targets)

    if in_parallel is None:
        in_parallel = False
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="interleaved_purity_benchmarking",
    )

    if in_parallel:
        result = ipb_experiment(
            exp,
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
            n_shots=n_shots,
            shot_interval=shot_interval,
            plot=plot,
            save_image=save_image,
        )
    else:
        results = {}
        for target in targets:
            result = ipb_experiment(
                exp,
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
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=plot,
                save_image=save_image,
            )
            results[target] = result[target]
        result = Result(data=results)

    return Result(data=result.data)
