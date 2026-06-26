"""Benchmarking service for randomized benchmarking experiments."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from qxpulse import PulseArray, PulseSchedule, VirtualZ, Waveform

import qubex.visualization as viz
from qubex.analysis import fitting
from qubex.clifford.clifford import Clifford
from qubex.clifford.clifford_generator import CliffordGenerator
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_MAX_N_CLIFFORDS_1Q,
    DEFAULT_MAX_N_CLIFFORDS_2Q,
    DEFAULT_RB_N_TRIALS,
    DEFAULT_SHOTS,
)
from qubex.experiment.experiment_context import ExperimentContext
from qubex.experiment.models.result import Result
from qubex.system import TargetType
from qubex.typing import TargetMap

from .measurement_service import MeasurementService
from .pulse_service import PulseService

logger = logging.getLogger(__name__)

Native2QGate = Literal["ZX90", "BSWAP"]


def _n_cliffords_ranges_by_target(
    rb_result: Result,
    targets: Collection[str],
) -> dict[str, np.ndarray]:
    return {
        target: np.asarray(rb_result[target]["n_cliffords"], dtype=int)
        for target in targets
    }


def _shared_n_cliffords_range(
    ranges_by_target: Mapping[str, np.ndarray],
) -> np.ndarray | None:
    ranges = list(ranges_by_target.values())
    if not ranges:
        return None

    first_range = ranges[0]
    if all(np.array_equal(first_range, n_cliffords) for n_cliffords in ranges[1:]):
        return first_range
    return None


def _rb_curve_data(entry: Mapping[str, object]) -> dict[str, object]:
    """Return RB curve payload fields that should be preserved in IRB results."""
    return {
        key: entry[key]
        for key in ("n_cliffords", "mean", "std", "trials", "seeds")
        if key in entry
    }


class BenchmarkingService:
    """Service for randomized benchmarking workflows."""

    def __init__(
        self,
        *,
        experiment_context: ExperimentContext,
        measurement_service: MeasurementService,
        pulse_service: PulseService,
    ):
        self._experiment_context: ExperimentContext = experiment_context
        self._measurement_service = measurement_service
        self._pulse_service = pulse_service
        self._clifford_generator_dict: dict[str, CliffordGenerator] = {}

    @property
    def ctx(self) -> ExperimentContext:
        """Return the experiment context."""
        return self._experiment_context

    @property
    def pulse(self) -> PulseService:
        """Return the pulse service."""
        return self._pulse_service

    @property
    def measurement_service(self) -> MeasurementService:
        """Return the measurement service."""
        return self._measurement_service

    @property
    def clifford_generator(self) -> CliffordGenerator:
        """Return the Clifford generator instance."""
        return self._get_clifford_generator()

    @property
    def clifford(self) -> dict[str, Clifford]:
        """Return the Clifford dictionary."""
        return self.clifford_generator.cliffords

    def _get_clifford_generator(
        self,
        file_name: str | None = None,
    ) -> CliffordGenerator:
        """Return a Clifford generator cached by its 2Q table."""
        key = file_name or "default"
        if key not in self._clifford_generator_dict:
            if file_name is None:
                generator = CliffordGenerator()
            else:
                generator = CliffordGenerator(auto_load=False)
                generator.load("2Q", file_name=file_name)
            self._clifford_generator_dict[key] = generator
        return self._clifford_generator_dict[key]

    @staticmethod
    def _float_mapping(
        value: object,
        *,
        field_name: str,
        target: str,
    ) -> dict[str, float]:
        """Return a string-keyed float mapping from calibration-note data."""
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"{field_name} for `{target}` must be a mapping.")

        result: dict[str, float] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} for `{target}` must use string keys.")
            try:
                result[key] = float(item)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{field_name}[{key!r}] for `{target}` must be a number."
                ) from None
        return result

    def _get_bswap_post_z(
        self,
        target: str,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return logical post-Z bSWAP calibration data for a target."""
        # Wei et al. 2024 writes the calibrated Stark bSWAP as
        # U_g ; Z_0(omega_s * t_end + phi_1) ; Z_1(omega_s * t_end + phi_2).
        # Store phi_q in post_z_offsets and omega_s-like coefficients in
        # post_z_update_rates, using logical Z angles.
        param = self.ctx.calib_note.get_bswap_param(target)
        if param is None:
            raise ValueError(
                f"bSWAP calibration parameters are missing for `{target}`."
            )
        return (
            self._float_mapping(
                param.get("post_z_offsets"),
                field_name="post_z_offsets",
                target=target,
            ),
            self._float_mapping(
                param.get("post_z_update_rates"),
                field_name="post_z_update_rates",
                target=target,
            ),
        )

    def rb_sequence(
        self,
        target: str,
        *,
        n: int,
        x90: Waveform | TargetMap[Waveform] | None = None,
        zx90: PulseSchedule | None = None,
        interleaved_waveform: Waveform | PulseSchedule | None = None,
        interleaved_clifford: Clifford | None = None,
        native_2q_gate: Native2QGate | None = None,
        native_2q_waveform: PulseSchedule | None = None,
        seed: int | None = None,
    ) -> PulseSchedule:
        """Build a randomized benchmarking sequence."""
        target_object = self.ctx.experiment_system.get_target(target)
        if target_object.is_2q:
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
                native_2q_gate=native_2q_gate,
                native_2q_waveform=native_2q_waveform,
                seed=seed,
            )
            return sched
        else:
            if native_2q_gate is not None or native_2q_waveform is not None:
                raise ValueError("Native 2Q options are only valid for 2Q RB.")
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
        """Build a single-qubit RB pulse sequence."""
        x90 = x90 or self.pulse.x90(target)
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
                    interleaved_waveform = self.pulse.x90(target)
                elif interleaved_clifford.name == "X180":
                    interleaved_waveform = self.pulse.x180(target)
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
        native_2q_gate: Native2QGate | None = None,
        native_2q_waveform: PulseSchedule | None = None,
        seed: int | None = None,
    ) -> PulseSchedule:
        """Build a two-qubit RB pulse schedule."""
        target_object = self.ctx.experiment_system.get_target(target)
        if not target_object.is_2q:
            raise ValueError(f"`{target}` is not a 2Q target.")

        control_qubit, target_qubit = self.ctx.resolve_2q_qubits(target)

        xi90 = x90.get(control_qubit) if x90 is not None else None
        ix90 = x90.get(target_qubit) if x90 is not None else None
        xi90 = xi90 or self.pulse.x90(control_qubit)
        ix90 = ix90 or self.pulse.x90(target_qubit)

        if native_2q_gate is None:
            if target_object.is_bswap:
                native_2q_gate = "BSWAP"
            elif target_object.type == TargetType.CTRL_2Q:
                raise ValueError(
                    "native_2q_gate must be provided for generic 2Q targets."
                )
            else:
                native_2q_gate = "ZX90"
        elif native_2q_gate not in ("ZX90", "BSWAP"):
            raise ValueError(f"Unsupported native 2Q gate: {native_2q_gate}")

        if native_2q_gate == "ZX90":
            cr_label = target
            if native_2q_waveform is not None:
                zx90 = native_2q_waveform
            elif zx90 is None:
                zx90 = self.pulse.zx90(control_qubit, target_qubit)
            return self._rb_sequence_2q_zx90(
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                cr_label=cr_label,
                n=n,
                xi90=xi90,
                ix90=ix90,
                zx90=zx90,
                interleaved_clifford=interleaved_clifford,
                interleaved_waveform=interleaved_waveform,
                seed=seed,
            )

        if native_2q_gate == "BSWAP":
            if native_2q_waveform is None:
                raise ValueError("native_2q_waveform must be provided for BSWAP.")
            post_z_offsets, post_z_update_rates = self._get_bswap_post_z(target)
            return self._rb_sequence_2q_bswap(
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                bswap_label=target,
                n=n,
                xi90=xi90,
                ix90=ix90,
                bswap=native_2q_waveform,
                post_z_offsets=post_z_offsets,
                post_z_update_rates=post_z_update_rates,
                interleaved_clifford=interleaved_clifford,
                interleaved_waveform=interleaved_waveform,
                seed=seed,
            )

        raise ValueError("Only ZX90 and BSWAP native 2Q RB are supported.")

    def _rb_sequence_2q_zx90(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        cr_label: str,
        n: int,
        xi90: Waveform,
        ix90: Waveform,
        zx90: PulseSchedule,
        interleaved_clifford: Clifford | None,
        interleaved_waveform: PulseSchedule | None,
        seed: int | None,
    ) -> PulseSchedule:
        """Build a ZX90-native two-qubit RB pulse schedule."""
        z90 = VirtualZ(np.pi / 2)
        clifford_generator = self._get_clifford_generator()

        if interleaved_clifford is None:
            cliffords, inverse = clifford_generator.create_rb_sequences(
                n=n,
                type="2Q",
                seed=seed,
            )
        else:
            if interleaved_waveform is None:
                if interleaved_clifford.name == "ZX90":
                    interleaved_waveform = zx90
                else:
                    raise ValueError("interleaved_waveform must be provided.")
            cliffords, inverse = clifford_generator.create_irb_sequences(
                n=n,
                interleave=interleaved_clifford,
                type="2Q",
                seed=seed,
            )

        with PulseSchedule([control_qubit, cr_label, target_qubit]) as ps:

            def add_2q_gate(waveform: PulseSchedule):
                ps.barrier()
                ps.call(waveform)
                ps.barrier()

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
                    add_2q_gate(zx90)
                else:
                    raise ValueError("Invalid gate.")

            for clifford in cliffords:
                for gate in clifford:
                    add_gate(gate)
                if interleaved_waveform is not None:
                    add_2q_gate(interleaved_waveform)

            for gate in inverse:
                add_gate(gate)
        return ps

    def _rb_sequence_2q_bswap(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        bswap_label: str,
        n: int,
        xi90: Waveform,
        ix90: Waveform,
        bswap: PulseSchedule,
        post_z_offsets: Mapping[str, float],
        post_z_update_rates: Mapping[str, float],
        interleaved_clifford: Clifford | None,
        interleaved_waveform: PulseSchedule | None,
        seed: int | None,
    ) -> PulseSchedule:
        """Build a bSWAP-native two-qubit RB pulse schedule."""
        clifford_generator = self._get_clifford_generator("clifford_list_2q_bswap")
        post_z_offsets = dict(post_z_offsets)
        post_z_update_rates = dict(post_z_update_rates)

        if interleaved_clifford is None:
            cliffords, inverse = clifford_generator.create_rb_sequences(
                n=n,
                type="2Q",
                seed=seed,
            )
        else:
            if interleaved_clifford.name != "BSWAP":
                raise ValueError("BSWAP native RB can only interleave BSWAP.")
            if interleaved_waveform is None:
                interleaved_waveform = bswap
            cliffords, inverse = clifford_generator.create_irb_sequences(
                n=n,
                interleave=interleaved_clifford,
                type="2Q",
                seed=seed,
            )

        with PulseSchedule([control_qubit, bswap_label, target_qubit]) as ps:
            pending_z_angles = {
                control_qubit: 0.0,
                target_qubit: 0.0,
            }

            def is_zero_angle(theta: float) -> bool:
                return bool(np.isclose(np.angle(np.exp(1j * theta)), 0.0))

            def add_pending_z_shifted_pulse(qubit: str, pulse: Waveform):
                z_angle = float(pending_z_angles[qubit])
                if is_zero_angle(z_angle):
                    ps.add(qubit, pulse)
                    return
                # Move the pending logical Z to the right of this physical 1Q
                # pulse by rotating the pulse axis: Z(f) X90 = X90.shifted(f) Z(f).
                ps.add(qubit, pulse.shifted(z_angle))

            def add_pending_z(qubit: str):
                z_angle = float(pending_z_angles[qubit])
                if not is_zero_angle(z_angle):
                    ps.add(qubit, VirtualZ(z_angle))

            def post_z_update(qubit: str, t_end: float) -> float:
                return (
                    post_z_offsets.get(qubit, 0.0)
                    + post_z_update_rates.get(qubit, 0.0) * t_end
                )

            def add_bswap_gate(waveform: PulseSchedule):
                ps.barrier()
                ps.call(waveform)
                ps.barrier()
                t_end = ps.duration
                control_z_angle = pending_z_angles[control_qubit]
                target_z_angle = pending_z_angles[target_qubit]
                # Logical-Z bSWAP crossing rule:
                # Z_c(a) Z_t(b) bSWAP -> bSWAP Z_c(-b) Z_t(-a), followed by
                # the calibrated post-Z terms from Wei et al. 2024.
                pending_z_angles[control_qubit] = -target_z_angle + post_z_update(
                    control_qubit, t_end
                )
                pending_z_angles[target_qubit] = -control_z_angle + post_z_update(
                    target_qubit, t_end
                )

            def add_gate(gate: str):
                if gate == "XI90":
                    add_pending_z_shifted_pulse(control_qubit, xi90)
                elif gate == "IX90":
                    add_pending_z_shifted_pulse(target_qubit, ix90)
                elif gate == "ZI90":
                    pending_z_angles[control_qubit] += np.pi / 2
                elif gate == "IZ90":
                    pending_z_angles[target_qubit] += np.pi / 2
                elif gate == "BSWAP":
                    add_bswap_gate(bswap)
                else:
                    raise ValueError("Invalid gate.")

            for clifford in cliffords:
                for gate in clifford:
                    add_gate(gate)
                if interleaved_waveform is not None:
                    add_bswap_gate(interleaved_waveform)

            for gate in inverse:
                add_gate(gate)

            add_pending_z(control_qubit)
            add_pending_z(target_qubit)
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
        in_parallel: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
        time_integration: bool | None = None,
        xaxis_type: Literal["linear", "log"] | None = None,
        plot: bool | None = None,
        save_image: bool | None = None,
        reset_awg_and_capunits: bool | None = None,
    ) -> Result:
        """Run single-qubit randomized benchmarking."""
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
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True

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

        if shots is None:
            shots = DEFAULT_SHOTS

        if interval is None:
            interval = DEFAULT_INTERVAL

        if time_integration is None:
            time_integration = True

        if xaxis_type is None:
            xaxis_type = "linear"

        for target in targets:
            target_object = self.ctx.experiment_system.get_target(target)
            if target_object.is_2q:
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
            trial_matrix_data = defaultdict(list)
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
                    result = self.measurement_service.measure(
                        sequence=rb_sequence(
                            n_clifford=n_clifford,
                            targets=target_group,
                            seed=seed,
                        ),
                        mode="avg",
                        shots=shots,
                        interval=interval,
                        time_integration=time_integration,
                        reset_awg_and_capunits=reset_awg_and_capunits,
                        plot=False,
                    )
                    for target, data in result.data.items():
                        iq = data.kerneled
                        z = self.pulse.rabi_params[target].normalize(iq)
                        trial_data[target].append((z + 1) / 2)

                check_vals = {}

                for target in target_group:
                    trial_values = np.asarray(trial_data[target], dtype=float)
                    mean = np.mean(trial_values)
                    std = np.std(trial_values)
                    trial_matrix_data[target].append(trial_values)
                    mean_data[target].append(mean)
                    std_data[target].append(std)
                    check_vals[target] = mean - std * 0.5

                max_check_val = np.max(list(check_vals.values()))
                if n_cliffords_range is None and max_check_val < 0.5:
                    break

            sweep_range = np.array(sweep_range, dtype=int)

            mean_data = {target: np.array(data) for target, data in mean_data.items()}
            std_data = {target: np.array(data) for target, data in std_data.items()}
            trial_matrix_data = {
                target: np.vstack(data) for target, data in trial_matrix_data.items()
            }

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
                    fig = fit_result.get_figure()
                    viz.save_figure(
                        fig,
                        name=f"rb_experiment_1q_{target}",
                    )

                return_data[target] = {
                    "n_cliffords": sweep_range,
                    "mean": mean,
                    "std": std,
                    "trials": trial_matrix_data[target],
                    "seeds": np.asarray(seeds, dtype=int),
                    **fit_result,
                }

        return Result(
            data=return_data,
            figures={target: result["fig"] for target, result in return_data.items()},
        )

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
        native_2q_gate: Native2QGate | None = None,
        native_2q_waveform: TargetMap[PulseSchedule] | None = None,
        interleaved_clifford: Clifford | None = None,
        interleaved_waveform: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool | None = None,
        mitigate_readout: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
        time_integration: bool | None = None,
        xaxis_type: Literal["linear", "log"] | None = None,
        plot: bool | None = None,
        save_image: bool | None = None,
        reset_awg_and_capunits: bool | None = None,
    ) -> Result:
        """Run two-qubit randomized benchmarking."""
        if in_parallel is None:
            in_parallel = False
        if mitigate_readout is None:
            mitigate_readout = True
        if plot is None:
            plot = True
        if save_image is None:
            save_image = True
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True

        if self.ctx.state_centers is None:
            raise ValueError("State classifiers are not built.")

        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        def has_2q_waveform(target: str) -> bool:
            return (
                target in self.ctx.calib_note.cr_params
                or (zx90 is not None and target in zx90)
                or (native_2q_waveform is not None and target in native_2q_waveform)
            )

        targets = [
            target
            for target in targets
            if self.ctx.experiment_system.get_target(target).is_2q
            and has_2q_waveform(target)
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

        if time_integration is None:
            time_integration = True

        if in_parallel:
            target_groups = [targets]
        else:
            target_groups = [[target] for target in targets]

        for target in targets:
            target_object = self.ctx.experiment_system.get_target(target)
            if not target_object.is_2q:
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
                        native_2q_gate=native_2q_gate,
                        native_2q_waveform=native_2q_waveform.get(target)
                        if native_2q_waveform
                        else None,
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
            trial_matrix_data = defaultdict(list)
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
                    result = self.measurement_service.measure(
                        sequence=rb_sequence(
                            n_clifford=n_clifford,
                            targets=target_group,
                            seed=seed,
                        ),
                        mode="single",
                        shots=shots,
                        interval=interval,
                        time_integration=time_integration,
                        reset_awg_and_capunits=reset_awg_and_capunits,
                        plot=False,
                    )

                    for target in target_group:
                        control_qubit, target_qubit = self.ctx.resolve_2q_qubits(target)
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
                    trial_values = np.asarray(trial_data[target], dtype=float)
                    mean = np.mean(trial_values)
                    std = np.std(trial_values)
                    trial_matrix_data[target].append(trial_values)
                    mean_data[target].append(mean)
                    std_data[target].append(std)
                    check_vals[target] = mean - std * 0.5

                max_check_val = np.max(list(check_vals.values()))
                if n_cliffords_range is None and max_check_val < 0.25:
                    break

            sweep_range = np.array(sweep_range, dtype=int)

            mean_data = {target: np.array(data) for target, data in mean_data.items()}
            std_data = {target: np.array(data) for target, data in std_data.items()}
            trial_matrix_data = {
                target: np.vstack(data) for target, data in trial_matrix_data.items()
            }

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
                    fig = fit_result.get_figure()
                    viz.save_figure(
                        fig,
                        name=f"rb_experiment_1q_{target}",
                    )

                return_data[target] = {
                    "n_cliffords": sweep_range,
                    "mean": mean,
                    "std": std,
                    "trials": trial_matrix_data[target],
                    "seeds": np.asarray(seeds, dtype=int),
                    **fit_result,
                }

        return Result(
            data=return_data,
            figures={target: result["fig"] for target, result in return_data.items()},
        )

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
        native_2q_gate: Native2QGate | None = None,
        native_2q_waveform: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
        time_integration: bool | None = None,
        plot: bool | None = None,
        save_image: bool | None = None,
    ) -> Result:
        """Run interleaved randomized benchmarking."""
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
        if time_integration is None:
            time_integration = True

        if isinstance(interleaved_clifford, str):
            clifford = self.clifford.get(interleaved_clifford)
            if clifford is None:
                raise ValueError(f"Invalid Clifford: {interleaved_clifford}")
            interleaved_clifford = clifford
        else:
            clifford = interleaved_clifford

        reset_qubits: set[str] = set()
        for target in targets:
            target_object = self.ctx.experiment_system.get_target(target)
            if target_object.is_2q:
                control_qubit, target_qubit = self.ctx.resolve_2q_qubits(target)
                reset_qubits.update([control_qubit, target_qubit])
            else:
                reset_qubits.add(self.ctx.resolve_qubit_label(target))
        self.ctx.reset_awg_and_capunits(qubits=reset_qubits)

        is_2q = self.ctx.experiment_system.get_target(targets[0]).is_2q

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
                native_2q_gate=native_2q_gate,
                native_2q_waveform=native_2q_waveform,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                time_integration=time_integration,
                plot=False,
                save_image=False,
                reset_awg_and_capunits=False,
            )
            reference_ranges = _n_cliffords_ranges_by_target(rb_result, targets)
            irb_n_cliffords_range = (
                n_cliffords_range
                if n_cliffords_range is not None
                else _shared_n_cliffords_range(reference_ranges)
            )
            if irb_n_cliffords_range is None:
                irb_data = {}
                for target in targets:
                    target_irb_result = self.rb_experiment_2q(
                        targets=target,
                        n_cliffords_range=reference_ranges[target],
                        n_trials=n_trials,
                        seeds=seeds,
                        max_n_cliffords=max_n_cliffords,
                        x90=x90,
                        zx90=zx90,
                        native_2q_gate=native_2q_gate,
                        native_2q_waveform=native_2q_waveform,
                        interleaved_waveform=interleaved_waveform,  # type: ignore
                        interleaved_clifford=interleaved_clifford,
                        in_parallel=False,
                        shots=shots,
                        interval=interval,
                        time_integration=time_integration,
                        plot=False,
                        save_image=False,
                        reset_awg_and_capunits=False,
                    )
                    irb_data[target] = target_irb_result[target]
                irb_result = Result(data=irb_data)
            else:
                irb_result = self.rb_experiment_2q(
                    targets=targets,
                    n_cliffords_range=irb_n_cliffords_range,
                    n_trials=n_trials,
                    seeds=seeds,
                    max_n_cliffords=max_n_cliffords,
                    x90=x90,
                    zx90=zx90,
                    native_2q_gate=native_2q_gate,
                    native_2q_waveform=native_2q_waveform,
                    interleaved_waveform=interleaved_waveform,  # type: ignore
                    interleaved_clifford=interleaved_clifford,
                    in_parallel=in_parallel,
                    shots=shots,
                    interval=interval,
                    time_integration=time_integration,
                    plot=False,
                    save_image=False,
                    reset_awg_and_capunits=False,
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
                time_integration=time_integration,
                plot=False,
                save_image=False,
                reset_awg_and_capunits=False,
            )
            reference_ranges = _n_cliffords_ranges_by_target(rb_result, targets)
            irb_n_cliffords_range = (
                n_cliffords_range
                if n_cliffords_range is not None
                else _shared_n_cliffords_range(reference_ranges)
            )
            if irb_n_cliffords_range is None:
                irb_data = {}
                for target in targets:
                    target_irb_result = self.rb_experiment_1q(
                        targets=target,
                        n_cliffords_range=reference_ranges[target],
                        n_trials=n_trials,
                        seeds=seeds,
                        max_n_cliffords=max_n_cliffords,
                        x90=x90,
                        interleaved_waveform=interleaved_waveform,  # type: ignore
                        interleaved_clifford=interleaved_clifford,
                        in_parallel=False,
                        shots=shots,
                        interval=interval,
                        time_integration=time_integration,
                        plot=False,
                        save_image=False,
                        reset_awg_and_capunits=False,
                    )
                    irb_data[target] = target_irb_result[target]
                irb_result = Result(data=irb_data)
            else:
                irb_result = self.rb_experiment_1q(
                    targets=targets,
                    n_cliffords_range=irb_n_cliffords_range,
                    n_trials=n_trials,
                    seeds=seeds,
                    max_n_cliffords=max_n_cliffords,
                    x90=x90,
                    interleaved_waveform=interleaved_waveform,  # type: ignore
                    interleaved_clifford=interleaved_clifford,
                    in_parallel=in_parallel,
                    shots=shots,
                    interval=interval,
                    time_integration=time_integration,
                    plot=False,
                    save_image=False,
                    reset_awg_and_capunits=False,
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
                viz.save_figure(
                    fig,
                    name=f"interleaved_randomized_benchmarking_{target}",
                )

            logger.info("")
            logger.info(
                f"Average gate fidelity (RB)  : {avg_gate_fidelity_rb * 100:.3f} ± {avg_gate_fidelity_err_rb * 100:.3f}%"
            )
            logger.info(
                f"Average gate fidelity (IRB) : {avg_gate_fidelity_irb * 100:.3f} ± {avg_gate_fidelity_err_irb * 100:.3f}%"
            )
            logger.info("")
            logger.info(
                f"Gate error    : {gate_error * 100:.3f} ± {gate_fidelity_err * 100:.3f}%"
            )
            logger.info(
                f"Gate fidelity : {gate_fidelity * 100:.3f} ± {gate_fidelity_err * 100:.3f}%"
            )
            logger.info("")

            if gate_error < 0.1 * avg_gate_error_rb:
                # TODO: use a more appropriate threshold based on the system.
                # NOTE: average number of gates per 2Q Clifford: 1Q=2.589, 2Q=1.5
                logger.warning(
                    f"Warning: Gate error ({gate_error * 100:.3f}%) is too low compared to the average gate error (RB) ({avg_gate_error_rb * 100:.3f}%)."
                )

            results[target] = {
                "gate_error": gate_error,
                "gate_fidelity": gate_fidelity,
                "gate_fidelity_err": gate_fidelity_err,
                "rb_fit_result": rb_fit_result,
                "irb_fit_result": irb_fit_result,
                "rb_data": _rb_curve_data(rb_result[target]),
                "irb_data": _rb_curve_data(irb_result[target]),
                # TODO: Remove this legacy payload key after callers migrate to result.figures.
                "fig": fig,
            }
        return Result(
            data=results,
            figures={target: result["fig"] for target, result in results.items()},
        )

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
        native_2q_gate: Native2QGate | None = None,
        native_2q_waveform: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool | None = None,
        xaxis_type: Literal["linear", "log"] | None = None,
        shots: int | None = None,
        interval: float | None = None,
        time_integration: bool | None = None,
        plot: bool | None = None,
        save_image: bool | None = None,
    ) -> Result:
        """Dispatch randomized benchmarking based on target type."""
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        target_object = self.ctx.experiment_system.get_target(targets[0])
        is_2q = target_object.is_2q

        if is_2q:
            return self.rb_experiment_2q(
                targets=targets,
                n_cliffords_range=n_cliffords_range,
                n_trials=n_trials,
                seeds=seeds,
                max_n_cliffords=max_n_cliffords,
                x90=x90,
                zx90=zx90,
                native_2q_gate=native_2q_gate,
                native_2q_waveform=native_2q_waveform,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                time_integration=time_integration,
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
                time_integration=time_integration,
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
        native_2q_gate: Native2QGate | None = None,
        native_2q_waveform: TargetMap[PulseSchedule] | None = None,
        in_parallel: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
        time_integration: bool | None = None,
        plot: bool | None = None,
        save_image: bool | None = None,
    ) -> Result:
        """Dispatch interleaved randomized benchmarking."""
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if in_parallel is None:
            in_parallel = False

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
                native_2q_gate=native_2q_gate,
                native_2q_waveform=native_2q_waveform,
                in_parallel=in_parallel,
                shots=shots,
                interval=interval,
                time_integration=time_integration,
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
                    native_2q_gate=native_2q_gate,
                    native_2q_waveform=native_2q_waveform,
                    in_parallel=in_parallel,
                    shots=shots,
                    interval=interval,
                    time_integration=time_integration,
                    plot=plot,
                    save_image=save_image,
                )
                results[target] = result[target]
            result = Result(
                data=results,
                figures={target: entry["fig"] for target, entry in results.items()},
            )

        return Result(
            data=result.data,
            figures=result.figures,
        )

    def benchmark_1q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_trials: int | None = None,
        in_parallel: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
        save_image: bool | None = None,
    ) -> None:
        """Run standard 1Q benchmarking suite."""
        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if in_parallel is None:
            in_parallel = False
        if plot is None:
            plot = True
        if save_image is None:
            save_image = True

        def _run_irb(
            benchmark_targets: Collection[str] | str,
            *,
            interleaved_clifford: str,
            label: object,
        ) -> None:
            try:
                self.interleaved_randomized_benchmarking(
                    benchmark_targets,
                    interleaved_clifford=interleaved_clifford,
                    n_trials=n_trials,
                    in_parallel=in_parallel,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                    save_image=save_image,
                )
            except Exception as e:
                print(f"Failed to benchmark {label} with {interleaved_clifford}: {e}")

        if in_parallel:
            for interleaved_clifford in ("X90", "X180"):
                _run_irb(
                    targets,
                    interleaved_clifford=interleaved_clifford,
                    label=targets,
                )
        else:
            for target in targets:
                for interleaved_clifford in ("X90", "X180"):
                    _run_irb(
                        target,
                        interleaved_clifford=interleaved_clifford,
                        label=target,
                    )

    def benchmark_2q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_trials: int | None = None,
        in_parallel: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
        save_image: bool | None = None,
    ) -> None:
        """Run standard 2Q benchmarking suite."""
        if targets is None:
            targets = self.ctx.cr_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if in_parallel is None:
            in_parallel = False
        if plot is None:
            plot = True
        if save_image is None:
            save_image = True

        def _run_zx90(benchmark_targets: Collection[str] | str, label: object) -> None:
            try:
                self.interleaved_randomized_benchmarking(
                    benchmark_targets,
                    interleaved_clifford="ZX90",
                    n_trials=n_trials,
                    in_parallel=in_parallel,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                    save_image=save_image,
                )
            except Exception as e:
                print(f"Failed to benchmark {label} with ZX90: {e}")

        if in_parallel:
            _run_zx90(targets, targets)
        else:
            for target in targets:
                _run_zx90(target, target)
