from __future__ import annotations

from collections import defaultdict
from typing import Collection, Literal

import cma
import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from tqdm import tqdm

from ...analysis import fitting
from ...analysis import visualization as vis
from ...backend import SAMPLING_PERIOD, Target
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import (
    CrossResonance,
    Drag,
    FlatTop,
    Pulse,
    PulseSchedule,
    PulseSequence,
    Waveform,
)
from ...typing import TargetMap
from ..experiment_constants import (
    CALIBRATION_SHOTS,
    DRAG_COEFF,
    DRAG_HPI_DURATION,
    DRAG_PI_DURATION,
    HPI_DURATION,
    HPI_RAMPTIME,
    PI_DURATION,
    PI_RAMPTIME,
)
from ..experiment_result import AmplCalibData, ExperimentResult
from ..protocol import BaseProtocol, CalibrationProtocol, MeasurementProtocol


class CalibrationMixin(
    BaseProtocol,
    MeasurementProtocol,
    CalibrationProtocol,
):
    def calibrate_default_pulse(
        self,
        targets: Collection[str],
        pulse_type: Literal["pi", "hpi"],
        n_rotations: int = 1,
        plot: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        targets = list(targets)
        rabi_params = self.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        def calibrate(target: str) -> AmplCalibData:
            if pulse_type == "hpi":
                pulse = FlatTop(
                    duration=HPI_DURATION,
                    amplitude=1,
                    tau=HPI_RAMPTIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area
            elif pulse_type == "pi":
                pulse = FlatTop(
                    duration=PI_DURATION,
                    amplitude=1,
                    tau=PI_RAMPTIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area
            else:
                raise ValueError("Invalid pulse type.")
            ampl = self.calc_control_amplitudes(
                rabi_rate=rabi_rate,
                print_result=False,
            )[target]
            ampl_min = ampl * (1 - 0.5 / n_rotations)
            ampl_max = ampl * (1 + 0.5 / n_rotations)
            ampl_min = 0 if ampl_min < 0 else ampl_min
            ampl_max = 1 if ampl_max > 1 else ampl_max
            ampl_range = np.linspace(ampl_min, ampl_max, 20)
            n_per_rotation = 2 if pulse_type == "pi" else 4
            sweep_data = self.sweep_parameter(
                sequence=lambda x: {target: pulse.scaled(x)},
                sweep_range=ampl_range,
                repetitions=n_per_rotation * n_rotations,
                shots=shots,
                interval=interval,
                plot=False,
            ).data[target]

            fit_result = fitting.fit_ampl_calib_data(
                target=target,
                amplitude_range=ampl_range,
                data=sweep_data.normalized,
                plot=plot,
                title=f"{pulse_type} pulse calibration",
                yaxis_title="Normalized signal",
            )

            return AmplCalibData.new(
                sweep_data=sweep_data,
                calib_value=fit_result["amplitude"],
            )

        data: dict[str, AmplCalibData] = {}
        for target in targets:
            print(f"Calibrating {target}...\n")
            data[target] = calibrate(target)

        print(f"Calibration results for {pulse_type} pulse:")
        for target, calib_data in data.items():
            print(f"  {target}: {calib_data.calib_value:.6f}")

        return ExperimentResult(data=data)

    def calibrate_ef_pulse(
        self,
        targets: Collection[str],
        pulse_type: Literal["pi", "hpi"],
        n_rotations: int = 1,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        targets = list(targets)
        rabi_params = self.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        ef_labels = [Target.ef_label(label) for label in targets]

        def calibrate(target: str) -> AmplCalibData:
            ge_label = Target.ge_label(target)
            ef_label = Target.ef_label(target)

            if pulse_type == "hpi":
                pulse = FlatTop(
                    duration=HPI_DURATION,
                    amplitude=1,
                    tau=HPI_RAMPTIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area
            elif pulse_type == "pi":
                pulse = FlatTop(
                    duration=PI_DURATION,
                    amplitude=1,
                    tau=PI_RAMPTIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area
            else:
                raise ValueError("Invalid pulse type.")
            ampl = self.calc_control_amplitudes(
                rabi_rate=rabi_rate,
                print_result=False,
            )[ef_label]
            ampl_min = ampl * (1 - 0.5 / n_rotations)
            ampl_max = ampl * (1 + 0.5 / n_rotations)
            ampl_min = 0 if ampl_min < 0 else ampl_min
            ampl_max = 1 if ampl_max > 1 else ampl_max
            ampl_range = np.linspace(ampl_min, ampl_max, 20)
            n_per_rotation = 2 if pulse_type == "pi" else 4
            repetitions = n_per_rotation * n_rotations

            def sequence(x: float) -> PulseSchedule:
                with PulseSchedule([ge_label, ef_label]) as ps:
                    ps.add(ge_label, self.hpi_pulse[ge_label].repeated(2))
                    ps.barrier()
                    ps.add(ef_label, pulse.scaled(x).repeated(repetitions))
                return ps

            sweep_data = self.sweep_parameter(
                sequence=sequence,
                sweep_range=ampl_range,
                repetitions=1,
                rabi_level="ef",
                shots=shots,
                interval=interval,
                plot=True,
            ).data[ge_label]

            fit_result = fitting.fit_ampl_calib_data(
                target=ef_label,
                amplitude_range=ampl_range,
                data=sweep_data.normalized,
                title=f"ef {pulse_type} pulse calibration",
                yaxis_title="Normalized signal",
            )

            return AmplCalibData.new(
                sweep_data=sweep_data,
                calib_value=fit_result["amplitude"],
            )

        data: dict[str, AmplCalibData] = {}
        for target in ef_labels:
            print(f"Calibrating {target}...\n")
            data[target] = calibrate(target)

        print(f"Calibration results for {pulse_type} pulse:")
        for target, calib_data in data.items():
            print(f"  {target}: {calib_data.calib_value:.6f}")

        return ExperimentResult(data=data)

    def calibrate_drag_amplitude(
        self,
        targets: Collection[str],
        *,
        spectator_state: str = "+",
        pulse_type: Literal["pi", "hpi"],
        n_rotations: int = 4,
        duration: float | None = None,
        drag_coeff: float = DRAG_COEFF,
        use_stored_amplitude: bool = False,
        use_stored_beta: bool = False,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict[str, float]:
        targets = list(targets)
        rabi_params = self.rabi_params
        self.validate_rabi_params(rabi_params)

        def calibrate(target: str) -> float:
            if pulse_type == "hpi":
                hpi_param = self.calib_note.get_drag_hpi_param(target)
                if hpi_param is not None and use_stored_beta:
                    beta = hpi_param["beta"]
                else:
                    beta = -drag_coeff / self.qubits[target].alpha

                pulse = Drag(
                    duration=duration or DRAG_HPI_DURATION,
                    amplitude=1,
                    beta=beta,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area

                if hpi_param is not None and use_stored_amplitude:
                    ampl = hpi_param["amplitude"]
                else:
                    ampl = self.calc_control_amplitudes(
                        rabi_rate=rabi_rate,
                        print_result=False,
                    )[target]

            elif pulse_type == "pi":
                pi_param = self.calib_note.get_drag_pi_param(target)
                if pi_param is not None and use_stored_beta:
                    beta = pi_param["beta"]
                else:
                    beta = -drag_coeff / self.qubits[target].alpha

                pulse = Drag(
                    duration=duration or DRAG_PI_DURATION,
                    amplitude=1,
                    beta=beta,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area

                if pi_param is not None and use_stored_amplitude:
                    ampl = pi_param["amplitude"]
                else:
                    ampl = self.calc_control_amplitudes(
                        rabi_rate=rabi_rate,
                        print_result=False,
                    )[target]
            else:
                raise ValueError("Invalid pulse type.")

            ampl_min = ampl * (1 - 0.5 / n_rotations)
            ampl_max = ampl * (1 + 0.5 / n_rotations)
            ampl_min = 0 if ampl_min < 0 else ampl_min
            ampl_max = 1 if ampl_max > 1 else ampl_max
            ampl_range = np.linspace(ampl_min, ampl_max, 20)
            n_per_rotation = 2 if pulse_type == "pi" else 4

            spectators = self.get_spectators(target)
            all_targets = [target] + [
                spectator.label
                for spectator in spectators
                if spectator.label in self.qubit_labels
            ]

            def sequence(x: float) -> PulseSchedule:
                with PulseSchedule(all_targets) as ps:
                    for spectator in spectators:
                        if spectator.label in self.qubit_labels:
                            ps.add(
                                spectator.label,
                                self.get_pulse_for_state(
                                    target=spectator.label,
                                    state=spectator_state,
                                ),
                            )
                    ps.barrier()
                    ps.add(
                        target, pulse.scaled(x).repeated(n_per_rotation * n_rotations)
                    )
                return ps

            sweep_data = self.sweep_parameter(
                sequence=sequence,
                sweep_range=ampl_range,
                shots=shots,
                interval=interval,
                plot=False,
            ).data[target]

            fit_result = fitting.fit_ampl_calib_data(
                target=target,
                amplitude_range=ampl_range,
                data=sweep_data.normalized,
                title=f"DRAG {pulse_type} amplitude calibration",
                yaxis_title="Normalized signal",
            )

            if pulse_type == "hpi":
                self.calib_note.drag_hpi_params = {
                    target: {
                        "target": target,
                        "duration": pulse.duration,
                        "amplitude": fit_result["amplitude"],
                        "beta": beta,
                    }
                }
            elif pulse_type == "pi":
                self.calib_note.drag_pi_params = {
                    target: {
                        "target": target,
                        "duration": pulse.duration,
                        "amplitude": fit_result["amplitude"],
                        "beta": beta,
                    }
                }

            return fit_result["amplitude"]

        result: dict[str, float] = {}
        for target in targets:
            print(f"Calibrating {target}...\n")
            result[target] = calibrate(target)

        print(f"Calibration results for DRAG {pulse_type} amplitude:")
        for target, amplitude in result.items():
            print(f"  {target}: {amplitude:.6f}")
        return result

    def calibrate_drag_beta(
        self,
        targets: Collection[str] | None = None,
        *,
        spectator_state: str = "+",
        pulse_type: Literal["pi", "hpi"] = "hpi",
        beta_range: ArrayLike = np.linspace(-1.5, 1.5, 41),
        n_turns: int = 1,
        duration: float | None = None,
        degree: int = 3,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict[str, float]:
        if targets is None:
            targets = self.qubit_labels
        else:
            targets = list(targets)
        beta_range = np.array(beta_range, dtype=np.float64)
        rabi_params = self.rabi_params
        self.validate_rabi_params(rabi_params)

        def calibrate(target: str) -> float:
            spectators = self.get_spectators(target)
            all_targets = [target] + [
                spectator.label
                for spectator in spectators
                if spectator.label in self.qubit_labels
            ]

            if pulse_type == "hpi":
                param = self.calib_note.get_drag_hpi_param(target)
            elif pulse_type == "pi":
                param = self.calib_note.get_drag_pi_param(target)
            if param is None:
                raise ValueError("DRAG parameters are not stored.")

            drag_duration = duration or param["duration"]
            drag_amplitude = param["amplitude"]

            def sequence(beta: float) -> PulseSchedule:
                with PulseSchedule(all_targets) as ps:
                    for spectator in spectators:
                        if spectator.label in self.qubit_labels:
                            ps.add(
                                spectator.label,
                                self.get_pulse_for_state(
                                    target=spectator.label,
                                    state=spectator_state,
                                ),
                            )
                    ps.barrier()
                    if pulse_type == "hpi":
                        x90p = Drag(
                            duration=drag_duration,
                            amplitude=drag_amplitude,
                            beta=beta,
                        )
                        x90m = x90p.scaled(-1)
                        y90m = self.hpi_pulse[target].shifted(-np.pi / 2)
                        ps.add(
                            target,
                            PulseSequence(
                                [
                                    x90p,
                                    PulseSequence([x90m, x90p] * n_turns),
                                    y90m,
                                ]
                            ),
                        )
                    elif pulse_type == "pi":
                        x180p = Drag(
                            duration=drag_duration,
                            amplitude=drag_amplitude,
                            beta=beta,
                        )
                        x180m = x180p.scaled(-1)
                        y90m = self.hpi_pulse[target].shifted(-np.pi / 2)
                        ps.add(
                            target,
                            PulseSequence(
                                [
                                    PulseSequence([x180p, x180m] * n_turns),
                                    y90m,
                                ]
                            ),
                        )
                return ps

            sweep_data = self.sweep_parameter(
                sequence=sequence,
                sweep_range=beta_range,
                shots=shots,
                interval=interval,
                plot=False,
            ).data[target]
            values = sweep_data.normalized
            fit_result = fitting.fit_polynomial(
                target=target,
                x=beta_range,
                y=values,
                degree=degree,
                title=f"DRAG {pulse_type} beta calibration",
                xaxis_title="Beta",
                yaxis_title="Normalized signal",
            )
            beta = fit_result["root"]
            if np.isnan(beta):
                beta = 0.0
            print(f"Calibrated beta: {beta:.6f}")

            if pulse_type == "hpi":
                self.calib_note.drag_hpi_params = {
                    target: {
                        "target": target,
                        "duration": drag_duration,
                        "amplitude": drag_amplitude,
                        "beta": beta,
                    }
                }
            elif pulse_type == "pi":
                self.calib_note.drag_pi_params = {
                    target: {
                        "target": target,
                        "duration": drag_duration,
                        "amplitude": drag_amplitude,
                        "beta": beta,
                    }
                }

            return beta

        result = {}
        for target in targets:
            print(f"Calibrating {target}...\n")
            result[target] = calibrate(target)

        print(f"Calibration results for DRAG {pulse_type} beta:")
        for target, beta in result.items():
            print(f"  {target}: {beta:.6f}")

        return result

    def calibrate_hpi_pulse(
        self,
        targets: Collection[str] | None = None,
        n_rotations: int = 1,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        if targets is None:
            targets = self.qubit_labels
        else:
            targets = list(targets)

        result = self.calibrate_default_pulse(
            targets=targets,
            pulse_type="hpi",
            n_rotations=n_rotations,
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        # self.system_note.put(HPI_AMPLITUDE, ampl)  # deprecated
        self.calib_note.hpi_params = {
            target: {
                "target": target,
                "duration": HPI_DURATION,
                "amplitude": ampl[target],
                "tau": HPI_RAMPTIME,
            }
            for target in targets
        }
        return result

    def calibrate_pi_pulse(
        self,
        targets: Collection[str] | None = None,
        n_rotations: int = 1,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        if targets is None:
            targets = self.qubit_labels
        else:
            targets = list(targets)

        result = self.calibrate_default_pulse(
            targets=targets,
            pulse_type="pi",
            n_rotations=n_rotations,
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        # self.system_note.put(PI_AMPLITUDE, ampl)  # deprecated
        self.calib_note.pi_params = {
            target: {
                "target": target,
                "duration": PI_DURATION,
                "amplitude": ampl[target],
                "tau": PI_RAMPTIME,
            }
            for target in targets
        }
        return result

    def calibrate_ef_hpi_pulse(
        self,
        targets: Collection[str] | None = None,
        n_rotations: int = 1,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        if targets is None:
            targets = self.qubit_labels
        else:
            targets = list(targets)

        result = self.calibrate_ef_pulse(
            targets=targets,
            pulse_type="hpi",
            n_rotations=n_rotations,
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        # self.system_note.put(HPI_AMPLITUDE, ampl)  # deprecated
        self.calib_note.hpi_params = {
            target: {
                "target": target,
                "duration": HPI_DURATION,
                "amplitude": ampl[target],
                "tau": HPI_RAMPTIME,
            }
            for target in targets
        }
        return result

    def calibrate_ef_pi_pulse(
        self,
        targets: Collection[str] | None = None,
        n_rotations: int = 1,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        if targets is None:
            targets = self.qubit_labels
        else:
            targets = list(targets)

        result = self.calibrate_ef_pulse(
            targets=targets,
            pulse_type="pi",
            n_rotations=n_rotations,
            shots=shots,
            interval=interval,
        )

        ampl = {target: data.calib_value for target, data in result.data.items()}
        # self.system_note.put(PI_AMPLITUDE, ampl)  # deprecated
        self.calib_note.pi_params = {
            target: {
                "target": target,
                "duration": PI_DURATION,
                "amplitude": ampl[target],
                "tau": PI_RAMPTIME,
            }
            for target in targets
        }
        return result

    def calibrate_drag_hpi_pulse(
        self,
        targets: Collection[str] | None = None,
        n_rotations: int = 4,
        n_turns: int = 1,
        n_iterations: int = 2,
        calibrate_beta: bool = True,
        beta_range: ArrayLike = np.linspace(-1.5, 1.5, 20),
        duration: float | None = None,
        drag_coeff: float = DRAG_COEFF,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        else:
            targets = list(targets)

        for i in range(n_iterations):
            print(f"\nIteration {i + 1}/{n_iterations}")

            use_stored_amplitude = True if i > 0 else False
            use_stored_beta = True if i > 0 else False

            print("Calibrating DRAG amplitude:")
            amplitude = self.calibrate_drag_amplitude(
                targets=targets,
                pulse_type="hpi",
                n_rotations=n_rotations,
                duration=duration,
                use_stored_amplitude=use_stored_amplitude,
                use_stored_beta=use_stored_beta,
                shots=shots,
                interval=interval,
            )

            # self.system_note.put(DRAG_HPI_AMPLITUDE, amplitude)  # deprecated

            if calibrate_beta:
                print("\nCalibrating DRAG beta:")
                beta = self.calibrate_drag_beta(
                    targets=targets,
                    pulse_type="hpi",
                    beta_range=beta_range,
                    n_turns=n_turns,
                    duration=duration,
                    degree=3,
                    shots=shots,
                    interval=interval,
                )
            else:
                beta = {
                    target: -drag_coeff / self.qubits[target].alpha
                    for target in targets
                }

            # self.system_note.put(DRAG_HPI_BETA, beta)  # deprecated

        return {
            "amplitude": amplitude,
            "beta": beta,
        }

    def calibrate_drag_pi_pulse(
        self,
        targets: Collection[str] | None = None,
        *,
        spectator_state: str = "+",
        n_rotations: int = 4,
        n_turns: int = 1,
        n_iterations: int = 2,
        calibrate_beta: bool = True,
        beta_range: ArrayLike = np.linspace(-1.5, 1.5, 20),
        duration: float | None = None,
        drag_coeff: float = DRAG_COEFF,
        degree: int = 3,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        else:
            targets = list(targets)

        for i in range(n_iterations):
            print(f"\nIteration {i + 1}/{n_iterations}")

            use_stored_amplitude = True if i > 0 else False
            use_stored_beta = True if i > 0 else False

            print("Calibrating DRAG amplitude:")
            amplitude = self.calibrate_drag_amplitude(
                targets=targets,
                spectator_state=spectator_state,
                pulse_type="pi",
                n_rotations=n_rotations,
                duration=duration,
                use_stored_amplitude=use_stored_amplitude,
                use_stored_beta=use_stored_beta,
                shots=shots,
                interval=interval,
            )

            # self.system_note.put(DRAG_PI_AMPLITUDE, amplitude)  # deprecated

            if calibrate_beta:
                print("Calibrating DRAG beta:")
                beta = self.calibrate_drag_beta(
                    targets=targets,
                    spectator_state=spectator_state,
                    pulse_type="pi",
                    beta_range=beta_range,
                    n_turns=n_turns,
                    duration=duration,
                    degree=degree,
                    shots=shots,
                    interval=interval,
                )
            else:
                beta = {
                    target: -drag_coeff / self.qubits[target].alpha
                    for target in targets
                }

            # self.system_note.put(DRAG_PI_BETA, beta)  # deprecated

        return {
            "amplitude": amplitude,
            "beta": beta,
        }

    def measure_cr_dynamics(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        time_range: ArrayLike | None = None,
        cr_amplitude: float = 1.0,
        cr_phase: float = 0.0,
        cancel_amplitude: float = 0.0,
        cancel_phase: float = 0.0,
        echo: bool = False,
        control_state: str = "0",
        x90: TargetMap[Waveform] | None = None,
        x180: TargetMap[Waveform] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ) -> dict:
        if time_range is None:
            time_range = np.arange(0, 1001, 20)
        else:
            time_range = np.array(time_range)

        if x90 is None:
            x90 = self.hpi_pulse
            if control_qubit in self.drag_hpi_pulse:
                x90[control_qubit] = self.drag_hpi_pulse[control_qubit]
            if target_qubit in self.drag_hpi_pulse:
                x90[target_qubit] = self.drag_hpi_pulse[target_qubit]

        if x180 is None:
            if control_qubit in self.drag_pi_pulse:
                x180 = self.drag_pi_pulse
            else:
                x180 = self.pi_pulse

        control_states = []
        target_states = []
        for T in time_range:
            result = self.state_tomography(
                CrossResonance(
                    control_qubit=control_qubit,
                    target_qubit=target_qubit,
                    cr_amplitude=cr_amplitude,
                    cr_duration=T,
                    cr_ramptime=0.0,
                    cr_phase=cr_phase,
                    cancel_amplitude=cancel_amplitude,
                    cancel_phase=cancel_phase,
                    echo=echo,
                    pi_pulse=x180[control_qubit],
                ),
                x90=x90,
                initial_state={control_qubit: control_state},
                shots=shots,
                interval=interval,
                plot=False,
            )
            control_states.append(np.array(result[control_qubit]))
            target_states.append(np.array(result[target_qubit]))

        return {
            "time_range": time_range,
            "control_states": np.array(control_states),
            "target_states": np.array(target_states),
        }

    def cr_hamiltonian_tomography(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        time_range: ArrayLike | None = None,
        cr_amplitude: float = 1.0,
        cr_phase: float = 0.0,
        cancel_amplitude: float = 0.0,
        cancel_phase: float = 0.0,
        x90: TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = False,
    ) -> dict:
        if time_range is None:
            time_range = np.arange(0, 1001, 20)
        else:
            time_range = np.array(time_range)

        result_0 = self.measure_cr_dynamics(
            time_range=time_range,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            echo=False,
            control_state="0",
            x90=x90,
            shots=shots,
            interval=interval,
        )
        result_1 = self.measure_cr_dynamics(
            time_range=time_range,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            echo=False,
            control_state="1",
            x90=x90,
            shots=shots,
            interval=interval,
        )

        target_states_0 = result_0["target_states"]
        target_states_1 = result_1["target_states"]

        fit_0 = fitting.fit_rotation(
            time_range,
            target_states_0,
            plot=plot,
            title="Cross resonance dynamics : |0〉",
            xlabel="Drive time (ns)",
            ylabel="Bloch vector",
        )
        fit_1 = fitting.fit_rotation(
            time_range,
            target_states_1,
            plot=plot,
            title="Cross resonance dynamics : |1〉",
            xlabel="Drive time (ns)",
            ylabel="Bloch vector",
        )
        if plot:
            vis.display_bloch_sphere(target_states_0)
            vis.display_bloch_sphere(target_states_1)
        Omega_0 = fit_0["Omega"]
        Omega_1 = fit_1["Omega"]
        Omega = np.concatenate(
            [
                0.5 * (Omega_0 + Omega_1),
                0.5 * (Omega_0 - Omega_1),
            ]
        )
        coeffs = dict(
            zip(
                ["IX", "IY", "IZ", "ZX", "ZY", "ZZ"],
                Omega / (2 * np.pi),  # GHz
            )
        )

        print("Rotation coefficients:")
        for key, value in coeffs.items():
            print(f"  {key} : {value * 1e3:+.3f} MHz")
        print()

        # xt (cross-talk) rotation
        xt_rotation = coeffs["IX"] + 1j * coeffs["IY"]
        xt_rotation_amplitude = np.abs(xt_rotation)
        xt_rotation_amplitude_hw = self.calc_control_amplitudes(
            rabi_rate=xt_rotation_amplitude,
            print_result=False,
        )[target_qubit]
        xt_rotation_phase = np.angle(xt_rotation)
        xt_rotation_phase_deg = np.angle(xt_rotation, deg=True)
        print("XT (crosstalk) rotation:")
        print(
            f"  rate  : {xt_rotation_amplitude * 1e3:.3f} MHz ({xt_rotation_amplitude_hw:.6f})"
        )
        print(
            f"  phase : {xt_rotation_phase:.3f} rad ({xt_rotation_phase_deg:.1f} deg)"
        )
        print()

        # cr (cross-resonance) rotation
        cr_rotation = coeffs["ZX"] + 1j * coeffs["ZY"]
        cr_rotation_amplitude = np.abs(cr_rotation)
        cr_rotation_amplitude_hw = self.calc_control_amplitudes(
            rabi_rate=cr_rotation_amplitude,
            print_result=False,
        )[target_qubit]
        cr_rotation_phase = np.angle(cr_rotation)
        cr_rotation_phase_deg = np.angle(cr_rotation, deg=True)
        zx90_duration = 1 / (4 * cr_rotation_amplitude)

        print("CR (cross-resonance) rotation:")
        print(
            f"  rate  : {cr_rotation_amplitude * 1e3:.3f} MHz ({cr_rotation_amplitude_hw:.6f})"
        )
        print(
            f"  phase : {cr_rotation_phase:.3f} rad ({cr_rotation_phase_deg:.1f} deg)"
        )
        print()

        # ZX90 gate
        cr_rabi_rate = self.calc_rabi_rates(
            control_amplitude=cr_amplitude,
            print_result=False,
        )[control_qubit]
        print("Estimated ZX90 gate:")
        print(f"  drive    : {cr_rabi_rate * 1e3:.1f} MHz ({cr_amplitude:.3f})")
        print(f"  duration : {zx90_duration:.1f} ns")
        print()

        return {
            "Omega": Omega,
            "coeffs": coeffs,
            "cr_rotation_amplitude": cr_rotation_amplitude,
            "cr_rotation_amplitude_hw": cr_rotation_amplitude_hw,
            "cr_rotation_phase": cr_rotation_phase,
            "xt_rotation_amplitude": xt_rotation_amplitude,
            "xt_rotation_amplitude_hw": xt_rotation_amplitude_hw,
            "xt_rotation_phase": xt_rotation_phase,
            "cr_drive_amplitude": cr_rabi_rate,
            "cr_drive_amplitude_hw": cr_amplitude,
            "zx90_duration": zx90_duration,
        }

    def update_cr_params(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        time_range: ArrayLike | None = None,
        cr_amplitude: float = 1.0,
        cr_ramptime: float = 20.0,
        cr_phase: float = 0.0,
        cancel_amplitude: float = 0.0,
        cancel_phase: float = 0.0,
        safe_factor: float = 1.1,
        x90: TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = False,
    ) -> dict:
        current_cr_pulse = cr_amplitude * np.exp(1j * cr_phase)
        current_cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase)

        result = self.cr_hamiltonian_tomography(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            time_range=time_range,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            x90=x90,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        shift = -result["cr_rotation_phase"]
        cancel_pulse = -result["xt_rotation_amplitude_hw"] * np.exp(
            1j * result["xt_rotation_phase"]
        )
        new_cr_pulse = current_cr_pulse * np.exp(1j * shift)
        new_cancel_pulse = (current_cancel_pulse + cancel_pulse) * np.exp(1j * shift)

        new_cr_amplitude = np.abs(new_cr_pulse)
        new_cr_phase = np.angle(new_cr_pulse)
        new_cancel_amplitude = np.abs(new_cancel_pulse)
        new_cancel_phase = np.angle(new_cancel_pulse)

        print("Updated CR params:")
        print(f"  CR amplitude     : {cr_amplitude:+.3f} -> {new_cr_amplitude:+.3f}")
        print(f"  CR phase         : {cr_phase:+.3f} -> {new_cr_phase:+.3f}")
        print(
            f"  Cancel amplitude : {cancel_amplitude:+.3f} -> {new_cancel_amplitude:+.3f}"
        )
        print(f"  Cancel phase     : {cancel_phase:+.3f} -> {new_cancel_phase:+.3f}")
        print()

        cr_label = f"{control_qubit}-{target_qubit}"
        half_duration = 0.5 * result["zx90_duration"] + cr_ramptime
        cr_duration = (safe_factor * half_duration // 10 + 1) * 10

        self.calib_note.cr_params = {
            cr_label: {
                "target": cr_label,
                "duration": cr_duration,
                "ramptime": cr_ramptime,
                "cr_amplitude": new_cr_amplitude,
                "cr_phase": new_cr_phase,
                "cancel_amplitude": new_cancel_amplitude,
                "cancel_phase": new_cancel_phase,
                "cr_cancel_ratio": new_cancel_amplitude / new_cr_amplitude,
            }
        }

        return {
            **result,
            "cr_param": self.calib_note.cr_params[cr_label],
        }

    def obtain_cr_params(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        cr_amplitude: float = 1.0,
        cr_ramptime: float = 20.0,
        n_iterations: int = 4,
        time_range: ArrayLike = np.arange(0, 501, 20),
        use_stored_params: bool = True,
        safe_factor: float = 1.1,
        x90: TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        time_range = np.array(time_range, dtype=float)

        cr_label = f"{control_qubit}-{target_qubit}"
        current_cr_param = self.calib_note.cr_params.get(cr_label)
        if use_stored_params and current_cr_param is not None:
            cr_phase = current_cr_param["cr_phase"]
            cancel_amplitude = current_cr_param["cr_cancel_ratio"] * cr_amplitude
            cancel_phase = current_cr_param["cancel_phase"]
        else:
            cr_phase = 0.0
            cancel_amplitude = 0.0
            cancel_phase = 0.0

        params_history = [
            {
                "time_range": time_range,
                "cr_phase": cr_phase,
                "cancel_amplitude": cancel_amplitude,
                "cancel_phase": cancel_phase,
            }
        ]

        coeffs_history = defaultdict(list)

        for _ in tqdm(range(n_iterations)):
            params = params_history[-1]

            result = self.update_cr_params(
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                time_range=params["time_range"],
                cr_amplitude=cr_amplitude,
                cr_ramptime=cr_ramptime,
                cr_phase=params["cr_phase"],
                cancel_amplitude=params["cancel_amplitude"],
                cancel_phase=params["cancel_phase"],
                safe_factor=safe_factor,
                x90=x90,
                shots=shots,
                interval=interval,
                plot=plot,
            )

            period = result["zx90_duration"] * 4
            n_points_per_period = 10
            dt = (period / n_points_per_period) // SAMPLING_PERIOD * SAMPLING_PERIOD
            n_cycles = 2
            duration = period * n_cycles
            time_range = np.arange(0, duration + 1, dt)

            params_history.append(
                {
                    "time_range": time_range,
                    "cr_phase": result["cr_param"]["cr_phase"],
                    "cancel_amplitude": result["cr_param"]["cancel_amplitude"],
                    "cancel_phase": result["cr_param"]["cancel_phase"],
                }
            )
            for key, value in result["coeffs"].items():
                coeffs_history[key].append(value)

        hamiltonian_coeffs = {
            key: np.array(value) for key, value in coeffs_history.items()
        }

        fig = go.Figure()
        for key, value in hamiltonian_coeffs.items():
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(value)),
                    y=value * 1e3,
                    mode="lines+markers",
                    name=f"{key}/2",
                )
            )

        fig.update_layout(
            title="CR Hamiltonian coefficients",
            xaxis_title="Number of steps",
            yaxis_title="Coefficient (MHz)",
            xaxis=dict(tickmode="array", tickvals=np.arange(len(value))),
        )
        if plot:
            fig.show()

        return {
            "params_history": params_history,
            "coeffs_history": hamiltonian_coeffs,
        }

    def calibrate_zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        duration: float | None = None,
        ramptime: float | None = None,
        amplitude_range: ArrayLike | None = None,
        initial_state: str = "0",
        n_repetitions: int = 1,
        degree: int = 3,
        x180: TargetMap[Waveform] | Waveform | None = None,
        use_zvalues: bool = False,
        store_params: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
    ):
        cr_label = f"{control_qubit}-{target_qubit}"
        cr_param = self.calib_note.get_cr_param(cr_label)
        if cr_param is None:
            raise ValueError("CR parameters are not stored.")
        if duration is None:
            duration = cr_param["duration"]
        if ramptime is None:
            ramptime = cr_param["ramptime"]
        if amplitude_range is None:
            cr_amplitude = cr_param.get("cr_amplitude")
            if cr_amplitude is None:
                amplitude_range = np.linspace(0, 1, 50)
            else:
                min_amplitude = np.clip(cr_amplitude * 0.5, 0.0, 1.0)
                max_amplitude = np.clip(cr_amplitude * 1.5, 0.0, 1.0)
                amplitude_range = np.linspace(min_amplitude, max_amplitude, 50)
        amplitude_range = np.array(amplitude_range)
        cr_phase = cr_param["cr_phase"]
        cancel_phase = cr_param["cancel_phase"]
        cr_cancel_ratio = cr_param["cr_cancel_ratio"]

        if x180 is None:
            if control_qubit in self.drag_pi_pulse:
                x180 = self.drag_pi_pulse
            else:
                x180 = self.pi_pulse
        elif isinstance(x180, Waveform):
            x180 = {control_qubit: x180}

        def ecr_sequence(amplitude: float) -> PulseSchedule:
            ecr = CrossResonance(
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                cr_amplitude=amplitude,
                cr_duration=duration,
                cr_ramptime=ramptime,
                cr_phase=cr_phase,
                cancel_amplitude=amplitude * cr_cancel_ratio,
                cancel_phase=cancel_phase,
                echo=True,
                pi_pulse=x180[control_qubit],
            ).repeated(n_repetitions)
            with PulseSchedule([control_qubit, cr_label, target_qubit]) as ps:
                if initial_state != "0":
                    ps.add(
                        control_qubit,
                        self.get_pulse_for_state(control_qubit, initial_state),
                    )
                    ps.barrier()
                ps.call(ecr)
            return ps

        sweep_result = self.sweep_parameter(
            ecr_sequence,
            sweep_range=amplitude_range,
            shots=shots,
            interval=interval,
            plot=plot,
        )

        if use_zvalues:
            signal = sweep_result.data[target_qubit].zvalues
        else:
            signal = sweep_result.data[target_qubit].normalized

        fit_result = fitting.fit_polynomial(
            target=target_qubit,
            x=amplitude_range,
            y=signal,
            degree=degree,
            title="ZX90 calibration",
            xaxis_title="Amplitude (arb. unit)",
            yaxis_title="Signal",
        )

        amplitude = fit_result["root"]

        if amplitude is not None and store_params:
            self.calib_note.cr_params = {
                cr_label: {
                    "target": cr_label,
                    "duration": duration,
                    "ramptime": ramptime,
                    "cr_amplitude": amplitude,
                    "cr_phase": cr_phase,
                    "cancel_amplitude": amplitude * cr_cancel_ratio,
                    "cancel_phase": cancel_phase,
                    "cr_cancel_ratio": cr_cancel_ratio,
                },
            }

        return {
            "amplitude_range": amplitude_range,
            "signal": signal,
            **fit_result,
        }

    def optimize_x90(
        self,
        qubit: str,
        *,
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform:
        pulse = self.drag_hpi_pulse[qubit]
        N = pulse.length
        initial_params = list(pulse.real) + list(pulse.imag)
        es = cma.CMAEvolutionStrategy(
            initial_params,
            sigma0,
            {
                "seed": seed,
                "ftarget": ftarget,
                "timeout": timeout,
                "bounds": [[-1] * 2 * N, [1] * 2 * N],
            },
        )

        def objective_func(params):
            pulse = Pulse(params[:N] + 1j * params[N:])
            result = self.state_tomography(
                {qubit: pulse.repeated(2)},
                x90={qubit: pulse},
            )
            loss = np.linalg.norm(result[qubit] - np.array((0, 0, -1)))
            return loss

        es.optimize(objective_func)
        x = es.result.xbest
        opt_pulse = Pulse(x[:N] + 1j * x[N:])
        return opt_pulse

    def optimize_drag_x90(
        self,
        qubit: str,
        *,
        duration: float = 16,
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform:
        param = self.calib_note.get_drag_hpi_param(qubit)
        if param is None:
            raise ValueError("DRAG HPI parameters are not stored.")
        initial_params = [param["amplitude"], param["beta"]]
        es = cma.CMAEvolutionStrategy(
            initial_params,
            sigma0,
            {
                "seed": seed,
                "ftarget": ftarget,
                "timeout": timeout,
                "bounds": [[-1, -1], [1, 1]],
            },
        )

        def objective_func(params):
            pulse = Drag(
                duration=duration,
                amplitude=params[0],
                beta=params[1],
            )
            result = self.state_tomography(
                {qubit: pulse.repeated(2)},
                x90={qubit: pulse},
            )
            loss = np.linalg.norm(result[qubit] - np.array((0, 0, -1)))
            return loss

        es.optimize(objective_func)
        x = es.result.xbest
        opt_pulse = Drag(duration=duration, amplitude=x[0], beta=x[1])
        return opt_pulse

    def optimize_pulse(
        self,
        qubit: str,
        *,
        pulse: Waveform,
        x90: Waveform,
        target_state: tuple[float, float, float],
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform:
        N = pulse.length
        initial_params = list(pulse.real) + list(pulse.imag)
        es = cma.CMAEvolutionStrategy(
            initial_params,
            sigma0,
            {
                "seed": seed,
                "ftarget": ftarget,
                "timeout": timeout,
                "bounds": [[-1] * 2 * N, [1] * 2 * N],
            },
        )

        def objective_func(params):
            pulse = Pulse(params[:N] + 1j * params[N:])
            result = self.state_tomography({qubit: pulse}, x90={qubit: x90})
            loss = np.linalg.norm(result[qubit] - np.array(target_state))
            return loss

        es.optimize(objective_func)
        x = es.result.xbest
        opt_pulse = Pulse(x[:N] + 1j * x[N:])
        return opt_pulse

    def optimize_zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        duration: float = 100,
        ramptime: float = 20,
        x180: TargetMap[Waveform] | Waveform | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
    ):
        cr_label = f"{control_qubit}-{target_qubit}"
        cr_param = self.calib_note.get_cr_param(cr_label)
        if cr_param is None:
            raise ValueError("CR parameters are not stored.")
        cr_ramptime = ramptime
        cr_amplitude = cr_param["cr_amplitude"]
        cr_phase = cr_param["cr_phase"]
        cancel_amplitude = cr_param["cancel_amplitude"]
        cancel_phase = cr_param["cancel_phase"]

        if x180 is None:
            if control_qubit in self.drag_pi_pulse:
                x180 = self.drag_pi_pulse
            else:
                x180 = self.pi_pulse
        elif isinstance(x180, Waveform):
            x180 = {control_qubit: x180}

        def objective_func(params):
            ecr_0 = CrossResonance(
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                cr_amplitude=params[0],
                cr_duration=duration,
                cr_ramptime=cr_ramptime,
                cr_phase=params[1],
                cancel_amplitude=params[2],
                cancel_phase=params[3],
                echo=True,
                pi_pulse=x180[control_qubit],
            )
            with PulseSchedule([control_qubit, cr_label, target_qubit]) as ecr_1:
                ecr_1.add(
                    control_qubit,
                    self.get_pulse_for_state(control_qubit, "1"),
                )
                ecr_1.barrier()
                ecr_1.call(ecr_0)

            result_0 = self.state_tomography(
                ecr_0,
                shots=shots,
                interval=interval,
            )
            result_1 = self.state_tomography(
                ecr_1,
                shots=shots,
                interval=interval,
            )

            loss_c0 = np.linalg.norm(result_0[control_qubit] - np.array((0, 0, 1)))
            loss_t0 = np.linalg.norm(result_0[target_qubit] - np.array((0, -1, 0)))
            loss_c1 = np.linalg.norm(result_1[control_qubit] - np.array((0, 0, -1)))
            loss_t1 = np.linalg.norm(result_1[target_qubit] - np.array((0, 1, 0)))

            loss = loss_c0 + loss_t0 + loss_c1 + loss_t1
            return loss

        initial_params = [cr_amplitude, cr_phase, cancel_amplitude, cancel_phase]
        es = cma.CMAEvolutionStrategy(
            initial_params,
            1.0,
            {
                "seed": 42,
                "ftarget": 1e-3,
                "timeout": 300,
                "bounds": [[0, -np.pi, 0, -np.pi], [1, np.pi, 1, np.pi]],
                "CMA_stds": [
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                ],
            },
        )
        es.optimize(objective_func)
        x = es.result.xbest

        # self.system_note.put(  # deprecated
        #     CR_PARAMS,
        #     {
        #         cr_label: {
        #             "duration": duration,
        #             "ramptime": cr_ramptime,
        #             "cr_pulse": {
        #                 "amplitude": x[0],
        #                 "phase": x[1],
        #             },
        #             "cancel_pulse": {
        #                 "amplitude": x[2],
        #                 "phase": x[3],
        #             },
        #         },
        #     },
        # )
        self.calib_note.cr_params = {
            cr_label: {
                "target": cr_label,
                "duration": duration,
                "ramptime": cr_ramptime,
                "cr_amplitude": x[0],
                "cr_phase": x[1],
                "cancel_amplitude": x[2],
                "cancel_phase": x[3],
                "cr_cancel_ratio": x[2] / x[0],
            },
        }

        return {
            "cr_amplitude": x[0],
            "cr_phase": x[1],
            "cancel_amplitude": x[2],
            "cancel_phase": x[3],
        }
