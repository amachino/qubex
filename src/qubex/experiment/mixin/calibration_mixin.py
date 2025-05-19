from __future__ import annotations

from collections import defaultdict
from typing import Collection, Literal

import cma
import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

from ...analysis import fitting
from ...analysis import visualization as viz
from ...backend import SAMPLING_PERIOD, Target
from ...measurement.measurement import DEFAULT_INTERVAL, DEFAULT_SHOTS
from ...pulse import (
    CrossResonance,
    Drag,
    FlatTop,
    Pulse,
    PulseArray,
    PulseSchedule,
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
        targets: Collection[str] | str | None = None,
        *,
        pulse_type: Literal["pi", "hpi"],
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        rabi_params = self.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        def calibrate(target: str) -> AmplCalibData:
            if pulse_type == "hpi":
                pulse = FlatTop(
                    duration=duration if duration is not None else HPI_DURATION,
                    amplitude=1,
                    tau=ramptime if ramptime is not None else HPI_RAMPTIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area
            elif pulse_type == "pi":
                pulse = FlatTop(
                    duration=duration if duration is not None else PI_DURATION,
                    amplitude=1,
                    tau=ramptime if ramptime is not None else PI_RAMPTIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area
            else:
                raise ValueError("Invalid pulse type.")

            # calculate the control amplitude for the target rabi rate
            ampl = self.calc_control_amplitude(target, rabi_rate)

            # create a range of amplitudes around the estimated value
            ampl_min = ampl * (1 - 0.5 / n_rotations)
            ampl_max = ampl * (1 + 0.5 / n_rotations)
            ampl_min = np.clip(ampl_min, 0, 1)
            ampl_max = np.clip(ampl_max, 0, 1)
            if ampl_min == ampl_max:
                ampl_min = 0
                ampl_max = 1
            ampl_range = np.linspace(ampl_min, ampl_max, n_points)

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
                ylabel="Normalized signal",
            )

            r2 = fit_result["r2"]
            if r2 > r2_threshold:
                if pulse_type == "hpi":
                    self.calib_note.update_hpi_param(
                        target,
                        {
                            "target": target,
                            "duration": pulse.duration,
                            "amplitude": fit_result["amplitude"],
                            "tau": pulse.tau,
                        },
                    )
                elif pulse_type == "pi":
                    self.calib_note.update_pi_param(
                        target,
                        {
                            "target": target,
                            "duration": pulse.duration,
                            "amplitude": fit_result["amplitude"],
                            "tau": pulse.tau,
                        },
                    )
            else:
                print(f"Error: R² value is too low ({r2:.3f})")
                print(f"Calibration data not stored for {target}.")

            return AmplCalibData.new(
                sweep_data=sweep_data,
                calib_value=fit_result["amplitude"],
                r2=r2,
            )

        data: dict[str, AmplCalibData] = {}
        for target in targets:
            if target not in self.calib_note.rabi_params:
                print(f"Rabi parameters are not stored for {target}.")
                continue
            print(f"Calibrating {pulse_type} pulse for {target}...")
            data[target] = calibrate(target)

        print("")
        print(f"Calibration results for {pulse_type} pulse:")
        for target, calib_data in data.items():
            print(f"  {target}: {calib_data.calib_value:.6f}")

        return ExperimentResult(data=data)

    def calibrate_hpi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        return self.calibrate_default_pulse(
            targets=targets,
            pulse_type="hpi",
            duration=duration,
            ramptime=ramptime,
            n_points=n_points,
            n_rotations=n_rotations,
            r2_threshold=r2_threshold,
            plot=plot,
            shots=shots,
            interval=interval,
        )

    def calibrate_pi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        return self.calibrate_default_pulse(
            targets=targets,
            pulse_type="pi",
            duration=duration,
            ramptime=ramptime,
            n_points=n_points,
            n_rotations=n_rotations,
            r2_threshold=r2_threshold,
            plot=plot,
            shots=shots,
            interval=interval,
        )

    def calibrate_ef_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        pulse_type: Literal["pi", "hpi"],
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        rabi_params = self.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        ef_labels = [
            Target.ef_label(label) for label in targets if label in self.ef_rabi_params
        ]

        def calibrate(target: str) -> AmplCalibData:
            ge_label = Target.ge_label(target)
            ef_label = Target.ef_label(target)

            if pulse_type == "hpi":
                pulse = FlatTop(
                    duration=duration if duration is not None else HPI_DURATION,
                    amplitude=1,
                    tau=ramptime if ramptime is not None else HPI_RAMPTIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area
            elif pulse_type == "pi":
                pulse = FlatTop(
                    duration=duration if duration is not None else PI_DURATION,
                    amplitude=1,
                    tau=ramptime if ramptime is not None else PI_RAMPTIME,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area
            else:
                raise ValueError("Invalid pulse type.")

            ampl = self.calc_control_amplitude(ef_label, rabi_rate)

            ampl_min = ampl * (1 - 0.5 / n_rotations)
            ampl_max = ampl * (1 + 0.5 / n_rotations)
            ampl_min = np.clip(ampl_min, 0, 1)
            ampl_max = np.clip(ampl_max, 0, 1)
            if ampl_min == ampl_max:
                ampl_min = 0
                ampl_max = 1
            ampl_range = np.linspace(ampl_min, ampl_max, n_points)

            n_per_rotation = 2 if pulse_type == "pi" else 4
            repetitions = n_per_rotation * n_rotations

            def sequence(x: float) -> PulseSchedule:
                with PulseSchedule() as ps:
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
                plot=plot,
            ).data[ge_label]

            fit_result = fitting.fit_ampl_calib_data(
                target=ef_label,
                amplitude_range=ampl_range,
                data=sweep_data.normalized,
                plot=plot,
                title=f"ef {pulse_type} pulse calibration",
                ylabel="Normalized signal",
            )

            r2 = fit_result["r2"]

            if r2 > r2_threshold:
                if pulse_type == "hpi":
                    self.calib_note.update_hpi_param(
                        ef_label,
                        {
                            "target": ef_label,
                            "duration": pulse.duration,
                            "amplitude": fit_result["amplitude"],
                            "tau": pulse.tau,
                        },
                    )
                elif pulse_type == "pi":
                    self.calib_note.update_pi_param(
                        ef_label,
                        {
                            "target": ef_label,
                            "duration": pulse.duration,
                            "amplitude": fit_result["amplitude"],
                            "tau": pulse.tau,
                        },
                    )
            else:
                print(f"Error: R² value is too low ({r2:.3f})")
                print(f"Calibration data not stored for {ef_label}.")

            return AmplCalibData.new(
                sweep_data=sweep_data,
                calib_value=fit_result["amplitude"],
                r2=r2,
            )

        data: dict[str, AmplCalibData] = {}
        for target in ef_labels:
            data[target] = calibrate(target)

        print("")
        print(f"Calibration results for {pulse_type} pulse:")
        for target, calib_data in data.items():
            print(f"  {target}: {calib_data.calib_value:.6f}")

        return ExperimentResult(data=data)

    def calibrate_ef_hpi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        return self.calibrate_ef_pulse(
            targets=targets,
            pulse_type="hpi",
            duration=duration,
            ramptime=ramptime,
            n_points=n_points,
            n_rotations=n_rotations,
            r2_threshold=r2_threshold,
            plot=plot,
            shots=shots,
            interval=interval,
        )

    def calibrate_ef_pi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int = 20,
        n_rotations: int = 1,
        r2_threshold: float = 0.5,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        return self.calibrate_ef_pulse(
            targets=targets,
            pulse_type="pi",
            duration=duration,
            ramptime=ramptime,
            n_points=n_points,
            n_rotations=n_rotations,
            r2_threshold=r2_threshold,
            plot=plot,
            shots=shots,
            interval=interval,
        )

    def calibrate_drag_amplitude(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str = "0",
        pulse_type: Literal["pi", "hpi"],
        duration: float | None = None,
        n_points: int = 20,
        n_rotations: int = 4,
        r2_threshold: float = 0.5,
        drag_coeff: float = DRAG_COEFF,
        use_stored_amplitude: bool = False,
        use_stored_beta: bool = False,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> dict[str, dict]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        rabi_params = self.rabi_params
        self.validate_rabi_params(rabi_params)

        def calibrate(target: str) -> dict:
            # hpi
            if pulse_type == "hpi":
                hpi_param = self.calib_note.get_drag_hpi_param(target)
                if hpi_param is not None and use_stored_beta:
                    beta = hpi_param["beta"]
                else:
                    beta = -drag_coeff / self.qubits[target].alpha

                pulse = Drag(
                    duration=duration if duration is not None else DRAG_HPI_DURATION,
                    amplitude=1,
                    beta=beta,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.25 / area

                if hpi_param is not None and use_stored_amplitude:
                    ampl = hpi_param["amplitude"]
                else:
                    ampl = self.calc_control_amplitude(target, rabi_rate)
            # pi
            elif pulse_type == "pi":
                pi_param = self.calib_note.get_drag_pi_param(target)
                if pi_param is not None and use_stored_beta:
                    beta = pi_param["beta"]
                else:
                    beta = -drag_coeff / self.qubits[target].alpha

                pulse = Drag(
                    duration=duration if duration is not None else DRAG_PI_DURATION,
                    amplitude=1,
                    beta=beta,
                )
                area = pulse.real.sum() * pulse.SAMPLING_PERIOD
                rabi_rate = 0.5 / area

                if pi_param is not None and use_stored_amplitude:
                    ampl = pi_param["amplitude"]
                else:
                    ampl = self.calc_control_amplitude(target, rabi_rate)
            else:
                raise ValueError("Invalid pulse type.")

            ampl_min = ampl * (1 - 0.5 / n_rotations)
            ampl_max = ampl * (1 + 0.5 / n_rotations)
            ampl_min = np.clip(ampl_min, 0, 1)
            ampl_max = np.clip(ampl_max, 0, 1)
            if ampl_min == ampl_max:
                ampl_min = 0
                ampl_max = 1
            ampl_range = np.linspace(ampl_min, ampl_max, n_points)

            n_per_rotation = 2 if pulse_type == "pi" else 4

            spectators = self.get_spectators(target)

            def sequence(x: float) -> PulseSchedule:
                with PulseSchedule() as ps:
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
                plot=plot,
                title=f"DRAG {pulse_type} amplitude calibration",
                ylabel="Normalized signal",
            )

            r2 = fit_result["r2"]
            if r2 > r2_threshold:
                if pulse_type == "hpi":
                    self.calib_note.update_drag_hpi_param(
                        target,
                        {
                            "target": target,
                            "duration": pulse.duration,
                            "amplitude": fit_result["amplitude"],
                            "beta": beta,
                        },
                    )
                elif pulse_type == "pi":
                    self.calib_note.update_drag_pi_param(
                        target,
                        {
                            "target": target,
                            "duration": pulse.duration,
                            "amplitude": fit_result["amplitude"],
                            "beta": beta,
                        },
                    )
            else:
                print(f"Error: R² value is too low ({r2:.3f})")
                print(f"Calibration data not stored for {target}.")

            return fit_result

        result: dict[str, dict] = {}
        for target in targets:
            result[target] = calibrate(target)

        return result

    def calibrate_drag_beta(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str = "0",
        pulse_type: Literal["pi", "hpi"] = "hpi",
        beta_range: ArrayLike = np.linspace(-2.0, 2.0, 20),
        duration: float | None = None,
        n_turns: int = 1,
        degree: int = 3,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> dict[str, float]:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        rabi_params = self.rabi_params
        self.validate_rabi_params(rabi_params)

        def calibrate(target: str) -> float:
            if pulse_type == "hpi":
                param = self.calib_note.get_drag_hpi_param(target)
            elif pulse_type == "pi":
                param = self.calib_note.get_drag_pi_param(target)
            if param is None:
                raise ValueError("DRAG parameters are not stored.")

            drag_duration = duration or param["duration"]
            drag_amplitude = param["amplitude"]
            drag_beta = param["beta"]

            sweep_range = np.array(beta_range) + drag_beta

            spectators = self.get_spectators(target)

            def sequence(beta: float) -> PulseSchedule:
                with PulseSchedule() as ps:
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
                            PulseArray(
                                [
                                    x90p,
                                    PulseArray([x90m, x90p] * n_turns),
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
                            PulseArray(
                                [
                                    PulseArray([x180p, x180m] * n_turns),
                                    y90m,
                                ]
                            ),
                        )
                return ps

            sweep_data = self.sweep_parameter(
                sequence=sequence,
                sweep_range=sweep_range,
                shots=shots,
                interval=interval,
                plot=False,
            ).data[target]

            values = sweep_data.normalized

            fit_result = fitting.fit_polynomial(
                target=target,
                x=sweep_range,
                y=values,
                degree=degree,
                plot=plot,
                title=f"DRAG {pulse_type} beta calibration",
                xlabel="Beta",
                ylabel="Normalized signal",
            )
            beta = fit_result["root"]
            if np.isnan(beta):
                beta = 0.0

            if pulse_type == "hpi":
                self.calib_note.update_drag_hpi_param(
                    target,
                    {
                        "target": target,
                        "duration": drag_duration,
                        "amplitude": drag_amplitude,
                        "beta": beta,
                    },
                )
            elif pulse_type == "pi":
                self.calib_note.update_drag_pi_param(
                    target,
                    {
                        "target": target,
                        "duration": drag_duration,
                        "amplitude": drag_amplitude,
                        "beta": beta,
                    },
                )

            return beta

        result = {}
        for target in targets:
            result[target] = calibrate(target)

        return result

    def calibrate_drag_hpi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str = "0",
        n_points: int = 20,
        n_rotations: int = 4,
        n_turns: int = 1,
        n_iterations: int = 2,
        degree: int = 3,
        r2_threshold: float = 0.5,
        calibrate_beta: bool = True,
        beta_range: ArrayLike = np.linspace(-2.0, 2.0, 20),
        duration: float | None = None,
        drag_coeff: float = DRAG_COEFF,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        for i in range(n_iterations):
            print(f"\nIteration {i + 1}/{n_iterations}")

            if i == 0:
                amplitude = self.calibrate_drag_amplitude(
                    targets=targets,
                    spectator_state=spectator_state,
                    pulse_type="hpi",
                    n_points=n_points,
                    n_rotations=1,
                    r2_threshold=r2_threshold,
                    duration=duration,
                    use_stored_amplitude=False,
                    use_stored_beta=False,
                    plot=plot,
                    shots=shots,
                    interval=interval,
                )
            else:
                amplitude = self.calibrate_drag_amplitude(
                    targets=targets,
                    spectator_state=spectator_state,
                    pulse_type="hpi",
                    n_points=n_points,
                    n_rotations=n_rotations,
                    r2_threshold=r2_threshold,
                    duration=duration,
                    use_stored_amplitude=True,
                    use_stored_beta=True,
                    plot=plot,
                    shots=shots,
                    interval=interval,
                )

            if calibrate_beta:
                beta = self.calibrate_drag_beta(
                    targets=targets,
                    spectator_state=spectator_state,
                    pulse_type="hpi",
                    beta_range=beta_range,
                    n_turns=n_turns,
                    duration=duration,
                    degree=degree,
                    plot=plot,
                    shots=shots,
                    interval=interval,
                )
            else:
                beta = {
                    target: -drag_coeff / self.qubits[target].alpha
                    for target in targets
                }

        return {
            "amplitude": amplitude,
            "beta": beta,
        }

    def calibrate_drag_pi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str = "0",
        n_points: int = 20,
        n_rotations: int = 4,
        n_turns: int = 1,
        n_iterations: int = 2,
        degree: int = 3,
        r2_threshold: float = 0.5,
        calibrate_beta: bool = True,
        beta_range: ArrayLike = np.linspace(-2.0, 2.0, 20),
        duration: float | None = None,
        drag_coeff: float = DRAG_COEFF,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> dict:
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        for i in range(n_iterations):
            print(f"\nIteration {i + 1}/{n_iterations}")

            if i == 0:
                amplitude = self.calibrate_drag_amplitude(
                    targets=targets,
                    spectator_state=spectator_state,
                    pulse_type="pi",
                    n_points=n_points,
                    n_rotations=1,
                    r2_threshold=r2_threshold,
                    duration=duration,
                    use_stored_amplitude=False,
                    use_stored_beta=False,
                    plot=plot,
                    shots=shots,
                    interval=interval,
                )
            else:
                amplitude = self.calibrate_drag_amplitude(
                    targets=targets,
                    spectator_state=spectator_state,
                    pulse_type="pi",
                    n_points=n_points,
                    n_rotations=n_rotations,
                    r2_threshold=r2_threshold,
                    duration=duration,
                    use_stored_amplitude=True,
                    use_stored_beta=True,
                    plot=plot,
                    shots=shots,
                    interval=interval,
                )

            if calibrate_beta:
                beta = self.calibrate_drag_beta(
                    targets=targets,
                    spectator_state=spectator_state,
                    pulse_type="pi",
                    beta_range=beta_range,
                    n_turns=n_turns,
                    duration=duration,
                    degree=degree,
                    plot=plot,
                    shots=shots,
                    interval=interval,
                )
            else:
                beta = {
                    target: -drag_coeff / self.qubits[target].alpha
                    for target in targets
                }

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
        ramptime: float = 0.0,
        cr_amplitude: float = 1.0,
        cr_phase: float = 0.0,
        cancel_amplitude: float = 0.0,
        cancel_phase: float = 0.0,
        echo: bool = False,
        control_state: str = "0",
        x90: TargetMap[Waveform] | None = None,
        x180: TargetMap[Waveform] | None = None,
        ramp_type: Literal[
            "Gaussian",
            "RaisedCosine",
            "Sintegral",
            "Bump",
        ] = "RaisedCosine",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
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
                    cr_duration=T + ramptime * 2,
                    cr_ramptime=ramptime,
                    cr_phase=cr_phase,
                    cancel_amplitude=cancel_amplitude,
                    cancel_phase=cancel_phase,
                    echo=echo,
                    pi_pulse=x180[control_qubit],
                    ramp_type=ramp_type,
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
        ramptime: float | None = None,
        cr_amplitude: float | None = None,
        cr_phase: float | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        x90: TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = False,
    ) -> dict:
        cr_label = f"{control_qubit}-{target_qubit}"

        if time_range is None:
            time_range = np.arange(0, 1001, 50)
        else:
            time_range = np.array(time_range)

        if ramptime is None:
            ramptime = 0.0
        if cr_amplitude is None:
            cr_amplitude = 1.0
        if cr_phase is None:
            cr_phase = 0.0
        if cancel_amplitude is None:
            cancel_amplitude = 0.0
        if cancel_phase is None:
            cancel_phase = 0.0

        effective_drive_range = time_range + ramptime

        result_0 = self.measure_cr_dynamics(
            time_range=time_range,
            ramptime=ramptime,
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

        control_states_0 = result_0["control_states"]
        target_states_0 = result_0["target_states"]

        fit_0 = fitting.fit_rotation(
            effective_drive_range,
            target_states_0,
            plot=False,
            title=f"Target qubit dynamics of {cr_label} : |0〉",
            xlabel="Drive time (ns)",
            ylabel=f"Expectation value : {target_qubit}",
        )

        if plot:
            print("Control = |0〉")
            print(f"Control qubit : {control_qubit}")
            viz.plot_bloch_vectors(
                effective_drive_range,
                control_states_0,
                title=f"Control qubit dynamics of {cr_label} : |0〉",
                xlabel="Drive time (ns)",
                ylabel=f"Expectation value : {control_qubit}",
            )
            viz.display_bloch_sphere(control_states_0)

            print(f"Target qubit : {target_qubit}")
            fit_0["fig"].show()
            fit_0["fig3d"].show()
            viz.display_bloch_sphere(target_states_0)

        result_1 = self.measure_cr_dynamics(
            time_range=time_range,
            ramptime=ramptime,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            echo=False,
            control_state="1",
            x90=x90,
            ramp_type="RaisedCosine",
            shots=shots,
            interval=interval,
        )

        control_states_1 = result_1["control_states"]
        target_states_1 = result_1["target_states"]

        fit_1 = fitting.fit_rotation(
            effective_drive_range,
            target_states_1,
            plot=False,
            title=f"Target qubit dynamics of {cr_label} : |1〉",
            xlabel="Drive time (ns)",
            ylabel=f"Expectation value : {target_qubit}",
        )

        if plot:
            print("Control = |1〉")
            print(f"Target qubit : {control_qubit}")
            viz.plot_bloch_vectors(
                effective_drive_range,
                control_states_1,
                title=f"Control qubit dynamics of {cr_label} : |1〉",
                xlabel="Drive time (ns)",
                ylabel=f"Expectation value : {control_qubit}",
            )
            viz.display_bloch_sphere(control_states_1)

            print(f"Target qubit : {target_qubit}")
            fit_1["fig"].show()
            fit_1["fig3d"].show()
            viz.display_bloch_sphere(target_states_1)

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

        f_control = self.qubits[control_qubit].frequency
        f_target = self.qubits[target_qubit].frequency
        f_delta = f_control - f_target

        print("Qubit frequencies:")
        print(f"  control ({control_qubit}) : {f_control * 1e3:.3f} MHz")
        print(f"  target  ({target_qubit}) : {f_target * 1e3:.3f} MHz")
        print(f"  Δ ({cr_label}) : {f_delta * 1e3:.3f} MHz")
        print()

        print("Rotation coefficients:")
        for key, value in coeffs.items():
            print(f"  {key} : {value * 1e3:+.3f} MHz")
        print()

        # xt (cross-talk) rotation
        xt_rotation = coeffs["IX"] + 1j * coeffs["IY"]
        xt_rotation_amplitude = np.abs(xt_rotation)
        xt_rotation_amplitude_hw = self.calc_control_amplitude(
            target=target_qubit,
            rabi_rate=xt_rotation_amplitude,
        )
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
        cr_rotation_amplitude_hw = self.calc_control_amplitude(
            target=target_qubit,
            rabi_rate=cr_rotation_amplitude,
        )
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
        cr_rabi_rate = self.calc_rabi_rate(control_qubit, cr_amplitude)
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
        ramptime: float | None = None,
        cr_amplitude: float | None = None,
        cr_phase: float | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        update_cr_phase: bool = True,
        update_cancel_pulse: bool = True,
        x90: TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = False,
    ) -> dict:
        if ramptime is None:
            ramptime = 0.0
        if cr_amplitude is None:
            cr_amplitude = 1.0
        if cr_phase is None:
            cr_phase = 0.0
        if cancel_amplitude is None:
            cancel_amplitude = 0.0
        if cancel_phase is None:
            cancel_phase = 0.0

        current_cr_pulse = cr_amplitude * np.exp(1j * cr_phase)
        current_cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase)

        result = self.cr_hamiltonian_tomography(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            time_range=time_range,
            ramptime=ramptime,
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

        if update_cr_phase:
            new_cr_pulse = current_cr_pulse * np.exp(1j * shift)
        else:
            new_cr_pulse = current_cr_pulse

        if update_cancel_pulse:
            new_cancel_pulse = (current_cancel_pulse + cancel_pulse) * np.exp(
                1j * shift
            )
        else:
            new_cancel_pulse = current_cancel_pulse

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
        duration = (
            (0.5 * result["zx90_duration"]) // SAMPLING_PERIOD + 1
        ) * SAMPLING_PERIOD

        zx_rotation_rate = result["coeffs"]["ZX"] / cr_amplitude

        self.calib_note.cr_params = {
            cr_label: {
                "target": cr_label,
                "duration": duration,
                "ramptime": ramptime,
                "cr_amplitude": new_cr_amplitude,
                "cr_phase": new_cr_phase,
                "cr_beta": 0.0,
                "cancel_amplitude": new_cancel_amplitude,
                "cancel_phase": new_cancel_phase,
                "cancel_beta": 0.0,
                "rotary_amplitude": 0.0,
                "zx_rotation_rate": zx_rotation_rate,
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
        time_range: ArrayLike | None = None,
        ramptime: float = 16,
        cr_amplitude: float | None = None,
        n_iterations: int = 4,
        n_cycles: int = 2,
        n_points_per_cycle: int = 10,
        use_stored_params: bool = False,
        tolerance: float = 10e-6,
        adiabatic_safe_factor: float = 0.75,
        max_amplitude: float = 1.0,
        x90: TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        def _create_time_range(
            zx90_duration: float,
        ) -> NDArray:
            period = 4 * zx90_duration
            dt = (period / n_points_per_cycle) // SAMPLING_PERIOD * SAMPLING_PERIOD
            duration = period * n_cycles
            return np.arange(0, duration + 1, dt)

        cr_label = f"{control_qubit}-{target_qubit}"

        f_control = self.qubits[control_qubit].frequency
        f_target = self.qubits[target_qubit].frequency
        f_delta = np.abs(f_target - f_control)
        max_cr_rabi = adiabatic_safe_factor * f_delta
        max_cr_amplitude = self.calc_control_amplitude(control_qubit, max_cr_rabi)
        max_cr_amplitude: float = np.clip(max_cr_amplitude, 0.0, max_amplitude)

        current_cr_param = self.calib_note.cr_params.get(cr_label)

        if use_stored_params and current_cr_param is not None:
            cr_amplitude = current_cr_param["cr_amplitude"]
            cr_phase = current_cr_param["cr_phase"]
            cancel_amplitude = current_cr_param["cancel_amplitude"]
            cancel_phase = current_cr_param["cancel_phase"]
            time_range = _create_time_range(current_cr_param["duration"] * 2)
        else:
            cr_amplitude = (
                cr_amplitude if cr_amplitude is not None else max_cr_amplitude
            )
            cr_phase = 0.0
            cancel_amplitude = 0.0
            cancel_phase = 0.0
            time_range = (
                time_range if time_range is not None else np.arange(0, 1001, 20)
            )

        params_history = [
            {
                "time_range": time_range,
                "cr_phase": cr_phase,
                "cancel_amplitude": cancel_amplitude,
                "cancel_phase": cancel_phase,
            }
        ]

        coeffs_history = defaultdict(list)

        print(f"Conducting CR Hamiltonian tomography for {cr_label}...")
        for i in range(n_iterations):
            print(f"Iteration {i + 1}/{n_iterations}")
            params = params_history[-1]

            result = self.update_cr_params(
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                time_range=params["time_range"],
                ramptime=ramptime,
                cr_amplitude=cr_amplitude,
                cr_phase=params["cr_phase"],
                cancel_amplitude=params["cancel_amplitude"],
                cancel_phase=params["cancel_phase"],
                x90=x90,
                shots=shots,
                interval=interval,
                plot=plot,
            )

            params_history.append(
                {
                    "time_range": _create_time_range(result["zx90_duration"]),
                    "cr_phase": result["cr_param"]["cr_phase"],
                    "cancel_amplitude": result["cr_param"]["cancel_amplitude"],
                    "cancel_phase": result["cr_param"]["cancel_phase"],
                }
            )
            for key, value in result["coeffs"].items():
                coeffs_history[key].append(value)

            if i > 0:
                IX = coeffs_history["IX"][-1]
                IY = coeffs_history["IY"][-1]
                IX_diff = coeffs_history["IX"][-2] - IX
                IY_diff = coeffs_history["IY"][-2] - IY

                if abs(IX) < tolerance and abs(IY) < tolerance:
                    print("Convergence reached.")
                    print(f"  IX : {IX * 1e3:.3f} MHz")
                    print(f"  IY : {IY * 1e3:.3f} MHz")
                    break
                if abs(IX_diff) < tolerance and abs(IY_diff) < tolerance:
                    print("Convergence reached.")
                    print(f"  IX_diff : {IX_diff * 1e3:.3f} MHz")
                    print(f"  IY_diff : {IY_diff * 1e3:.3f} MHz")
                    break

        hamiltonian_coeffs = {
            key: np.array(value) for key, value in coeffs_history.items()
        }

        fig = go.Figure()
        for key, value in hamiltonian_coeffs.items():
            fig.add_trace(
                go.Scatter(
                    x=np.arange(1, len(value) + 1),
                    y=value * 1e3,
                    mode="lines+markers",
                    name=f"{key}/2",
                )
            )

        fig.update_layout(
            title=f"CR Hamiltonian coefficients : {cr_label}",
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
        ramptime: float = 16.0,
        duration: float | None = None,
        amplitude_range: ArrayLike | None = None,
        initial_state: str = "0",
        degree: int = 3,
        adiabatic_safe_factor: float = 0.75,
        max_amplitude: float = 1.0,
        rotary_multiple: float = 9.0,
        use_drag: bool = True,
        duration_unit: float = 16.0,
        duration_buffer: float = 1.05,
        n_repetitions: int = 1,
        x180: TargetMap[Waveform] | Waveform | None = None,
        x180_margin: float = 0.0,
        use_zvalues: bool = False,
        store_params: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> dict:
        if x180 is None:
            x180 = self.x180(control_qubit)
        elif not isinstance(x180, Waveform):
            x180 = x180[control_qubit]

        cr_label = f"{control_qubit}-{target_qubit}"
        cr_param = self.calib_note.get_cr_param(cr_label)

        if cr_param is None:
            raise ValueError("CR parameters are not stored.")

        cr_amplitude = cr_param["cr_amplitude"]
        cr_phase = cr_param["cr_phase"]
        cancel_amplitude = cr_param["cancel_amplitude"]
        cancel_phase = cr_param["cancel_phase"]
        zx_rotation_rate = cr_param["zx_rotation_rate"]
        zx_frequency = zx_rotation_rate * cr_amplitude
        rotary_amplitude = self.calc_control_amplitude(
            target=target_qubit,
            rabi_rate=zx_frequency * rotary_multiple,
        )
        cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase) + rotary_amplitude

        f_control = self.qubits[control_qubit].frequency
        f_target = self.qubits[target_qubit].frequency
        f_delta = np.abs(f_target - f_control)
        max_cr_rabi = adiabatic_safe_factor * f_delta
        max_cr_amplitude = self.calc_control_amplitude(control_qubit, max_cr_rabi)
        max_cr_amplitude: float = np.clip(max_cr_amplitude, 0.0, max_amplitude)

        if duration is None:
            duration = duration_buffer / (8 * zx_frequency) + ramptime
            if duration % duration_unit != 0:
                duration = (duration // duration_unit + 1) * duration_unit

        def ecr_sequence(amplitude: float, n_repetitions: int) -> PulseSchedule:
            scaled_cancel_pulse = amplitude / cr_amplitude * cancel_pulse
            ecr = CrossResonance(
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                cr_amplitude=amplitude,
                cr_duration=duration,
                cr_ramptime=ramptime,
                cr_phase=cr_phase,
                cancel_amplitude=np.abs(scaled_cancel_pulse),
                cancel_phase=np.angle(scaled_cancel_pulse),
                echo=True,
                pi_pulse=x180,
                pi_margin=x180_margin,
            ).repeated(n_repetitions)
            with PulseSchedule() as ps:
                if initial_state != "0":
                    ps.add(
                        control_qubit,
                        self.get_pulse_for_state(control_qubit, initial_state),
                    )
                    ps.barrier()
                ps.call(ecr)
            return ps

        def calibrate(n_repetitions, amplitude_range) -> dict:
            min_amplitude = np.clip(amplitude_range[0], 0.0, max_cr_amplitude)
            max_amplitude = np.clip(amplitude_range[-1], 0.0, max_cr_amplitude)
            amplitude_range = np.linspace(
                min_amplitude,
                max_amplitude,
                len(amplitude_range),
            )

            sweep_result = self.sweep_parameter(
                lambda x: ecr_sequence(x, n_repetitions),
                sweep_range=amplitude_range,
                shots=shots,
                interval=interval,
                plot=False,
            )

            if use_zvalues:
                signal = sweep_result.data[target_qubit].zvalues
            else:
                signal = sweep_result.data[target_qubit].normalized

            fit_result = fitting.fit_polynomial(
                target=cr_label,
                x=amplitude_range,
                y=signal,
                degree=degree,
                title=f"ZX90 calibration (n = {n_repetitions})",
                xlabel="Amplitude (arb. units)",
                ylabel="Signal",
            )

            root = fit_result["root"]

            if np.isnan(root):
                root = None

            return {
                "amplitude_range": amplitude_range,
                "signal": signal,
                "root": root,
                "fit_result": fit_result,
            }

        if amplitude_range is None:
            print(f"Estimating CR amplitude of {cr_label} (n_repetitions = 1)")
            rough_result = calibrate(
                n_repetitions=n_repetitions,
                amplitude_range=np.linspace(0.0, cr_amplitude * 2, 20),
            )
            rough_amplitude = rough_result["root"]
            if rough_amplitude is None:
                raise ValueError("Could not find a root for the rough calibration.")
            else:
                min_amplitude = float(rough_amplitude * 0.8)
                max_amplitude = float(rough_amplitude * 1.2)
                amplitude_range = np.linspace(min_amplitude, max_amplitude, 50)
        else:
            amplitude_range = np.asarray(amplitude_range)

        print(
            f"Calibrating CR amplitude of {cr_label} (n_repetitions = {n_repetitions})"
        )
        result_n1 = calibrate(
            n_repetitions=n_repetitions,
            amplitude_range=amplitude_range,
        )
        amplitude_range = np.asarray(result_n1["amplitude_range"])
        signal_n1 = result_n1["signal"]
        fit_result_n1 = result_n1["fit_result"]

        print(
            f"Calibrating CR amplitude of {cr_label} (n_repetitions = {n_repetitions + 2})"
        )
        result_n3 = calibrate(
            n_repetitions=n_repetitions + 2,
            amplitude_range=amplitude_range,
        )
        signal_n3 = result_n3["signal"]
        fit_result_n3 = result_n3["fit_result"]

        signal = signal_n1 - signal_n3
        fit_result = fitting.fit_polynomial(
            target=cr_label,
            x=amplitude_range,
            y=signal,
            degree=degree,
            title="ZX90 calibration",
            xlabel="Amplitude (arb. units)",
            ylabel="Signal",
        )

        calibrated_cr_amplitude = fit_result["root"]

        if np.isnan(calibrated_cr_amplitude):
            print("Could not find a root for the CR amplitude calibration.")
            calibrated_cr_amplitude = 1.0

        calibrated_cancel_amplitude = (
            calibrated_cr_amplitude / cr_amplitude * cancel_amplitude
        )

        calibrated_rotary_amplitude = (
            calibrated_cr_amplitude / cr_amplitude * rotary_amplitude
        )

        if calibrated_cr_amplitude is not None and store_params:
            if use_drag:
                f_control = self.qubits[control_qubit].frequency
                f_target = self.qubits[target_qubit].frequency
                Delta_ct = 2 * np.pi * (f_control - f_target)
                cr_beta = -1 / Delta_ct
                # cancel_beta = -1 / self.qubits[target_qubit].alpha
                cancel_beta = 0.0
            else:
                cr_beta = 0.0
                cancel_beta = 0.0
            self.calib_note.cr_params = {
                cr_label: {
                    "target": cr_label,
                    "duration": duration,
                    "ramptime": ramptime,
                    "cr_amplitude": calibrated_cr_amplitude,
                    "cr_phase": cr_phase,
                    "cr_beta": cr_beta,
                    "cancel_amplitude": calibrated_cancel_amplitude,
                    "cancel_phase": cancel_phase,
                    "cancel_beta": cancel_beta,
                    "rotary_amplitude": calibrated_rotary_amplitude,
                    "zx_rotation_rate": zx_rotation_rate,
                },
            }

        print()
        print("Calibrated CR parameters:")
        print(f"  CR duration      : {duration:.1f} ns")
        print(f"  CR ramptime      : {ramptime:.1f} ns")
        print(f"  CR amplitude     : {calibrated_cr_amplitude:.6f}")
        print(f"  CR phase         : {cr_phase:.6f}")
        print(f"  CR beta          : {cr_beta:.6f}")
        print(f"  Cancel amplitude : {calibrated_cancel_amplitude:.6f}")
        print(f"  Cancel phase     : {cancel_phase:.6f}")
        print(f"  Cancel beta      : {cancel_beta:.6f}")
        print(f"  Rotary amplitude : {rotary_amplitude:.6f}")
        print()
        if plot:
            zx90 = self.zx90(control_qubit, target_qubit, x180=x180)
            zx90.plot(
                title=f"ZX90 sequence : {cr_label}",
                show_physical_pulse=True,
            )

        return {
            "amplitude_range": amplitude_range,
            "signal": signal,
            **fit_result,
            "n1": {
                "signal": signal_n1,
                **fit_result_n1,
            },
            "n3": {
                "signal": signal_n3,
                **fit_result_n3,
            },
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
        opt_params: Collection[str] | None = None,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
        duration: float | None = None,
        ramptime: float | None = None,
        x180: TargetMap[Waveform] | Waveform | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ):
        if opt_params is None:
            opt_params = [
                "cr_amplitude",
                "cr_phase",
                "cr_beta",
                "cancel_phase",
                "cancel_beta",
            ]

        if x180 is None:
            x180 = self.x180(control_qubit)
        elif not isinstance(x180, Waveform):
            x180 = x180[control_qubit]

        cr_label = f"{control_qubit}-{target_qubit}"
        cr_param = self.calib_note.get_cr_param(cr_label)
        if cr_param is None:
            raise ValueError("CR parameters are not stored.")

        if duration is None:
            duration = cr_param["duration"]
        if ramptime is None:
            ramptime = cr_param["ramptime"]

        defaults = {
            "cr_amplitude": {
                "initial": cr_param["cr_amplitude"],
                "bounds": [0.0, 1.0],
                "std": 0.01,
            },
            "cr_phase": {
                "initial": cr_param["cr_phase"],
                "bounds": [-np.pi, np.pi],
                "std": 0.01,
            },
            "cr_beta": {
                "initial": cr_param["cr_beta"],
                "bounds": [-10.0, 10.0],
                "std": 0.1,
            },
            "cancel_amplitude": {
                "initial": cr_param["cancel_amplitude"],
                "bounds": [0.0, 1.0],
                "std": 0.01,
            },
            "cancel_phase": {
                "initial": cr_param["cancel_phase"],
                "bounds": [-np.pi, np.pi],
                "std": 0.01,
            },
            "cancel_beta": {
                "initial": cr_param["cancel_beta"],
                "bounds": [-10.0, 10.0],
                "std": 0.1,
            },
            "rotary_amplitude": {
                "initial": cr_param["rotary_amplitude"],
                "bounds": [0.0, 1.0],
                "std": 0.1,
            },
        }

        for opt_param in opt_params:
            if opt_param not in defaults:
                raise ValueError(f"Invalid optimization parameter: {opt_param}")

        if isinstance(opt_params, list):
            opt_params_dict = {p: defaults[p] for p in opt_params}
        elif isinstance(opt_params, dict):
            opt_params_dict = opt_params
        else:
            raise ValueError("opt_params must be a list or dictionary.")
        opt_params = list(opt_params_dict.keys())

        def objective_func(params_vec):
            params = {k: v["initial"] for k, v in defaults.items()}
            for k, v in zip(opt_params, params_vec):
                params[k] = v

            cr_amplitude = params["cr_amplitude"]
            cr_phase = params["cr_phase"]
            cr_beta = params["cr_beta"]
            cancel_amplitude = params["cancel_amplitude"]
            cancel_phase = params["cancel_phase"]
            cancel_beta = params["cancel_beta"]
            rotary_amplitude = params["rotary_amplitude"]

            cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase) + (
                rotary_amplitude + 0j
            )

            ecr_0 = CrossResonance(
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                cr_amplitude=cr_amplitude,
                cr_duration=duration,
                cr_ramptime=ramptime,
                cr_phase=cr_phase,
                cr_beta=cr_beta,
                cancel_amplitude=np.abs(cancel_pulse),
                cancel_phase=np.angle(cancel_pulse),
                cancel_beta=cancel_beta,
                echo=True,
                pi_pulse=x180,
            )

            with PulseSchedule([control_qubit, cr_label, target_qubit]) as ecr_1:
                ecr_1.add(control_qubit, x180)
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

        initial_params = [opt_params_dict[k]["initial"] for k in opt_params]
        bounds0 = [opt_params_dict[k]["bounds"][0] for k in opt_params]
        bounds1 = [opt_params_dict[k]["bounds"][1] for k in opt_params]
        bounds = [bounds0, bounds1]
        stds = [opt_params_dict[k]["std"] for k in opt_params]

        es = cma.CMAEvolutionStrategy(
            initial_params,
            1.0,
            {
                "seed": seed,
                "ftarget": ftarget,
                "timeout": timeout,
                "bounds": bounds,
                "CMA_stds": stds,
            },
        )
        es.optimize(objective_func)

        print("Optimized parameters:")
        opt_result = {}
        for key, value in zip(opt_params, es.result.xbest):
            opt_result[key] = value
            print(f"  {key} : {value:.6f}")

        cr_param.update(opt_result)

        self.calib_note.update_cr_param(cr_label, cr_param)

        return {
            "cr_param": cr_param,
            "result": es.result,
        }
