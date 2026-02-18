"""Calibration service for pulse and readout parameters."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
from typing import Literal, no_type_check

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from qxpulse import (
    CrossResonance,
    Drag,
    FlatTop,
    MultiDerivativeCrossResonance,
    PulseArray,
    PulseSchedule,
    RampType,
    Waveform,
)

from qubex.analysis import FitResult, fitting, util, visualization as viz
from qubex.backend import Target
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_CR_RAMPTIME,
    DEFAULT_CR_TIME_RANGE,
    DEFAULT_INTERVAL,
    DEFAULT_SHOTS,
    DRAG_COEFF,
    DRAG_HPI_DURATION,
    DRAG_PI_DURATION,
    HPI_DURATION,
    HPI_RAMPTIME,
    PI_DURATION,
    PI_RAMPTIME,
)
from qubex.experiment.experiment_context import ExperimentContext
from qubex.experiment.models.experiment_result import AmplCalibData, ExperimentResult
from qubex.experiment.models.result import Result
from qubex.typing import TargetMap

from .measurement_service import MeasurementService
from .pulse_service import PulseService


class CalibrationService:
    """Service for calibration workflows."""

    def __init__(
        self,
        *,
        experiment_context: ExperimentContext,
        measurement_service: MeasurementService,
        pulse_service: PulseService,
    ):
        self._experiment_context = experiment_context
        self._measurement_service = measurement_service
        self._pulse_service = pulse_service

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

    def _resolve_ge_label(self, label: str) -> str:
        """Resolve GE label via target registry with legacy fallback."""
        target_registry = getattr(self.ctx.experiment_system, "target_registry", None)
        if target_registry is not None:
            try:
                return target_registry.resolve_ge_label(label)
            except ValueError:
                pass
        return Target.ge_label(label)

    def _resolve_ef_label(self, label: str) -> str:
        """Resolve EF label via target registry with legacy fallback."""
        target_registry = getattr(self.ctx.experiment_system, "target_registry", None)
        if target_registry is not None:
            try:
                return target_registry.resolve_ef_label(label)
            except ValueError:
                pass
        return Target.ef_label(label)

    def correct_rabi_params(
        self,
        targets: Collection[str] | str | None = None,
        *,
        reference_phases: dict[str, float] | None = None,
        save: bool | None = None,
    ) -> None:
        """Correct stored Rabi parameters using reference phases."""
        if save is None:
            save = True

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if reference_phases is None:
            phases = self.measurement_service.obtain_reference_points(targets=targets)[
                "phase"
            ]
        else:
            phases = reference_phases

        for target, phase in phases.items():
            try:
                rabi_param = self.pulse.rabi_params.get(target)
                if rabi_param is None:
                    print(f"Rabi parameters for {target} are not stored.")
                    continue
                else:
                    rabi_param.correct(new_reference_phase=phase)

                self.ctx.calib_note.update_rabi_param(
                    target,
                    {
                        "target": rabi_param.target,
                        "frequency": rabi_param.frequency,
                        "amplitude": rabi_param.amplitude,
                        "phase": rabi_param.phase,
                        "offset": rabi_param.offset,
                        "noise": rabi_param.noise,
                        "angle": rabi_param.angle,
                        "distance": rabi_param.distance,
                        "r2": rabi_param.r2,
                        "reference_phase": rabi_param.reference_phase,
                    },
                )
            except Exception as e:
                print(f"Failed to correct Rabi parameters for {target}: {e}")
                continue
        if save:
            self.ctx.save_calib_note()

    def correct_classifiers(
        self,
        targets: Collection[str] | str | None = None,
        *,
        reference_phases: dict[str, float] | None = None,
        save: bool | None = None,
    ) -> None:
        """Correct stored state classifiers using reference phases."""
        if save is None:
            save = True

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if reference_phases is None:
            phases = self.measurement_service.obtain_reference_points(targets=targets)[
                "phase"
            ]
        else:
            phases = reference_phases

        for target, phase in phases.items():
            classifier = self.ctx.classifiers.get(target)
            if classifier is not None:
                classifier.phase = phase
                if save:
                    classifier.save(
                        path=self.ctx.classifier_dir
                        / self.ctx.chip_id
                        / f"{target}.pkl"
                    )

        for target, phase in phases.items():
            try:
                state_param = self.ctx.calib_note.get_state_param(target)
                if state_param is not None:
                    reference_phase = state_param.get("reference_phase")
                    if reference_phase is None:
                        state_param["reference_phase"] = phase
                        continue
                    else:
                        centers = state_param["centers"]
                        phase_diff = phase - reference_phase
                        for state, points in centers.items():
                            iq = complex(points[0], points[1])
                            iq *= np.exp(1j * phase_diff)
                            centers[str(state)] = [iq.real, iq.imag]
                        state_param["reference_phase"] = phase
                    self.ctx.calib_note.update_state_param(
                        target,
                        state_param,
                    )
            except Exception as e:
                print(f"Failed to correct state parameters for {target}: {e}")
                continue
        if save:
            self.ctx.save_calib_note()

    def correct_cr_params(
        self,
        cr_labels: Collection[str] | str | None = None,
        *,
        shots: int | None = None,
        save: bool | None = None,
    ) -> None:
        """Correct stored CR phase parameters using tomography."""
        if shots is None:
            shots = 10000
        if save is None:
            save = True

        if cr_labels is None:
            cr_labels = self.ctx.cr_labels
        elif isinstance(cr_labels, str):
            cr_labels = [cr_labels]
        else:
            cr_labels = list(cr_labels)

        for label in cr_labels:
            try:
                control_qubit, target_qubit = self.ctx.cr_pair(label)
                if label not in self.ctx.calib_note.cr_params:
                    continue
                result = self.measurement_service.state_tomography(
                    self.pulse.zx90(control_qubit, target_qubit),
                    shots=shots,
                )
                x, y, _ = result[target_qubit]
                phase = np.arctan2(y, x)
                cr_param = self.ctx.calib_note.get_cr_param(label)
                if cr_param is not None:
                    cr_param["cr_phase"] = cr_param["cr_phase"] - phase - np.pi / 2
                    self.ctx.calib_note.update_cr_param(label, cr_param)
            except Exception as e:
                print(f"Failed to correct CR parameters for {label}: {e}")
                continue
        if save:
            self.ctx.save_calib_note()

    def correct_calibration(
        self,
        qubit_labels: Collection[str] | str | None = None,
        cr_labels: Collection[str] | str | None = None,
        *,
        save: bool | None = None,
    ) -> None:
        """Correct stored calibration data for qubits and CR pairs."""
        if save is None:
            save = False

        if qubit_labels is None:
            qubit_labels = self.ctx.qubit_labels
        elif isinstance(qubit_labels, str):
            qubit_labels = [qubit_labels]
        else:
            qubit_labels = list(qubit_labels)

        if cr_labels is None:
            cr_labels = self.ctx.cr_labels
        elif isinstance(cr_labels, str):
            cr_labels = [cr_labels]
        else:
            cr_labels = list(cr_labels)

        reference_phases = self.measurement_service.obtain_reference_points(
            qubit_labels
        )["phase"]

        self.correct_rabi_params(
            qubit_labels,
            reference_phases=reference_phases,
            save=save,
        )
        self.correct_classifiers(
            qubit_labels,
            reference_phases=reference_phases,
            save=save,
        )
        self.correct_cr_params(
            cr_labels,
            save=save,
        )

    def calibrate_default_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        pulse_type: Literal["pi", "hpi"],
        duration: float | None = None,
        ramptime: float | None = None,
        n_points: int | None = None,
        n_rotations: int | None = None,
        r2_threshold: float | None = None,
        update_params: bool | None = None,
        plot: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> ExperimentResult[AmplCalibData]:
        """Calibrate default pulse amplitude for targets."""
        if n_points is None:
            n_points = 20
        if n_rotations is None:
            n_rotations = 1
        if r2_threshold is None:
            r2_threshold = 0.5
        if update_params is None:
            update_params = True
        if plot is None:
            plot = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        rabi_params = self.pulse.rabi_params
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
            ampl = self.pulse.calc_control_amplitude(target, rabi_rate)

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

            sweep_data = self.measurement_service.sweep_parameter(
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
                if update_params:
                    if pulse_type == "hpi":
                        self.ctx.calib_note.update_hpi_param(
                            target,
                            {
                                "target": target,
                                "duration": pulse.duration,
                                "amplitude": fit_result["amplitude"],
                                "tau": pulse.tau,
                            },
                        )
                    elif pulse_type == "pi":
                        self.ctx.calib_note.update_pi_param(
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
            if target not in self.ctx.calib_note.rabi_params:
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
        n_points: int | None = None,
        n_rotations: int | None = None,
        r2_threshold: float | None = None,
        plot: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> ExperimentResult[AmplCalibData]:
        """Calibrate half-pi pulse amplitude for targets."""
        if n_points is None:
            n_points = 20
        if n_rotations is None:
            n_rotations = 1
        if r2_threshold is None:
            r2_threshold = 0.5
        if plot is None:
            plot = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        """Calibrate a ZX90 gate for a qubit pair."""

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
        n_points: int | None = None,
        n_rotations: int | None = None,
        r2_threshold: float | None = None,
        plot: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> ExperimentResult[AmplCalibData]:
        """Calibrate pi pulse amplitude for targets."""
        if n_points is None:
            n_points = 20
        if n_rotations is None:
            n_rotations = 1
        if r2_threshold is None:
            r2_threshold = 0.5
        if plot is None:
            plot = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL

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
        n_points: int | None = None,
        n_rotations: int | None = None,
        r2_threshold: float | None = None,
        plot: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> ExperimentResult[AmplCalibData]:
        """Calibrate EF pulse amplitude for targets."""
        if n_points is None:
            n_points = 20
        if n_rotations is None:
            n_rotations = 1
        if r2_threshold is None:
            r2_threshold = 0.5
        if plot is None:
            plot = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        rabi_params = self.pulse.rabi_params
        if rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        ef_labels = [
            self._resolve_ef_label(label)
            for label in targets
            if label in self.pulse.ef_rabi_params
        ]

        def calibrate(target: str) -> AmplCalibData:
            ge_label = self._resolve_ge_label(target)
            ef_label = self._resolve_ef_label(target)

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

            ampl = self.pulse.calc_control_amplitude(ef_label, rabi_rate)

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
                    ps.add(ge_label, self.pulse.get_hpi_pulse(ge_label).repeated(2))
                    ps.barrier()
                    ps.add(ef_label, pulse.scaled(x).repeated(repetitions))
                return ps

            sweep_data = self.measurement_service.sweep_parameter(
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
                    self.ctx.calib_note.update_hpi_param(
                        ef_label,
                        {
                            "target": ef_label,
                            "duration": pulse.duration,
                            "amplitude": fit_result["amplitude"],
                            "tau": pulse.tau,
                        },
                    )
                elif pulse_type == "pi":
                    self.ctx.calib_note.update_pi_param(
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
        n_points: int | None = None,
        n_rotations: int | None = None,
        r2_threshold: float | None = None,
        plot: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> ExperimentResult[AmplCalibData]:
        """Calibrate EF half-pi pulse amplitude for targets."""
        if n_points is None:
            n_points = 20
        if n_rotations is None:
            n_rotations = 1
        if r2_threshold is None:
            r2_threshold = 0.5
        if plot is None:
            plot = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL

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
        n_points: int | None = None,
        n_rotations: int | None = None,
        r2_threshold: float | None = None,
        plot: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> ExperimentResult[AmplCalibData]:
        """Calibrate EF pi pulse amplitude for targets."""
        if n_points is None:
            n_points = 20
        if n_rotations is None:
            n_rotations = 1
        if r2_threshold is None:
            r2_threshold = 0.5
        if plot is None:
            plot = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL

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
        spectator_state: str | None = None,
        pulse_type: Literal["pi", "hpi"],
        duration: float | None = None,
        n_points: int | None = None,
        n_rotations: int | None = None,
        r2_threshold: float | None = None,
        drag_coeff: float | None = None,
        use_stored_amplitude: bool | None = None,
        use_stored_beta: bool | None = None,
        plot: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> Result:
        """Calibrate DRAG amplitude for targets."""
        if n_points is None:
            n_points = 20
        if n_rotations is None:
            n_rotations = 4
        if r2_threshold is None:
            r2_threshold = 0.5
        if drag_coeff is None:
            drag_coeff = DRAG_COEFF
        if use_stored_amplitude is None:
            use_stored_amplitude = False
        if use_stored_beta is None:
            use_stored_beta = False
        if plot is None:
            plot = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        rabi_params = self.pulse.rabi_params
        self.pulse.validate_rabi_params(rabi_params)

        def calibrate(target: str) -> FitResult:
            # hpi
            if pulse_type == "hpi":
                hpi_param = self.ctx.calib_note.get_drag_hpi_param(target)
                if hpi_param is not None and use_stored_beta:
                    beta = hpi_param["beta"]
                else:
                    beta = -drag_coeff / self.ctx.qubits[target].alpha

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
                    ampl = self.pulse.calc_control_amplitude(target, rabi_rate)
            # pi
            elif pulse_type == "pi":
                pi_param = self.ctx.calib_note.get_drag_pi_param(target)
                if pi_param is not None and use_stored_beta:
                    beta = pi_param["beta"]
                else:
                    beta = -drag_coeff / self.ctx.qubits[target].alpha

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
                    ampl = self.pulse.calc_control_amplitude(target, rabi_rate)
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

            if spectator_state is not None:
                spectators = self.ctx.get_spectators(target)

            def sequence(x: float) -> PulseSchedule:
                with PulseSchedule() as ps:
                    if spectator_state is not None:
                        for spectator in spectators:
                            if spectator.label in self.ctx.qubit_labels:
                                ps.add(
                                    spectator.label,
                                    self.pulse.get_pulse_for_state(
                                        target=spectator.label,
                                        state=spectator_state,
                                    ),
                                )
                        ps.barrier()
                    ps.add(
                        target, pulse.scaled(x).repeated(n_per_rotation * n_rotations)
                    )
                return ps

            sweep_data = self.measurement_service.sweep_parameter(
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
                    self.ctx.calib_note.update_drag_hpi_param(
                        target,
                        {
                            "target": target,
                            "duration": pulse.duration,
                            "amplitude": fit_result["amplitude"],
                            "beta": beta,
                        },
                    )
                elif pulse_type == "pi":
                    self.ctx.calib_note.update_drag_pi_param(
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

        result: dict[str, FitResult] = {}
        for target in targets:
            result[target] = calibrate(target)

        return Result(data=result)

    def calibrate_drag_beta(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str | None = None,
        pulse_type: Literal["pi", "hpi"] | None = None,
        beta_range: ArrayLike | None = None,
        duration: float | None = None,
        n_turns: int | None = None,
        degree: int | None = None,
        plot: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> Result:
        """Calibrate DRAG beta for targets."""
        if pulse_type is None:
            pulse_type = "hpi"
        if n_turns is None:
            n_turns = 1
        if degree is None:
            degree = 3
        if plot is None:
            plot = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if beta_range is None:
            beta_range = np.linspace(-2.0, 2.0, 20)

        rabi_params = self.pulse.rabi_params
        self.pulse.validate_rabi_params(rabi_params)

        def calibrate(target: str) -> float:
            if pulse_type == "hpi":
                param = self.ctx.calib_note.get_drag_hpi_param(target)
            elif pulse_type == "pi":
                param = self.ctx.calib_note.get_drag_pi_param(target)
            if param is None:
                raise ValueError("DRAG parameters are not stored.")

            drag_duration = duration or param["duration"]
            drag_amplitude = param["amplitude"]
            drag_beta = param["beta"]

            sweep_range = np.array(beta_range) + drag_beta

            if spectator_state is not None:
                spectators = self.ctx.get_spectators(target)

            def sequence(beta: float) -> PulseSchedule:
                with PulseSchedule() as ps:
                    if spectator_state is not None:
                        for spectator in spectators:
                            if spectator.label in self.ctx.qubit_labels:
                                ps.add(
                                    spectator.label,
                                    self.pulse.get_pulse_for_state(
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
                        y90m = self.pulse.get_hpi_pulse(target).shifted(-np.pi / 2)
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
                        y90m = self.pulse.get_hpi_pulse(target).shifted(-np.pi / 2)
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

            sweep_data = self.measurement_service.sweep_parameter(
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
                self.ctx.calib_note.update_drag_hpi_param(
                    target,
                    {
                        "target": target,
                        "duration": drag_duration,
                        "amplitude": drag_amplitude,
                        "beta": beta,
                    },
                )
            elif pulse_type == "pi":
                self.ctx.calib_note.update_drag_pi_param(
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

        return Result(data=result)

    def calibrate_drag_hpi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str | None = None,
        n_points: int | None = None,
        n_rotations: int | None = None,
        n_turns: int | None = None,
        n_iterations: int | None = None,
        degree: int | None = None,
        r2_threshold: float | None = None,
        calibrate_beta: bool | None = None,
        beta_range: ArrayLike | None = None,
        duration: float | None = None,
        drag_coeff: float | None = None,
        plot: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> Result:
        """
        Calibrate DRAG half-pi pulses for targets.

        Parameters
        ----------
        targets
            Target qubits to calibrate.
        calibrate_beta
            Whether to tune DRAG beta.
        n_iterations
            Number of calibration iterations.
        """
        if n_points is None:
            n_points = 20
        if n_rotations is None:
            n_rotations = 4
        if n_turns is None:
            n_turns = 1
        if n_iterations is None:
            n_iterations = 2
        if degree is None:
            degree = 3
        if r2_threshold is None:
            r2_threshold = 0.5
        if calibrate_beta is None:
            calibrate_beta = True
        if drag_coeff is None:
            drag_coeff = DRAG_COEFF
        if plot is None:
            plot = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if beta_range is None:
            beta_range = np.linspace(-2.0, 2.0, 20)

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
                    target: -drag_coeff / self.ctx.qubits[target].alpha
                    for target in targets
                }

        return Result(
            data={
                "amplitude": amplitude,
                "beta": beta,
            }
        )

    def calibrate_drag_pi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str | None = None,
        n_points: int | None = None,
        n_rotations: int | None = None,
        n_turns: int | None = None,
        n_iterations: int | None = None,
        degree: int | None = None,
        r2_threshold: float | None = None,
        calibrate_beta: bool | None = None,
        beta_range: ArrayLike | None = None,
        duration: float | None = None,
        drag_coeff: float | None = None,
        plot: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
    ) -> Result:
        """
        Calibrate DRAG pi pulses for targets.

        Parameters
        ----------
        targets
            Target qubits to calibrate.
        calibrate_beta
            Whether to tune DRAG beta.
        n_iterations
            Number of calibration iterations.
        """
        if n_points is None:
            n_points = 20
        if n_rotations is None:
            n_rotations = 4
        if n_turns is None:
            n_turns = 1
        if n_iterations is None:
            n_iterations = 2
        if degree is None:
            degree = 3
        if r2_threshold is None:
            r2_threshold = 0.5
        if calibrate_beta is None:
            calibrate_beta = True
        if drag_coeff is None:
            drag_coeff = DRAG_COEFF
        if plot is None:
            plot = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL

        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        if beta_range is None:
            beta_range = np.linspace(-2.0, 2.0, 20)

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
                    target: -drag_coeff / self.ctx.qubits[target].alpha
                    for target in targets
                }

        return Result(
            data={
                "amplitude": amplitude,
                "beta": beta,
            }
        )

    def measure_cr_dynamics(
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
        echo: bool | None = None,
        control_state: str | None = None,
        x90: TargetMap[Waveform] | None = None,
        x180: TargetMap[Waveform] | None = None,
        ramp_type: Literal[
            "Gaussian",
            "RaisedCosine",
            "Sintegral",
            "Bump",
        ]
        | None = None,
        x180_margin: float | None = None,
        shots: int | None = None,
        interval: float | None = None,
        reset_awg_and_capunits: bool | None = None,
        plot: bool | None = None,
    ) -> Result:
        """Measure CR dynamics for a control/target pair."""
        if echo is None:
            echo = False
        if control_state is None:
            control_state = "0"
        if ramp_type is None:
            ramp_type = "RaisedCosine"
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if plot is None:
            plot = True

        cr_label = f"{control_qubit}-{target_qubit}"
        if time_range is None:
            time_range = np.array(DEFAULT_CR_TIME_RANGE, dtype=float)
        else:
            time_range = np.array(time_range, dtype=float)
        if ramptime is None:
            ramptime = DEFAULT_CR_RAMPTIME
        if cr_amplitude is None:
            cr_amplitude = 1.0
        if cr_phase is None:
            cr_phase = 0.0
        if cancel_amplitude is None:
            cancel_amplitude = 0.0
        if cancel_phase is None:
            cancel_phase = 0.0
        if x180_margin is None:
            x180_margin = 0.0
        if x90 is None:
            x90 = {
                control_qubit: self.pulse.x90(control_qubit),
                target_qubit: self.pulse.x90(target_qubit),
            }
        if x180 is None:
            x180 = {
                control_qubit: self.pulse.x180(control_qubit),
            }

        if reset_awg_and_capunits:
            self.ctx.reset_awg_and_capunits(qubits=[control_qubit, target_qubit])

        control_states = []
        target_states = []
        for T in time_range:
            result = self.measurement_service.state_tomography(
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
                    pi_margin=x180_margin,
                    ramp_type=ramp_type,
                ),
                x90=x90,
                initial_state={control_qubit: control_state},
                shots=shots,
                interval=interval,
                reset_awg_and_capunits=False,
                plot=False,
            )
            control_states.append(np.array(result[control_qubit]))
            target_states.append(np.array(result[target_qubit]))

        control_states = np.array(control_states)
        target_states = np.array(target_states)

        effective_drive_range = time_range + ramptime

        fit_result = fitting.fit_rotation(
            effective_drive_range,
            target_states,
            plot=False,
            title=f"Target qubit dynamics of {cr_label} : |{control_state}〉",
            xlabel="Drive time (ns)",
            ylabel=f"Target qubit : {target_qubit}",
        )

        if plot:
            viz.plot_bloch_vectors(
                effective_drive_range,
                control_states,
                title=f"Control qubit dynamics of {cr_label} : |{control_state}〉",
                xlabel="Drive time (ns)",
                ylabel=f"Control qubit : {control_qubit}",
            )
            viz.display_bloch_sphere(control_states)

            fit_result["fig"].show()
            fit_result["fig3d"].show()
            viz.display_bloch_sphere(target_states)

        return Result(
            data={
                "time_range": time_range,
                "effective_drive_range": effective_drive_range,
                "control_states": control_states,
                "target_states": target_states,
                "fit_result": fit_result,
                "cr_amplitude": cr_amplitude,
                "ramptime": ramptime,
            }
        )

    def _ramptime(self, control_qubit: str, target_qubit: str) -> float:
        f_ge_control = self.ctx.qubits[control_qubit].frequency
        f_ef_target = self.ctx.qubits[target_qubit].control_frequency_ef

        if f_ge_control < f_ef_target:
            return DEFAULT_CR_RAMPTIME
        else:
            return DEFAULT_CR_RAMPTIME * 2

    def _adiabatic_safe_factor(
        self,
        control_qubit: str,
        target_qubit: str,
    ) -> float:
        # f_ge_control = self.ctx.qubits[control_qubit].frequency
        # f_ef_target = self.ctx.qubits[target_qubit].ef_frequency

        # if f_ge_control < f_ef_target:
        #     return 0.75
        # else:
        #     return 0.5
        return 0.75

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
        x180_margin: float | None = None,
        shots: int | None = None,
        interval: float | None = None,
        reset_awg_and_capunits: bool | None = None,
        plot: bool | None = None,
    ) -> Result:
        """Run CR Hamiltonian tomography for a qubit pair."""
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if plot is None:
            plot = True

        cr_label = f"{control_qubit}-{target_qubit}"

        if cr_amplitude is None:
            cr_amplitude = 1.0

        if ramptime is None:
            ramptime = self._ramptime(control_qubit, target_qubit)

        if reset_awg_and_capunits:
            self.ctx.reset_awg_and_capunits(qubits=[control_qubit, target_qubit])

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
            ramp_type="RaisedCosine",
            x180_margin=x180_margin,
            shots=shots,
            interval=interval,
            reset_awg_and_capunits=False,
            plot=False,
        )

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
            x180_margin=x180_margin,
            shots=shots,
            interval=interval,
            reset_awg_and_capunits=False,
            plot=False,
        )

        Omega_0 = result_0["fit_result"]["Omega"]
        Omega_1 = result_1["fit_result"]["Omega"]
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
                strict=True,
            )
        )

        f_control = self.ctx.qubits[control_qubit].frequency
        f_target = self.ctx.qubits[target_qubit].frequency
        f_delta = f_control - f_target

        # xt (cross-talk) rotation
        xt_rotation = coeffs["IX"] + 1j * coeffs["IY"]
        xt_rotation_amplitude = np.abs(xt_rotation)
        xt_rotation_amplitude_hw = self.pulse.calc_control_amplitude(
            target=target_qubit,
            rabi_rate=xt_rotation_amplitude,
        )
        xt_rotation_phase = np.angle(xt_rotation)
        xt_rotation_phase_deg = np.angle(xt_rotation, deg=True)

        # cr (cross-resonance) rotation
        cr_rotation = coeffs["ZX"] + 1j * coeffs["ZY"]
        cr_rotation_amplitude = np.abs(cr_rotation)
        cr_rotation_amplitude_hw = self.pulse.calc_control_amplitude(
            target=target_qubit,
            rabi_rate=cr_rotation_amplitude,
        )
        cr_rotation_phase = np.angle(cr_rotation)
        cr_rotation_phase_deg = np.angle(cr_rotation, deg=True)
        zx90_duration = 1 / (4 * cr_rotation_amplitude)

        # ZX90 gate
        cr_rabi_rate = self.pulse.calc_rabi_rate(control_qubit, cr_amplitude)

        fig_c = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
        )
        fig_c_0 = viz.plot_bloch_vectors(
            result_0["effective_drive_range"],
            result_0["control_states"],
            return_figure=True,
        )
        fig_c_1 = viz.plot_bloch_vectors(
            result_1["effective_drive_range"],
            result_1["control_states"],
            return_figure=True,
        )
        for data in fig_c_0.data:  # type: ignore
            data: go.Scatter
            fig_c.add_trace(
                go.Scatter(
                    x=data.x,
                    y=data.y,
                    mode=data.mode,
                    line=data.line,
                    marker=data.marker,
                    name=data.name,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        for data in fig_c_1.data:  # type: ignore
            data: go.Scatter
            fig_c.add_trace(
                go.Scatter(
                    x=data.x,
                    y=data.y,
                    mode=data.mode,
                    line=data.line,
                    marker=data.marker,
                    name=data.name,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig_c.update_xaxes(
            title_text="Drive time (ns)",
            row=2,
            col=1,
        )
        fig_c.update_yaxes(
            title_text="Control : |0〉",
            range=[-1.1, 1.1],
            row=1,
            col=1,
        )
        fig_c.update_yaxes(
            title_text="Control : |1〉",
            range=[-1.1, 1.1],
            row=2,
            col=1,
        )
        fig_c.update_layout(
            title=dict(
                text=f"Control qubit dynamics : {cr_label}",
                subtitle=dict(
                    text=f"Δ = {f_delta * 1e3:.0f} MHz , Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
                ),
            ),
            height=400,
            width=600,
            showlegend=True,
            margin=dict(t=90),
        )

        fig_t = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
        )
        fig_t_0 = result_0["fit_result"]["fig"]
        fig_t_1 = result_1["fit_result"]["fig"]

        for data in fig_t_0.data:
            data: go.Scatter
            fig_t.add_trace(
                go.Scatter(
                    x=data.x,
                    y=data.y,
                    mode=data.mode,
                    line=data.line,
                    marker=data.marker,
                    name=data.name,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        for data in fig_t_1.data:
            data: go.Scatter
            fig_t.add_trace(
                go.Scatter(
                    x=data.x,
                    y=data.y,
                    mode=data.mode,
                    line=data.line,
                    marker=data.marker,
                    name=data.name,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig_t.update_xaxes(
            title_text="Drive time (ns)",
            row=2,
            col=1,
        )
        fig_t.update_yaxes(
            title_text="Control : |0〉",
            range=[-1.1, 1.1],
            row=1,
            col=1,
        )
        fig_t.update_yaxes(
            title_text="Control : |1〉",
            range=[-1.1, 1.1],
            row=2,
            col=1,
        )
        fig_t.update_layout(
            title=dict(
                text=f"Target qubit dynamics : {cr_label}",
                subtitle=dict(
                    text=f"Δ = {f_delta * 1e3:.0f} MHz , Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
                ),
            ),
            height=400,
            width=600,
            showlegend=True,
            margin=dict(t=90),
        )

        fig_t_3d = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Control : |0〉",
                "Control : |1〉",
            ],
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
            horizontal_spacing=0.01,
        )
        fig_t_3d_0 = result_0["fit_result"]["fig3d"]
        fig_t_3d_1 = result_1["fit_result"]["fig3d"]
        for data in fig_t_3d_0.data:
            fig_t_3d.add_trace(
                data,
                row=1,
                col=1,
            )
        for data in fig_t_3d_1.data:
            fig_t_3d.add_trace(
                data,
                row=1,
                col=2,
            )
        fig_t_3d.update_annotations(
            dict(
                font=dict(size=13),
                yshift=-20,
            )
        )
        fig_t_3d.update_layout(
            title=dict(
                text=f"Target qubit dynamics : {cr_label}",
                subtitle=dict(
                    text=f"Δ = {f_delta * 1e3:.0f} MHz , Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
                ),
            ),
            height=400,
            width=600,
            showlegend=False,
            margin=dict(t=90, b=10, l=10, r=10),
        )

        if plot:
            fig_c.show()
            fig_t.show()
            fig_t_3d.show()

            print("Qubit frequencies:")
            print(f"  ω_c ({control_qubit}) : {f_control * 1e3:.3f} MHz")
            print(f"  ω_t ({target_qubit}) : {f_target * 1e3:.3f} MHz")
            print(f"  Δ ({cr_label}) : {f_delta * 1e3:.3f} MHz")

            print("CR drive:")
            print(f"  Ω : {cr_rabi_rate * 1e3:.3f} MHz ({cr_amplitude:.4f})")

            print("Rotation rates:")
            for key, value in coeffs.items():
                print(f"  {key} : {value * 1e3:+.4f} MHz")

            print("XT (crosstalk) rotation:")
            print(
                f"  rate  : {xt_rotation_amplitude * 1e3:.4f} MHz ({xt_rotation_amplitude_hw:.6f})"
            )
            print(
                f"  phase : {xt_rotation_phase:.4f} rad ({xt_rotation_phase_deg:.1f} deg)"
            )

            print("CR (cross-resonance) rotation:")
            print(
                f"  rate  : {cr_rotation_amplitude * 1e3:.4f} MHz ({cr_rotation_amplitude_hw:.6f})"
            )
            print(
                f"  phase : {cr_rotation_phase:.4f} rad ({cr_rotation_phase_deg:.1f} deg)"
            )

            print(f"Estimated ZX90 gate length : {zx90_duration:.1f} ns")

        return Result(
            data={
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
                "result_0": result_0,
                "result_1": result_1,
                "fig_c": fig_c,
                "fig_t": fig_t,
            }
        )

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
        update_cr_phase: bool | None = None,
        update_cancel_pulse: bool | None = None,
        x90: TargetMap[Waveform] | None = None,
        x180_margin: float | None = None,
        shots: int | None = None,
        interval: float | None = None,
        reset_awg_and_capunits: bool | None = None,
        plot: bool | None = None,
    ) -> Result:
        """Update CR calibration parameters for a qubit pair."""
        if update_cr_phase is None:
            update_cr_phase = True
        if update_cancel_pulse is None:
            update_cancel_pulse = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if plot is None:
            plot = True
        if ramptime is None:
            ramptime = self._ramptime(control_qubit, target_qubit)
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
            x180_margin=x180_margin,
            shots=shots,
            interval=interval,
            reset_awg_and_capunits=reset_awg_and_capunits,
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

        cr_amplitude_diff = new_cr_amplitude - cr_amplitude
        cr_phase_diff = new_cr_phase - cr_phase
        cancel_amplitude_diff = new_cancel_amplitude - cancel_amplitude
        cancel_phase_diff = new_cancel_phase - cancel_phase

        if plot:
            print("Updated CR params:")
            print(
                f"  CR amplitude     : {cr_amplitude:+.4f} -> {new_cr_amplitude:+.4f} (diff: {cr_amplitude_diff:+.4f})"
            )
            print(
                f"  CR phase         : {cr_phase:+.4f} -> {new_cr_phase:+.4f} (diff: {cr_phase_diff:+.4f})"
            )
            print(
                f"  Cancel amplitude : {cancel_amplitude:+.4f} -> {new_cancel_amplitude:+.4f} (diff: {cancel_amplitude_diff:+.4f})"
            )
            print(
                f"  Cancel phase     : {cancel_phase:+.4f} -> {new_cancel_phase:+.4f} (diff: {cancel_phase_diff:+.4f})"
            )

        cr_label = f"{control_qubit}-{target_qubit}"

        zx_rotation_rate = result["coeffs"]["ZX"] / cr_amplitude

        self.ctx.calib_note.update_cr_param(
            cr_label,
            {
                "target": cr_label,
                "duration": 0.0,
                "ramptime": ramptime,
                "cr_amplitude": new_cr_amplitude,
                "cr_phase": new_cr_phase,
                "cr_beta": 0.0,
                "cancel_amplitude": new_cancel_amplitude,
                "cancel_phase": new_cancel_phase,
                "cancel_beta": 0.0,
                "rotary_amplitude": 0.0,
                "zx_rotation_rate": zx_rotation_rate,
            },
        )

        return Result(
            data={
                **result,
                "cr_param": self.ctx.calib_note.cr_params[cr_label],
            }
        )

    def obtain_cr_params(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        time_range: ArrayLike | None = None,
        ramptime: float | None = None,
        cr_amplitude: float | None = None,
        n_iterations: int | None = None,
        n_cycles: int | None = None,
        n_points_per_cycle: int | None = None,
        use_stored_params: bool | None = None,
        tolerance: float | None = None,
        adiabatic_safe_factor: float | None = None,
        max_amplitude: float | None = None,
        max_time_range: float | None = None,
        x90: TargetMap[Waveform] | None = None,
        x180_margin: float | None = None,
        shots: int | None = None,
        interval: float | None = None,
        reset_awg_and_capunits: bool | None = None,
        plot: bool | None = None,
    ) -> Result:
        """Obtain CR parameters for a qubit pair."""
        if n_iterations is None:
            n_iterations = 4
        if n_cycles is None:
            n_cycles = 2
        if n_points_per_cycle is None:
            n_points_per_cycle = 6
        if use_stored_params is None:
            use_stored_params = False
        if tolerance is None:
            tolerance = 0.005e-3
        if max_amplitude is None:
            max_amplitude = 1.0
        if max_time_range is None:
            max_time_range = 4096.0
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if plot is None:
            plot = True
        if ramptime is None:
            ramptime = self._ramptime(control_qubit, target_qubit)
        if adiabatic_safe_factor is None:
            adiabatic_safe_factor = self._adiabatic_safe_factor(
                control_qubit, target_qubit
            )
        sampling_period = self.ctx.measurement.sampling_period

        def _create_time_range(
            zx90_duration: float,
        ) -> NDArray:
            period = 4 * zx90_duration
            dt = (period / n_points_per_cycle) // sampling_period * sampling_period
            duration = min(period * n_cycles, max_time_range)
            return np.arange(0, duration + 1, dt)

        cr_label = f"{control_qubit}-{target_qubit}"

        f_control = self.ctx.qubits[control_qubit].frequency
        f_target = self.ctx.qubits[target_qubit].frequency
        f_delta = np.abs(f_target - f_control)
        max_cr_rabi = adiabatic_safe_factor * f_delta
        max_cr_amplitude = self.pulse.calc_control_amplitude(control_qubit, max_cr_rabi)
        max_cr_amplitude: float = np.clip(max_cr_amplitude, 0.0, max_amplitude)

        current_cr_param = self.ctx.calib_note.get_cr_param(cr_label)

        if use_stored_params and current_cr_param is not None:
            cr_amplitude = current_cr_param["cr_amplitude"]
            cr_phase = current_cr_param["cr_phase"]
            cancel_amplitude = current_cr_param["cancel_amplitude"]
            cancel_phase = current_cr_param["cancel_phase"]
            zx90_duration = 1 / (
                4 * cr_amplitude * current_cr_param["zx_rotation_rate"]
            )
            time_range = _create_time_range(zx90_duration)
        else:
            cr_amplitude = (
                cr_amplitude if cr_amplitude is not None else max_cr_amplitude
            )
            cr_phase = 0.0
            cancel_amplitude = 0.0
            cancel_phase = 0.0
            if time_range is None:
                time_range = DEFAULT_CR_TIME_RANGE
            time_range = np.array(time_range, dtype=float)

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
                cr_phase=float(params["cr_phase"]),
                cancel_amplitude=float(params["cancel_amplitude"]),
                cancel_phase=float(params["cancel_phase"]),
                x90=x90,
                x180_margin=x180_margin,
                shots=shots,
                interval=interval,
                reset_awg_and_capunits=reset_awg_and_capunits,
                plot=plot,
            )
            next_time_range = _create_time_range(result["zx90_duration"])
            params_history.append(
                {
                    "time_range": next_time_range,
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
                    print(f"  IX : {IX * 1e3:.4f} MHz")
                    print(f"  IY : {IY * 1e3:.4f} MHz")
                    break
                if abs(IX_diff) < tolerance and abs(IY_diff) < tolerance:
                    print("Convergence reached.")
                    print(f"  IX_diff : {IX_diff * 1e3:.4f} MHz")
                    print(f"  IY_diff : {IY_diff * 1e3:.4f} MHz")
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

        return Result(
            data={
                "params_history": params_history,
                "coeffs_history": hamiltonian_coeffs,
            }
        )

    def calibrate_zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        ramptime: float | None = None,
        duration: float | None = None,
        amplitude_range: ArrayLike | None = None,
        initial_state: str | None = None,
        degree: int | None = None,
        adiabatic_safe_factor: float | None = None,
        max_amplitude: float | None = None,
        rotary_multiple: float | None = None,
        use_drag: bool | None = None,
        duration_unit: float | None = None,
        duration_buffer: float | None = None,
        n_repetitions: int | None = None,
        x180: TargetMap[Waveform] | Waveform | None = None,
        x180_margin: float | None = None,
        use_zvalues: bool | None = None,
        store_params: bool | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
    ) -> Result:
        """Calibrate the ZX90 gate for a qubit pair."""
        if initial_state is None:
            initial_state = "0"
        if degree is None:
            degree = 3
        if max_amplitude is None:
            max_amplitude = 1.0
        if rotary_multiple is None:
            rotary_multiple = 9.0
        if use_drag is None:
            use_drag = True
        if duration_unit is None:
            duration_unit = 16.0
        if duration_buffer is None:
            duration_buffer = 1.05
        if n_repetitions is None:
            n_repetitions = 1
        if x180_margin is None:
            x180_margin = 0.0
        if use_zvalues is None:
            use_zvalues = False
        if store_params is None:
            store_params = True
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if plot is None:
            plot = True

        if ramptime is None:
            ramptime = self._ramptime(control_qubit, target_qubit)
        if adiabatic_safe_factor is None:
            adiabatic_safe_factor = self._adiabatic_safe_factor(
                control_qubit, target_qubit
            )
        if x180 is None:
            x180 = self.pulse.x180(control_qubit)
        elif not isinstance(x180, Waveform):
            x180 = x180[control_qubit]

        cr_label = f"{control_qubit}-{target_qubit}"
        cr_param = self.ctx.calib_note.get_cr_param(cr_label)

        if cr_param is None:
            raise ValueError("CR parameters are not stored.")

        cr_amplitude = cr_param["cr_amplitude"]
        cr_phase = cr_param["cr_phase"]
        cancel_amplitude = cr_param["cancel_amplitude"]
        cancel_phase = cr_param["cancel_phase"]
        zx_rotation_rate = cr_param["zx_rotation_rate"]
        zx_frequency = zx_rotation_rate * cr_amplitude
        rotary_amplitude = self.pulse.calc_control_amplitude(
            target=target_qubit,
            rabi_rate=zx_frequency * rotary_multiple,
        )
        cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase) + rotary_amplitude

        f_control = self.ctx.qubits[control_qubit].frequency
        f_target = self.ctx.qubits[target_qubit].frequency
        f_delta = np.abs(f_target - f_control)
        max_cr_rabi = adiabatic_safe_factor * f_delta
        max_cr_amplitude = self.pulse.calc_control_amplitude(control_qubit, max_cr_rabi)
        max_cr_amplitude: float = np.clip(max_cr_amplitude, 0.0, max_amplitude)

        if duration is None:
            if cr_param["duration"] == 0.0:
                duration = duration_buffer / (8 * zx_frequency) + ramptime
                if duration % duration_unit != 0:
                    duration = (duration // duration_unit + 1) * duration_unit
            else:
                duration = cr_param["duration"]

        if duration % duration_unit != 0:
            print(
                f"Warning: Duration {duration} ns is not a multiple of duration_unit {duration_unit} ns."
            )

        print(f"duration : {duration} ns")
        print(f"ramptime : {ramptime} ns")

        def ecr_sequence(
            amplitude: float,
            duration: float,
            n_repetitions: int,
        ) -> PulseSchedule:
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
                        self.pulse.get_pulse_for_state(control_qubit, initial_state),
                    )
                    ps.barrier()
                ps.call(ecr)
            return ps

        def calibrate(
            amplitude_range: ArrayLike,
            duration: float,
            n_repetitions: int,
        ) -> dict:
            amplitude_array = np.asarray(amplitude_range, dtype=float)
            min_amplitude = np.clip(amplitude_array[0], 0.0, max_cr_amplitude)
            max_amplitude = np.clip(amplitude_array[-1], 0.0, max_cr_amplitude)
            amplitude_range = np.linspace(
                min_amplitude,
                max_amplitude,
                len(amplitude_array),
            )
            sweep_result = self.measurement_service.sweep_parameter(
                lambda x: ecr_sequence(
                    amplitude=x,
                    duration=duration,
                    n_repetitions=n_repetitions,
                ),
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
            print(
                f"Estimating CR amplitude of {cr_label} (n_repetitions = {n_repetitions})"
            )
            rough_result = calibrate(
                amplitude_range=np.linspace(0.0, cr_amplitude * 2, 20),
                duration=duration,
                n_repetitions=n_repetitions,
            )
            rough_amplitude = rough_result["root"]
            if rough_amplitude is None:
                duration = (
                    duration * duration_buffer // duration_unit + 1
                ) * duration_unit
                print(f"Retrying with duration = {duration} ns")
                rough_result = calibrate(
                    amplitude_range=np.linspace(0.0, cr_amplitude * 2, 20),
                    duration=duration,
                    n_repetitions=n_repetitions,
                )
                rough_amplitude = rough_result["root"]
                if rough_amplitude is None:
                    raise ValueError(
                        "Could not find a root for the CR amplitude calibration."
                    )
            min_amplitude = float(rough_amplitude * 0.8)
            max_amplitude = float(rough_amplitude * 1.2)
            amplitude_range = np.linspace(min_amplitude, max_amplitude, 50)
        else:
            amplitude_range = np.asarray(amplitude_range)

        print(
            f"Calibrating CR amplitude of {cr_label} (n_repetitions = {n_repetitions})"
        )
        result_n1 = calibrate(
            amplitude_range=amplitude_range,
            duration=duration,
            n_repetitions=n_repetitions,
        )
        amplitude_range = np.asarray(result_n1["amplitude_range"])
        signal_n1 = result_n1["signal"]
        fit_result_n1 = result_n1["fit_result"]

        print(
            f"Calibrating CR amplitude of {cr_label} (n_repetitions = {n_repetitions + 2})"
        )
        result_n3 = calibrate(
            amplitude_range=amplitude_range,
            duration=duration,
            n_repetitions=n_repetitions + 2,
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
                f_control = self.ctx.qubits[control_qubit].frequency
                f_target = self.ctx.qubits[target_qubit].frequency
                Delta_ct = 2 * np.pi * (f_control - f_target)
                cr_beta = -1 / Delta_ct
                # cancel_beta = -1 / self.ctx.qubits[target_qubit].alpha
                cancel_beta = 0.0
            else:
                cr_beta = 0.0
                cancel_beta = 0.0
            self.ctx.calib_note.update_cr_param(
                cr_label,
                {
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
            )

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
        print(f"  Rotary amplitude : {calibrated_rotary_amplitude:.6f}")
        print()

        try:
            coherence_limit = self.calc_zx90_coherence_limit(
                control_qubit, target_qubit
            )
            print("ZX90 coherence limit:")
            print(f"  Gate time       : {coherence_limit['gate_time']:.0f} ns")
            print(f"  T1 (control)    : {coherence_limit['t1_control'] * 1e-3:.1f} μs")
            print(f"  T1 (target)     : {coherence_limit['t1_target'] * 1e-3:.1f} μs")
            print(f"  T2 (control)    : {coherence_limit['t2_control'] * 1e-3:.1f} μs")
            print(f"  T2 (target)     : {coherence_limit['t2_target'] * 1e-3:.1f} μs")
            print(f"  Coherence limit : {coherence_limit['fidelity'] * 100:.2f} %")
            print()
        except KeyError:
            coherence_limit = {}

        if plot:
            zx90 = self.pulse.zx90(control_qubit, target_qubit, x180=x180)
            zx90.plot(
                title=f"ZX90 sequence : {cr_label}",
                show_physical_pulse=True,
            )

        return Result(
            data={
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
                "coherence_limit": coherence_limit,
            }
        )

    def calc_zx90_coherence_limit(
        self,
        control_qubit: str,
        target_qubit: str,
    ) -> Result:
        """Estimate the coherence-limited ZX90 fidelity."""
        zx90 = self.pulse.zx90(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
        )
        gate_time = zx90.duration
        t1_dict = self.ctx.system_manager.config_loader.load_param_data("t1")
        t2_dict = self.ctx.system_manager.config_loader.load_param_data("t2_echo")
        t1 = (t1_dict[control_qubit], t1_dict[target_qubit])
        t2 = (t2_dict[control_qubit], t2_dict[target_qubit])
        return Result(
            data={
                "control_qubit": control_qubit,
                "target_qubit": target_qubit,
                "gate_time": gate_time,
                "t1_control": t1[0],
                "t1_target": t1[1],
                "t2_control": t2[0],
                "t2_target": t2[1],
                **util.calc_2q_gate_coherence_limit(
                    gate_time=gate_time,
                    t1=t1,
                    t2=t2,
                ),
            }
        )

    def calibrate_1q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
        coarse: bool | None = None,
    ) -> Result:
        """Run one-qubit calibration workflow."""
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if plot is None:
            plot = True
        if coarse is None:
            coarse = False
        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        data = {
            "obtain_rabi_params": {},
            "calibrate_hpi_pulse": {},
            "calibrate_drag_hpi_pulse": {},
            "calibrate_drag_pi_pulse": {},
            "build_classifier": {},
        }

        for target in targets:
            try:
                result = self.measurement_service.obtain_rabi_params(
                    target,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                )
                rabi_param = result.data[target].rabi_param
                if rabi_param.r2 < 0.9:
                    print(f"Warning: R² for {target} is low ({rabi_param.r2:.2f}).")
                elif rabi_param.r2 < 0.5:
                    print(f"Error: R² for {target} is very low ({rabi_param.r2:.2f}).")
                    continue
                data["obtain_rabi_params"][target] = result.data[target]

                result = self.calibrate_hpi_pulse(
                    target,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                )
                data["calibrate_hpi_pulse"][target] = result.data[target]

                if not coarse:
                    result = self.calibrate_drag_hpi_pulse(
                        target,
                        shots=shots,
                        interval=interval,
                        plot=plot,
                    )
                    data["calibrate_drag_hpi_pulse"][target] = result

                    result = self.calibrate_drag_pi_pulse(
                        target,
                        shots=shots,
                        interval=interval,
                        plot=plot,
                    )
                    data["calibrate_drag_pi_pulse"][target] = result

                result = self.measurement_service.build_classifier(
                    target,
                    shots=shots * 4,
                    interval=interval,
                    plot=plot,
                )
                data["build_classifier"][target] = result
                self.ctx.save_calib_note()

            except Exception as e:
                print(f"Error calibrating 1Q gates for {targets}: {e}")
                continue

        return Result(data=data)

    def calibrate_2q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        cr_calib_params: dict | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
    ) -> Result:
        """Run two-qubit calibration workflow."""
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if plot is None:
            plot = True
        if targets is None:
            targets = self.ctx.cr_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        pairs = [self.ctx.cr_pair(target) for target in targets]

        data = {
            "obtain_cr_params": {},
            "calibrate_zx90": {},
        }

        cr_calib_params = cr_calib_params or {}

        for control_qubit, target_qubit in pairs:
            cr_label = f"{control_qubit}-{target_qubit}"
            try:
                param = cr_calib_params.get(cr_label, {})
                result = self.obtain_cr_params(
                    control_qubit=control_qubit,
                    target_qubit=target_qubit,
                    time_range=param.get("time_range"),
                    ramptime=param.get("ramptime"),
                    cr_amplitude=param.get("cr_amplitude"),
                    n_iterations=param.get("n_iterations", 6),
                    n_cycles=param.get("n_cycles", 2),
                    use_stored_params=param.get("use_stored_params", False),
                    tolerance=param.get("tolerance", 10e-6),
                    adiabatic_safe_factor=param.get("adiabatic_safe_factor"),
                    max_amplitude=param.get("max_amplitude", 1.0),
                    max_time_range=param.get("max_time_range", 4096.0),
                    shots=shots,
                    interval=interval,
                    plot=plot,
                )
                data["obtain_cr_params"][cr_label] = result

                result = self.calibrate_zx90(
                    control_qubit=control_qubit,
                    target_qubit=target_qubit,
                    ramptime=param.get("ramptime"),
                    duration=param.get("duration"),
                    amplitude_range=param.get("amplitude_range"),
                    degree=param.get("degree", 3),
                    adiabatic_safe_factor=param.get("adiabatic_safe_factor"),
                    max_amplitude=param.get("max_amplitude", 1.0),
                    rotary_multiple=param.get("rotary_multiple", 9.0),
                    use_drag=param.get("use_drag", True),
                    duration_unit=param.get("duration_unit", 16.0),
                    shots=shots,
                    interval=interval,
                    plot=plot,
                )
                data["calibrate_zx90"][cr_label] = result

                self.ctx.save_calib_note()
            except Exception as e:
                print(f"Error calibrating {cr_label}: {e}")
                continue

        return Result(data=data)

    def calibrate_1q_ef(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool | None = None,
    ) -> Result:
        """Run one-qubit EF calibration workflow."""
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if plot is None:
            plot = True
        if targets is None:
            targets = self.ctx.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        return_data = {
            "obtain_ef_rabi_params": {},
            "calibrate_ef_hpi_pulse": {},
            "build_classifier": {},
        }

        for target in targets:
            try:
                result = self.measurement_service.obtain_ef_rabi_params(
                    target,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                )
                data = next(iter(result.data.values()))
                rabi_param = data.rabi_param
                if rabi_param.r2 < 0.9:
                    print(f"Warning: R² for {target} is low ({rabi_param.r2:.2f}).")
                elif rabi_param.r2 < 0.5:
                    print(f"Error: R² for {target} is very low ({rabi_param.r2:.2f}).")
                    continue
                return_data["obtain_ef_rabi_params"][target] = data

                result = self.calibrate_ef_hpi_pulse(
                    target,
                    shots=shots,
                    interval=interval,
                    plot=plot,
                )
                data = next(iter(result.data.values()))
                return_data["calibrate_ef_hpi_pulse"][target] = data

                self.ctx.save_calib_note()

                result = self.measurement_service.build_classifier(
                    target,
                    shots=shots * 4,
                    n_states=3,
                    interval=interval,
                    plot=plot,
                )
                data = next(iter(result.values()))
                return_data["build_classifier"][target] = data

                self.ctx.save_calib_note()

            except Exception as e:
                print(f"Error calibrating ef gates for {targets}: {e}")
                continue

        return Result(data=return_data)

    def measure_cr_crosstalk(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        spectator_qubits: list[str],
        time_range: ArrayLike | None = None,
        ramptime: float | None = None,
        cr_amplitude: float | None = None,
        cr_phase: float | None = None,
        cr_betas: dict[int, float] | None = None,
        cr_power: int | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        cancel_betas: dict[int, float] | None = None,
        cancel_power: int | None = None,
        echo: bool | None = None,
        control_state: str | None = None,
        x90: TargetMap[Waveform] | None = None,
        x180: TargetMap[Waveform] | None = None,
        ramp_type: RampType | None = None,
        x180_margin: float | None = None,
        shots: int | None = None,
        interval: float | None = None,
        reset_awg_and_capunits: bool | None = None,
        plot: bool | None = None,
    ) -> Result:
        """Measure CR crosstalk with spectator qubits."""
        if echo is None:
            echo = False
        if control_state is None:
            control_state = "0"
        if ramp_type is None:
            ramp_type = "RaisedCosine"
        if shots is None:
            shots = DEFAULT_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if plot is None:
            plot = True
        cr_label = f"{control_qubit}-{target_qubit}"
        if time_range is None:
            time_range = np.array(DEFAULT_CR_TIME_RANGE, dtype=float)
        else:
            time_range = np.array(time_range, dtype=float)
        if ramptime is None:
            ramptime = DEFAULT_CR_RAMPTIME
        if cr_amplitude is None:
            cr_amplitude = 1.0
        if cr_phase is None:
            cr_phase = 0.0
        if cr_betas is None:
            cr_betas = {}
        if cr_power is None:
            cr_power = 2
        if cancel_amplitude is None:
            cancel_amplitude = 0.0
        if cancel_phase is None:
            cancel_phase = 0.0
        if cancel_betas is None:
            cancel_betas = {}
        if cancel_power is None:
            cancel_power = 2
        if x180_margin is None:
            x180_margin = 0.0
        if x90 is None:
            x90 = {
                control_qubit: self.pulse.x90(control_qubit),
                target_qubit: self.pulse.x90(target_qubit),
            }
            x90.update(
                {spectator: self.pulse.x90(spectator) for spectator in spectator_qubits}
            )

        if x180 is None:
            x180 = {
                control_qubit: self.pulse.x180(control_qubit),
            }

        if reset_awg_and_capunits:
            self.ctx.reset_awg_and_capunits(
                qubits=[control_qubit, target_qubit, *spectator_qubits]
            )

        def cr_sequence(targets: list[str], T: float) -> PulseSchedule:
            cr = CrossResonance(
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
                pi_margin=x180_margin,
                ramp_type=ramp_type,
            )
            with PulseSchedule(targets) as ps:
                ps.call(cr)

            return ps

        def multi_derivative_cr_sequence(targets: list[str], T: float) -> PulseSchedule:
            cr = MultiDerivativeCrossResonance(
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                cr_amplitude=cr_amplitude,
                cr_duration=T + ramptime * 2,
                cr_ramptime=ramptime,
                cr_phase=cr_phase,
                cr_betas=cr_betas,
                cancel_amplitude=cancel_amplitude,
                cancel_phase=cancel_phase,
                cancel_betas=cancel_betas,
                echo=echo,
                pi_pulse=x180[control_qubit],
                pi_margin=x180_margin,
            )
            with PulseSchedule(targets) as ps:
                ps.call(cr)

            return ps

        if ramp_type == "MultiDerivativeSintegral":
            sequence_func = multi_derivative_cr_sequence
        else:
            sequence_func = cr_sequence

        control_states = []
        target_states = []
        spectators_states = defaultdict(list)

        with self.ctx.modified_frequencies(
            frequencies=dict.fromkeys(
                spectator_qubits, self.ctx.targets[cr_label].frequency
            )
        ):
            for T in time_range:
                result = self.measurement_service.state_tomography(
                    sequence=sequence_func(
                        targets=[control_qubit, target_qubit, *spectator_qubits], T=T
                    ),
                    x90=x90,
                    initial_state={control_qubit: control_state},
                    shots=shots,
                    interval=interval,
                    reset_awg_and_capunits=False,
                    plot=False,
                )
                control_states.append(np.array(result[control_qubit]))
                target_states.append(np.array(result[target_qubit]))

                for spectator in spectator_qubits:
                    spectators_states[spectator].append(np.array(result[spectator]))

        control_states = np.array(control_states)
        target_states = np.array(target_states)
        spectators_states = {
            spectator: np.array(states)
            for spectator, states in spectators_states.items()
        }

        effective_drive_range = time_range + ramptime

        fit_result = fitting.fit_rotation(
            effective_drive_range,
            target_states,
            plot=False,
            title=f"Target qubit dynamics of {cr_label} : |{control_state}〉",
            xlabel="Drive time (ns)",
            ylabel=f"Target qubit : {target_qubit}",
        )

        spectators_fit_result = {}
        for spectator, states in spectators_states.items():
            fit_spectator = fitting.fit_rotation(
                effective_drive_range,
                states,
                plot=False,
                title=f"Spectator qubit dynamics of {cr_label} : |{control_state}〉",
                xlabel="Drive time (ns)",
                ylabel=f"Spectator qubit : {spectator}",
            )
            spectators_fit_result[spectator] = fit_spectator

        if plot:
            viz.plot_bloch_vectors(
                effective_drive_range,
                control_states,
                title=f"Control qubit dynamics of {cr_label} : |{control_state}〉",
                xlabel="Drive time (ns)",
                ylabel=f"Control qubit : {control_qubit}",
            )
            viz.display_bloch_sphere(control_states)

            fit_result["fig"].show()
            fit_result["fig3d"].show()
            viz.display_bloch_sphere(target_states)

            for spectator, fit_spectator in spectators_fit_result.items():
                fit_spectator["fig"].show()
                fit_spectator["fig3d"].show()
                viz.display_bloch_sphere(spectators_states[spectator])

        return Result(
            data={
                "time_range": time_range,
                "effective_drive_range": effective_drive_range,
                "control_states": control_states,
                "target_states": target_states,
                "spectators_states": spectators_states,
                "fit_result": fit_result,
                "spectators_fit_result": spectators_fit_result,
                "cr_amplitude": cr_amplitude,
                "ramptime": ramptime,
            }
        )

    @no_type_check
    def cr_crosstalk_hamiltonian_tomography(
        self,
        *,
        control_qubit: str,
        target_qubit: str,
        spectator_qubits: list[str] | None = None,
        time_range: ArrayLike | None = None,
        ramptime: float | None = None,
        cr_amplitude: float | None = None,
        cr_phase: float | None = None,
        cr_betas: dict[int, float] | None = None,
        cr_power: int | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        cancel_betas: dict[int, float] | None = None,
        cancel_power: int | None = None,
        ramp_type: RampType | None = None,
        x90: TargetMap[Waveform] | None = None,
        x180_margin: float | None = None,
        shots: int | None = None,
        interval: float | None = None,
        reset_awg_and_capunits: bool | None = None,
        plot: bool | None = None,
    ) -> Result:
        """Perform CR crosstalk Hamiltonian tomography."""
        if ramp_type is None:
            ramp_type = "RaisedCosine"
        if shots is None:
            shots = CALIBRATION_SHOTS
        if interval is None:
            interval = DEFAULT_INTERVAL
        if reset_awg_and_capunits is None:
            reset_awg_and_capunits = True
        if plot is None:
            plot = True
        cr_label = f"{control_qubit}-{target_qubit}"

        if spectator_qubits is None:
            spectator_qubits = []
            for spectator in self.ctx.get_spectators(control_qubit):
                if (
                    spectator.label in self.ctx.qubit_labels
                    and spectator.label != target_qubit
                ):
                    spectator_qubits.append(spectator.label)

        if cr_amplitude is None:
            cr_amplitude = 1.0

        if ramptime is None:
            ramptime = self._ramptime(control_qubit, target_qubit)

        if reset_awg_and_capunits:
            self.ctx.reset_awg_and_capunits(
                qubits=[control_qubit, target_qubit, *spectator_qubits]
            )

        result_0 = self.measure_cr_crosstalk(
            time_range=time_range,
            ramptime=ramptime,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            spectator_qubits=spectator_qubits,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cr_betas=cr_betas,
            cr_power=cr_power,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            cancel_betas=cancel_betas,
            cancel_power=cancel_power,
            echo=False,
            control_state="0",
            x90=x90,
            ramp_type=ramp_type,
            x180_margin=x180_margin,
            shots=shots,
            interval=interval,
            reset_awg_and_capunits=False,
            plot=False,
        )

        result_1 = self.measure_cr_crosstalk(
            time_range=time_range,
            ramptime=ramptime,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            spectator_qubits=spectator_qubits,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cr_betas=cr_betas,
            cr_power=cr_power,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            cancel_betas=cancel_betas,
            cancel_power=cancel_power,
            echo=False,
            control_state="1",
            x90=x90,
            ramp_type=ramp_type,
            x180_margin=x180_margin,
            shots=shots,
            interval=interval,
            reset_awg_and_capunits=False,
            plot=False,
        )

        Omega_0 = result_0["fit_result"]["Omega"]
        Omega_1 = result_1["fit_result"]["Omega"]
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
                strict=True,
            )
        )

        f_control = self.ctx.qubits[control_qubit].frequency
        f_target = self.ctx.qubits[target_qubit].frequency
        f_delta = f_control - f_target

        # xt (cross-talk) rotation
        xt_rotation = coeffs["IX"] + 1j * coeffs["IY"]
        xt_rotation_amplitude = np.abs(xt_rotation)
        xt_rotation_amplitude_hw = self.pulse.calc_control_amplitude(
            target=target_qubit,
            rabi_rate=xt_rotation_amplitude,
        )
        xt_rotation_phase = np.angle(xt_rotation)
        xt_rotation_phase_deg = np.angle(xt_rotation, deg=True)

        # cr (cross-resonance) rotation
        cr_rotation = coeffs["ZX"] + 1j * coeffs["ZY"]
        cr_rotation_amplitude = np.abs(cr_rotation)
        cr_rotation_amplitude_hw = self.pulse.calc_control_amplitude(
            target=target_qubit,
            rabi_rate=cr_rotation_amplitude,
        )
        cr_rotation_phase = np.angle(cr_rotation)
        cr_rotation_phase_deg = np.angle(cr_rotation, deg=True)
        zx90_duration = 1 / (4 * cr_rotation_amplitude)

        # ZX90 gate
        cr_rabi_rate = self.pulse.calc_rabi_rate(control_qubit, cr_amplitude)

        fig_c = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
        )
        fig_c_0 = viz.plot_bloch_vectors(
            result_0["effective_drive_range"],
            result_0["control_states"],
            return_figure=True,
        )
        fig_c_1 = viz.plot_bloch_vectors(
            result_1["effective_drive_range"],
            result_1["control_states"],
            return_figure=True,
        )
        for data in fig_c_0.data:
            data: go.Scatter
            """Run CR crosstalk Hamiltonian tomography."""
            fig_c.add_trace(
                go.Scatter(
                    x=data.x,
                    y=data.y,
                    mode=data.mode,
                    line=data.line,
                    marker=data.marker,
                    name=data.name,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        for data in fig_c_1.data:
            data: go.Scatter
            fig_c.add_trace(
                go.Scatter(
                    x=data.x,
                    y=data.y,
                    mode=data.mode,
                    line=data.line,
                    marker=data.marker,
                    name=data.name,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig_c.update_xaxes(
            title_text="Drive time (ns)",
            row=2,
            col=1,
        )
        fig_c.update_yaxes(
            title_text="Control : |0〉",
            range=[-1.1, 1.1],
            row=1,
            col=1,
        )
        fig_c.update_yaxes(
            title_text="Control : |1〉",
            range=[-1.1, 1.1],
            row=2,
            col=1,
        )
        fig_c.update_layout(
            title=dict(
                text=f"Control qubit dynamics : {cr_label}",
                subtitle=dict(
                    text=f"Δ = {f_delta * 1e3:.0f} MHz , Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
                ),
            ),
            height=400,
            width=600,
            showlegend=True,
            margin=dict(t=90),
        )

        fig_t = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
        )
        fig_t_0 = result_0["fit_result"]["fig"]
        fig_t_1 = result_1["fit_result"]["fig"]

        for data in fig_t_0.data:
            data: go.Scatter
            fig_t.add_trace(
                go.Scatter(
                    x=data.x,
                    y=data.y,
                    mode=data.mode,
                    line=data.line,
                    marker=data.marker,
                    name=data.name,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        for data in fig_t_1.data:
            data: go.Scatter
            fig_t.add_trace(
                go.Scatter(
                    x=data.x,
                    y=data.y,
                    mode=data.mode,
                    line=data.line,
                    marker=data.marker,
                    name=data.name,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig_t.update_xaxes(
            title_text="Drive time (ns)",
            row=2,
            col=1,
        )
        fig_t.update_yaxes(
            title_text="Control : |0〉",
            range=[-1.1, 1.1],
            row=1,
            col=1,
        )
        fig_t.update_yaxes(
            title_text="Control : |1〉",
            range=[-1.1, 1.1],
            row=2,
            col=1,
        )
        fig_t.update_layout(
            title=dict(
                text=f"Target qubit dynamics : {cr_label}",
                subtitle=dict(
                    text=f"Δ = {f_delta * 1e3:.0f} MHz , Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
                ),
            ),
            height=400,
            width=600,
            showlegend=True,
            margin=dict(t=90),
        )

        fig_t_3d = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Control : |0〉",
                "Control : |1〉",
            ],
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
            horizontal_spacing=0.01,
        )
        fig_t_3d_0 = result_0["fit_result"]["fig3d"]
        fig_t_3d_1 = result_1["fit_result"]["fig3d"]
        for data in fig_t_3d_0.data:
            fig_t_3d.add_trace(
                data,
                row=1,
                col=1,
            )
        for data in fig_t_3d_1.data:
            fig_t_3d.add_trace(
                data,
                row=1,
                col=2,
            )
        fig_t_3d.update_annotations(
            dict(
                font=dict(size=13),
                yshift=-20,
            )
        )
        fig_t_3d.update_layout(
            title=dict(
                text=f"Target qubit dynamics : {cr_label}",
                subtitle=dict(
                    text=f"Δ = {f_delta * 1e3:.0f} MHz , Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
                ),
            ),
            height=400,
            width=600,
            showlegend=False,
            margin=dict(t=90, b=10, l=10, r=10),
        )

        spectators_fit_results_0 = result_0["spectators_fit_result"]
        spectators_fit_results_1 = result_1["spectators_fit_result"]
        figs_s = {}
        figs_s_3d = {}
        for label in spectators_fit_results_0:
            f_delta = (
                self.ctx.qubits[control_qubit].frequency
                - self.ctx.qubits[target_qubit].frequency
            )
            f_delta_st = (
                self.ctx.qubits[label].frequency
                - self.ctx.qubits[target_qubit].frequency
            )

            fig_s_0: go.Figure = spectators_fit_results_0[label]["fig"]
            fig_s_1: go.Figure = spectators_fit_results_1[label]["fig"]

            fig_s = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
            )

            for data in fig_s_0.data:
                data: go.Scatter
                fig_s.add_trace(
                    go.Scatter(
                        x=data.x,
                        y=data.y,
                        mode=data.mode,
                        line=data.line,
                        marker=data.marker,
                        name=data.name,
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

            for data in fig_s_1.data:
                data: go.Scatter
                fig_s.add_trace(
                    go.Scatter(
                        x=data.x,
                        y=data.y,
                        mode=data.mode,
                        line=data.line,
                        marker=data.marker,
                        name=data.name,
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

            fig_s.update_xaxes(
                title_text="Drive time (ns)",
                row=2,
                col=1,
            )

            fig_s.update_yaxes(
                title_text="control : |0〉",
                range=[-1.1, 1.1],
                row=1,
                col=1,
            )
            fig_s.update_yaxes(
                title_text="control : |1〉",
                range=[-1.1, 1.1],
                row=2,
                col=1,
            )
            fig_s.update_layout(
                title=dict(
                    text=f"Spectator qubit dynamics : {label} in {cr_label}",
                    subtitle=dict(
                        text=f"Δ = {f_delta * 1e3:.0f} MHz ,Δ_st = {f_delta_st * 1e3:.0f} MHz  Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
                    ),
                ),
                height=400,
                width=800,
                showlegend=True,
                margin=dict(t=90),
            )

            fig_s_3d_0 = spectators_fit_results_0[label]["fig3d"]
            fig_s_3d_1 = spectators_fit_results_1[label]["fig3d"]

            fig_s_3d = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=[
                    "Control |0〉",
                    "Control |1〉",
                ],
                specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
                horizontal_spacing=0.01,
            )

            @no_type_check
            def add_trace_with_color(
                fig: go.Figure,
                incoming_fig: go.Figure,
                row: int,
                col: int,
                name: str | None = None,
            ) -> None:
                for ddx, data in enumerate(incoming_fig.data):
                    if ddx % 2 == 0:
                        name_suffix = "data"
                    else:
                        name_suffix = "fit"
                    fig.add_trace(
                        data,
                        row=row,
                        col=col,
                    )
                    if not isinstance(data, go.Surface):
                        if data.mode == "markers":
                            fig.data[-1].marker.color = viz.COLORS[ddx]
                        if data.mode == "lines":
                            fig.data[-1].line.color = viz.COLORS[ddx]
                        if name is not None:
                            fig.data[-1].name = f"{name_suffix} ({name})"
                            fig.data[-1].showlegend = True

            add_trace_with_color(
                fig=fig_s_3d,
                incoming_fig=fig_s_3d_0,
                row=1,
                col=1,
            )

            add_trace_with_color(
                fig=fig_s_3d,
                incoming_fig=fig_s_3d_1,
                row=1,
                col=2,
                name="raw",
            )

            fig_s_3d.update_layout(
                title=dict(
                    text=f"Spectator qubit dynamics : {label} of {cr_label}",
                    subtitle=dict(
                        text=f"Δ = {f_delta_st * 1e3:.0f} MHz , Ω = {cr_rabi_rate * 1e3:.1f} MHz , τ = {ramptime:.0f} ns"
                    ),
                ),
                height=400,
                width=600,
                showlegend=False,
                margin=dict(t=90, b=10, l=10, r=10),
            )

            figs_s[label] = fig_s
            figs_s_3d[label] = fig_s_3d

        fig_c.show()
        fig_t.show()
        fig_t_3d.show()
        for fig_s_ in figs_s.values():
            fig_s_.show()
        for fig_s_3d in figs_s_3d.values():
            fig_s_3d.show()

        print("Qubit frequencies:")
        print(f"  ω_c ({control_qubit}) : {f_control * 1e3:.3f} MHz")
        print(f"  ω_t ({target_qubit}) : {f_target * 1e3:.3f} MHz")
        print(f"  Δ ({cr_label}) : {f_delta * 1e3:.3f} MHz")

        print("CR drive:")
        print(f"  Ω : {cr_rabi_rate * 1e3:.3f} MHz ({cr_amplitude:.4f})")

        print("Rotation rates:")
        for key, value in coeffs.items():
            print(f"  {key} : {value * 1e3:+.4f} MHz")

        print("XT (crosstalk) rotation:")
        print(
            f"  rate  : {xt_rotation_amplitude * 1e3:.4f} MHz ({xt_rotation_amplitude_hw:.6f})"
        )
        print(
            f"  phase : {xt_rotation_phase:.4f} rad ({xt_rotation_phase_deg:.1f} deg)"
        )

        print("CR (cross-resonance) rotation:")
        print(
            f"  rate  : {cr_rotation_amplitude * 1e3:.4f} MHz ({cr_rotation_amplitude_hw:.6f})"
        )
        print(
            f"  phase : {cr_rotation_phase:.4f} rad ({cr_rotation_phase_deg:.1f} deg)"
        )

        print(f"Estimated ZX90 gate length : {zx90_duration:.1f} ns")

        coeffs_ = {}
        for spectator in spectator_qubits:
            print(f" Spectator qubit: {spectator}")

            f_s = self.ctx.qubits[spectator].frequency

            print("")
            print(f"  ω_s ({spectator}) : {f_s * 1e3:.3f} MHz")
            print(f"  ω_t ({target_qubit}) : {f_target * 1e3:.3f} MHz")
            print(
                f"  Δ_st ({spectator}-{target_qubit}) : {(f_s - f_target) * 1e3:.3f} MHz"
            )

            spectator_Omega_0 = result_0["spectators_fit_result"][spectator]["Omega"]
            spectator_Omega_1 = result_1["spectators_fit_result"][spectator]["Omega"]

            spectator_Omega = np.concatenate(
                [
                    0.5 * (spectator_Omega_0 + spectator_Omega_1),
                    0.5 * (spectator_Omega_0 - spectator_Omega_1),
                ]
            )
            spectator_coeffs = dict(
                zip(
                    ["IX", "IY", "IZ", "ZX", "ZY", "ZZ"],
                    spectator_Omega / (2 * np.pi),  # GHz
                    strict=True,
                )
            )

            coeffs_[spectator] = spectator_coeffs

            print("")
            for key, value in spectator_coeffs.items():
                print(f"  {key} : {value * 1e3:+.4f} MHz")
            print("")

            print(
                f"  |IX + 1j * IY| : {np.abs(spectator_coeffs['IX'] + 1j * spectator_coeffs['IY']) * 1e3:.4f} MHz"
            )
            print(
                f"  |ZX + 1j * ZY| : {np.abs(spectator_coeffs['ZX'] + 1j * spectator_coeffs['ZY']) * 1e3:.4f} MHz"
            )
            print(
                f"  √ (|IX + 1j * IY|² + IZ²) : {np.sqrt(spectator_coeffs['IX'] ** 2 + spectator_coeffs['IY'] ** 2 + spectator_coeffs['IZ'] ** 2) * 1e3:.4f} MHz"
            )
            print(
                f"  √ (|ZX + 1j * ZY|² + ZZ²) : {np.sqrt(spectator_coeffs['ZX'] ** 2 + spectator_coeffs['ZY'] ** 2 + spectator_coeffs['ZZ'] ** 2) * 1e3:.4f} MHz"
            )

            print(
                f" r2 (control |0〉): {spectators_fit_results_0[spectator]['r2']:.4f}"
            )
            print(
                f" r2 (control |1〉): {spectators_fit_results_1[spectator]['r2']:.4f}"
            )
            print("")

        coeffs_[target_qubit] = coeffs
        return Result(
            data={
                "Omega": Omega,
                "coeffs": coeffs_,
                "cr_rotation_amplitude": cr_rotation_amplitude,
                "cr_rotation_amplitude_hw": cr_rotation_amplitude_hw,
                "cr_rotation_phase": cr_rotation_phase,
                "xt_rotation_amplitude": xt_rotation_amplitude,
                "xt_rotation_amplitude_hw": xt_rotation_amplitude_hw,
                "xt_rotation_phase": xt_rotation_phase,
                "cr_drive_amplitude": cr_rabi_rate,
                "cr_drive_amplitude_hw": cr_amplitude,
                "zx90_duration": zx90_duration,
                "result_0": result_0,
                "result_1": result_1,
                "fig_c": fig_c,
                "fig_t": fig_t,
                "figs_s": figs_s,
            }
        )
