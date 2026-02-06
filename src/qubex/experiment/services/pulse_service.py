"""Pulse construction helpers for experiment workflows."""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Literal

import numpy as np
from qxpulse import (
    Blank,
    CrossResonance,
    Drag,
    FlatTop,
    PulseArray,
    PulseSchedule,
    RampType,
    VirtualZ,
    Waveform,
)

from qubex.backend import Target, TargetType
from qubex.experiment.experiment_constants import (
    DEFAULT_RABI_FREQUENCY,
    HPI_DURATION,
    HPI_RAMPTIME,
)
from qubex.experiment.experiment_context import ExperimentContext
from qubex.experiment.experiment_exceptions import CalibrationMissingError
from qubex.experiment.models.rabi_param import RabiParam
from qubex.typing import TargetMap

logger = logging.getLogger(__name__)


class PulseService:
    """Service for constructing calibrated pulses."""

    def __init__(
        self,
        experiment_context: ExperimentContext,
    ):
        self._experiment_context = experiment_context

    @property
    def ctx(self) -> ExperimentContext:
        """Return the experiment context."""
        return self._experiment_context

    @property
    def readout_duration(self) -> float:
        """Return the default readout duration."""
        return self.ctx.readout_duration

    @property
    def readout_pre_margin(self) -> float:
        """Return the default pre-readout margin."""
        return self.ctx.readout_pre_margin

    @property
    def readout_post_margin(self) -> float:
        """Return the default post-readout margin."""
        return self.ctx.readout_post_margin

    @property
    def drag_hpi_duration(self) -> float:
        """Return the DRAG half-pi duration."""
        return self.ctx.drag_hpi_duration

    @property
    def drag_pi_duration(self) -> float:
        """Return the DRAG pi duration."""
        return self.ctx.drag_pi_duration

    @property
    def rabi_params(self) -> dict[str, RabiParam]:
        """Return stored Rabi parameters for available targets."""
        params = {}
        for target in self.ctx.ge_targets | self.ctx.ef_targets:
            param = self.ctx.get_rabi_param(target)
            if param is not None:
                params[target] = param
        return params

    @property
    def ge_rabi_params(self) -> dict[str, RabiParam]:
        """Return Rabi parameters for ge targets."""
        return {
            target: param
            for target, param in self.rabi_params.items()
            if self.ctx.targets[target].is_ge
        }

    @property
    def ef_rabi_params(self) -> dict[str, RabiParam]:
        """Return Rabi parameters for ef targets."""
        return {
            Target.ge_label(target): param
            for target, param in self.rabi_params.items()
            if self.ctx.targets[target].is_ef
        }

    def validate_rabi_params(
        self,
        targets: Collection[str] | None = None,
    ) -> None:
        """
        Validate that Rabi parameters exist for targets.

        Parameters
        ----------
        targets : Collection[str] | None, optional
            Targets to validate. If None, validates all known targets.
        """
        rabi_params = self.rabi_params
        if len(rabi_params) == 0:
            raise ValueError("Rabi parameters are not stored.")
        if targets is not None:
            for target in targets:
                if target not in rabi_params:
                    raise ValueError(f"Rabi parameters for {target} are not stored.")
        if targets is not None:
            for target in targets:
                if target not in rabi_params:
                    raise ValueError(f"Rabi parameters for {target} are not stored.")

    def calc_control_amplitude(
        self,
        target: str,
        rabi_rate: float,
        *,
        rabi_amplitude_ratio: float | None = None,
    ) -> float:
        """
        Convert a desired Rabi rate into a control amplitude.

        Parameters
        ----------
        target : str
            Target label.
        rabi_rate : float
            Desired Rabi rate.
        rabi_amplitude_ratio : float | None, optional
            Optional ratio override.

        Returns
        -------
        float
            Control amplitude.
        """
        qubit = Target.qubit_label(target)
        if rabi_amplitude_ratio is None:
            rabi_param = self.rabi_params.get(target)
            if self.ctx.targets[target].type == TargetType.CTRL_EF:
                default_amplitude = self.ctx.params.get_ef_control_amplitude(qubit)
            else:
                default_amplitude = self.ctx.params.get_control_amplitude(qubit)

            if rabi_param is None:
                raise ValueError(f"Rabi parameters for {target} are not stored.")
            if default_amplitude is None:
                raise ValueError(f"Control amplitude for {qubit} is not defined.")

            rabi_amplitude_ratio = rabi_param.frequency / default_amplitude

        if rabi_amplitude_ratio is None:
            raise ValueError("rabi_amplitude_ratio must be provided or derivable.")
        return rabi_rate / rabi_amplitude_ratio

    def calc_control_amplitudes(
        self,
        rabi_rate: float | None = None,
        *,
        current_amplitudes: dict[str, float] | None = None,
        current_rabi_params: dict[str, RabiParam] | None = None,
        print_result: bool | None = None,
    ) -> dict[str, float]:
        """
        Calculate control amplitudes for available targets.

        Parameters
        ----------
        rabi_rate : float | None, optional
            Rabi rate to use for all targets.
        current_amplitudes : dict[str, float] | None, optional
            Override amplitudes for targets.
        current_rabi_params : dict[str, RabiParam] | None, optional
            Override Rabi parameters for targets.
        print_result : bool | None, optional
            Whether to log results.

        Returns
        -------
        dict[str, float]
            Calculated control amplitudes.
        """
        if rabi_rate is None:
            rabi_rate = DEFAULT_RABI_FREQUENCY
        if current_amplitudes is None:
            current_amplitudes = {}

        if current_rabi_params is None:
            current_rabi_params = self.rabi_params

        if current_rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        if len(current_amplitudes) == 0:
            for target in current_rabi_params:
                qubit = Target.qubit_label(target)
                current_amplitudes[target] = self.ctx.params.get_control_amplitude(
                    qubit
                )

        if print_result is None:
            print_result = True

        amplitudes = {
            target: current_amplitudes[target]
            * rabi_rate
            / current_rabi_params[target].frequency
            for target in current_rabi_params
        }

        if print_result:
            logger.info(f"Control amplitude for rabi rate {rabi_rate * 1e3:.3f} MHz\n")
            for target, amplitude in amplitudes.items():
                logger.info(f"{target}: {amplitude:.6f}")

        return amplitudes

    def calc_rabi_rate(
        self,
        target: str,
        control_amplitude: float,
    ) -> float:
        """Return the Rabi rate for a control amplitude."""
        # TODO: Support ef targets
        default_amplitude = self.ctx.params.control_amplitude.get(target)
        if default_amplitude is None:
            raise ValueError(f"Control amplitude for {target} is not defined.")

        rabi_param = self.rabi_params.get(target)
        if rabi_param is None:
            raise ValueError(f"Rabi parameters for {target} are not stored.")

        return control_amplitude * rabi_param.frequency / default_amplitude

    def calc_rabi_rates(
        self,
        control_amplitude: float | None = None,
        *,
        print_result: bool | None = None,
    ) -> dict[str, float]:
        """Calculate Rabi rates for available targets."""
        if control_amplitude is None:
            control_amplitude = 1.0
        if print_result is None:
            print_result = True

        default_ampl = self.ctx.params.control_amplitude

        rabi_rates = {
            target: control_amplitude * rabi_param.frequency / default_ampl[target]
            for target, rabi_param in self.rabi_params.items()
        }

        if print_result:
            logger.info(f"Rabi rate for control amplitude {control_amplitude}\n")
            for target, rabi_rate in rabi_rates.items():
                logger.info(f"{target}: {rabi_rate * 1e3:.3f} MHz")

        return rabi_rates

    def get_hpi_pulse(
        self,
        target: str,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """Get the π/2 pulse for the given target."""
        param = self.ctx.calib_note.get_hpi_param(
            target,
            valid_days=valid_days or self.ctx.calibration_valid_days,
        )
        if param is not None:
            return FlatTop(
                duration=param["duration"],
                amplitude=param["amplitude"],
                tau=param["tau"],
            )
        else:
            return FlatTop(
                duration=HPI_DURATION,
                amplitude=self.ctx.params.get_control_amplitude(target),
                tau=HPI_RAMPTIME,
            )

    def get_pi_pulse(
        self,
        target: str,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """Get the π pulse for the given target."""
        param = self.ctx.calib_note.get_pi_param(
            target,
            valid_days=valid_days or self.ctx.calibration_valid_days,
        )
        if param is not None:
            return FlatTop(
                duration=param["duration"],
                amplitude=param["amplitude"],
                tau=param["tau"],
            )
        else:
            raise CalibrationMissingError(
                message="π pulse parameters are not stored.",
                target=target,
            )

    def get_drag_hpi_pulse(
        self,
        target: str,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """Get the DRAG π/2 pulse for the given target."""
        param = self.ctx.calib_note.get_drag_hpi_param(
            target,
            valid_days=valid_days or self.ctx.calibration_valid_days,
        )
        if param is not None:
            return Drag(
                duration=param["duration"],
                amplitude=param["amplitude"],
                beta=param["beta"],
            )
        else:
            raise CalibrationMissingError(
                message="DRAG π/2 pulse parameters are not stored.",
                target=target,
            )

    def get_drag_pi_pulse(
        self,
        target: str,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """Get the DRAG π pulse for the given target."""
        param = self.ctx.calib_note.get_drag_pi_param(
            target,
            valid_days=valid_days or self.ctx.calibration_valid_days,
        )
        if param is not None:
            return Drag(
                duration=param["duration"],
                amplitude=param["amplitude"],
                beta=param["beta"],
            )
        else:
            raise CalibrationMissingError(
                message="DRAG π pulse parameters are not stored.",
                target=target,
            )

    def get_pulse_for_state(
        self,
        target: str,
        state: str,  # ["0", "1", "+", "-", "+i", "-i"],
    ) -> Waveform:
        """Return a preparation pulse for a target state label."""
        if state == "0":
            return Blank(0)
        elif state == "1":
            return self.x180(target)
        else:
            if state == "+":
                return self.y90(target)
            elif state == "-":
                return self.y90m(target)
            elif state == "+i":
                return self.x90m(target)
            elif state == "-i":
                return self.x90(target)
            else:
                raise ValueError("Invalid state.")

    def x90(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """Return an X pi/2 pulse for the target."""
        try:
            x90 = self.get_drag_hpi_pulse(target)
        except CalibrationMissingError:
            x90 = self.get_hpi_pulse(target, valid_days=valid_days)
        return x90

    def x90m(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """Return a negative X pi/2 pulse for the target."""
        return self.x90(target, valid_days=valid_days).scaled(-1)

    def x180(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """Return an X pi pulse for the target."""
        try:
            x180 = self.get_drag_pi_pulse(target, valid_days=valid_days)
        except CalibrationMissingError:
            try:
                x180 = self.get_pi_pulse(target, valid_days=valid_days)
            except CalibrationMissingError:
                x90 = self.x90(target, valid_days=valid_days)
                x180 = x90.repeated(2)
        return x180

    def y90(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """Return a Y pi/2 pulse for the target."""
        return self.x90(target, valid_days=valid_days).shifted(np.pi / 2)

    def y90m(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """Return a negative Y pi/2 pulse for the target."""
        return self.x90(target, valid_days=valid_days).shifted(-np.pi / 2)

    def y180(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """Return a Y pi pulse for the target."""
        return self.x180(target, valid_days=valid_days).shifted(np.pi / 2)

    def z90(
        self,
    ) -> VirtualZ:
        """Return a virtual Z pi/2 rotation."""
        return VirtualZ(np.pi / 2)

    def z180(
        self,
    ) -> VirtualZ:
        """Return a virtual Z pi rotation."""
        return VirtualZ(np.pi)

    def hadamard(
        self,
        target: str,
        *,
        decomposition: Literal["Z180-Y90", "Y90-X180"] | None = None,
    ) -> PulseArray:
        """Return a Hadamard pulse sequence for the target."""
        if decomposition is None:
            decomposition = "Z180-Y90"
        if decomposition == "Z180-Y90":
            return PulseArray(
                [
                    # TODO: Need phase correction for CR targets
                    self.z180(),
                    self.y90(target),
                ]
            )
        elif decomposition == "Y90-X180":
            return PulseArray(
                [
                    self.y90(target),
                    self.x180(target),
                ]
            )
        else:
            raise ValueError(f"Invalid decomposition: {decomposition}. ")

    def readout(
        self,
        target: str,
        /,
        *,
        duration: float | None = None,
        amplitude: float | None = None,
        ramptime: float | None = None,
        type: RampType | None = None,
        drag_coeff: float | None = None,
        pre_margin: float | None = None,
        post_margin: float | None = None,
    ) -> Waveform:
        """Return a readout pulse for the target."""
        if duration is None:
            duration = self.readout_duration
        if pre_margin is None:
            pre_margin = self.readout_pre_margin
        if post_margin is None:
            post_margin = self.readout_post_margin

        return self.ctx.measurement.pulse_factory.readout_pulse(
            target=target,
            duration=duration,
            amplitude=amplitude,
            ramptime=ramptime,
            type=type,
            drag_coeff=drag_coeff,
            pre_margin=pre_margin,
            post_margin=post_margin,
        )

    def zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        cr_duration: float | None = None,
        cr_ramptime: float | None = None,
        cr_amplitude: float | None = None,
        cr_phase: float | None = None,
        cr_beta: float | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        cancel_beta: float | None = None,
        rotary_amplitude: float | None = None,
        echo: bool | None = None,
        x180: TargetMap[Waveform] | Waveform | None = None,
        x180_margin: float | None = None,
    ) -> PulseSchedule:
        """Return a ZX90 cross-resonance pulse schedule."""
        if echo is None:
            echo = True
        if x180_margin is None:
            x180_margin = 0.0

        cr_label = f"{control_qubit}-{target_qubit}"
        cr_param = self.ctx.calib_note.get_cr_param(
            cr_label,
            valid_days=self.ctx.calibration_valid_days,
        )
        if cr_param is None:
            raise ValueError(f"CR parameters for {cr_label} are not stored.")

        if x180 is None:
            pi_pulse = self.x180(control_qubit)
        elif isinstance(x180, Waveform):
            pi_pulse = x180
        else:
            pi_pulse = x180[control_qubit]

        if cr_amplitude is None:
            cr_amplitude = cr_param["cr_amplitude"]
        if cr_duration is None:
            cr_duration = cr_param["duration"]
        if cr_ramptime is None:
            cr_ramptime = cr_param["ramptime"]
        if cr_phase is None:
            cr_phase = cr_param["cr_phase"]
        if cr_beta is None:
            cr_beta = cr_param["cr_beta"]
        if cancel_amplitude is None:
            cancel_amplitude = cr_param["cancel_amplitude"]
        if cancel_phase is None:
            cancel_phase = cr_param["cancel_phase"]
        if cancel_beta is None:
            cancel_beta = cr_param["cancel_beta"]
        if rotary_amplitude is None:
            rotary_amplitude = cr_param["rotary_amplitude"]

        cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase) + rotary_amplitude

        return CrossResonance(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            cr_amplitude=cr_amplitude,
            cr_duration=cr_duration,
            cr_ramptime=cr_ramptime,
            cr_phase=cr_phase,
            cr_beta=cr_beta,
            cancel_amplitude=np.abs(cancel_pulse),
            cancel_phase=np.angle(cancel_pulse),
            cancel_beta=cancel_beta,
            echo=echo,
            pi_pulse=pi_pulse,
            pi_margin=x180_margin,
        )

    def rzx(
        self,
        control_qubit: str,
        target_qubit: str,
        angle: float,
        *,
        cr_duration: float | None = None,
        cr_ramptime: float | None = None,
        cr_amplitude: float | None = None,
        cr_phase: float | None = None,
        cr_beta: float | None = None,
        cancel_amplitude: float | None = None,
        cancel_phase: float | None = None,
        cancel_beta: float | None = None,
        rotary_amplitude: float | None = None,
        echo: bool | None = None,
        x180: TargetMap[Waveform] | Waveform | None = None,
        x180_margin: float | None = None,
    ) -> PulseSchedule:
        """Return an RZX pulse schedule for a given angle."""
        if echo is None:
            echo = True
        if x180_margin is None:
            x180_margin = 0.0

        # Reference angle for RZX gate normalization (half pi)
        REFERENCE_ANGLE = np.pi / 2
        coeff_value = angle / REFERENCE_ANGLE
        cr_label = f"{control_qubit}-{target_qubit}"
        cr_param = self.ctx.calib_note.get_cr_param(
            cr_label,
            valid_days=self.ctx.calibration_valid_days,
        )
        if cr_param is None:
            raise ValueError(f"CR parameters for {cr_label} are not stored.")

        if x180 is None:
            pi_pulse = self.x180(control_qubit)
        elif isinstance(x180, Waveform):
            pi_pulse = x180
        else:
            pi_pulse = x180[control_qubit]

        if cr_amplitude is None:
            cr_amplitude = cr_param["cr_amplitude"] * coeff_value
        if cr_duration is None:
            cr_duration = cr_param["duration"]
        if cr_ramptime is None:
            cr_ramptime = cr_param["ramptime"]
        if cr_phase is None:
            cr_phase = cr_param["cr_phase"]
        if cr_beta is None:
            cr_beta = cr_param["cr_beta"]
        if cancel_amplitude is None:
            cancel_amplitude = cr_param["cancel_amplitude"] * coeff_value
        if cancel_phase is None:
            cancel_phase = cr_param["cancel_phase"]
        if cancel_beta is None:
            cancel_beta = cr_param["cancel_beta"]
        if rotary_amplitude is None:
            rotary_amplitude = cr_param["rotary_amplitude"]

        cancel_pulse = cancel_amplitude * np.exp(1j * cancel_phase) + rotary_amplitude

        return CrossResonance(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            cr_amplitude=cr_amplitude,
            cr_duration=cr_duration,
            cr_ramptime=cr_ramptime,
            cr_phase=cr_phase,
            cr_beta=cr_beta,
            cancel_amplitude=np.abs(cancel_pulse),
            cancel_phase=np.angle(cancel_pulse),
            cancel_beta=cancel_beta,
            echo=echo,
            pi_pulse=pi_pulse,
            pi_margin=x180_margin,
        )

    def cnot(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        zx90: PulseSchedule | None = None,
        x90: Waveform | None = None,
        only_low_to_high: bool | None = None,
    ) -> PulseSchedule:
        """Return a CNOT pulse schedule."""
        if only_low_to_high is None:
            only_low_to_high = False

        cr_label = f"{control_qubit}-{target_qubit}"

        is_low_to_high = self.ctx.qubits[control_qubit].index % 4 in [0, 3]

        if (only_low_to_high and is_low_to_high) or (
            not only_low_to_high and cr_label in self.ctx.calib_note.cr_params
        ):
            if x90 is None:
                x90 = self.x90(target_qubit)
            zx90 = zx90 or self.zx90(control_qubit, target_qubit)
            with PulseSchedule([control_qubit, cr_label, target_qubit]) as cnot:
                cnot.call(zx90)
                cnot.add(control_qubit, VirtualZ(-np.pi / 2))
                cnot.add(target_qubit, x90.scaled(-1))
            return cnot
        else:
            if x90 is None:
                x90 = self.x90(control_qubit)
            zx90 = zx90 or self.zx90(target_qubit, control_qubit)
            cr_label = f"{target_qubit}-{control_qubit}"
            with PulseSchedule([control_qubit, cr_label, target_qubit]) as cnot_tc:
                cnot_tc.call(zx90)
                cnot_tc.add(target_qubit, VirtualZ(-np.pi / 2))
                cnot_tc.add(control_qubit, x90.scaled(-1))
            z180 = self.z180()
            hadamard_c = PulseArray([z180, self.y90(control_qubit)])
            hadamard_t = PulseArray([z180, self.y90(target_qubit)])
            with PulseSchedule([control_qubit, cr_label, target_qubit]) as cnot_ct:
                cnot_ct.add(control_qubit, hadamard_c)
                cnot_ct.add(target_qubit, hadamard_t)
                cnot_ct.add(cr_label, z180)
                cnot_ct.call(cnot_tc)
                cnot_ct.add(cr_label, z180)
                cnot_ct.add(control_qubit, hadamard_c)
                cnot_ct.add(target_qubit, hadamard_t)
            return cnot_ct

    def cx(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        zx90: PulseSchedule | None = None,
        x90: Waveform | None = None,
        only_low_to_high: bool | None = None,
    ) -> PulseSchedule:
        """Alias for `cnot`."""
        return self.cnot(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            zx90=zx90,
            x90=x90,
            only_low_to_high=only_low_to_high,
        )

    def cz(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        zx90: PulseSchedule | None = None,
        x90: Waveform | None = None,
        only_low_to_high: bool | None = None,
    ) -> PulseSchedule:
        """Return a CZ pulse schedule."""
        if only_low_to_high is None:
            only_low_to_high = False

        cr_label = f"{control_qubit}-{target_qubit}"

        is_low_to_high = self.ctx.qubits[control_qubit].index % 4 in [0, 3]

        if (only_low_to_high and is_low_to_high) or (
            not only_low_to_high and cr_label in self.ctx.calib_note.cr_params
        ):
            if x90 is None:
                x90 = self.x90(target_qubit)
            zx90 = zx90 or self.zx90(control_qubit, target_qubit)
            with PulseSchedule([control_qubit, cr_label, target_qubit]) as cnot:
                cnot.call(zx90)
                cnot.add(control_qubit, VirtualZ(-np.pi / 2))
                cnot.add(target_qubit, x90.scaled(-1))
            z180 = self.z180()
            hadamard_t = PulseArray([z180, self.y90(target_qubit)])
            with PulseSchedule([control_qubit, cr_label, target_qubit]) as cz:
                cz.add(target_qubit, hadamard_t)
                cz.add(cr_label, z180)
                cz.call(cnot)
                cz.add(cr_label, z180)
                cz.add(target_qubit, hadamard_t)
            return cz
        else:
            if x90 is None:
                x90 = self.x90(control_qubit)
            zx90 = zx90 or self.zx90(target_qubit, control_qubit)
            cr_label = f"{target_qubit}-{control_qubit}"
            with PulseSchedule([control_qubit, cr_label, target_qubit]) as cnot_tc:
                cnot_tc.call(zx90)
                cnot_tc.add(target_qubit, VirtualZ(-np.pi / 2))
                cnot_tc.add(control_qubit, x90.scaled(-1))
            z180 = self.z180()
            hadamard_c = PulseArray([z180, self.y90(control_qubit)])
            hadamard_t = PulseArray([z180, self.y90(target_qubit)])
            cr_label = f"{target_qubit}-{control_qubit}"
            with PulseSchedule([control_qubit, cr_label, target_qubit]) as cz:
                cz.add(control_qubit, hadamard_c)
                cz.add(cr_label, z180)
                cz.call(cnot_tc)
                cz.add(cr_label, z180)
                cz.add(control_qubit, hadamard_c)
            return cz

    @property
    def ef_hpi_pulse(self) -> dict[str, Waveform]:
        """Return calibrated EF half-pi pulses."""
        result = {}
        for target in self.ctx.ef_targets:
            param = self.ctx.calib_note.get_hpi_param(
                target,
                valid_days=self.ctx.calibration_valid_days,
            )
            if param is not None and None not in param.values():
                result[target] = FlatTop(
                    duration=param["duration"],
                    amplitude=param["amplitude"],
                    tau=param["tau"],
                )
        return result

    @property
    def ef_pi_pulse(self) -> dict[str, Waveform]:
        """Return calibrated EF pi pulses."""
        result = {}
        for target in self.ctx.ef_targets:
            param = self.ctx.calib_note.get_pi_param(
                target,
                valid_days=self.ctx.calibration_valid_days,
            )
            if param is not None and None not in param.values():
                result[target] = FlatTop(
                    duration=param["duration"],
                    amplitude=param["amplitude"],
                    tau=param["tau"],
                )
        return result

    @property
    def cr_pulse(self) -> dict[str, PulseSchedule]:
        """Return calibrated cross-resonance pulse schedules."""
        result = {}
        for cr_label in self.ctx.cr_targets:
            control_qubit, target_qubit = Target.cr_qubit_pair(cr_label)
            cr_param = self.ctx.calib_note.get_cr_param(cr_label)
            if cr_param is not None and None not in cr_param.values():
                cancel_amplitude = cr_param["cancel_amplitude"]
                cancel_phase = cr_param["cancel_phase"]
                rotary_amplitude = cr_param["rotary_amplitude"]
                cancel_pulse = (
                    cancel_amplitude * np.exp(1j * cancel_phase) + rotary_amplitude
                )
                result[cr_label] = CrossResonance(
                    control_qubit=control_qubit,
                    target_qubit=target_qubit,
                    cr_amplitude=cr_param["cr_amplitude"],
                    cr_duration=cr_param["duration"],
                    cr_ramptime=cr_param["ramptime"],
                    cr_phase=cr_param["cr_phase"],
                    cr_beta=cr_param["cr_beta"],
                    cancel_amplitude=np.abs(cancel_pulse),
                    cancel_phase=np.angle(cancel_pulse),
                    cancel_beta=cr_param["cancel_beta"],
                    echo=True,
                    pi_pulse=self.x180(control_qubit),
                    pi_margin=0.0,
                )

        return result

    @property
    def drag_hpi_pulse(self) -> dict[str, Waveform]:
        """Return calibrated DRAG half-pi pulses."""
        result = {}
        for target in self.ctx.ge_targets:
            param = self.ctx.calib_note.get_drag_hpi_param(
                target,
                valid_days=self.ctx.calibration_valid_days,
            )
            if param is not None and None not in param.values():
                result[target] = Drag(
                    duration=param["duration"],
                    amplitude=param["amplitude"],
                    beta=param["beta"],
                )
        return result

    @property
    def drag_pi_pulse(self) -> dict[str, Waveform]:
        """Return calibrated DRAG pi pulses."""
        result = {}
        for target in self.ctx.ge_targets:
            param = self.ctx.calib_note.get_drag_pi_param(
                target,
                valid_days=self.ctx.calibration_valid_days,
            )
            if param is not None and None not in param.values():
                result[target] = Drag(
                    duration=param["duration"],
                    amplitude=param["amplitude"],
                    beta=param["beta"],
                )
        return result

    @property
    def hpi_pulse(self) -> dict[str, Waveform]:
        """Return calibrated half-pi pulses."""
        result = {}
        for target in self.ctx.ge_targets:
            param = self.ctx.calib_note.get_hpi_param(
                target,
                valid_days=self.ctx.calibration_valid_days,
            )
            if param is not None and None not in param.values():
                result[target] = FlatTop(
                    duration=param["duration"],
                    amplitude=param["amplitude"],
                    tau=param["tau"],
                )
        return result

    @property
    def pi_pulse(self) -> dict[str, Waveform]:
        """Return calibrated pi pulses."""
        result = {}
        for target in self.ctx.ge_targets:
            param = self.ctx.calib_note.get_pi_param(
                target,
                valid_days=self.ctx.calibration_valid_days,
            )
            if param is not None and None not in param.values():
                result[target] = FlatTop(
                    duration=param["duration"],
                    amplitude=param["amplitude"],
                    tau=param["tau"],
                )
        return result
