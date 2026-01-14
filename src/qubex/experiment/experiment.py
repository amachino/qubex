"""
This module provides the Experiment class, which serves as a facade for users conducting experiments using Notebooks.
It manages which methods act as the public interface for conducting experiments.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Collection, Literal

from numpy.typing import ArrayLike, NDArray
from typing_extensions import deprecated

from ..backend import (
    Box,
    Chip,
    ConfigLoader,
    ControlParams,
    ControlSystem,
    DeviceController,
    ExperimentSystem,
    QuantumSystem,
    Qubit,
    Resonator,
    SystemManager,
    Target,
    TargetType,
)
from ..clifford import Clifford, CliffordGenerator
from ..measurement import (
    Measurement,
    MeasureResult,
    MultipleMeasureResult,
    StateClassifier,
)
from ..measurement.measurement import (
    DEFAULT_INTERVAL,
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_POST_MARGIN,
    DEFAULT_READOUT_PRE_MARGIN,
    DEFAULT_SHOTS,
)
from ..pulse import (
    PulseArray,
    PulseSchedule,
    RampType,
    VirtualZ,
    Waveform,
)
from ..typing import TargetMap
from .calibration_note import CalibrationNote
from .experiment_constants import (
    CALIBRATION_VALID_DAYS,
    CLASSIFIER_DIR,
    DEFAULT_RABI_TIME_RANGE,
    DRAG_HPI_DURATION,
    DRAG_PI_DURATION,
    PROPERTY_DIR,
)
from .experiment_context import ExperimentContext
from .experiment_note import ExperimentNote
from .experiment_record import ExperimentRecord
from .experiment_result import ExperimentResult, RabiData
from .mixin import (
    BenchmarkingMixin,
    CalibrationMixin,
    CharacterizationMixin,
    MeasurementMixin,
    OptimizationMixin,
)
from .rabi_param import RabiParam


class Experiment(
    BenchmarkingMixin,
    CharacterizationMixin,
    CalibrationMixin,
    MeasurementMixin,
    OptimizationMixin,
):
    """
    Class representing an experiment.

    Parameters
    ----------
    chip_id : str
        Identifier of the quantum chip.
    muxes : Collection[str | int], optional
        Mux labels to use in the experiment.
    qubits : Collection[str | int], optional
        Qubit labels to use in the experiment.
    exclude_qubits : Collection[str | int], optional
        Qubit labels to exclude in the experiment.
    config_dir : str, optional
        Path to the configuration directory containing:
          - box.yaml
          - chip.yaml
          - wiring.yaml
          - skew.yaml
    params_dir : str, optional
        Path to the parameters directory containing:
          - params.yaml
          - props.yaml
    calib_note_path : Path | str, optional
        Path to the calibration note file.
    calibration_valid_days : int, optional
        Number of days for which the calibration is valid.
    drag_hpi_duration : int, optional
        Duration of the DRAG HPI pulse.
    drag_pi_duration : int, optional
        Duration of the DRAG π pulse.
    readout_duration : int, optional
        Duration of the readout pulse.
    readout_pre_margin : int, optional
        Pre-margin of the readout pulse.
    readout_post_margin : int, optional
        Post-margin of the readout pulse.
    classifier_dir : Path | str, optional
        Directory of the state classifiers.
    classifier_type : Literal["kmeans", "gmm"], optional
        Type of the state classifier. Defaults to "gmm".
    configuration_mode : Literal["ge-ef-cr", "ge-cr-cr"], optional
        Configuration mode of the experiment. Defaults to "ge-cr-cr".

    Examples
    --------
    >>> from qubex import Experiment
    >>> ex = Experiment(
    ...     chip_id="64Q",
    ...     qubits=["Q00", "Q01"],
    ... )
    """

    def __init__(
        self,
        *,
        chip_id: str,
        muxes: Collection[str | int] | None = None,
        qubits: Collection[str | int] | None = None,
        exclude_qubits: Collection[str | int] | None = None,
        config_dir: Path | str | None = None,
        params_dir: Path | str | None = None,
        calib_note_path: Path | str | None = None,
        calibration_valid_days: int = CALIBRATION_VALID_DAYS,
        drag_hpi_duration: float = DRAG_HPI_DURATION,
        drag_pi_duration: float = DRAG_PI_DURATION,
        readout_duration: float = DEFAULT_READOUT_DURATION,
        readout_pre_margin: float = DEFAULT_READOUT_PRE_MARGIN,
        readout_post_margin: float = DEFAULT_READOUT_POST_MARGIN,
        property_dir: Path | str = PROPERTY_DIR,
        classifier_dir: Path | str = CLASSIFIER_DIR,
        classifier_type: Literal["kmeans", "gmm"] = "gmm",
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] = "ge-cr-cr",
        mock_mode: bool = False,
    ):
        self._ctx = ExperimentContext(
            chip_id=chip_id,
            muxes=muxes,
            qubits=qubits,
            exclude_qubits=exclude_qubits,
            config_dir=config_dir,
            params_dir=params_dir,
            calib_note_path=calib_note_path,
            calibration_valid_days=calibration_valid_days,
            drag_hpi_duration=drag_hpi_duration,
            drag_pi_duration=drag_pi_duration,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            property_dir=property_dir,
            classifier_dir=classifier_dir,
            classifier_type=classifier_type,
            configuration_mode=configuration_mode,
            mock_mode=mock_mode,
        )

    @property
    def ctx(self) -> ExperimentContext:
        return self._ctx

    @property
    def tool(self):
        return self.ctx.tool

    @property
    def util(self):
        return self.ctx.util

    @property
    def measurement(self) -> Measurement:
        return self.ctx.measurement

    @property
    def system_manager(self) -> SystemManager:
        return self.ctx.system_manager

    @property
    def config_loader(self) -> ConfigLoader:
        return self.ctx.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        return self.ctx.experiment_system

    @property
    def quantum_system(self) -> QuantumSystem:
        return self.ctx.quantum_system

    @property
    def control_system(self) -> ControlSystem:
        return self.ctx.control_system

    @property
    def device_controller(self) -> DeviceController:
        return self.ctx.device_controller

    @property
    def params(self) -> ControlParams:
        return self.ctx.params

    @property
    def chip(self) -> Chip:
        return self.ctx.chip

    @property
    def chip_id(self) -> str:
        return self.ctx.chip_id

    @property
    def qubit_labels(self) -> list[str]:
        return self.ctx.qubit_labels

    @property
    def mux_labels(self) -> list[str]:
        return self.ctx.mux_labels

    @property
    def qubits(self) -> dict[str, Qubit]:
        return self.ctx.qubits

    @property
    def resonators(self) -> dict[str, Resonator]:
        return self.ctx.resonators

    @property
    def targets(self) -> dict[str, Target]:
        return self.ctx.targets

    @property
    def available_targets(self) -> dict[str, Target]:
        return self.ctx.available_targets

    @property
    def ge_targets(self) -> dict[str, Target]:
        return self.ctx.ge_targets

    @property
    def ef_targets(self) -> dict[str, Target]:
        return self.ctx.ef_targets

    @property
    def cr_targets(self) -> dict[str, Target]:
        return self.ctx.cr_targets

    @property
    def cr_labels(self) -> list[str]:
        return self.ctx.cr_labels

    @property
    def cr_pairs(self) -> list[tuple[str, str]]:
        return self.ctx.cr_pairs

    @property
    def edge_pairs(self) -> list[tuple[str, str]]:
        return self.ctx.edge_pairs

    @property
    def edge_labels(self) -> list[str]:
        return self.ctx.edge_labels

    @property
    def boxes(self) -> dict[str, Box]:
        return self.ctx.boxes

    @property
    def box_ids(self) -> list[str]:
        return self.ctx.box_ids

    @property
    def config_path(self) -> str:
        return self.ctx.config_path

    @property
    def params_path(self) -> str:
        return self.ctx.params_path

    @property
    def calib_note(self) -> CalibrationNote:
        return self.ctx.calib_note

    @property
    def note(self) -> ExperimentNote:
        return self.ctx.note

    @property
    def readout_duration(self) -> float:
        return self.ctx.readout_duration

    @property
    def readout_pre_margin(self) -> float:
        return self.ctx.readout_pre_margin

    @property
    def readout_post_margin(self) -> float:
        return self.ctx.readout_post_margin

    @property
    def drag_hpi_duration(self) -> float:
        return self.ctx.drag_hpi_duration

    @property
    def drag_pi_duration(self) -> float:
        return self.ctx.drag_pi_duration

    @property
    def hpi_pulse(self) -> dict[str, Waveform]:
        return self.ctx.hpi_pulse

    @property
    def pi_pulse(self) -> dict[str, Waveform]:
        return self.ctx.pi_pulse

    @property
    def drag_hpi_pulse(self) -> dict[str, Waveform]:
        return self.ctx.drag_hpi_pulse

    @property
    def drag_pi_pulse(self) -> dict[str, Waveform]:
        return self.ctx.drag_pi_pulse

    @property
    def ef_hpi_pulse(self) -> dict[str, Waveform]:
        return self.ctx.ef_hpi_pulse

    @property
    def ef_pi_pulse(self) -> dict[str, Waveform]:
        return self.ctx.ef_pi_pulse

    @property
    def cr_pulse(self) -> dict[str, PulseSchedule]:
        return self.ctx.cr_pulse

    @property
    def rabi_params(self) -> dict[str, RabiParam]:
        return self.ctx.rabi_params

    @property
    def ge_rabi_params(self) -> dict[str, RabiParam]:
        return self.ctx.ge_rabi_params

    @property
    def ef_rabi_params(self) -> dict[str, RabiParam]:
        return self.ctx.ef_rabi_params

    @property
    def property_dir(self) -> Path:
        return self.ctx.property_dir

    @property
    def classifier_dir(self) -> Path:
        return self.ctx.classifier_dir

    @property
    def classifier_type(self) -> Literal["kmeans", "gmm"]:
        return self.ctx.classifier_type

    @property
    def classifiers(self) -> TargetMap[StateClassifier]:
        return self.ctx.classifiers

    @property
    def state_centers(self) -> dict[str, dict[int, complex]]:
        return self.ctx.state_centers

    @property
    def clifford_generator(self) -> CliffordGenerator:
        return self.ctx.clifford_generator

    @property
    def clifford(self) -> dict[str, Clifford]:
        return self.ctx.clifford

    @property
    def configuration_mode(self) -> Literal["ge-ef-cr", "ge-cr-cr"]:
        return self.ctx.configuration_mode

    @property
    def reference_phases(self) -> dict[str, float]:
        return self.ctx.reference_phases

    def load_property(self, property_name: str) -> dict:
        return self.ctx.load_property(property_name)

    def save_property(
        self,
        property_name: str,
        data: dict,
        *,
        save_path: Path | str | None = None,
    ):
        return self.ctx.save_property(
            property_name,
            data,
            save_path=save_path,
        )

    def load_calib_note(self, path: Path | str | None = None):
        """
        Load the calibration data from the given path or from the default calibration note file.
        """
        return self.ctx.load_calib_note(path=path)

    def get_qubit_label(self, index: int) -> str:
        """
        Get the qubit label from the given qubit index.
        """
        return self.ctx.get_qubit_label(index)

    def get_resonator_label(self, index: int) -> str:
        """
        Get the resonator label from the given resonator index.
        """
        return self.ctx.get_resonator_label(index)

    def get_cr_label(
        self,
        control_index: int,
        target_index: int,
    ) -> str:
        """
        Get the cross-resonance label from the given control and target qubit indices.
        """
        return self.ctx.get_cr_label(control_index, target_index)

    def get_cr_pairs(
        self,
        low_to_high: bool = True,
        high_to_low: bool = False,
    ) -> list[tuple[str, str]]:
        """
        Get the cross-resonance pairs.
        """
        return self.ctx.get_cr_pairs(low_to_high, high_to_low)

    def get_cr_labels(
        self,
        low_to_high: bool = True,
        high_to_low: bool = False,
    ) -> list[str]:
        """
        Get the cross-resonance labels.
        """
        return self.ctx.get_cr_labels(low_to_high, high_to_low)

    def get_edge_pairs(
        self,
    ) -> list[tuple[str, str]]:
        """
        Get the qubit edge pairs.
        """
        return self.ctx.get_edge_pairs()

    def get_edge_labels(
        self,
    ) -> list[str]:
        """
        Get the qubit edge labels.
        """
        return self.ctx.get_edge_labels()

    @staticmethod
    def cr_pair(cr_label: str) -> tuple[str, str]:
        return Target.cr_qubit_pair(cr_label)

    def validate_rabi_params(
        self,
        targets: Collection[str] | None = None,
    ):
        return self.ctx.validate_rabi_params(targets=targets)

    def get_rabi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> RabiParam | None:
        """
        Get the Rabi parameters for the given target.
        """
        return self.ctx.get_rabi_param(target, valid_days=valid_days)

    def store_rabi_params(
        self,
        rabi_params: dict[str, RabiParam],
        r2_threshold: float = 0.5,
    ):
        self.ctx.store_rabi_params(
            rabi_params,
            r2_threshold=r2_threshold,
        )

    def correct_rabi_params(
        self,
        targets: Collection[str] | str | None = None,
        *,
        reference_phases: dict[str, float] | None = None,
        save: bool = True,
    ):
        return self.ctx.correct_rabi_params(
            targets=targets,
            reference_phases=reference_phases,
            save=save,
        )

    def correct_classifiers(
        self,
        targets: Collection[str] | str | None = None,
        *,
        reference_phases: dict[str, float] | None = None,
        save: bool = True,
    ):
        return self.ctx.correct_classifiers(
            targets=targets,
            reference_phases=reference_phases,
            save=save,
        )

    def correct_cr_params(
        self,
        cr_labels: Collection[str] | str | None = None,
        *,
        shots: int = 10000,
        save: bool = True,
    ):
        return self.ctx.correct_cr_params(
            cr_labels=cr_labels,
            shots=shots,
            save=save,
        )

    def correct_calibration(
        self,
        qubit_labels: Collection[str] | str | None = None,
        cr_labels: Collection[str] | str | None = None,
        *,
        save: bool = False,
    ):
        return self.ctx.correct_calibration(
            qubit_labels=qubit_labels,
            cr_labels=cr_labels,
            save=save,
        )

    def get_hpi_pulse(
        self,
        target: str,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """
        Get the π/2 pulse for the given target.
        """
        return self.ctx.get_hpi_pulse(
            target,
            valid_days=valid_days,
        )

    def get_pi_pulse(
        self,
        target: str,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """
        Get the π pulse for the given target.
        """
        return self.ctx.get_pi_pulse(
            target,
            valid_days=valid_days,
        )

    def get_drag_hpi_pulse(
        self,
        target: str,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """
        Get the DRAG π/2 pulse for the given target.
        """
        return self.ctx.get_drag_hpi_pulse(
            target,
            valid_days=valid_days,
        )

    def get_drag_pi_pulse(
        self,
        target: str,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """
        Get the DRAG π pulse for the given target.
        """
        return self.ctx.get_drag_pi_pulse(
            target,
            valid_days=valid_days,
        )

    def get_pulse_for_state(
        self,
        target: str,
        state: str,  # ["0", "1", "+", "-", "+i", "-i"],
    ) -> Waveform:
        return self.ctx.get_pulse_for_state(target, state)

    def get_spectators(
        self,
        qubit: str,
        in_same_mux: bool = False,
    ) -> list[Qubit]:
        return self.ctx.get_spectators(qubit, in_same_mux=in_same_mux)

    def get_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray:
        return self.ctx.get_confusion_matrix(targets)

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray:
        return self.ctx.get_inverse_confusion_matrix(targets)

    def is_connected(self) -> bool:
        return self.ctx.is_connected()

    def check_status(self):
        return self.ctx.check_status()

    def connect(
        self,
        *,
        sync_clocks: bool = True,
    ) -> None:
        return self.ctx.connect(sync_clocks=sync_clocks)

    def linkup(
        self,
        box_ids: list[str] | None = None,
        noise_threshold: int | None = None,
    ) -> None:
        return self.ctx.linkup(
            box_ids=box_ids,
            noise_threshold=noise_threshold,
        )

    def resync_clocks(
        self,
        box_ids: list[str] | None = None,
    ) -> None:
        return self.ctx.resync_clocks(box_ids=box_ids)

    def configure(
        self,
        box_ids: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
        mode: Literal["ge-ef-cr", "ge-cr-cr"] | None = None,
    ):
        return self.ctx.configure(
            box_ids=box_ids,
            exclude=exclude,
            mode=mode,
        )

    def reload(self):
        return self.ctx.reload()

    def reset_awg_and_capunits(
        self,
        box_ids: str | Collection[str] | None = None,
        qubits: Collection[str] | None = None,
    ):
        return self.ctx.reset_awg_and_capunits(
            box_ids=box_ids,
            qubits=qubits,
        )

    @deprecated("This method is tentative. It may be removed in the future.")
    def register_custom_target(
        self,
        *,
        label: str,
        frequency: float,
        box_id: str,
        port_number: int,
        channel_number: int,
        target_type: TargetType = TargetType.CTRL_GE,
        update_lsi: bool = False,
    ):
        return self.ctx.register_custom_target(
            label=label,
            frequency=frequency,
            box_id=box_id,
            port_number=port_number,
            channel_number=channel_number,
            target_type=target_type,
            update_lsi=update_lsi,
        )

    @contextmanager
    def modified_frequencies(
        self,
        frequencies: dict[str, float] | None,
    ):
        with self.ctx.modified_frequencies(frequencies):
            yield

    def save_calib_note(
        self,
        file_path: Path | str | None = None,
    ):
        return self.ctx.save_calib_note(file_path=file_path)

    @deprecated("Use `calib_note.save()` instead.")
    def save_defaults(self):
        return self.ctx.save_defaults()

    @deprecated("Use `calib_note.clear()` instead.")
    def clear_defaults(self):
        return self.ctx.clear_defaults()

    @deprecated("")
    def delete_defaults(self):
        return self.ctx.delete_defaults()

    def load_record(
        self,
        name: str,
    ) -> ExperimentRecord:
        return self.ctx.load_record(name)

    def check_noise(
        self,
        targets: Collection[str] | str | None = None,
        *,
        duration: int = 10240,
        plot: bool = True,
    ) -> MeasureResult:
        """
        Checks the noise level of the system.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to check the noise.
        duration : int, optional
            Duration of the noise measurement. Defaults to 2048.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = ex.check_noise(["Q00", "Q01"])
        """
        return self.ctx.check_noise(
            targets=targets,
            duration=duration,
            plot=plot,
        )

    def check_waveform(
        self,
        targets: Collection[str] | str | None = None,
        *,
        method: Literal["measure", "execute"] = "measure",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitude: float | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool = False,
        plot: bool = True,
    ) -> MeasureResult | MultipleMeasureResult:
        """
        Checks the readout waveforms of the given targets.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to check the waveforms.
        shots : int, optional
            Number of shots.
        interval : int, optional
            Interval between shots.
        readout_amplitude : float, optional
            Amplitude of the readout pulse.
        readout_duration : float, optional
            Duration of the readout pulse in ns.
        readout_pre_margin : float, optional
            Pre-margin of the readout pulse in ns.
        readout_post_margin : float, optional
            Post-margin of the readout pulse in ns.
        add_pump_pulses : bool, optional
            Whether to add pump pulses to the readout sequence. Defaults to False.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = ex.check_waveform(["Q00", "Q01"])
        """
        return self.ctx.check_waveform(
            targets=targets,
            method=method,
            shots=shots,
            interval=interval,
            readout_amplitude=readout_amplitude,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )

    def check_rabi(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        store_params: bool = False,
        rabi_level: Literal["ge", "ef"] = "ge",
        plot: bool = True,
    ) -> ExperimentResult[RabiData]:
        """
        Checks the Rabi oscillation of the given targets.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to check the Rabi oscillation.
        time_range : ArrayLike, optional
            Time range of the experiment in ns. Defaults to RABI_TIME_RANGE.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        store_params : bool, optional
            Whether to store the Rabi parameters. Defaults to False.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[RabiData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.check_rabi(["Q00", "Q01"])
        """
        return self.ctx.check_rabi(
            targets=targets,
            time_range=time_range,
            shots=shots,
            interval=interval,
            store_params=store_params,
            rabi_level=rabi_level,
            plot=plot,
        )

    def calc_control_amplitude(
        self,
        target: str,
        rabi_rate: float,
        *,
        rabi_amplitude_ratio: float | None = None,
    ) -> float:
        return self.ctx.calc_control_amplitude(
            target,
            rabi_rate,
            rabi_amplitude_ratio=rabi_amplitude_ratio,
        )

    def calc_control_amplitudes(
        self,
        rabi_rate: float | None = None,
        *,
        current_amplitudes: dict[str, float] | None = None,
        current_rabi_params: dict[str, RabiParam] | None = None,
        print_result: bool = True,
    ) -> dict[str, float]:
        return self.ctx.calc_control_amplitudes(
            rabi_rate,
            current_amplitudes=current_amplitudes,
            current_rabi_params=current_rabi_params,
            print_result=print_result,
        )

    def calc_rabi_rate(
        self,
        target: str,
        control_amplitude,
    ) -> float:
        return self.ctx.calc_rabi_rate(target, control_amplitude)

    def calc_rabi_rates(
        self,
        control_amplitude: float = 1.0,
        *,
        print_result: bool = True,
    ) -> dict[str, float]:
        return self.ctx.calc_rabi_rates(
            control_amplitude,
            print_result=print_result,
        )

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
        return self.ctx.readout(
            target,
            duration=duration,
            amplitude=amplitude,
            ramptime=ramptime,
            type=type,
            drag_coeff=drag_coeff,
            pre_margin=pre_margin,
            post_margin=post_margin,
        )

    def x90(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.ctx.x90(target, valid_days=valid_days)

    def x90m(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.ctx.x90m(target, valid_days=valid_days)

    def x180(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.ctx.x180(target, valid_days=valid_days)

    def y90(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.ctx.y90(target, valid_days=valid_days)

    def y90m(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.ctx.y90m(target, valid_days=valid_days)

    def y180(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.ctx.y180(target, valid_days=valid_days)

    def z90(
        self,
    ) -> VirtualZ:
        return self.ctx.z90()

    def z180(
        self,
    ) -> VirtualZ:
        return self.ctx.z180()

    def hadamard(
        self,
        target: str,
        *,
        decomposition: Literal["Z180-Y90", "Y90-X180"] = "Z180-Y90",
    ) -> PulseArray:
        return self.ctx.hadamard(target, decomposition=decomposition)

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
        echo: bool = True,
        x180: TargetMap[Waveform] | Waveform | None = None,
        x180_margin: float = 0.0,
    ) -> PulseSchedule:
        return self.ctx.zx90(
            control_qubit,
            target_qubit,
            cr_duration=cr_duration,
            cr_ramptime=cr_ramptime,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cr_beta=cr_beta,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            cancel_beta=cancel_beta,
            rotary_amplitude=rotary_amplitude,
            echo=echo,
            x180=x180,
            x180_margin=x180_margin,
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
        echo: bool = True,
        x180: TargetMap[Waveform] | Waveform | None = None,
        x180_margin: float = 0.0,
    ) -> PulseSchedule:
        return self.ctx.rzx(
            control_qubit,
            target_qubit,
            angle,
            cr_duration=cr_duration,
            cr_ramptime=cr_ramptime,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cr_beta=cr_beta,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            cancel_beta=cancel_beta,
            rotary_amplitude=rotary_amplitude,
            echo=echo,
            x180=x180,
            x180_margin=x180_margin,
        )

    def cnot(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        zx90: PulseSchedule | None = None,
        x90: Waveform | None = None,
        only_low_to_high: bool = False,
    ) -> PulseSchedule:
        return self.ctx.cnot(
            control_qubit,
            target_qubit,
            zx90=zx90,
            x90=x90,
            only_low_to_high=only_low_to_high,
        )

    def cx(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        zx90: PulseSchedule | None = None,
        x90: Waveform | None = None,
        only_low_to_high: bool = False,
    ) -> PulseSchedule:
        return self.ctx.cx(
            control_qubit,
            target_qubit,
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
        only_low_to_high: bool = False,
    ) -> PulseSchedule:
        return self.ctx.cz(
            control_qubit,
            target_qubit,
            zx90=zx90,
            x90=x90,
            only_low_to_high=only_low_to_high,
        )
