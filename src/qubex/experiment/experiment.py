"""
This module provides the Experiment class, which serves as a facade for users conducting experiments using Notebooks.
It manages which methods act as the public interface for conducting experiments.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Collection, Literal, Optional, Sequence

import numpy as np
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
from ..typing import IQArray, ParametricPulseSchedule, ParametricWaveformDict, TargetMap
from .calibration_note import CalibrationNote
from .experiment_constants import (
    CALIBRATION_SHOTS,
    CALIBRATION_VALID_DAYS,
    CLASSIFIER_DIR,
    DEFAULT_RABI_FREQUENCY,
    DEFAULT_RABI_TIME_RANGE,
    DRAG_COEFF,
    DRAG_HPI_DURATION,
    DRAG_PI_DURATION,
    PROPERTY_DIR,
)
from .experiment_context import ExperimentContext
from .experiment_note import ExperimentNote
from .experiment_record import ExperimentRecord
from .experiment_result import (
    AmplCalibData,
    AmplRabiData,
    ExperimentResult,
    FreqRabiData,
    RabiData,
    RamseyData,
    SweepData,
    T1Data,
    T2Data,
)
from .rabi_param import RabiParam
from .result import Result
from .services import (
    BenchmarkingService,
    CalibrationService,
    CharacterizationService,
    MeasurementService,
    OptimizationService,
    PulseService,
)


class Experiment:
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
        self._experiment_context = ExperimentContext(
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
        self._measurement_service = MeasurementService(
            experiment_context=self._experiment_context,
        )
        self._pulse_service = PulseService(
            experiment_context=self._experiment_context,
        )
        self._calibration_service = CalibrationService(
            experiment_context=self._experiment_context,
            measurement_service=self._measurement_service,
        )
        self._characterization_service = CharacterizationService(
            experiment_context=self._experiment_context,
            measurement_service=self._measurement_service,
            calibration_service=self._calibration_service,
        )
        self._benchmarking_service = BenchmarkingService(
            experiment_context=self._experiment_context,
            measurement_service=self._measurement_service,
        )
        self._optimization_service = OptimizationService(
            experiment_context=self._experiment_context,
            measurement_service=self._measurement_service,
            calibration_service=self._calibration_service,
            characterization_service=self._characterization_service,
            benchmarking_service=self._benchmarking_service,
        )

    @property
    def ctx(self) -> ExperimentContext:
        return self._experiment_context

    @property
    def pulse(self) -> PulseService:
        return self._pulse_service

    @property
    def measurement_service(self) -> MeasurementService:
        return self._measurement_service

    @property
    def calibration_service(self) -> CalibrationService:
        return self._calibration_service

    @property
    def characterization_service(self) -> CharacterizationService:
        return self._characterization_service

    @property
    def benchmarking_service(self) -> BenchmarkingService:
        return self._benchmarking_service

    @property
    def optimization_service(self) -> OptimizationService:
        return self._optimization_service

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
        return self.calibration_service.correct_rabi_params(
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
        return self.calibration_service.correct_classifiers(
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
        return self.calibration_service.correct_cr_params(
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
        return self.calibration_service.correct_calibration(
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
        return self.pulse.get_hpi_pulse(
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
        return self.pulse.get_pi_pulse(
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
        return self.pulse.get_drag_hpi_pulse(
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
        return self.pulse.get_drag_pi_pulse(
            target,
            valid_days=valid_days,
        )

    def get_pulse_for_state(
        self,
        target: str,
        state: str,  # ["0", "1", "+", "-", "+i", "-i"],
    ) -> Waveform:
        return self.pulse.get_pulse_for_state(target, state)

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
        return self.measurement_service.check_waveform(
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
        return self.measurement_service.check_rabi(
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
        return self.pulse.readout(
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
        return self.pulse.x90(target, valid_days=valid_days)

    def x90m(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.pulse.x90m(target, valid_days=valid_days)

    def x180(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.pulse.x180(target, valid_days=valid_days)

    def y90(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.pulse.y90(target, valid_days=valid_days)

    def y90m(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.pulse.y90m(target, valid_days=valid_days)

    def y180(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.pulse.y180(target, valid_days=valid_days)

    def z90(
        self,
    ) -> VirtualZ:
        return self.pulse.z90()

    def z180(
        self,
    ) -> VirtualZ:
        return self.pulse.z180()

    def hadamard(
        self,
        target: str,
        *,
        decomposition: Literal["Z180-Y90", "Y90-X180"] = "Z180-Y90",
    ) -> PulseArray:
        return self.pulse.hadamard(target, decomposition=decomposition)

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
        return self.pulse.zx90(
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
        return self.pulse.rzx(
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
        return self.pulse.cnot(
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
        return self.pulse.cx(
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
        return self.pulse.cz(
            control_qubit,
            target_qubit,
            zx90=zx90,
            x90=x90,
            only_low_to_high=only_low_to_high,
        )

    # measurement_service methods

    def execute(
        self,
        schedule: PulseSchedule,
        *,
        frequencies: Optional[dict[str, float]] = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        add_last_measurement: bool = False,
        add_pump_pulses: bool = False,
        enable_dsp_demodulation: bool = True,
        enable_dsp_sum: bool | None = None,
        enable_dsp_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
        reset_awg_and_capunits: bool = True,
        plot: bool = False,
    ) -> MultipleMeasureResult:
        """
        Execute the given schedule.

        Parameters
        ----------
        schedule : PulseSchedule
            Schedule to execute.
        mode : Literal["single", "avg"], optional
            Measurement mode. Defaults to "avg".
        shots : int, optional
            Number of shots
        interval : float, optional
            Interval between shots in ns.
        frequencies : Optional[dict[str, float]], optional
            Frequencies of the qubits.
        readout_amplitudes : dict[str, float], optional
            Readout amplitude for each target.
        readout_duration : float, optional
            Readout duration in ns.
        readout_pre_margin : float, optional
            Readout pre-margin in ns.
        readout_post_margin : float, optional
            Readout post-margin in ns.
        readout_ramptime : float, optional
            Readout ramp time in ns.
        readout_drag_coeff : float, optional
            Readout DRAG coefficient.
        readout_ramp_type : RampType, optional
            Readout ramp type. Defaults to "RaisedCosine".
        add_last_measurement : bool, optional
            Whether to add the last measurement to the result. Defaults to False.
        add_pump_pulses : bool, optional
            Whether to add pump pulses to the sequence. Defaults to False.
        reset_awg_and_capunits : bool, optional
            Whether to reset the AWG and capture units before the experiment. Defaults to True.
        enable_dsp_demodulation : bool, optional
            Whether to enable DSP demodulation. Defaults to True.
        enable_dsp_sum : bool | None, optional
            Whether to enable DSP summation. Defaults to None.
        enable_dsp_classification : bool, optional
            Whether to enable DSP classification. Defaults to False
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        MultipleMeasureResult
            Result of the experiment.

        Examples
        --------
        >>> with pulse.PulseSchedule(["Q00", "Q01"]) as ps:
        ...     ps.add("Q00", pulse.Rect(...))
        ...     ps.add("Q01", pulse.Gaussian(...))
        >>> result = ex.execute(
        ...     schedule=ps,
        ...     mode="avg",
        ...     shots=1024,
        ...     interval=150 * 1024,
        ... )
        """
        return self.measurement_service.execute(
            schedule,
            frequencies=frequencies,
            mode=mode,
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramptime=readout_ramptime,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            add_last_measurement=add_last_measurement,
            add_pump_pulses=add_pump_pulses,
            enable_dsp_demodulation=enable_dsp_demodulation,
            enable_dsp_sum=enable_dsp_sum,
            enable_dsp_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
            reset_awg_and_capunits=reset_awg_and_capunits,
            plot=plot,
        )

    def measure(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        frequencies: Optional[dict[str, float]] = None,
        initial_states: dict[str, str] | None = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramptime: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        add_pump_pulses: bool = False,
        enable_dsp_demodulation: bool = True,
        enable_dsp_sum: bool | None = None,
        enable_dsp_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
        reset_awg_and_capunits: bool = True,
        plot: bool = False,
    ) -> MeasureResult:
        """
        Measures the signals using the given sequence.

        Parameters
        ----------
        sequence : TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule
            Sequence of the experiment.
        frequencies : Optional[dict[str, float]]
            Frequencies of the qubits.
        initial_states : dict[str, str], optional
            Initial states of the qubits.
        mode : Literal["single", "avg"], optional
            Measurement mode. Defaults to "avg".
        shots : int, optional
            Number of shots.
        interval : float, optional
            Interval between shots in ns.
        readout_amplitudes : dict[str, float], optional
            Readout amplitude for each target.
        readout_duration : float, optional
            Readout duration in ns.
        readout_pre_margin : float, optional
            Readout pre-margin in ns.
        readout_post_margin : float, optional
            Readout post-margin in ns.
        readout_ramptime : float, optional
            Readout ramp time in ns.
        readout_drag_coeff : float, optional
            Readout DRAG coefficient.
        readout_ramp_type : RampType, optional
            Readout ramp type. Defaults to "RaisedCosine".
        add_pump_pulses : bool, optional
            Whether to add pump pulses to the sequence. Defaults to False.
        enable_dsp_demodulation : bool, optional
            Whether to enable DSP demodulation. Defaults to True.
        enable_dsp_sum : bool | None, optional
            Whether to enable DSP summation. Defaults to None.
        enable_dsp_classification : bool, optional
            Whether to enable DSP classification. Defaults to False.
        reset_awg_and_capunits : bool, optional
            Whether to reset the AWG and capture units before the experiment. Defaults to True.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = ex.measure(
        ...     sequence={"Q00": [0.1+0.0j, 0.3+0.0j, 0.1+0.0j]},
        ...     mode="avg",
        ...     shots=1024,
        ...     interval=150 * 1024,
        ...     plot=True,
        ... )
        """
        return self.measurement_service.measure(
            sequence=sequence,
            frequencies=frequencies,
            initial_states=initial_states,
            mode=mode,
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramptime=readout_ramptime,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            add_pump_pulses=add_pump_pulses,
            enable_dsp_demodulation=enable_dsp_demodulation,
            enable_dsp_sum=enable_dsp_sum,
            enable_dsp_classification=enable_dsp_classification,
            line_param0=line_param0,
            line_param1=line_param1,
            reset_awg_and_capunits=reset_awg_and_capunits,
            plot=plot,
        )

    def measure_state(
        self,
        states: dict[
            str, Literal["0", "1", "+", "-", "+i", "-i"] | Literal["g", "e", "f"]
        ],
        *,
        mode: Literal["single", "avg"] = "single",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool = False,
        plot: bool = False,
    ) -> MeasureResult:
        """
        Measures the signals using the given states.

        Parameters
        ----------
        states : dict[str, Literal["0", "1", "+", "-", "+i", "-i"] | Literal["g", "e", "f"]]
            States to prepare.
        mode : Literal["single", "avg"], optional
            Measurement mode. Defaults to "single".
        shots : int, optional
            Number of shots.
        interval : float, optional
            Interval between shots in ns.
        readout_amplitudes : dict[str, float], optional
            Readout amplitude for each target.
        readout_duration : float, optional
            Readout duration in ns.
        readout_pre_margin : float, optional
            Readout pre-margin in ns.
        readout_post_margin : float, optional
            Readout post-margin in ns.
        add_pump_pulses : bool, optional
            Whether to add pump pulses to the sequence. Defaults to False.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        MeasureResult
            Result of the experiment.

        Examples
        --------
        >>> result = ex.measure_state(
        ...     states={"Q00": "0", "Q01": "1"},
        ...     mode="single",
        ...     shots=1024,
        ...     interval=150 * 1024,
        ...     plot=True,
        ... )
        """
        return self.measurement_service.measure_state(
            states=states,
            mode=mode,
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )

    def measure_idle_states(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool = False,
        plot: bool = True,
    ) -> Result:
        """
        Measures the idle states of the given targets.

        Parameters
        ----------
        targets : Collection[str] | str | None, optional
            Targets to measure. Defaults to None (all targets).
        shots : int, optional
            Number of shots.
        interval : float, optional
            Interval between shots in ns.
        readout_amplitudes : dict[str, float], optional
            Readout amplitude for each target.
        readout_duration : float, optional
            Readout duration in ns.
        readout_pre_margin : float, optional
            Readout pre-margin in ns.
        readout_post_margin : float, optional
            Readout post-margin in ns.
        add_pump_pulses : bool, optional
            Whether to add pump pulses to the sequence. Defaults to False.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        dict
            Dictionary containing the idle states for each target.

        Examples
        --------
        >>> idle_states = ex.measure_idle_states(targets=["Q00", "Q01"])
        """
        return self.measurement_service.measure_idle_states(
            targets=targets,
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            add_pump_pulses=add_pump_pulses,
            plot=plot,
        )

    def obtain_reference_points(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int | None = None,
        interval: float | None = None,
        store_reference_points: bool = True,
    ) -> Result:
        """
        Obtains the reference points for the given targets.

        Parameters
        ----------
        targets : Collection[str] | str | None, optional
            Targets to obtain reference points for. Defaults to None (all targets).
        shots : int, optional
            Number of shots.
        interval : float, optional
            Interval between shots in ns.
        store_reference_points : bool, optional
            Whether to store the reference points. Defaults to True.

        Returns
        -------
        dict
            Dictionary containing the reference points for each target.

        Examples
        --------
        >>> ref_points = ex.obtain_reference_points(targets=["Q00", "Q01"])
        """
        return self.measurement_service.obtain_reference_points(
            targets=targets,
            shots=shots,
            interval=interval,
            store_reference_points=store_reference_points,
        )

    def sweep_parameter(
        self,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        *,
        sweep_range: ArrayLike,
        repetitions: int = 1,
        frequencies: dict[str, float] | None = None,
        rabi_level: Literal["ge", "ef"] = "ge",
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        plot: bool = True,
        title: str = "Sweep result",
        xlabel: str = "Sweep value",
        ylabel: str = "Measured value",
        xaxis_type: Literal["linear", "log"] = "linear",
        yaxis_type: Literal["linear", "log"] = "linear",
    ) -> ExperimentResult[SweepData]:
        """
        Sweeps a parameter and measures the signals.

        Parameters
        ----------
        sequence : ParametricPulseSchedule | ParametricWaveformMap
            Parametric sequence to sweep.
        sweep_range : ArrayLike
            Range of the parameter to sweep.
        repetitions : int, optional
            Number of repetitions. Defaults to 1.
        frequencies : dict[str, float], optional
            Frequencies of the qubits.
        rabi_level : Literal["ge", "ef"], optional
            Rabi level to use. Defaults to "ge".
        shots : int, optional
            Number of shots.
        interval : float, optional
            Interval between shots in ns.
        readout_amplitudes : dict[str, float], optional
            Readout amplitude for each target.
        readout_duration : float, optional
            Readout duration in ns.
        readout_pre_margin : float, optional
            Readout pre-margin in ns.
        readout_post_margin : float, optional
            Readout post-margin in ns.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        title : str, optional
            Title of the plot. Defaults to "Sweep result".
        xlabel : str, optional
            Label of the x-axis. Defaults to "Sweep value".
        ylabel : str, optional
            Label of the y-axis. Defaults to "Measured value".
        xaxis_type : Literal["linear", "log"], optional
            Type of the x-axis. Defaults to "linear".
        yaxis_type : Literal["linear", "log"], optional
            Type of the y-axis. Defaults to "linear".

        Returns
        -------
        ExperimentResult[SweepData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.sweep_parameter(
        ...     sequence=lambda x: {"Q00": Rect(duration=30, amplitude=x)},
        ...     sweep_range=np.arange(0, 101, 4),
        ...     repetitions=4,
        ...     shots=1024,
        ...     plot=True,
        ... )
        """
        return self.measurement_service.sweep_parameter(
            sequence=sequence,
            sweep_range=sweep_range,
            repetitions=repetitions,
            frequencies=frequencies,
            rabi_level=rabi_level,
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            plot=plot,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xaxis_type=xaxis_type,
            yaxis_type=yaxis_type,
        )

    def repeat_sequence(
        self,
        sequence: TargetMap[Waveform] | PulseSchedule,
        *,
        repetitions: int = 20,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
    ) -> ExperimentResult[SweepData]:
        """
        Repeats the pulse sequence n times.

        Parameters
        ----------
        sequence : TargetMap[Waveform] | PulseSchedule
            Pulse sequence to repeat.
        repetitions : int, optional
            Number of repetitions. Defaults to 20.
        shots : int, optional
            Number of shots.
        interval : float, optional
            Interval between shots in ns.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        ExperimentResult[SweepData]
            Result of the experiment.

        Examples
        --------
        >>> result = ex.repeat_sequence(
        ...     sequence={"Q00": Rect(duration=64, amplitude=0.1)},
        ...     repetitions=4,
        ... )
        """
        return self.measurement_service.repeat_sequence(
            sequence=sequence,
            repetitions=repetitions,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def obtain_rabi_params(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
        amplitudes: dict[str, float] | None = None,
        frequencies: dict[str, float] | None = None,
        is_damped: bool = False,
        fit_threshold: float = 0.5,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = True,
        simultaneous: bool = False,
    ) -> ExperimentResult[RabiData]:
        return self.measurement_service.obtain_rabi_params(
            targets=targets,
            time_range=time_range,
            amplitudes=amplitudes,
            frequencies=frequencies,
            is_damped=is_damped,
            fit_threshold=fit_threshold,
            shots=shots,
            interval=interval,
            plot=plot,
            store_params=store_params,
            simultaneous=simultaneous,
        )

    def obtain_ef_rabi_params(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
        is_damped: bool = False,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[RabiData]:
        return self.measurement_service.obtain_ef_rabi_params(
            targets=targets,
            time_range=time_range,
            is_damped=is_damped,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
        ramptime: float | None = None,
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool = True,
        fit_threshold: float = 0.5,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[RabiData]:
        return self.measurement_service.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
            ramptime=ramptime,
            frequencies=frequencies,
            detuning=detuning,
            is_damped=is_damped,
            fit_threshold=fit_threshold,
            shots=shots,
            interval=interval,
            plot=plot,
            store_params=store_params,
        )

    def ef_rabi_experiment(
        self,
        *,
        amplitudes: dict[str, float],
        time_range: ArrayLike,
        frequencies: dict[str, float] | None = None,
        detuning: float | None = None,
        is_damped: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        store_params: bool = False,
    ) -> ExperimentResult[RabiData]:
        return self.measurement_service.ef_rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
            frequencies=frequencies,
            detuning=detuning,
            is_damped=is_damped,
            shots=shots,
            interval=interval,
            plot=plot,
            store_params=store_params,
        )

    def measure_state_distribution(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_states: Literal[2, 3] = 2,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        plot: bool = True,
    ) -> list[MeasureResult]:
        return self.measurement_service.measure_state_distribution(
            targets=targets,
            n_states=n_states,
            shots=shots,
            interval=interval,
            readout_duration=readout_duration,
            readout_amplitudes=readout_amplitudes,
            plot=plot,
        )

    def build_classifier(
        self,
        targets: Collection[str] | str | None = None,
        *,
        n_states: Literal[2, 3] | None = None,
        save_classifier: bool = True,
        save_dir: Path | str | None = None,
        shots: int | None = None,
        interval: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        add_pump_pulses: bool = False,
        simultaneous: bool = False,
        plot: bool = True,
    ) -> Result:
        return self.measurement_service.build_classifier(
            targets=targets,
            n_states=n_states,
            save_classifier=save_classifier,
            save_dir=save_dir,
            shots=shots,
            interval=interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            add_pump_pulses=add_pump_pulses,
            simultaneous=simultaneous,
            plot=plot,
        )

    def state_tomography(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        method: Literal["measure", "execute"] = "measure",
        use_zvalues: bool = False,
        plot: bool = False,
    ) -> Result:
        """
        Conducts a state tomography experiment.

        Parameters
        ----------
        sequence : TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule
            Sequence to measure for each target.
        x90 : TargetMap[Waveform], optional
            π/2 pulse.
        initial_state : TargetMap[str], optional
            Initial state of each target.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        reset_awg_and_capunits : bool, optional
            Whether to reset the AWG and capture units before the experiment. Defaults to True.
        method : Literal["measure", "execute"], optional
            Measurement method. Defaults to "measure".
        use_zvalues : bool, optional
            Whether to use Z-values. Defaults to False.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        Result
            Results of the experiment.
        """
        return self.measurement_service.state_tomography(
            sequence=sequence,
            x90=x90,
            initial_state=initial_state,
            shots=shots,
            interval=interval,
            reset_awg_and_capunits=reset_awg_and_capunits,
            method=method,
            use_zvalues=use_zvalues,
            plot=plot,
        )

    def state_evolution_tomography(
        self,
        *,
        sequences: (
            Sequence[TargetMap[IQArray]]
            | Sequence[TargetMap[Waveform]]
            | Sequence[PulseSchedule]
        ),
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> Result:
        """
        Conducts a state evolution tomography experiment.

        Parameters
        ----------
        sequences : Sequence[TargetMap[IQArray]] | Sequence[TargetMap[Waveform]] | Sequence[PulseSchedule]
            Sequences to measure for each target.
        x90 : TargetMap[Waveform], optional
            π/2 pulse.
        initial_state : TargetMap[str], optional
            Initial state of each target.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to False.

        Returns
        -------
        Result
            Results of the experiment.
        """
        return self.measurement_service.state_evolution_tomography(
            sequences=sequences,
            x90=x90,
            initial_state=initial_state,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def pulse_tomography(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        x90: TargetMap[Waveform] | None = None,
        initial_state: TargetMap[str] | None = None,
        n_samples: int | None = 100,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        method: Literal["measure", "execute"] = "measure",
        plot: bool = True,
    ) -> Result:
        """
        Conducts a pulse tomography experiment.

        Parameters
        ----------
        sequence : TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule
            Waveforms to measure for each target.
        x90 : TargetMap[Waveform], optional
            π/2 pulse.
        initial_state : TargetMap[str], optional
            Initial state of each target.
        n_samples : int, optional
            Number of samples. Defaults to 100.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        method : Literal["measure", "execute"], optional
            Measurement method. Defaults to "measure".
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.

        Returns
        -------
        Result
            Results of the experiment.
        """
        return self.measurement_service.pulse_tomography(
            sequence=sequence,
            x90=x90,
            initial_state=initial_state,
            n_samples=n_samples,
            shots=shots,
            interval=interval,
            method=method,
            plot=plot,
        )

    def measure_population(
        self,
        sequence: TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule,
        *,
        fit_gmm: bool = False,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        """
        Measures the state populations of the target qubits.

        Parameters
        ----------
        sequence : TargetMap[IQArray] | TargetMap[Waveform] | PulseSchedule
            Sequence to measure for each target.
        fit_gmm : bool, optional
            Whether to fit the data with a Gaussian mixture model. Defaults to False
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]
            State probabilities and standard deviations.

        Examples
        --------
        >>> sequence = {
        ...     "Q00": ex.hpi_pulse["Q00"],
        ...     "Q01": ex.hpi_pulse["Q01"],
        ... }
        >>> result = ex.measure_population(sequence)
        """
        return self.measurement_service.measure_population(
            sequence=sequence,
            fit_gmm=fit_gmm,
            shots=shots,
            interval=interval,
        )

    def measure_population_dynamics(
        self,
        *,
        sequence: ParametricPulseSchedule | ParametricWaveformDict,
        params_list: Sequence | NDArray,
        fit_gmm: bool = False,
        xlabel: str = "Index",
        scatter_mode: str = "lines+markers",
        show_error: bool = True,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        """
        Measures the population dynamics of the target qubits.

        Parameters
        ----------
        sequence : ParametricPulseSchedule | ParametricWaveformDict
            Parametric sequence to measure.
        params_list : Sequence | NDArray
            List of parameters.
        fit_gmm : bool, optional
            Whether to fit the data with a Gaussian mixture model. Defaults to False.
        xlabel : str, optional
            Label of the x-axis. Defaults to "Index".
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]
            State probabilities and standard deviations.

        Examples
        --------
        >>> sequence = lambda x: {
        ...     "Q00": ex.hpi_pulse["Q00"].scaled(x),
        ...     "Q01": ex.hpi_pulse["Q01"].scaled(x),
        >>> params_list = np.linspace(0.5, 1.5, 100)
        >>> result = ex.measure_population_dynamics(sequence, params_list)
        """
        return self.measurement_service.measure_population_dynamics(
            sequence=sequence,
            params_list=params_list,
            fit_gmm=fit_gmm,
            xlabel=xlabel,
            scatter_mode=scatter_mode,
            show_error=show_error,
            shots=shots,
            interval=interval,
        )

    def measure_bell_state(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        control_basis: str = "Z",
        target_basis: str = "Z",
        zx90: PulseSchedule | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        plot_sequence: bool = False,
        plot_raw: bool = True,
        plot_mitigated: bool = True,
        save_image: bool = True,
        reset_awg_and_capunits: bool = True,
    ) -> Result:
        return self.measurement_service.measure_bell_state(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            control_basis=control_basis,
            target_basis=target_basis,
            zx90=zx90,
            shots=shots,
            interval=interval,
            plot=plot,
            plot_sequence=plot_sequence,
            plot_raw=plot_raw,
            plot_mitigated=plot_mitigated,
            save_image=save_image,
            reset_awg_and_capunits=reset_awg_and_capunits,
        )

    def bell_state_tomography(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        readout_mitigation: bool = True,
        zx90: PulseSchedule | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
        mle_fit: bool = True,
    ) -> Result:
        return self.measurement_service.bell_state_tomography(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            readout_mitigation=readout_mitigation,
            zx90=zx90,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
            mle_fit=mle_fit,
        )

    # calibration_service methods

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
        update_params: bool = True,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> ExperimentResult[AmplCalibData]:
        """
        Calibrates the default pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        return self.calibration_service.calibrate_default_pulse(
            targets=targets,
            pulse_type=pulse_type,
            duration=duration,
            ramptime=ramptime,
            n_points=n_points,
            n_rotations=n_rotations,
            r2_threshold=r2_threshold,
            update_params=update_params,
            plot=plot,
            shots=shots,
            interval=interval,
        )

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
        """
        Calibrates the π/2 pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        return self.calibration_service.calibrate_hpi_pulse(
            targets=targets,
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
        """
        Calibrates the π pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        return self.calibration_service.calibrate_pi_pulse(
            targets=targets,
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
        """
        Calibrates the default pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        pulse_type : Literal["pi", "hpi"]
            Type of the pulse to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        return self.calibration_service.calibrate_ef_pulse(
            targets=targets,
            pulse_type=pulse_type,
            duration=duration,
            ramptime=ramptime,
            n_points=n_points,
            n_rotations=n_rotations,
            r2_threshold=r2_threshold,
            plot=plot,
            shots=shots,
            interval=interval,
        )

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
        """
        Calibrates the ef π/2 pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        return self.calibration_service.calibrate_ef_hpi_pulse(
            targets=targets,
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
        """
        Calibrates the ef π pulse.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target qubits to calibrate.
        duration : float, optional
            Duration of the pulse.
        ramptime : float, optional
            Ramp time of the pulse.
        n_points : int, optional
            Number of points to sweep. Defaults to 20.
        n_rotations : int, optional
            Number of rotations. Defaults to 1.
        r2_threshold : float, optional
            Threshold for R² value. Defaults to 0.5.
        plot : bool, optional
            Whether to plot the measured signals. Defaults to True.
        shots : int, optional
            Number of shots. Defaults to CALIBRATION_SHOTS.
        interval : float, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.

        Returns
        -------
        ExperimentResult[AmplCalibData]
            Result of the experiment.
        """
        return self.calibration_service.calibrate_ef_pi_pulse(
            targets=targets,
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
        n_points: int = 20,
        n_rotations: int = 4,
        r2_threshold: float = 0.5,
        drag_coeff: float = DRAG_COEFF,
        use_stored_amplitude: bool = False,
        use_stored_beta: bool = False,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> Result:
        return self.calibration_service.calibrate_drag_amplitude(
            targets=targets,
            spectator_state=spectator_state,
            pulse_type=pulse_type,
            duration=duration,
            n_points=n_points,
            n_rotations=n_rotations,
            r2_threshold=r2_threshold,
            drag_coeff=drag_coeff,
            use_stored_amplitude=use_stored_amplitude,
            use_stored_beta=use_stored_beta,
            plot=plot,
            shots=shots,
            interval=interval,
        )

    def calibrate_drag_beta(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str | None = None,
        pulse_type: Literal["pi", "hpi"] = "hpi",
        beta_range: ArrayLike = np.linspace(-2.0, 2.0, 20),
        duration: float | None = None,
        n_turns: int = 1,
        degree: int = 3,
        plot: bool = True,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> Result:
        return self.calibration_service.calibrate_drag_beta(
            targets=targets,
            spectator_state=spectator_state,
            pulse_type=pulse_type,
            beta_range=beta_range,
            duration=duration,
            n_turns=n_turns,
            degree=degree,
            plot=plot,
            shots=shots,
            interval=interval,
        )

    def calibrate_drag_hpi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str | None = None,
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
    ) -> Result:
        return self.calibration_service.calibrate_drag_hpi_pulse(
            targets=targets,
            spectator_state=spectator_state,
            n_points=n_points,
            n_rotations=n_rotations,
            n_turns=n_turns,
            n_iterations=n_iterations,
            degree=degree,
            r2_threshold=r2_threshold,
            calibrate_beta=calibrate_beta,
            beta_range=beta_range,
            duration=duration,
            drag_coeff=drag_coeff,
            plot=plot,
            shots=shots,
            interval=interval,
        )

    def calibrate_drag_pi_pulse(
        self,
        targets: Collection[str] | str | None = None,
        *,
        spectator_state: str | None = None,
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
    ) -> Result:
        return self.calibration_service.calibrate_drag_pi_pulse(
            targets=targets,
            spectator_state=spectator_state,
            n_points=n_points,
            n_rotations=n_rotations,
            n_turns=n_turns,
            n_iterations=n_iterations,
            degree=degree,
            r2_threshold=r2_threshold,
            calibrate_beta=calibrate_beta,
            beta_range=beta_range,
            duration=duration,
            drag_coeff=drag_coeff,
            plot=plot,
            shots=shots,
            interval=interval,
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
        x180_margin: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        plot: bool = True,
    ) -> Result:
        return self.calibration_service.measure_cr_dynamics(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            time_range=time_range,
            ramptime=ramptime,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            echo=echo,
            control_state=control_state,
            x90=x90,
            x180=x180,
            ramp_type=ramp_type,
            x180_margin=x180_margin,
            shots=shots,
            interval=interval,
            reset_awg_and_capunits=reset_awg_and_capunits,
            plot=plot,
        )

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
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        plot: bool = True,
    ) -> Result:
        return self.calibration_service.cr_hamiltonian_tomography(
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
        x180_margin: float | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        plot: bool = True,
    ) -> Result:
        return self.calibration_service.update_cr_params(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            time_range=time_range,
            ramptime=ramptime,
            cr_amplitude=cr_amplitude,
            cr_phase=cr_phase,
            cancel_amplitude=cancel_amplitude,
            cancel_phase=cancel_phase,
            update_cr_phase=update_cr_phase,
            update_cancel_pulse=update_cancel_pulse,
            x90=x90,
            x180_margin=x180_margin,
            shots=shots,
            interval=interval,
            reset_awg_and_capunits=reset_awg_and_capunits,
            plot=plot,
        )

    def obtain_cr_params(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        time_range: ArrayLike | None = None,
        ramptime: float | None = None,
        cr_amplitude: float | None = None,
        n_iterations: int = 4,
        n_cycles: int = 2,
        n_points_per_cycle: int = 6,
        use_stored_params: bool = False,
        tolerance: float = 0.005e-3,
        adiabatic_safe_factor: float | None = None,
        max_amplitude: float = 1.0,
        max_time_range: float = 4096.0,
        x90: TargetMap[Waveform] | None = None,
        x180_margin: float | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        reset_awg_and_capunits: bool = True,
        plot: bool = True,
    ) -> Result:
        return self.calibration_service.obtain_cr_params(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            time_range=time_range,
            ramptime=ramptime,
            cr_amplitude=cr_amplitude,
            n_iterations=n_iterations,
            n_cycles=n_cycles,
            n_points_per_cycle=n_points_per_cycle,
            use_stored_params=use_stored_params,
            tolerance=tolerance,
            adiabatic_safe_factor=adiabatic_safe_factor,
            max_amplitude=max_amplitude,
            max_time_range=max_time_range,
            x90=x90,
            x180_margin=x180_margin,
            shots=shots,
            interval=interval,
            reset_awg_and_capunits=reset_awg_and_capunits,
            plot=plot,
        )

    def calibrate_zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        ramptime: float | None = None,
        duration: float | None = None,
        amplitude_range: ArrayLike | None = None,
        initial_state: str = "0",
        degree: int = 3,
        adiabatic_safe_factor: float | None = None,
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
    ) -> Result:
        return self.calibration_service.calibrate_zx90(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            ramptime=ramptime,
            duration=duration,
            amplitude_range=amplitude_range,
            initial_state=initial_state,
            degree=degree,
            adiabatic_safe_factor=adiabatic_safe_factor,
            max_amplitude=max_amplitude,
            rotary_multiple=rotary_multiple,
            use_drag=use_drag,
            duration_unit=duration_unit,
            duration_buffer=duration_buffer,
            n_repetitions=n_repetitions,
            x180=x180,
            x180_margin=x180_margin,
            use_zvalues=use_zvalues,
            store_params=store_params,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    # characterization_service methods

    def measure_readout_snr(
        self,
        targets: Collection[str] | str | None = None,
        *,
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        readout_duration: float | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> Result:
        return self.characterization_service.measure_readout_snr(
            targets=targets,
            initial_state=initial_state,
            readout_duration=readout_duration,
            readout_amplitudes=readout_amplitudes,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def sweep_readout_amplitude(
        self,
        targets: Collection[str] | str | None = None,
        *,
        amplitude_range: ArrayLike | None = None,
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        readout_duration: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> Result:
        return self.characterization_service.sweep_readout_amplitude(
            targets=targets,
            amplitude_range=amplitude_range,
            initial_state=initial_state,
            readout_duration=readout_duration,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def sweep_readout_duration(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = np.arange(128, 2048, 128),
        initial_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> Result:
        return self.characterization_service.sweep_readout_duration(
            targets=targets,
            time_range=time_range,
            initial_state=initial_state,
            readout_amplitudes=readout_amplitudes,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def chevron_pattern(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.05, 0.05, 51),
        time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
        frequencies: dict[str, float] | None = None,
        amplitudes: dict[str, float] | None = None,
        rabi_params: dict[str, RabiParam] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        return self.characterization_service.chevron_pattern(
            targets=targets,
            detuning_range=detuning_range,
            time_range=time_range,
            frequencies=frequencies,
            amplitudes=amplitudes,
            rabi_params=rabi_params,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def obtain_freq_rabi_relation(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = np.arange(0, 101, 4),
        rabi_level: Literal["ge", "ef"] = "ge",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[FreqRabiData]:
        return self.characterization_service.obtain_freq_rabi_relation(
            targets=targets,
            detuning_range=detuning_range,
            time_range=time_range,
            rabi_level=rabi_level,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def obtain_ampl_rabi_relation(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = DEFAULT_RABI_TIME_RANGE,
        amplitude_range: ArrayLike = np.linspace(0.01, 0.1, 10),
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> ExperimentResult[AmplRabiData]:
        return self.characterization_service.obtain_ampl_rabi_relation(
            targets=targets,
            time_range=time_range,
            amplitude_range=amplitude_range,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def calibrate_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = range(0, 101, 4),
        frequencies: dict[str, float] | None = None,
        amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> Result:
        return self.characterization_service.calibrate_control_frequency(
            targets=targets,
            detuning_range=detuning_range,
            time_range=time_range,
            frequencies=frequencies,
            amplitudes=amplitudes,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def calibrate_ef_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = np.arange(0, 101, 4),
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> Result:
        return self.characterization_service.calibrate_ef_control_frequency(
            targets=targets,
            detuning_range=detuning_range,
            time_range=time_range,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def calibrate_readout_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
        time_range: ArrayLike = range(0, 101, 4),
        readout_amplitudes: dict[str, float] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> Result:
        return self.characterization_service.calibrate_readout_frequency(
            targets=targets,
            detuning_range=detuning_range,
            time_range=time_range,
            readout_amplitudes=readout_amplitudes,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def t1_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
        xaxis_type: Literal["linear", "log"] = "log",
    ) -> ExperimentResult[T1Data]:
        return self.characterization_service.t1_experiment(
            targets=targets,
            time_range=time_range,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
            xaxis_type=xaxis_type,
        )

    def t2_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        n_cpmg: int = 1,
        pi_cpmg: Waveform | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
        xaxis_type: Literal["linear", "log"] = "log",
    ) -> ExperimentResult[T2Data]:
        return self.characterization_service.t2_experiment(
            targets=targets,
            time_range=time_range,
            n_cpmg=n_cpmg,
            pi_cpmg=pi_cpmg,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
            xaxis_type=xaxis_type,
        )

    def ramsey_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        detuning: float | None = None,
        second_rotation_axis: Literal["X", "Y"] = "Y",
        spectator_state: Literal["0", "1", "+", "-", "+i", "-i"] = "0",
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> ExperimentResult[RamseyData]:
        return self.characterization_service.ramsey_experiment(
            targets=targets,
            time_range=time_range,
            detuning=detuning,
            second_rotation_axis=second_rotation_axis,
            spectator_state=spectator_state,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def _simultaneous_measurement_coherence(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike | None = None,
        detuning: float | None = None,
        second_rotation_axis: Literal["X", "Y"] = "Y",
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
    ) -> dict[str, ExperimentResult]:
        return self.characterization_service._simultaneous_measurement_coherence(
            targets=targets,
            time_range=time_range,
            detuning=detuning,
            second_rotation_axis=second_rotation_axis,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def _stark_t1_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        stark_detuning: float | dict[str, float] | None = None,
        stark_amplitude: float | dict[str, float] | None = None,
        stark_ramptime: float | dict[str, float] | None = None,
        time_range: ArrayLike | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = False,
        xaxis_type: Literal["linear", "log"] = "log",
    ) -> ExperimentResult[T1Data]:
        return self.characterization_service._stark_t1_experiment(
            targets=targets,
            stark_detuning=stark_detuning,
            stark_amplitude=stark_amplitude,
            stark_ramptime=stark_ramptime,
            time_range=time_range,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
            xaxis_type=xaxis_type,
        )

    def _stark_ramsey_experiment(
        self,
        targets: Collection[str] | str | None = None,
        *,
        stark_detuning: float | dict[str, float] | None = None,
        stark_amplitude: float | dict[str, float] | None = None,
        stark_ramptime: float | dict[str, float] | None = None,
        time_range: ArrayLike | None = None,
        second_rotation_axis: Literal["X", "Y"] = "Y",
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        envelope_region: Literal["full", "flat"] = "full",
        plot: bool = True,
        save_image: bool = False,
    ) -> ExperimentResult[RamseyData]:
        return self.characterization_service._stark_ramsey_experiment(
            targets=targets,
            stark_detuning=stark_detuning,
            stark_amplitude=stark_amplitude,
            stark_ramptime=stark_ramptime,
            time_range=time_range,
            second_rotation_axis=second_rotation_axis,
            shots=shots,
            interval=interval,
            envelope_region=envelope_region,
            plot=plot,
            save_image=save_image,
        )

    def obtain_effective_control_frequency(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = np.arange(0, 10001, 100),
        detuning: float = 0.001,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> Result:
        return self.characterization_service.obtain_effective_control_frequency(
            targets=targets,
            time_range=time_range,
            detuning=detuning,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def jazz_experiment(
        self,
        target_qubit: str,
        spectator_qubit: str,
        *,
        time_range: ArrayLike = np.arange(0, 2001, 100),
        x90: TargetMap[Waveform] | None = None,
        x180: TargetMap[Waveform] | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ):
        return self.characterization_service.jazz_experiment(
            target_qubit=target_qubit,
            spectator_qubit=spectator_qubit,
            time_range=time_range,
            x90=x90,
            x180=x180,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def obtain_coupling_strength(
        self,
        target_qubit: str,
        spectator_qubit: str,
        *,
        time_range: ArrayLike = np.arange(0, 5001, 200),
        x90: TargetMap[Waveform] | None = None,
        x180: TargetMap[Waveform] | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
    ) -> Result:
        return self.characterization_service.obtain_coupling_strength(
            target_qubit=target_qubit,
            spectator_qubit=spectator_qubit,
            time_range=time_range,
            x90=x90,
            x180=x180,
            shots=shots,
            interval=interval,
            plot=plot,
        )

    def measure_electrical_delay(
        self,
        target: str,
        *,
        f_start: float | None = None,
        df: float | None = None,
        n_samples: int | None = None,
        readout_amplitude: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = 0,
        plot: bool = True,
        confirm: bool = True,
    ) -> float:
        return self.characterization_service.measure_electrical_delay(
            target=target,
            f_start=f_start,
            df=df,
            n_samples=n_samples,
            readout_amplitude=readout_amplitude,
            shots=shots,
            interval=interval,
            plot=plot,
            confirm=confirm,
        )

    def scan_resonator_frequencies(
        self,
        target: str,
        *,
        frequency_range: ArrayLike | None = None,
        readout_amplitude: float | None = None,
        electrical_delay: float | None = None,
        subrange_width: float = 0.3,
        peak_height: float | None = None,
        peak_distance: int | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = 0,
        plot: bool = True,
        save_image: bool = False,
    ) -> Result:
        return self.characterization_service.scan_resonator_frequencies(
            target=target,
            frequency_range=frequency_range,
            readout_amplitude=readout_amplitude,
            electrical_delay=electrical_delay,
            subrange_width=subrange_width,
            peak_height=peak_height,
            peak_distance=peak_distance,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def resonator_spectroscopy(
        self,
        target: str,
        *,
        frequency_range: ArrayLike | None = None,
        power_range: ArrayLike = np.arange(-60, 5, 5),
        electrical_delay: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = 0,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        return self.characterization_service.resonator_spectroscopy(
            target=target,
            frequency_range=frequency_range,
            power_range=power_range,
            electrical_delay=electrical_delay,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def measure_reflection_coefficient(
        self,
        target: str,
        *,
        center_frequency: float | None = None,
        df: float | None = None,
        frequency_width: float | None = None,
        readout_amplitude: float | None = None,
        electrical_delay: float | None = None,
        qubit_state: str = "0",
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        return self.characterization_service.measure_reflection_coefficient(
            target=target,
            center_frequency=center_frequency,
            df=df,
            frequency_width=frequency_width,
            readout_amplitude=readout_amplitude,
            electrical_delay=electrical_delay,
            qubit_state=qubit_state,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def scan_qubit_frequencies(
        self,
        target: str,
        *,
        frequency_range: ArrayLike | None = None,
        control_amplitude: float | None = None,
        readout_amplitude: float | None = None,
        readout_frequency: float | None = None,
        subrange_width: float | None = None,
        peak_height: float | None = None,
        peak_distance: int | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = False,
    ) -> Result:
        return self.characterization_service.scan_qubit_frequencies(
            target=target,
            frequency_range=frequency_range,
            control_amplitude=control_amplitude,
            readout_amplitude=readout_amplitude,
            readout_frequency=readout_frequency,
            subrange_width=subrange_width,
            peak_height=peak_height,
            peak_distance=peak_distance,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def estimate_control_amplitude(
        self,
        target: str,
        *,
        frequency_range: ArrayLike,
        control_amplitude: float | None = None,
        readout_amplitude: float | None = None,
        target_rabi_rate: float = DEFAULT_RABI_FREQUENCY,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ):
        return self.characterization_service.estimate_control_amplitude(
            target=target,
            frequency_range=frequency_range,
            control_amplitude=control_amplitude,
            readout_amplitude=readout_amplitude,
            target_rabi_rate=target_rabi_rate,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def qubit_spectroscopy(
        self,
        target: str,
        frequency_range: ArrayLike | None = None,
        power_range: ArrayLike = np.arange(-60, 0, 5),
        readout_amplitude: float | None = None,
        readout_frequency: float | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        return self.characterization_service.qubit_spectroscopy(
            target=target,
            frequency_range=frequency_range,
            power_range=power_range,
            readout_amplitude=readout_amplitude,
            readout_frequency=readout_frequency,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def measure_dispersive_shift(
        self,
        target: str,
        *,
        df: float | None = None,
        frequency_width: float | None = None,
        readout_amplitude: float | None = None,
        electrical_delay: float | None = None,
        threshold: float = 0.5,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        return self.characterization_service.measure_dispersive_shift(
            target=target,
            df=df,
            frequency_width=frequency_width,
            readout_amplitude=readout_amplitude,
            electrical_delay=electrical_delay,
            threshold=threshold,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def find_optimal_readout_frequency(
        self,
        target: str,
        *,
        df: float | None = None,
        frequency_width: float | None = None,
        readout_amplitude: float | None = None,
        electrical_delay: float | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        return self.characterization_service.find_optimal_readout_frequency(
            target=target,
            df=df,
            frequency_width=frequency_width,
            readout_amplitude=readout_amplitude,
            electrical_delay=electrical_delay,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def find_optimal_readout_amplitude(
        self,
        target: str,
        *,
        amplitude_range: ArrayLike | None = None,
        shots: int = CALIBRATION_SHOTS,
        interval: float = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        return self.characterization_service.find_optimal_readout_amplitude(
            target=target,
            amplitude_range=amplitude_range,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def ckp_sequence(
        self,
        target: str,
        qubit_initial_state: str | None = None,
        qubit_drive_detuning: float | None = None,
        qubit_pi_pulse: Waveform | None = None,
        qubit_drive_scale: float | None = None,
        resonator_drive_detuning: float | None = None,
        resonator_drive_amplitude: float | None = None,
        resonator_drive_duration: float | None = None,
        resonator_drive_ramptime: float | None = None,
    ) -> PulseSchedule:
        return self.characterization_service.ckp_sequence(
            target=target,
            qubit_initial_state=qubit_initial_state,
            qubit_drive_detuning=qubit_drive_detuning,
            qubit_pi_pulse=qubit_pi_pulse,
            qubit_drive_scale=qubit_drive_scale,
            resonator_drive_detuning=resonator_drive_detuning,
            resonator_drive_amplitude=resonator_drive_amplitude,
            resonator_drive_duration=resonator_drive_duration,
            resonator_drive_ramptime=resonator_drive_ramptime,
        )

    def ckp_measurement(
        self,
        target: str,
        qubit_initial_state: str,
        qubit_detuning_range: ArrayLike | None = None,
        qubit_pi_pulse: Waveform | None = None,
        qubit_drive_scale: float | None = None,
        resonator_detuning_range: ArrayLike | None = None,
        resonator_drive_amplitude: float | None = None,
        resonator_drive_duration: float | None = None,
        plot: bool = True,
        verbose: bool = False,
        save_image: bool = True,
    ) -> Result:
        return self.characterization_service.ckp_measurement(
            target=target,
            qubit_initial_state=qubit_initial_state,
            qubit_detuning_range=qubit_detuning_range,
            qubit_pi_pulse=qubit_pi_pulse,
            qubit_drive_scale=qubit_drive_scale,
            resonator_detuning_range=resonator_detuning_range,
            resonator_drive_amplitude=resonator_drive_amplitude,
            resonator_drive_duration=resonator_drive_duration,
            plot=plot,
            verbose=verbose,
            save_image=save_image,
        )

    def ckp_experiment(
        self,
        target: str,
        qubit_detuning_range: ArrayLike | None = None,
        qubit_pi_pulse: Waveform | None = None,
        qubit_drive_scale: float | None = None,
        resonator_detuning_range: ArrayLike | None = None,
        resonator_drive_amplitude: float | None = None,
        resonator_drive_duration: float | None = None,
        plot: bool = True,
        verbose: bool = False,
        save_image: bool = True,
    ) -> Result:
        return self.characterization_service.ckp_experiment(
            target=target,
            qubit_detuning_range=qubit_detuning_range,
            qubit_pi_pulse=qubit_pi_pulse,
            qubit_drive_scale=qubit_drive_scale,
            resonator_detuning_range=resonator_detuning_range,
            resonator_drive_amplitude=resonator_drive_amplitude,
            resonator_drive_duration=resonator_drive_duration,
            plot=plot,
            verbose=verbose,
            save_image=save_image,
        )

    def characterize_1q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        return self.characterization_service.characterize_1q(
            targets=targets,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
        )

    def characterize_2q(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int = CALIBRATION_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        return self.characterization_service.characterize_2q(
            targets=targets,
            shots=shots,
            interval=interval,
            plot=plot,
            save_image=save_image,
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
        seed: int | None = None,
    ) -> PulseSchedule:
        return self.benchmarking_service.rb_sequence(
            target=target,
            n=n,
            x90=x90,
            zx90=zx90,
            interleaved_waveform=interleaved_waveform,
            interleaved_clifford=interleaved_clifford,
            seed=seed,
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
        in_parallel: bool = False,
        xaxis_type: Literal["linear", "log"] | None = None,
        shots: int | None = None,
        interval: float | None = None,
        plot: bool = True,
        save_image: bool = True,
    ) -> Result:
        return self.benchmarking_service.randomized_benchmarking(
            targets=targets,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            zx90=zx90,
            in_parallel=in_parallel,
            xaxis_type=xaxis_type,
            shots=shots,
            interval=interval,
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
        return self.benchmarking_service.interleaved_randomized_benchmarking(
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
        return self.benchmarking_service.purity_benchmarking(
            targets=targets,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            zx90=zx90,
            in_parallel=in_parallel,
            xaxis_type=xaxis_type,
            shots=shots,
            interval=interval,
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
        return self.benchmarking_service.interleaved_purity_benchmarking(
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

    def optimize_x90(
        self,
        qubit: str,
        *,
        sigma0: float = 0.001,
        seed: int = 42,
        ftarget: float = 1e-3,
        timeout: int = 300,
    ) -> Waveform:
        return self.optimization_service.optimize_x90(
            qubit=qubit,
            sigma0=sigma0,
            seed=seed,
            ftarget=ftarget,
            timeout=timeout,
        )

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
        return self.optimization_service.optimize_drag_x90(
            qubit=qubit,
            duration=duration,
            sigma0=sigma0,
            seed=seed,
            ftarget=ftarget,
            timeout=timeout,
        )

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
        return self.optimization_service.optimize_pulse(
            qubit=qubit,
            pulse=pulse,
            x90=x90,
            target_state=target_state,
            sigma0=sigma0,
            seed=seed,
            ftarget=ftarget,
            timeout=timeout,
        )

    def optimize_zx90(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        objective_type: str = "st",
        optimize_method: str = "cma",
        update_cr_param: bool = True,
        opt_params: Collection[str] | None = None,
        seed: int | None = None,
        ftarget: float | None = None,
        timeout: int | None = None,
        maxiter: int | None = None,
        n_cliffords: int | None = None,
        n_trials: int | None = None,
        duration: float | None = None,
        ramptime: float | None = None,
        x180: TargetMap[Waveform] | None = None,
        x180_margin: float | None = None,
        shots: int = DEFAULT_SHOTS,
        interval: float = DEFAULT_INTERVAL,
    ) -> dict:
        return self.optimization_service.optimize_zx90(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            objective_type=objective_type,
            optimize_method=optimize_method,
            update_cr_param=update_cr_param,
            opt_params=opt_params,
            seed=seed,
            ftarget=ftarget,
            timeout=timeout,
            maxiter=maxiter,
            n_cliffords=n_cliffords,
            n_trials=n_trials,
            duration=duration,
            ramptime=ramptime,
            x180=x180,
            x180_margin=x180_margin,
            shots=shots,
            interval=interval,
        )
