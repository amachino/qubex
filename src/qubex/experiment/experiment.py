from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Collection, Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from typing_extensions import deprecated

from ..analysis.fitting import RabiParam
from ..backend import (
    Box,
    Chip,
    ControlParams,
    ControlSystem,
    DeviceController,
    ExperimentSystem,
    MixingUtil,
    QuantumSystem,
    Qubit,
    Resonator,
    StateManager,
    Target,
    TargetType,
)
from ..clifford import Clifford, CliffordGenerator
from ..measurement import Measurement, MeasureResult, StateClassifier
from ..measurement.measurement import (
    DEFAULT_CAPTURE_MARGIN,
    DEFAULT_CAPTURE_WINDOW,
    DEFAULT_CONFIG_DIR,
    DEFAULT_INTERVAL,
    DEFAULT_PARAMS_DIR,
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_RAMPTIME,
    DEFAULT_SHOTS,
)
from ..pulse import (
    Blank,
    CrossResonance,
    Drag,
    FlatTop,
    PulseArray,
    PulseSchedule,
    VirtualZ,
    Waveform,
)
from ..typing import TargetMap
from ..version import get_package_version
from . import experiment_tool
from .calibration_note import CalibrationNote
from .experiment_constants import (
    CALIBRATION_VALID_DAYS,
    CLASSIFIER_DIR,
    DRAG_HPI_DURATION,
    DRAG_PI_DURATION,
    HPI_DURATION,
    HPI_RAMPTIME,
    RABI_FREQUENCY,
    RABI_TIME_RANGE,
    SYSTEM_NOTE_PATH,
    USER_NOTE_PATH,
)
from .experiment_exceptions import CalibrationMissingError
from .experiment_note import ExperimentNote
from .experiment_record import ExperimentRecord
from .experiment_result import ExperimentResult, RabiData
from .experiment_util import ExperimentUtil
from .mixin import (
    BenchmarkingMixin,
    CalibrationMixin,
    CharacterizationMixin,
    MeasurementMixin,
)

console = Console()


class Experiment(
    BenchmarkingMixin,
    CharacterizationMixin,
    CalibrationMixin,
    MeasurementMixin,
):
    """
    Class representing an experiment.

    Parameters
    ----------
    chip_id : str
        Identifier of the quantum chip.
    muxes : Collection[str | int], optional
        Mux labels to use in the experiment. Defaults to None.
    qubits : Collection[str | int], optional
        Qubit labels to use in the experiment. Defaults to None.
    exclude_qubits : Collection[str | int], optional
        Qubit labels to exclude in the experiment. Defaults to None.
    config_dir : str, optional
        Directory of the configuration files. Defaults to DEFAULT_CONFIG_DIR.
    params_dir : str, optional
        Directory of the parameter files. Defaults to DEFAULT_PARAMS_DIR.
    connect_devices : bool, optional
        Whether to connect the devices. Defaults to True.
    drag_hpi_duration : int, optional
        Duration of the DRAG HPI pulse. Defaults to DRAG_HPI_DURATION.
    drag_pi_duration : int, optional
        Duration of the DRAG π pulse. Defaults to DRAG_PI_DURATION.
    capture_window : int, optional
        Capture window. Defaults to DEFAULT_CAPTURE_WINDOW.
    readout_duration : int, optional
        Readout duration. Defaults to DEFAULT_READOUT_DURATION.
    classifier_dir : Path | str, optional
        Directory of the state classifiers. Defaults to CLASSIFIER_DIR.
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
        config_dir: str = DEFAULT_CONFIG_DIR,
        params_dir: str = DEFAULT_PARAMS_DIR,
        calib_note_path: Path | str | None = None,
        connect_devices: bool = True,
        drag_hpi_duration: int = DRAG_HPI_DURATION,
        drag_pi_duration: int = DRAG_PI_DURATION,
        control_window: int | None = None,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        capture_margin: int = DEFAULT_CAPTURE_MARGIN,
        readout_duration: int = DEFAULT_READOUT_DURATION,
        classifier_dir: Path | str = CLASSIFIER_DIR,
        classifier_type: Literal["kmeans", "gmm"] = "gmm",
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] = "ge-cr-cr",
        calibration_valid_days: int = CALIBRATION_VALID_DAYS,
    ):
        self._load_config(
            chip_id=chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
        )
        qubits = self._create_qubit_labels(
            muxes=muxes,
            qubits=qubits,
            exclude_qubits=exclude_qubits,
        )
        self._chip_id: Final = chip_id
        self._qubits: Final = qubits
        self._config_dir: Final = config_dir
        self._params_dir: Final = params_dir
        self._drag_hpi_duration: Final = drag_hpi_duration
        self._drag_pi_duration: Final = drag_pi_duration
        self._control_window: Final = control_window
        self._capture_window: Final = capture_window
        self._capture_margin: Final = capture_margin
        self._readout_duration: Final = readout_duration
        self._classifier_dir: Final = classifier_dir
        self._classifier_type: Final = classifier_type
        self._configuration_mode: Final = configuration_mode
        self._calibration_valid_days: Final = calibration_valid_days
        self._measurement = Measurement(
            chip_id=chip_id,
            qubits=qubits,
            config_dir=self._config_dir,
            params_dir=self._params_dir,
            connect_devices=connect_devices,
            configuration_mode=configuration_mode,
        )
        self._clifford_generator: CliffordGenerator | None = None
        self._user_note: Final = ExperimentNote(file_path=USER_NOTE_PATH)
        self._system_note: Final = ExperimentNote(file_path=SYSTEM_NOTE_PATH)
        self._calib_note: Final = CalibrationNote(
            chip_id=chip_id,
            file_path=calib_note_path,
        )
        self.print_environment(verbose=False)
        self._load_classifiers()

    def _load_classifiers(self):
        for qubit in self.qubit_labels:
            classifier_path = self.classifier_dir / self.chip_id / f"{qubit}.pkl"
            if classifier_path.exists():
                self._measurement.classifiers[qubit] = StateClassifier.load(  # type: ignore
                    classifier_path
                )

    def _load_config(
        self,
        chip_id: str,
        config_dir: str,
        params_dir: str,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"],
    ):
        """Load the configuration files."""
        self.state_manager.load(
            chip_id=chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
        )

    def _create_qubit_labels(
        self,
        muxes: Collection[str | int] | None,
        qubits: Collection[str | int] | None,
        exclude_qubits: Collection[str | int] | None,
    ) -> list[str]:
        """Create the list of qubit labels."""
        if muxes is None and qubits is None:
            return []
        quantum_system = self.quantum_system
        qubit_labels = []
        if muxes is not None:
            for mux in muxes:
                labels = [
                    qubit.label for qubit in quantum_system.get_qubits_in_mux(mux)
                ]
                qubit_labels.extend(labels)
        if qubits is not None:
            for qubit in qubits:
                qubit_labels.append(quantum_system.get_qubit(qubit).label)
        if exclude_qubits is not None:
            for qubit in exclude_qubits:
                label = quantum_system.get_qubit(qubit).label
                if label in qubit_labels:
                    qubit_labels.remove(label)
        qubit_labels = sorted(list(set(qubit_labels)))

        available_qubits = [
            target.qubit for target in self.experiment_system.ge_targets
        ]
        unavailable_qubits = [
            qubit for qubit in qubit_labels if qubit not in available_qubits
        ]
        if len(unavailable_qubits) > 0:
            print(f"Unavailable qubits: {unavailable_qubits}")

        qubit_labels = [qubit for qubit in qubit_labels if qubit in available_qubits]
        return qubit_labels

    def print_environment(self, verbose: bool = True):
        """Print the environment information."""
        print("========================================")
        print("date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("python:", sys.version.split()[0])
        if verbose:
            print("numpy:", get_package_version("numpy"))
            print("quel_ic_config:", get_package_version("quel_ic_config"))
            print("quel_clock_master:", get_package_version("quel_clock_master"))
            print("qubecalib:", get_package_version("qubecalib"))
        print("qubex:", get_package_version("qubex"))
        print("env:", sys.prefix)
        print("config:", self.config_path)
        print("params:", self.params_path)
        print("chip:", self.chip_id, f"({self.chip.name})")
        print("qubits:", self.qubit_labels)
        print("muxes:", self.mux_labels)
        print("boxes:", self.box_ids)
        print("========================================")

    def print_boxes(self):
        """Print the box information."""
        table = Table(header_style="bold")
        table.add_column("ID", justify="left")
        table.add_column("NAME", justify="left")
        table.add_column("ADDRESS", justify="left")
        table.add_column("ADAPTER", justify="left")
        for box in self.boxes.values():
            table.add_row(box.id, box.name, box.address, box.adapter)
        console.print(table)

    @property
    def tool(self):
        return experiment_tool

    @property
    def util(self):
        return ExperimentUtil

    @property
    def measurement(self) -> Measurement:
        return self._measurement

    @property
    def state_manager(self) -> StateManager:
        return StateManager.shared()

    @property
    def experiment_system(self) -> ExperimentSystem:
        return self.state_manager.experiment_system

    @property
    def quantum_system(self) -> QuantumSystem:
        return self.experiment_system.quantum_system

    @property
    def control_system(self) -> ControlSystem:
        return self.experiment_system.control_system

    @property
    def device_controller(self) -> DeviceController:
        return self.state_manager.device_controller

    @property
    def params(self) -> ControlParams:
        return self.experiment_system.control_params

    @property
    def chip(self) -> Chip:
        return self.experiment_system.chip

    @property
    def chip_id(self) -> str:
        return self._chip_id

    @property
    def qubit_labels(self) -> list[str]:
        return self._qubits

    @property
    def mux_labels(self) -> list[str]:
        mux_set = set()
        for qubit in self.qubit_labels:
            mux = self.experiment_system.get_mux_by_qubit(qubit)
            mux_set.add(mux.label)
        return sorted(list(mux_set))

    @property
    def qubits(self) -> dict[str, Qubit]:
        return {
            qubit.label: qubit
            for qubit in self.experiment_system.qubits
            if qubit.label in self.qubit_labels
        }

    @property
    def resonators(self) -> dict[str, Resonator]:
        return {
            resonator.qubit: resonator
            for resonator in self.experiment_system.resonators
            if resonator.qubit in self.qubit_labels
        }

    @property
    def targets(self) -> dict[str, Target]:
        return {
            target.label: target
            for target in self.experiment_system.targets
            if target.is_related_to_qubits(self.qubit_labels)
        }

    @property
    def available_targets(self) -> dict[str, Target]:
        return {
            label: target
            for label, target in self.targets.items()
            if target.is_available
        }

    @property
    def ge_targets(self) -> dict[str, Target]:
        return {
            label: target
            for label, target in self.available_targets.items()
            if target.is_ge
        }

    @property
    def ef_targets(self) -> dict[str, Target]:
        return {
            label: target
            for label, target in self.available_targets.items()
            if target.is_ef
        }

    @property
    def cr_targets(self) -> dict[str, Target]:
        return {
            label: target
            for label, target in self.available_targets.items()
            if target.is_cr
        }

    @property
    def boxes(self) -> dict[str, Box]:
        boxes = self.experiment_system.get_boxes_for_qubits(self.qubit_labels)
        return {box.id: box for box in boxes}

    @property
    def box_ids(self) -> list[str]:
        return list(self.boxes.keys())

    @property
    def config_path(self) -> str:
        return str(Path(self._config_dir).resolve())

    @property
    def params_path(self) -> str:
        return str(Path(self._params_dir).resolve())

    @property
    def calib_note(self) -> CalibrationNote:
        return self._calib_note

    @property
    @deprecated("This property is deprecated. Use `calib_note` instead.")
    def system_note(self) -> ExperimentNote:
        return self._system_note

    @property
    def control_window(self) -> int | None:
        return self._control_window

    @property
    def capture_window(self) -> int:
        return self._capture_window

    @property
    def capture_margin(self) -> int:
        return self._capture_margin

    @property
    def readout_duration(self) -> int:
        return self._readout_duration

    @property
    def note(self) -> ExperimentNote:
        return self._user_note

    @property
    def drag_hpi_duration(self) -> int:
        return self._drag_hpi_duration

    @property
    def drag_pi_duration(self) -> int:
        return self._drag_pi_duration

    @property
    def hpi_pulse(self) -> dict[str, Waveform]:
        result = {}
        for target in self.ge_targets:
            param = self.calib_note.get_hpi_param(
                target,
                valid_days=self._calibration_valid_days,
            )
            if param is not None and None not in param.values():
                result[target] = FlatTop(
                    duration=param["duration"],
                    amplitude=param["amplitude"],
                    tau=param["tau"],
                )
            else:
                result[target] = FlatTop(
                    duration=HPI_DURATION,
                    amplitude=self.params.get_control_amplitude(target),
                    tau=HPI_RAMPTIME,
                )
        return result

    @property
    def pi_pulse(self) -> dict[str, Waveform]:
        result = {}
        for target in self.ge_targets:
            param = self.calib_note.get_pi_param(
                target,
                valid_days=self._calibration_valid_days,
            )
            if param is not None and None not in param.values():
                result[target] = FlatTop(
                    duration=param["duration"],
                    amplitude=param["amplitude"],
                    tau=param["tau"],
                )
        return result

    @property
    def drag_hpi_pulse(self) -> dict[str, Waveform]:
        result = {}
        for target in self.ge_targets:
            param = self.calib_note.get_drag_hpi_param(
                target,
                valid_days=self._calibration_valid_days,
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
        result = {}
        for target in self.ge_targets:
            param = self.calib_note.get_drag_pi_param(
                target,
                valid_days=self._calibration_valid_days,
            )
            if param is not None and None not in param.values():
                result[target] = Drag(
                    duration=param["duration"],
                    amplitude=param["amplitude"],
                    beta=param["beta"],
                )
        return result

    @property
    def ef_hpi_pulse(self) -> dict[str, Waveform]:
        result = {}
        for target in self.ef_targets:
            param = self.calib_note.get_hpi_param(
                target,
                valid_days=self._calibration_valid_days,
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
        result = {}
        for target in self.ef_targets:
            param = self.calib_note.get_pi_param(
                target,
                valid_days=self._calibration_valid_days,
            )
            if param is not None and None not in param.values():
                result[target] = FlatTop(
                    duration=param["duration"],
                    amplitude=param["amplitude"],
                    tau=param["tau"],
                )
        return result

    @property
    def rabi_params(self) -> dict[str, RabiParam]:
        result = {}
        for target in self.ge_targets | self.ef_targets:
            param = self.calib_note.get_rabi_param(
                target,
                valid_days=self._calibration_valid_days,
            )
            if param is not None and None not in param.values():
                result[target] = RabiParam(
                    target=param.get("target"),
                    frequency=param.get("frequency"),
                    amplitude=param.get("amplitude"),
                    phase=param.get("phase"),
                    offset=param.get("offset"),
                    noise=param.get("noise"),
                    angle=param.get("angle"),
                    r2=param.get("r2"),
                )
        return result

    @property
    def ge_rabi_params(self) -> dict[str, RabiParam]:
        return {
            target: param
            for target, param in self.rabi_params.items()
            if self.targets[target].is_ge
        }

    @property
    def ef_rabi_params(self) -> dict[str, RabiParam]:
        return {
            Target.ge_label(target): param
            for target, param in self.rabi_params.items()
            if self.targets[target].is_ef
        }

    @property
    def classifier_dir(self) -> Path:
        return Path(self._classifier_dir)

    @property
    def classifier_type(self) -> Literal["kmeans", "gmm"]:
        return self._classifier_type

    @property
    def classifiers(self) -> TargetMap[StateClassifier]:
        return self._measurement.classifiers

    @property
    def state_centers(self) -> dict[str, dict[int, complex]]:
        result = {}
        for target in self.qubit_labels:
            param = self.calib_note.get_state_param(
                target,
                valid_days=self._calibration_valid_days,
            )
            if param is not None:
                result[target] = {
                    int(state): complex(center[0], center[1])
                    for state, center in param["centers"].items()
                }
        return result

    @property
    def clifford_generator(self) -> CliffordGenerator:
        if self._clifford_generator is None:
            self._clifford_generator = CliffordGenerator()
        return self._clifford_generator

    @property
    def clifford(self) -> dict[str, Clifford]:
        return self.clifford_generator.cliffords

    @property
    def configuration_mode(self) -> Literal["ge-ef-cr", "ge-cr-cr"]:
        return self._configuration_mode

    def validate_rabi_params(
        self,
        targets: Collection[str] | None = None,
    ):
        if len(self.rabi_params) == 0:
            raise ValueError("Rabi parameters are not stored.")
        if targets is not None:
            for target in targets:
                if target not in self.rabi_params:
                    raise ValueError(f"Rabi parameters for {target} are not stored.")
        if targets is not None:
            for target in targets:
                if target not in self.rabi_params:
                    raise ValueError(f"Rabi parameters for {target} are not stored.")

    def store_rabi_params(
        self,
        rabi_params: dict[str, RabiParam],
        r2_threshold: float = 0.5,
    ):
        not_stored = []
        for label, rabi_param in rabi_params.items():
            if rabi_param.r2 < r2_threshold:
                not_stored.append(label)
            else:
                self.calib_note.update_rabi_param(
                    label,
                    {
                        "target": rabi_param.target,
                        "frequency": rabi_param.frequency,
                        "amplitude": rabi_param.amplitude,
                        "phase": rabi_param.phase,
                        "offset": rabi_param.offset,
                        "noise": rabi_param.noise,
                        "angle": rabi_param.angle,
                        "r2": rabi_param.r2,
                    },
                )

        if len(not_stored) > 0:
            print(f"Rabi parameters are not stored for qubits: {not_stored}")

    def get_hpi_pulse(
        self,
        target: str,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        """
        Get the π/2 pulse for the given target.
        """
        param = self.calib_note.get_hpi_param(
            target,
            valid_days=valid_days or self._calibration_valid_days,
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
                amplitude=self.params.get_control_amplitude(target),
                tau=HPI_RAMPTIME,
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
        param = self.calib_note.get_pi_param(
            target,
            valid_days=valid_days or self._calibration_valid_days,
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
        """
        Get the DRAG π/2 pulse for the given target.
        """
        param = self.calib_note.get_drag_hpi_param(
            target,
            valid_days=valid_days or self._calibration_valid_days,
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
        """
        Get the DRAG π pulse for the given target.
        """
        param = self.calib_note.get_drag_pi_param(
            target,
            valid_days=valid_days or self._calibration_valid_days,
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

    def get_spectators(
        self,
        qubit: str,
        in_same_mux: bool = False,
    ) -> list[Qubit]:
        return self.quantum_system.get_spectator_qubits(qubit, in_same_mux=in_same_mux)

    def get_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray:
        return self.measurement.get_confusion_matrix(targets)

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray:
        return self.measurement.get_inverse_confusion_matrix(targets)

    def check_status(self):
        # link status
        link_status = self._measurement.check_link_status(self.box_ids)
        if link_status["status"]:
            print("Link status: OK")
        else:
            print("Link status: NG")
        print(link_status["links"])

        # clock status
        clock_status = self._measurement.check_clock_status(self.box_ids)
        if clock_status["status"]:
            print("Clock status: OK")
        else:
            print("Clock status: NG")
        print(clock_status["clocks"])

        # config status
        config_status = self.state_manager.is_synced(box_ids=self.box_ids)
        if config_status:
            print("Config status: OK")
        else:
            print("Config status: NG")
        print(self.state_manager.device_settings)

    def linkup(
        self,
        box_ids: list[str] | None = None,
        noise_threshold: int = 500,
    ) -> None:
        if box_ids is None:
            box_ids = self.box_ids
        self._measurement.linkup(box_ids, noise_threshold=noise_threshold)

    def resync_clocks(
        self,
        box_ids: list[str] | None = None,
    ) -> None:
        if box_ids is None:
            box_ids = self.box_ids
        self.device_controller.resync_clocks(box_ids)

    def configure(
        self,
        box_ids: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
        mode: Literal["ge-ef-cr", "ge-cr-cr"] | None = None,
    ):
        if isinstance(box_ids, str):
            box_ids = [box_ids]
        if isinstance(exclude, str):
            exclude = [exclude]
        if mode is None:
            mode = self.configuration_mode
        self.state_manager.load(
            chip_id=self.chip_id,
            config_dir=self.config_path,
            params_dir=self.params_path,
            targets_to_exclude=exclude,
            configuration_mode=mode,
        )
        self.state_manager.push(
            box_ids=box_ids or self.box_ids,
        )

    def reload(self):
        self._measurement.reload()

    def reset_awg_and_capunits(
        self,
        box_ids: str | Collection[str] | None = None,
    ):
        if box_ids is None:
            box_ids = self.box_ids
        self.device_controller.initialize_awg_and_capunits(box_ids)

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
        try:
            qubit_label = Target.qubit_label(label)
        except ValueError:
            raise ValueError(f"Invalid target label: {label}")

        port = self.control_system.get_port(box_id, port_number)
        channel = port.channels[channel_number]
        qubit = self.qubits[qubit_label]
        target = Target.new_target(
            label=label,
            frequency=frequency,
            object=qubit,
            channel=channel,  # type: ignore
            type=target_type,
        )
        self.experiment_system.add_target(target)
        self.device_controller.define_target(
            target_name=target.label,
            channel_name=target.channel.id,
            target_frequency=target.frequency,
        )
        if update_lsi:
            fnco, _ = MixingUtil.calc_fnco(
                f=frequency * 1e9,
                ssb="L",
                lo=port.lo_freq,  # type: ignore
                cnco=port.cnco_freq,
            )
            port.channels[channel_number].fnco_freq = fnco
            self.state_manager.push(box_ids=[box_id])

    @contextmanager
    def modified_frequencies(
        self,
        frequencies: dict[str, float] | None,
    ):
        if frequencies is None:
            yield
        else:
            with self.state_manager.modified_frequencies(frequencies):
                yield

    def save_calib_note(
        self,
        file_path: Path | str | None = None,
    ):
        self.calib_note.save(file_path=file_path)

    @deprecated("Use `calib_note.save()` instead.")
    def save_defaults(self):
        self._system_note.save()

    @deprecated("Use `calib_note.clear()` instead.")
    def clear_defaults(self):
        self._system_note.clear()

    @deprecated("")
    def delete_defaults(self):
        if Confirm.ask("Delete the default params?"):
            self._system_note.clear()
            self._system_note.save()

    def load_record(
        self,
        name: str,
    ) -> ExperimentRecord:
        record = ExperimentRecord.load(name)
        print(f"ExperimentRecord `{name}` is loaded.\n")
        print(f"description: {record.description}")
        print(f"created_at: {record.created_at}")
        return record

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
            Target labels to check the noise. Defaults to None.
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
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        result = self.measurement.measure_noise(targets, duration)
        for data in result.data.values():
            if plot:
                data.plot()
        return result

    def check_waveform(
        self,
        targets: Collection[str] | str | None = None,
        *,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        capture_window: int = DEFAULT_CAPTURE_WINDOW,
        readout_duration: int = DEFAULT_READOUT_DURATION,
        readout_amplitude: float | None = None,
        add_pump_pulses: bool = False,
        plot: bool = True,
    ) -> MeasureResult:
        """
        Checks the readout waveforms of the given targets.

        Parameters
        ----------
        targets : Collection[str] | str, optional
            Target labels to check the waveforms.
        shots : int, optional
            Number of shots. Defaults to DEFAULT_SHOTS.
        interval : int, optional
            Interval between shots. Defaults to DEFAULT_INTERVAL.
        capture_window : int, optional
            Capture window. Defaults to DEFAULT_CAPTURE_WINDOW.
        readout_duration : int, optional
            Readout duration. Defaults to DEFAULT_READOUT_DURATION.
        readout_amplitude : float, optional
            Readout amplitude. Defaults to None.
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
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)

        result = self.measure(
            sequence={target: np.zeros(0) for target in targets},
            shots=shots,
            interval=interval,
            capture_window=capture_window,
            readout_duration=readout_duration,
            readout_amplitudes={target: readout_amplitude for target in targets}
            if readout_amplitude is not None
            else None,
            add_pump_pulses=add_pump_pulses,
        )
        if plot:
            result.plot()
        return result

    def check_rabi(
        self,
        targets: Collection[str] | str | None = None,
        *,
        time_range: ArrayLike = RABI_TIME_RANGE,
        shots: int = DEFAULT_SHOTS,
        interval: int = DEFAULT_INTERVAL,
        store_params: bool = False,
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
        if targets is None:
            targets = self.qubit_labels
        elif isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)
        time_range = np.asarray(time_range)
        ampl = self.params.control_amplitude
        amplitudes = {target: ampl[target] for target in targets}
        result = self.rabi_experiment(
            amplitudes=amplitudes,
            time_range=time_range,
            shots=shots,
            interval=interval,
            store_params=store_params,
            plot=plot,
        )
        return result

    def calc_control_amplitude(
        self,
        target: str,
        rabi_rate: float,
        *,
        rabi_amplitude_ratio: float | None = None,
    ) -> float:
        if rabi_amplitude_ratio is None:
            rabi_param = self.rabi_params.get(target)
            default_amplitude = self.params.control_amplitude.get(target)

            if rabi_param is None:
                raise ValueError(f"Rabi parameters for {target} are not stored.")
            if default_amplitude is None:
                raise ValueError(f"Control amplitude for {target} is not defined.")

            rabi_amplitude_ratio = rabi_param.frequency / default_amplitude

        return rabi_rate / rabi_amplitude_ratio

    def calc_control_amplitudes(
        self,
        rabi_rate: float = RABI_FREQUENCY,
        *,
        current_amplitudes: dict[str, float] | None = None,
        current_rabi_params: dict[str, RabiParam] | None = None,
        print_result: bool = True,
    ) -> dict[str, float]:
        current_rabi_params = current_rabi_params or self.rabi_params

        if current_rabi_params is None:
            raise ValueError("Rabi parameters are not stored.")

        if current_amplitudes is None:
            current_amplitudes = {}
            default_ampl = self.params.control_amplitude
            for target in current_rabi_params:
                if self.targets[target].is_ge:
                    current_amplitudes[target] = default_ampl[target]
                elif self.targets[target].is_ef:
                    qubit = Target.qubit_label(target)
                    current_amplitudes[target] = default_ampl[qubit] / np.sqrt(2)
                else:
                    raise ValueError("Invalid target.")

        amplitudes = {
            target: current_amplitudes[target]
            * rabi_rate
            / current_rabi_params[target].frequency
            for target in current_rabi_params
        }

        if print_result:
            print(f"Control amplitude for rabi rate {rabi_rate * 1e3:.3f} MHz\n")
            for target, amplitude in amplitudes.items():
                print(f"{target}: {amplitude:.6f}")

        return amplitudes

    def calc_rabi_rate(
        self,
        target: str,
        control_amplitude,
    ) -> float:
        # TODO: Support ef targets
        default_amplitude = self.params.control_amplitude.get(target)
        if default_amplitude is None:
            raise ValueError(f"Control amplitude for {target} is not defined.")

        rabi_param = self.rabi_params.get(target)
        if rabi_param is None:
            raise ValueError(f"Rabi parameters for {target} are not stored.")

        return control_amplitude * rabi_param.frequency / default_amplitude

    def calc_rabi_rates(
        self,
        control_amplitude: float = 1.0,
        *,
        print_result: bool = True,
    ) -> dict[str, float]:
        default_ampl = self.params.control_amplitude

        rabi_rates = {
            target: control_amplitude * rabi_param.frequency / default_ampl[target]
            for target, rabi_param in self.rabi_params.items()
        }

        if print_result:
            print(f"Rabi rate for control amplitude {control_amplitude}\n")
            for target, rabi_rate in rabi_rates.items():
                print(f"{target}: {rabi_rate * 1e3:.3f} MHz")

        return rabi_rates

    def readout(
        self,
        target: str,
        /,
        *,
        amplitude: float | None = None,
        duration: float = DEFAULT_READOUT_DURATION,
        ramptime: float = DEFAULT_READOUT_RAMPTIME,
        capture_window: float = DEFAULT_CAPTURE_WINDOW,
        capture_margin: float = DEFAULT_CAPTURE_MARGIN,
    ) -> Waveform:
        return PulseArray(
            [
                Blank(capture_margin),
                self.measurement.readout_pulse(
                    target=target,
                    duration=duration,
                    amplitude=amplitude,
                    tau=ramptime,
                ).padded(
                    total_duration=capture_window,
                    pad_side="right",
                ),
            ]
        )

    def x90(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
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
        return self.x90(target, valid_days=valid_days).scaled(-1)

    def x180(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
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
        return self.x90(target, valid_days=valid_days).shifted(np.pi / 2)

    def y90m(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.x90(target, valid_days=valid_days).shifted(-np.pi / 2)

    def y180(
        self,
        target: str,
        /,
        *,
        valid_days: int | None = None,
    ) -> Waveform:
        return self.x180(target, valid_days=valid_days).shifted(np.pi / 2)

    def z90(
        self,
    ) -> VirtualZ:
        return VirtualZ(np.pi / 2)

    def z180(
        self,
    ) -> VirtualZ:
        return VirtualZ(np.pi)

    def hadamard(
        self,
        target: str,
        *,
        decomposition: Literal["Z180-Y90", "Y90-X180"] = "Z180-Y90",
    ) -> PulseArray:
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
        cr_label = f"{control_qubit}-{target_qubit}"
        cr_param = self.calib_note.get_cr_param(
            cr_label,
            valid_days=self._calibration_valid_days,
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

    def cnot(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        zx90: PulseSchedule | None = None,
        x90: Waveform | None = None,
    ) -> PulseSchedule:
        cr_label = f"{control_qubit}-{target_qubit}"

        if cr_label in self.calib_note.cr_params:
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
    ) -> PulseSchedule:
        return self.cnot(
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            zx90=zx90,
            x90=x90,
        )

    def cz(
        self,
        control_qubit: str,
        target_qubit: str,
        *,
        zx90: PulseSchedule | None = None,
        x90: Waveform | None = None,
    ) -> PulseSchedule:
        cr_label = f"{control_qubit}-{target_qubit}"

        if cr_label in self.calib_note.cr_params:
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
