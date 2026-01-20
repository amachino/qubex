from __future__ import annotations

import json
import logging
import sys
from collections.abc import Collection
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

from numpy.typing import NDArray
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from typing_extensions import deprecated

from qubex.backend import (
    Box,
    Chip,
    ConfigLoader,
    ControlParams,
    ControlSystem,
    DeviceController,
    ExperimentSystem,
    MixingUtil,
    QuantumSystem,
    Qubit,
    Resonator,
    SystemManager,
    Target,
    TargetType,
)
from qubex.clifford.clifford import Clifford
from qubex.clifford.clifford_generator import CliffordGenerator
from qubex.measurement import (
    Measurement,
    StateClassifier,
)
from qubex.measurement.measurement import (
    DEFAULT_READOUT_DURATION,
    DEFAULT_READOUT_POST_MARGIN,
    DEFAULT_READOUT_PRE_MARGIN,
)
from qubex.typing import TargetMap
from qubex.version import get_package_version

from . import experiment_tool
from .calibration_note import CalibrationNote
from .experiment_constants import (
    CALIBRATION_VALID_DAYS,
    CLASSIFIER_DIR,
    DRAG_HPI_DURATION,
    DRAG_PI_DURATION,
    PROPERTY_DIR,
    SYSTEM_NOTE_PATH,
    USER_NOTE_PATH,
)
from .experiment_exceptions import CalibrationMissingError
from .experiment_note import ExperimentNote
from .experiment_record import ExperimentRecord
from .experiment_util import ExperimentUtil
from .rabi_param import RabiParam

logger = logging.getLogger(__name__)

console = Console()


class ExperimentContext:
    """
    Class representing the context of an experiment.

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
        calibration_valid_days: int | None = None,
        drag_hpi_duration: float | None = None,
        drag_pi_duration: float | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        property_dir: Path | str | None = None,
        classifier_dir: Path | str | None = None,
        classifier_type: Literal["kmeans", "gmm"] | None = None,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"] | None = None,
        mock_mode: bool | None = None,
    ):
        if calibration_valid_days is None:
            calibration_valid_days = CALIBRATION_VALID_DAYS
        if drag_hpi_duration is None:
            drag_hpi_duration = DRAG_HPI_DURATION
        if drag_pi_duration is None:
            drag_pi_duration = DRAG_PI_DURATION
        if readout_duration is None:
            readout_duration = DEFAULT_READOUT_DURATION
        if readout_pre_margin is None:
            readout_pre_margin = DEFAULT_READOUT_PRE_MARGIN
        if readout_post_margin is None:
            readout_post_margin = DEFAULT_READOUT_POST_MARGIN
        if property_dir is None:
            property_dir = PROPERTY_DIR
        if classifier_dir is None:
            classifier_dir = CLASSIFIER_DIR
        if classifier_type is None:
            classifier_type = "gmm"
        if configuration_mode is None:
            configuration_mode = "ge-cr-cr"
        if mock_mode is None:
            mock_mode = False

        self._load_config(
            chip_id=chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
            mock_mode=mock_mode,
        )
        qubits = self._create_qubit_labels(
            muxes=muxes,
            qubits=qubits,
            exclude_qubits=exclude_qubits,
        )
        self._chip_id: Final = chip_id
        self._qubits: Final = qubits
        self._readout_duration: Final = readout_duration
        self._readout_pre_margin: Final = readout_pre_margin
        self._readout_post_margin: Final = readout_post_margin
        self._drag_hpi_duration: Final = drag_hpi_duration
        self._drag_pi_duration: Final = drag_pi_duration
        self._property_dir: Final = property_dir
        self._classifier_dir: Final = classifier_dir
        self._classifier_type: Final = classifier_type
        self._configuration_mode: Final = configuration_mode
        self._calibration_valid_days: Final = calibration_valid_days
        self._measurement = Measurement(
            chip_id=chip_id,
            qubits=qubits,
            load_configs=False,
            connect_devices=False,
        )
        self._clifford_generator: CliffordGenerator | None = None
        self._user_note: Final = ExperimentNote(file_path=USER_NOTE_PATH)
        self._system_note: Final = ExperimentNote(file_path=SYSTEM_NOTE_PATH)
        self._calib_note: Final = CalibrationNote(
            chip_id=chip_id,
            file_path=calib_note_path,
        )
        self.system_manager.load_skew_file(self.box_ids)
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
        config_dir: Path | str | None,
        params_dir: Path | str | None,
        configuration_mode: Literal["ge-ef-cr", "ge-cr-cr"],
        mock_mode: bool | None = None,
    ):
        """Load the configuration files."""
        if mock_mode is None:
            mock_mode = False
        self.system_manager.load(
            chip_id=chip_id,
            config_dir=config_dir,
            params_dir=params_dir,
            configuration_mode=configuration_mode,
            mock_mode=mock_mode,
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
            logger.warning(f"Unavailable qubits: {unavailable_qubits}")

        qubit_labels = [qubit for qubit in qubit_labels if qubit in available_qubits]

        return qubit_labels

    def print_environment(self, verbose: bool | None = None):
        """Print the environment information."""
        if verbose is None:
            verbose = True
        logger.info("========================================")
        logger.info(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"python: {sys.version.split()[0]}")
        if verbose:
            logger.info(f"numpy: {get_package_version('numpy')}")
            logger.info(f"quel_ic_config: {get_package_version('quel_ic_config')}")
            logger.info(
                f"quel_clock_master: {get_package_version('quel_clock_master')}"
            )
            logger.info(f"qubecalib: {get_package_version('qubecalib')}")
        logger.info(f"qubex: {get_package_version('qubex')}")
        logger.info(f"env: {sys.prefix}")
        logger.info(f"config: {self.config_path}")
        logger.info(f"params: {self.params_path}")
        logger.info(f"chip: {self.chip_id} ({self.chip.name})")
        logger.info(f"qubits: {self.qubit_labels}")
        logger.info(f"muxes: {self.mux_labels}")
        logger.info(f"boxes: {self.box_ids}")
        logger.info("========================================")

    def print_boxes(self):
        """Print the box information."""
        if not logger.isEnabledFor(logging.INFO):
            return

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
    def system_manager(self) -> SystemManager:
        return SystemManager.shared()

    @property
    def config_loader(self) -> ConfigLoader:
        return self.system_manager.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        return self.system_manager.experiment_system

    @property
    def quantum_system(self) -> QuantumSystem:
        return self.experiment_system.quantum_system

    @property
    def control_system(self) -> ControlSystem:
        return self.experiment_system.control_system

    @property
    def device_controller(self) -> DeviceController:
        return self.system_manager.device_controller

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
    def cr_labels(self) -> list[str]:
        return self.get_cr_labels()

    @property
    def cr_pairs(self) -> list[tuple[str, str]]:
        return self.get_cr_pairs()

    @property
    def edge_pairs(self) -> list[tuple[str, str]]:
        return self.get_edge_pairs()

    @property
    def edge_labels(self) -> list[str]:
        return self.get_edge_labels()

    @property
    def boxes(self) -> dict[str, Box]:
        boxes = self.experiment_system.get_boxes_for_qubits(self.qubit_labels)
        return {box.id: box for box in boxes}

    @property
    def box_ids(self) -> list[str]:
        return list(self.boxes.keys())

    @property
    def config_path(self) -> str:
        return str(Path(self.config_loader.config_path).resolve())

    @property
    def params_path(self) -> str:
        return str(Path(self.config_loader.params_path).resolve())

    @property
    def calib_note(self) -> CalibrationNote:
        return self._calib_note

    @property
    def note(self) -> ExperimentNote:
        return self._user_note

    @property
    def calibration_valid_days(self) -> int:
        return self._calibration_valid_days

    @property
    def readout_duration(self) -> float:
        return self._readout_duration

    @property
    def readout_pre_margin(self) -> float:
        return self._readout_pre_margin

    @property
    def readout_post_margin(self) -> float:
        return self._readout_post_margin

    @property
    def drag_hpi_duration(self) -> float:
        return self._drag_hpi_duration

    @property
    def drag_pi_duration(self) -> float:
        return self._drag_pi_duration

    @property
    def property_dir(self) -> Path:
        return Path(self._property_dir)

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

    @property
    def reference_phases(self) -> dict[str, float]:
        return self.calib_note.reference_phases

    def load_property(self, property_name: str) -> dict:
        property_path = self.property_dir / self.chip_id / f"{property_name}.json"
        if property_path.exists():
            with open(property_path) as f:
                property_data = json.load(f)
                return property_data
        else:
            raise FileNotFoundError(f"Property file not found: {property_path}")

    def save_property(
        self,
        property_name: str,
        data: dict,
        *,
        save_path: Path | str | None = None,
    ):
        if save_path is not None:
            property_path = Path(save_path)
        else:
            property_path = self.property_dir / self.chip_id / f"{property_name}.json"
        if not property_path.parent.exists():
            property_path.parent.mkdir(parents=True)
        try:
            with open(property_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Property '{property_name}' saved to {property_path}")
        except Exception as e:
            raise OSError(f"Failed to save property '{property_name}': {e}") from e

    def load_calib_note(self, path: Path | str | None = None):
        """
        Load the calibration data from the given path or from the default calibration note file.
        """
        if path is None:
            # TODO: Make this path configurable
            path = (
                f"/home/shared/qubex-config/{self.chip_id}/calibration/calib_note.json"
            )
        if not Path(path).exists():
            raise FileNotFoundError(f"Calibration file '{path}' does not exist.")
        try:
            self._calib_note.load(path)
            logger.info(f"Calibration data loaded from {path}")
        except Exception as e:
            raise CalibrationMissingError(
                f"Failed to load calibration data from {path}: {e}"
            ) from e

    def get_qubit_label(self, index: int) -> str:
        """
        Get the qubit label from the given qubit index.
        """
        return self.quantum_system.get_qubit(index).label

    def get_resonator_label(self, index: int) -> str:
        """
        Get the resonator label from the given resonator index.
        """
        return self.quantum_system.get_resonator(index).label

    def get_cr_label(
        self,
        control_index: int,
        target_index: int,
    ) -> str:
        """
        Get the cross-resonance label from the given control and target qubit indices.
        """
        control_qubit = self.quantum_system.get_qubit(control_index)
        target_qubit = self.quantum_system.get_qubit(target_index)
        label = Target.cr_label(control_qubit.label, target_qubit.label)
        return label

    def get_cr_pairs(
        self,
        low_to_high: bool | None = None,
        high_to_low: bool | None = None,
    ) -> list[tuple[str, str]]:
        """
        Get the cross-resonance pairs.
        """
        if low_to_high is None:
            low_to_high = True
        if high_to_low is None:
            high_to_low = False
        cr_pairs = []
        for label in self.cr_targets:
            try:
                pair = Target.cr_qubit_pair(label)
                control_qubit = self.quantum_system.get_qubit(pair[0])
                target_qubit = self.quantum_system.get_qubit(pair[1])
                if target_qubit.label not in self.available_targets:
                    continue
                # if control_qubit.frequency < target_qubit.frequency:
                if control_qubit.index % 4 in [0, 3]:
                    if low_to_high:
                        cr_pairs.append(pair)
                else:
                    if high_to_low:
                        cr_pairs.append(pair)
            except Exception:
                continue
        return cr_pairs

    def get_cr_labels(
        self,
        low_to_high: bool | None = None,
        high_to_low: bool | None = None,
    ) -> list[str]:
        """
        Get the cross-resonance labels.
        """
        if low_to_high is None:
            low_to_high = True
        if high_to_low is None:
            high_to_low = False
        return [
            Target.cr_label(*pair)
            for pair in self.get_cr_pairs(low_to_high, high_to_low)
        ]

    def get_edge_pairs(
        self,
    ) -> list[tuple[str, str]]:
        """
        Get the qubit edge pairs.
        """
        edge_pairs = []
        for qubit in self.qubit_labels:
            spectators = self.get_spectators(qubit, in_same_mux=True)
            for spectator in spectators:
                pair = (qubit, spectator.label)
                edge_pairs.append(pair)
        return edge_pairs

    def get_edge_labels(
        self,
    ) -> list[str]:
        """
        Get the qubit edge labels.
        """
        return [f"{pair[0]}-{pair[1]}" for pair in self.get_edge_pairs()]

    @staticmethod
    def cr_pair(cr_label: str) -> tuple[str, str]:
        return Target.cr_qubit_pair(cr_label)

    def get_rabi_param(
        self,
        target: str,
        valid_days: int | None = None,
    ) -> RabiParam | None:
        """
        Get the Rabi parameters for the given target.
        """
        if valid_days is None:
            valid_days = self._calibration_valid_days

        param = self.calib_note.get_rabi_param(
            target,
            valid_days=valid_days,
        )
        if param is not None:
            try:
                return RabiParam(
                    target=param.get("target"),
                    frequency=param.get("frequency"),
                    amplitude=param.get("amplitude"),
                    phase=param.get("phase"),
                    offset=param.get("offset"),
                    noise=param.get("noise"),
                    angle=param.get("angle"),
                    distance=param.get("distance"),
                    r2=param.get("r2"),
                    reference_phase=param.get("reference_phase"),
                )
            except TypeError:
                raise ValueError(
                    f"Invalid Rabi parameters for {target}: {param}"
                ) from None
        else:
            return None

    def store_rabi_params(
        self,
        rabi_params: dict[str, RabiParam],
        r2_threshold: float | None = None,
    ):
        if r2_threshold is None:
            r2_threshold = 0.5
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
                        "distance": rabi_param.distance,
                        "r2": rabi_param.r2,
                        "reference_phase": rabi_param.reference_phase,
                    },
                )

        if len(not_stored) > 0:
            logger.warning(f"Rabi parameters are not stored for qubits: {not_stored}")

    def get_spectators(
        self,
        qubit: str,
        in_same_mux: bool | None = None,
    ) -> list[Qubit]:
        if in_same_mux is None:
            in_same_mux = False
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

    def is_connected(self) -> bool:
        return self._measurement.is_connected()

    def check_status(self):
        if not self.is_connected():
            logger.warning(
                "Not connected to the devices. Call `connect()` method first."
            )
            return

        if len(self.box_ids) == 0:
            logger.warning("No boxes are selected.")
            return

        # link status
        link_status = self._measurement.check_link_status(self.box_ids)
        if link_status["status"]:
            logger.info("Link status: OK")
        else:
            logger.warning("Link status: NG")
        logger.info(link_status["links"])

        # clock status
        clock_status = self._measurement.check_clock_status(self.box_ids)
        if clock_status["status"]:
            logger.info("Clock status: OK")
        else:
            logger.warning("Clock status: NG")
        logger.info(clock_status["clocks"])

        # config status
        config_status = self.system_manager.is_synced(box_ids=self.box_ids)
        if config_status:
            logger.info("Config status: OK")
        else:
            logger.warning("Config status: NG")
        logger.info(self.system_manager.device_settings)

    def connect(
        self,
        *,
        sync_clocks: bool | None = None,
    ) -> None:
        if sync_clocks is None:
            sync_clocks = True
        try:
            self._measurement.connect(sync_clocks=sync_clocks)
            logger.info("Successfully connected.")
        except Exception as e:
            logger.error(f"Failed to connect to the devices: {e}")
            raise

    def linkup(
        self,
        box_ids: list[str] | None = None,
        noise_threshold: int | None = None,
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
        self.system_manager.load(
            chip_id=self.chip_id,
            config_dir=self.config_path,
            params_dir=self.params_path,
            targets_to_exclude=exclude,
            configuration_mode=mode,
        )
        self.system_manager.push(
            box_ids=box_ids or self.box_ids,
        )

    def reload(self):
        try:
            self._measurement.reload(configuration_mode=self.configuration_mode)
            logger.info("Successfully reloaded.")
        except Exception as e:
            logger.error(f"Failed to reload the devices: {e}")
            raise

    def reset_awg_and_capunits(
        self,
        box_ids: str | Collection[str] | None = None,
        qubits: Collection[str] | None = None,
    ):
        box_ids = []
        if qubits is not None:
            boxes = self.experiment_system.get_boxes_for_qubits(qubits)
            box_ids += [box.id for box in boxes]
        if len(box_ids) == 0:
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
        target_type: TargetType | None = None,
        update_lsi: bool | None = None,
    ):
        if target_type is None:
            target_type = TargetType.CTRL_GE
        if update_lsi is None:
            update_lsi = False
        try:
            qubit_label = Target.qubit_label(label)
        except ValueError:
            raise ValueError(f"Invalid target label: {label}") from None

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
                lo=port.lo_freq,
                cnco=port.cnco_freq,
            )
            port.channels[channel_number].fnco_freq = fnco
            self.system_manager.push(box_ids=[box_id])

    @contextmanager
    def modified_frequencies(
        self,
        frequencies: dict[str, float] | None,
    ):
        if frequencies is None:
            yield
        else:
            with self.system_manager.modified_frequencies(frequencies):
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
        logger.info(f"ExperimentRecord `{name}` is loaded.\n")
        logger.info(f"description: {record.description}")
        logger.info(f"created_at: {record.created_at}")
        return record
