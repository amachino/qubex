"""Execution services for measurement workflows."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike
from qxpulse import PulseSchedule, RampType

from qubex.backend import (
    BackendController,
)
from qubex.backend.quel1 import (
    ExecutionMode,
    Quel1BackendController,
)
from qubex.backend.quel3 import Quel3BackendController
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge
from qubex.measurement.classifiers.state_classifier import StateClassifier
from qubex.measurement.measurement_config_factory import MeasurementConfigFactory
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_context import MeasurementContext
from qubex.measurement.measurement_pulse_factory import MeasurementPulseFactory
from qubex.measurement.measurement_result_converter import MeasurementResultConverter
from qubex.measurement.measurement_schedule_builder import (
    CapturePlacement,
    MeasurementScheduleBuilder,
)
from qubex.measurement.measurement_schedule_runner import MeasurementScheduleRunner
from qubex.measurement.models.capture_data import CaptureData
from qubex.measurement.models.measure_result import (
    MeasureResult,
    MultipleMeasureResult,
)
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.measurement.models.quel1_measurement_options import Quel1MeasurementOptions
from qubex.measurement.models.sweep_measurement_result import (
    NDSweepMeasurementResult,
    SweepAxes,
    SweepKey,
    SweepMeasurementResult,
    SweepPoint,
    SweepValue,
)
from qubex.measurement.services.measurement_session_service import (
    MeasurementSessionService,
)
from qubex.measurement.services.measurement_stability_service import (
    MeasurementStabilityService,
)
from qubex.system import (
    ConfigLoader,
    ControlParameters,
    ExperimentSystem,
    Mux,
    PortType,
    SystemManager,
    Target,
)
from qubex.typing import IQArray, TargetMap

logger = logging.getLogger(__name__)

T = TypeVar("T")
OptionT = TypeVar("OptionT")
RFSwitchState = Literal["pass", "block", "open", "loop"]
_LOOPBACK_DEMODULATION_FILTER_TAPS = 129
_LOOPBACK_DEMODULATION_FILTER_CUTOFF_GHZ = 0.025


@dataclass(frozen=True)
class _LoopbackMonitorSourceSetting:
    """Frequency setting for one active source that can feed a monitor input."""

    label: str
    port: Any
    channel_number: int
    lo_freq_hz: int | None
    cnco_freq_hz: int | None
    fnco_freq_hz: int | None

    @property
    def key(self) -> tuple[str, int | tuple[int, int], int]:
        """Return a stable key for this physical source channel."""
        return (
            str(self.port.box_id),
            cast(int | tuple[int, int], self.port.number),
            self.channel_number,
        )


@dataclass(frozen=True)
class _LoopbackRunSpec:
    """One loopback execution with a fixed source schedule and monitor NCO map."""

    pulse_schedule: PulseSchedule
    capture_targets: list[str]
    monitor_source_settings: Mapping[str, _LoopbackMonitorSourceSetting]
    backend_demodulation: bool
    software_demodulation: bool


def _run_async(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """Run one awaitable factory from synchronous APIs."""
    bridge = get_shared_async_bridge(key="measurement")
    return bridge.run(factory, timeout=timeout)


class MeasurementExecutionService:
    """Own measurement execution orchestration in the measurement layer."""

    def __init__(
        self,
        *,
        context: MeasurementContext,
        session_service: MeasurementSessionService,
        stability_service: MeasurementStabilityService,
        classifiers: TargetMap[StateClassifier],
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> None:
        self._context = context
        self._session_service = session_service
        self._stability_service = stability_service
        self._classifiers = classifiers
        self._execution_mode: ExecutionMode | None = execution_mode
        self._clock_health_checks: bool | None = clock_health_checks

    @property
    def context(self) -> MeasurementContext:
        """Return the measurement context."""
        return self._context

    @property
    def session_service(self) -> MeasurementSessionService:
        """Return the session lifecycle service."""
        return self._session_service

    @property
    def stability_service(self) -> MeasurementStabilityService:
        """Return the measurement stability service."""
        return self._stability_service

    @property
    def classifiers(self) -> TargetMap[StateClassifier]:
        """Return classifier mapping used for result conversion."""
        return self._classifiers

    @property
    def system_manager(self) -> SystemManager:
        """Return the shared system manager."""
        return self.context.system_manager

    @property
    def config_loader(self) -> ConfigLoader:
        """Return the configuration loader."""
        return self.context.config_loader

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Return the active experiment system."""
        return self.context.experiment_system

    @property
    def backend_controller(self) -> BackendController:
        """Return the active backend controller."""
        return self.session_service.backend_controller

    @property
    def mux_dict(self) -> dict[str, Mux]:
        """Return MUX objects indexed by qubit label."""
        return self.context.mux_dict

    @property
    def pulse_factory(self) -> MeasurementPulseFactory:
        """Return a pulse factory bound to current system state."""
        target_registry = getattr(self.experiment_system, "target_registry", None)
        return MeasurementPulseFactory(
            control_params=self.control_params,
            mux_dict=self.mux_dict,
            target_registry=target_registry,
            measurement_defaults=getattr(
                self.experiment_system, "measurement_defaults", None
            ),
        )

    @property
    def schedule_builder(self) -> MeasurementScheduleBuilder:
        """Return a schedule builder bound to current system state."""
        target_registry = getattr(self.experiment_system, "target_registry", None)
        return MeasurementScheduleBuilder(
            control_params=self.control_params,
            pulse_factory=self.pulse_factory,
            targets=self.targets,
            mux_dict=self.mux_dict,
            target_registry=target_registry,
            constraint_profile=self.constraint_profile,
        )

    @property
    def measurement_config_factory(self) -> MeasurementConfigFactory:
        """Return a measurement-config factory."""
        return MeasurementConfigFactory(
            experiment_system=self.experiment_system,
        )

    @property
    def sampling_period(self) -> float:
        """Return sampling period in ns."""
        return self.constraint_profile.sampling_period_ns

    @property
    def constraint_profile(self) -> MeasurementConstraintProfile:
        """Return backend timing and alignment constraints."""
        sampling_period = self.backend_controller.sampling_period_ns
        if isinstance(self.backend_controller, Quel3BackendController):
            return MeasurementConstraintProfile.quel3(sampling_period)
        if isinstance(self.backend_controller, Quel1BackendController):
            return MeasurementConstraintProfile.quel1(sampling_period)
        raise TypeError(
            "Unsupported backend controller for constraint profile selection."
        )

    @property
    def measurement_schedule_runner(self) -> MeasurementScheduleRunner:
        """Return the schedule-execution runner."""
        return MeasurementScheduleRunner(
            backend_controller=self.backend_controller,
            experiment_system=self.experiment_system,
            execution_mode=self._execution_mode,
            clock_health_checks=self._clock_health_checks,
        )

    @property
    def control_params(self) -> ControlParameters:
        """Return active control parameters."""
        return self.experiment_system.control_params

    @property
    def chip_id(self) -> str:
        """Return the active chip identifier."""
        return self.experiment_system.chip.id

    @property
    def targets(self) -> dict[str, Target]:
        """Return available targets indexed by label."""
        return {target.label: target for target in self.experiment_system.targets}

    @property
    def nco_frequencies(self) -> dict[str, float]:
        """Return NCO frequencies indexed by target label."""
        return {
            target.label: self.experiment_system.get_nco_frequency(target.label)
            for target in self.experiment_system.targets
        }

    @property
    def awg_frequencies(self) -> dict[str, float]:
        """Return AWG frequencies indexed by target label."""
        return {
            target.label: self.experiment_system.get_awg_frequency(target.label)
            for target in self.experiment_system.targets
        }

    @staticmethod
    def _resolve_device_config(
        backend_controller: BackendController,
    ) -> dict:
        """Resolve backend device configuration mapping."""
        box_config = getattr(backend_controller, "box_config", None)
        if isinstance(box_config, dict):
            return box_config
        return {}

    @staticmethod
    def _warn_deprecated_alias(
        *,
        old_name: str,
        new_name: str | None = None,
        message: str | None = None,
    ) -> None:
        """Emit a deprecation warning for a legacy option."""
        if message is None:
            if new_name is None:
                message = f"`{old_name}` is deprecated."
            else:
                message = f"`{old_name}` is deprecated; use `{new_name}`."
        warnings.warn(
            message,
            DeprecationWarning,
            stacklevel=3,
        )

    @classmethod
    def _resolve_deprecated_alias(
        cls,
        *,
        new_value: OptionT | None,
        old_value: OptionT | None,
        old_name: str,
        new_name: str,
    ) -> OptionT | None:
        """Resolve an old/new alias pair and validate conflicts."""
        if old_value is None:
            return new_value
        cls._warn_deprecated_alias(
            old_name=old_name,
            new_name=new_name,
        )
        if new_value is not None and new_value != old_value:
            raise ValueError(
                f"`{old_name}` conflicts with `{new_name}`. Provide only `{new_name}`."
            )
        return old_value if new_value is None else new_value

    def get_awg_frequency(self, target: str) -> float:
        """
        Get the AWG frequency for the target.

        Parameters
        ----------
        target : str
            Target label.


        Returns
        -------
        float
            AWG frequency in Hz.

        """
        return self.experiment_system.get_awg_frequency(target)

    def get_diff_frequency(self, target: str) -> float:
        """
        Get the difference frequency for the target.

        Parameters
        ----------
        target : str
            Target label.


        Returns
        -------
        float
            Difference frequency in Hz.

        """
        return self.experiment_system.get_diff_frequency(target)

    def _resolve_loopback_capture_targets(
        self,
        *,
        schedule: PulseSchedule,
        include_read_in: bool,
    ) -> list[str]:
        """Resolve default loopback capture targets for the active schedule."""
        active_readout_qubits: list[str] = []
        active_boxes: list[str] = []
        for label in schedule.labels:
            target = self.targets.get(label)
            if target is not None:
                port = target.channel.port
                active_boxes.append(str(port.box_id))
                is_readout_target = getattr(
                    port, "type", None
                ) == PortType.READ_OUT or getattr(target, "is_read", False)
                if is_readout_target:
                    try:
                        qubit_label = self.experiment_system.resolve_qubit_label(label)
                    except ValueError:
                        continue
                    active_readout_qubits.append(str(qubit_label))
                continue

        def _resolve_read_in_capture_target(target: Any) -> str:
            """Prefer READ_IN port ID so loopback labels match monitor port labels."""
            channel = getattr(target, "channel", None)
            port = None if channel is None else getattr(channel, "port", None)
            port_id = None if port is None else getattr(port, "id", None)
            if isinstance(port_id, str) and port_id:
                return port_id
            return str(target.label)

        read_capture_targets: list[str] = []
        if include_read_in:
            read_in_target_by_qubit: dict[str, str] = {}
            for target in self.experiment_system.read_in_targets:
                try:
                    qubit_label = self.experiment_system.resolve_qubit_label(
                        target.label
                    )
                except ValueError:
                    continue
                if qubit_label not in read_in_target_by_qubit:
                    read_in_target_by_qubit[qubit_label] = (
                        _resolve_read_in_capture_target(target)
                    )

            read_capture_targets = [
                read_in_target_by_qubit[qubit]
                for qubit in dict.fromkeys(active_readout_qubits)
                if qubit in read_in_target_by_qubit
            ]

        active_box_set = set(active_boxes)
        if not active_box_set:
            active_box_set = {
                box.id for box in self.experiment_system.control_system.boxes
            }

        monitor_capture_targets: list[str] = []
        for box in self.experiment_system.control_system.boxes:
            if box.id not in active_box_set:
                continue
            monitor_capture_targets.extend(
                port.id for port in box.ports if port.type == PortType.MNTR_IN
            )

        return list(dict.fromkeys([*read_capture_targets, *monitor_capture_targets]))

    def _filter_loopback_capture_targets(
        self,
        *,
        capture_targets: Sequence[str],
        port_type: PortType,
    ) -> list[str]:
        """Filter loopback capture targets by resolved port type."""
        filtered: list[str] = []
        for target in capture_targets:
            port = self._resolve_loopback_capture_port(target_or_port_id=target)
            if port is None:
                continue
            if port.type == port_type:
                filtered.append(target)
        return list(dict.fromkeys(filtered))

    @staticmethod
    def _is_e7awg_capture_data_error(exc: Exception) -> bool:
        """Return whether an exception indicates broken captured data."""
        return type(exc).__name__ == "E7awgCaptureDataError"

    @staticmethod
    def _is_rfswitch_unsupported_error(exc: Exception) -> bool:
        """Return whether an exception indicates RF-switch is unsupported."""
        if type(exc).__name__ == "NoRfSwitchError":
            return True
        if isinstance(exc, ValueError):
            message = str(exc).lower()
            return "invalid port of" in message or "no switch available" in message
        return False

    def _resolve_loopback_box_ids(
        self,
        *,
        schedule: PulseSchedule,
        capture_targets: Sequence[str],
    ) -> list[str]:
        """Resolve box IDs involved in loopback capture execution."""
        box_ids: list[str] = []

        for label in schedule.labels:
            target = self.targets.get(label)
            if target is None:
                continue
            box_ids.append(str(target.channel.port.box_id))

        for target in capture_targets:
            port = self._resolve_loopback_capture_port(target_or_port_id=target)
            if port is None:
                continue
            box_ids.append(str(port.box_id))

        if not box_ids:
            box_ids.extend(self.context.box_ids)

        return list(dict.fromkeys(box_ids))

    def _initialize_loopback_capture_units(
        self,
        *,
        box_ids: Sequence[str],
    ) -> None:
        """Initialize AWG/CAP units for loopback capture when supported."""
        initialize_awg_and_capunits = getattr(
            self.backend_controller,
            "initialize_awg_and_capunits",
            None,
        )
        if not callable(initialize_awg_and_capunits):
            return

        resolved_box_ids = list(dict.fromkeys(box_ids))
        if not resolved_box_ids:
            return
        initialize_awg_and_capunits(resolved_box_ids)

    @staticmethod
    def _coerce_loopback_frequency_hz(value: object) -> int | None:
        """Return a finite frequency value in Hz."""
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            return None
        frequency = float(value)
        if not np.isfinite(frequency):
            return None
        return int(frequency)

    @classmethod
    def _first_loopback_frequency_hz(cls, *values: object) -> int | None:
        """Return the first finite frequency value in Hz."""
        for value in values:
            frequency = cls._coerce_loopback_frequency_hz(value)
            if frequency is not None:
                return frequency
        return None

    @staticmethod
    def _resolve_loopback_channel_number(channel: Any) -> int:
        """Resolve a channel number from channel metadata."""
        number = getattr(channel, "number", 0)
        if isinstance(number, bool) or not isinstance(number, int):
            return 0
        return number

    def _dump_loopback_port_config(self, *, port: Any) -> Mapping[str, Any]:
        """Return a backend port dump when the active backend supports it."""
        dump_port = getattr(self.backend_controller, "dump_port", None)
        if not callable(dump_port):
            return {}
        try:
            dump = dump_port(box_name=port.box_id, port_number=port.number)
        except Exception:
            logger.exception(
                "Failed to dump loopback port %s on %s.",
                getattr(port, "number", None),
                getattr(port, "box_id", None),
            )
            return {}
        if isinstance(dump, Mapping):
            return dump
        return {}

    @classmethod
    def _resolve_loopback_dump_fnco_frequency_hz(
        cls,
        *,
        dump: Mapping[str, Any],
        section_name: str,
        channel_number: int,
    ) -> int | None:
        """Resolve one FNCO frequency from a backend dump section."""
        section = dump.get(section_name)
        if not isinstance(section, Mapping):
            return None
        channel_config = section.get(channel_number, section.get(str(channel_number)))
        if not isinstance(channel_config, Mapping):
            return None
        return cls._coerce_loopback_frequency_hz(channel_config.get("fnco_freq"))

    def _resolve_loopback_channel_fnco_frequency_hz(
        self,
        *,
        port: Any,
        channel_number: int,
        dump: Mapping[str, Any],
        dump_section: str,
    ) -> int | None:
        """Resolve channel FNCO frequency from model metadata or a backend dump."""
        channels = getattr(port, "channels", ())
        channel = None
        if isinstance(channels, Sequence) and channel_number < len(channels):
            channel = channels[channel_number]
        model_frequency = (
            None if channel is None else getattr(channel, "fnco_freq", None)
        )
        return self._first_loopback_frequency_hz(
            model_frequency,
            self._resolve_loopback_dump_fnco_frequency_hz(
                dump=dump,
                section_name=dump_section,
                channel_number=channel_number,
            ),
        )

    def _resolve_loopback_monitor_source_ports(
        self,
        *,
        monitor_port: Any,
    ) -> set[int | tuple[int, int]] | None:
        """Return source output ports that can feed a monitor input."""
        get_loopbacks_of_port = getattr(
            self.backend_controller,
            "get_loopbacks_of_port",
            None,
        )
        if not callable(get_loopbacks_of_port):
            return None
        try:
            loopbacks = cast(
                Collection[int | tuple[int, int]],
                get_loopbacks_of_port(
                    box_name=monitor_port.box_id,
                    port_number=monitor_port.number,
                ),
            )
            return set(loopbacks)
        except Exception:
            logger.exception(
                "Failed to resolve loopback sources for monitor port %s.",
                getattr(monitor_port, "id", monitor_port),
            )
            return set()

    def _resolve_loopback_monitor_source_settings(
        self,
        *,
        capture_target: str,
        pulse_schedule: PulseSchedule,
    ) -> list[_LoopbackMonitorSourceSetting]:
        """Resolve active source channels and frequencies for one monitor input."""
        monitor_port = self._resolve_loopback_capture_port(
            target_or_port_id=capture_target,
        )
        if monitor_port is None or monitor_port.type != PortType.MNTR_IN:
            return []

        loopback_source_ports = self._resolve_loopback_monitor_source_ports(
            monitor_port=monitor_port,
        )
        source_settings: list[_LoopbackMonitorSourceSetting] = []
        for label in self._resolve_loopback_source_labels(
            pulse_schedule=pulse_schedule,
            capture_target=capture_target,
        ):
            target = self.targets.get(label)
            if target is None:
                continue
            channel = getattr(target, "channel", None)
            source_port = None if channel is None else getattr(channel, "port", None)
            if source_port is None:
                continue
            source_port_number = getattr(source_port, "number", None)
            if not isinstance(source_port_number, (int, tuple)):
                continue
            if (
                loopback_source_ports is not None
                and source_port_number not in loopback_source_ports
            ):
                continue
            channel_number = self._resolve_loopback_channel_number(channel)
            source_dump = self._dump_loopback_port_config(port=source_port)
            lo_freq_hz = self._first_loopback_frequency_hz(
                getattr(source_port, "lo_freq", None),
                source_dump.get("lo_freq"),
            )
            cnco_freq_hz = self._first_loopback_frequency_hz(
                getattr(source_port, "cnco_freq", None),
                source_dump.get("cnco_freq"),
            )
            fnco_freq_hz = self._first_loopback_frequency_hz(
                getattr(channel, "fnco_freq", None),
                self._resolve_loopback_channel_fnco_frequency_hz(
                    port=source_port,
                    channel_number=channel_number,
                    dump=source_dump,
                    dump_section="channels",
                ),
            )
            source_settings.append(
                _LoopbackMonitorSourceSetting(
                    label=label,
                    port=source_port,
                    channel_number=channel_number,
                    lo_freq_hz=lo_freq_hz,
                    cnco_freq_hz=cnco_freq_hz,
                    fnco_freq_hz=fnco_freq_hz,
                )
            )

        unique_settings: dict[
            tuple[str, int | tuple[int, int], int],
            _LoopbackMonitorSourceSetting,
        ] = {}
        for setting in source_settings:
            unique_settings[setting.key] = setting

        return list(unique_settings.values())

    def _resolve_loopback_monitor_source_setting(
        self,
        *,
        capture_target: str,
        pulse_schedule: PulseSchedule,
    ) -> _LoopbackMonitorSourceSetting | None:
        """Resolve a single active source setting for one monitor input."""
        unique_settings = self._resolve_loopback_monitor_source_settings(
            capture_target=capture_target,
            pulse_schedule=pulse_schedule,
        )
        if len(unique_settings) == 1:
            return unique_settings[0]
        if len(unique_settings) > 1:
            logger.warning(
                "Skip monitor NCO setup for %s because active source ports are ambiguous: %s",
                capture_target,
                [setting.key for setting in unique_settings],
            )
        return None

    def _configure_loopback_monitor_frequency_settings(
        self,
        *,
        pulse_schedule: PulseSchedule,
        capture_targets: Sequence[str],
        monitor_source_settings: Mapping[str, _LoopbackMonitorSourceSetting]
        | None = None,
    ) -> None:
        """Configure monitor input LO/CNCO/FNCO to follow active source outputs."""
        config_port = getattr(self.backend_controller, "config_port", None)
        config_runit = getattr(self.backend_controller, "config_runit", None)
        if not callable(config_port):
            return

        control_system = self.experiment_system.control_system
        for capture_target in capture_targets:
            monitor_port = self._resolve_loopback_capture_port(
                target_or_port_id=capture_target,
            )
            if monitor_port is None or monitor_port.type != PortType.MNTR_IN:
                continue
            source_setting = (
                monitor_source_settings.get(capture_target)
                if monitor_source_settings is not None
                else self._resolve_loopback_monitor_source_setting(
                    capture_target=capture_target,
                    pulse_schedule=pulse_schedule,
                )
            )
            if source_setting is None:
                continue

            source_port = source_setting.port
            lo_freq_hz = source_setting.lo_freq_hz
            cnco_freq_hz = source_setting.cnco_freq_hz
            fnco_freq_hz = source_setting.fnco_freq_hz
            if lo_freq_hz is None and cnco_freq_hz is None and fnco_freq_hz is None:
                continue

            config_port(
                box_name=monitor_port.box_id,
                port=monitor_port.number,
                lo_freq_hz=lo_freq_hz,
                cnco_locked_with=source_port.number,
            )

            model_updates: dict[str, Any] = {}
            if lo_freq_hz is not None:
                model_updates["lo_freq"] = lo_freq_hz
            if cnco_freq_hz is not None:
                model_updates["cnco_freq"] = cnco_freq_hz

            monitor_channels = getattr(monitor_port, "channels", ())
            if fnco_freq_hz is not None:
                if callable(config_runit):
                    runit = 0
                    if isinstance(monitor_channels, Sequence) and monitor_channels:
                        runit = self._resolve_loopback_channel_number(
                            monitor_channels[0]
                        )
                    config_runit(
                        box_name=monitor_port.box_id,
                        port=monitor_port.number,
                        runit=runit,
                        fnco_freq_hz=fnco_freq_hz,
                    )
                if isinstance(monitor_channels, Sequence) and monitor_channels:
                    model_updates["fnco_freqs"] = [
                        fnco_freq_hz for _ in monitor_channels
                    ]

            if model_updates:
                control_system.set_port_params(
                    box_id=monitor_port.box_id,
                    port_number=monitor_port.number,
                    **model_updates,
                )

    @staticmethod
    def _copy_loopback_schedule_for_labels(
        *,
        pulse_schedule: PulseSchedule,
        labels: Sequence[str],
    ) -> PulseSchedule:
        """Return a schedule that contains only selected source labels."""
        selected = PulseSchedule()
        sequences = pulse_schedule.get_sequences(copy=True)
        set_frequency = getattr(selected, "set_frequency", None)
        get_frequency = getattr(pulse_schedule, "get_frequency", None)
        set_target = getattr(selected, "set_target", None)
        get_target = getattr(pulse_schedule, "get_target", None)
        set_frame = getattr(selected, "set_frame", None)
        get_frame = getattr(pulse_schedule, "get_frame", None)
        for label in labels:
            if label not in sequences:
                continue
            selected.add(label, sequences[label])
            if callable(set_frequency) and callable(get_frequency):
                with suppress(KeyError):
                    set_frequency(label, get_frequency(label))
            if callable(set_target) and callable(get_target):
                with suppress(KeyError):
                    set_target(label, get_target(label))
            if callable(set_frame) and callable(get_frame):
                with suppress(KeyError):
                    set_frame(label, get_frame(label))

        if selected.duration < pulse_schedule.duration:
            return selected.padded(pulse_schedule.duration)
        return selected

    def _build_loopback_run_specs(
        self,
        *,
        pulse_schedule: PulseSchedule,
        capture_targets: Sequence[str],
        demodulation: bool,
    ) -> list[_LoopbackRunSpec]:
        """Build loopback executions whose monitor captures have one source each."""
        run_specs: list[_LoopbackRunSpec] = []

        read_in_targets = self._filter_loopback_capture_targets(
            capture_targets=capture_targets,
            port_type=PortType.READ_IN,
        )
        if read_in_targets:
            run_specs.append(
                _LoopbackRunSpec(
                    pulse_schedule=pulse_schedule.copy(),
                    capture_targets=read_in_targets,
                    monitor_source_settings={},
                    backend_demodulation=demodulation,
                    software_demodulation=False,
                )
            )

        monitor_targets = self._filter_loopback_capture_targets(
            capture_targets=capture_targets,
            port_type=PortType.MNTR_IN,
        )
        monitor_groups: dict[
            tuple[str, int | tuple[int, int], int],
            tuple[
                _LoopbackMonitorSourceSetting,
                list[str],
                dict[str, _LoopbackMonitorSourceSetting],
            ],
        ] = {}
        background_monitor_targets: list[str] = []
        for monitor_target in monitor_targets:
            source_settings = self._resolve_loopback_monitor_source_settings(
                capture_target=monitor_target,
                pulse_schedule=pulse_schedule,
            )
            if not source_settings:
                background_monitor_targets.append(monitor_target)
                continue
            for source_setting in source_settings:
                _, group_targets, group_settings = monitor_groups.setdefault(
                    source_setting.key,
                    (source_setting, [], {}),
                )
                group_targets.append(monitor_target)
                group_settings[monitor_target] = source_setting

        if background_monitor_targets:
            run_specs.append(
                _LoopbackRunSpec(
                    pulse_schedule=pulse_schedule.copy(),
                    capture_targets=list(dict.fromkeys(background_monitor_targets)),
                    monitor_source_settings={},
                    backend_demodulation=False,
                    software_demodulation=demodulation,
                )
            )

        for source_setting, group_targets, group_settings in monitor_groups.values():
            run_specs.append(
                _LoopbackRunSpec(
                    pulse_schedule=self._copy_loopback_schedule_for_labels(
                        pulse_schedule=pulse_schedule,
                        labels=[source_setting.label],
                    ),
                    capture_targets=list(dict.fromkeys(group_targets)),
                    monitor_source_settings=group_settings,
                    backend_demodulation=False,
                    software_demodulation=demodulation,
                )
            )

        if not run_specs:
            run_specs.append(
                _LoopbackRunSpec(
                    pulse_schedule=pulse_schedule.copy(),
                    capture_targets=list(dict.fromkeys(capture_targets)),
                    monitor_source_settings={},
                    backend_demodulation=demodulation,
                    software_demodulation=False,
                )
            )
        return run_specs

    @staticmethod
    def _merge_loopback_results(
        results: Sequence[MeasurementResult],
    ) -> MeasurementResult:
        """Merge captures returned by split loopback executions."""
        if len(results) == 0:
            raise ValueError("At least one loopback result is required.")
        if len(results) == 1:
            return results[0]

        merged_data: dict[str, list[CaptureData]] = {}
        device_config: dict[str, Any] | None = None
        for result in results:
            if device_config is None and result.device_config is not None:
                device_config = result.device_config
            for target, captures in result.data.items():
                merged_data.setdefault(target, []).extend(captures)

        return MeasurementResult(
            data=merged_data,
            measurement_config=results[0].measurement_config,
            device_config=device_config,
        )

    @staticmethod
    def _order_loopback_result_by_targets(
        result: MeasurementResult,
        *,
        target_order: Sequence[str],
    ) -> MeasurementResult:
        """Return loopback captures ordered by the user's schedule labels."""
        remaining = dict(result.data)
        ordered_data: dict[str, list[CaptureData]] = {}
        for target in dict.fromkeys(target_order):
            captures = remaining.pop(target, None)
            if captures is not None:
                ordered_data[target] = captures
        ordered_data.update(remaining)
        if list(ordered_data) == list(result.data):
            return result

        classifier_refs = None
        if result.classifier_refs is not None:
            classifier_refs = {
                target: result.classifier_refs[target]
                for target in ordered_data
                if target in result.classifier_refs
            }

        return MeasurementResult(
            data=ordered_data,
            measurement_config=result.measurement_config,
            device_config=result.device_config,
            classifier_refs=classifier_refs,
        )

    @staticmethod
    def _resolve_loopback_schedule_frequency(
        *,
        pulse_schedule: PulseSchedule,
        target: str,
    ) -> float | None:
        """Resolve schedule channel frequency metadata when available."""
        get_frequency = getattr(pulse_schedule, "get_frequency", None)
        if not callable(get_frequency):
            return None
        try:
            frequency = get_frequency(target)
        except KeyError:
            return None
        if isinstance(frequency, (int, float)):
            return float(frequency)
        return None

    @staticmethod
    def _port_frequency_to_ghz(frequency: float) -> float:
        """Return a port-channel frequency in GHz for software DSP."""
        return frequency * 1e-9 if abs(frequency) > 1.0 else frequency

    def _resolve_loopback_nco_frequency_ghz(
        self,
        *,
        target_or_port_id: str,
    ) -> float | None:
        """Resolve target or capture-port NCO frequency in GHz."""
        try:
            return float(self.experiment_system.get_nco_frequency(target_or_port_id))
        except (AttributeError, KeyError, ValueError):
            pass

        port = self._resolve_loopback_capture_port(
            target_or_port_id=target_or_port_id,
        )
        if port is None:
            return None
        channels = getattr(port, "channels", ())
        if not channels:
            return None
        fnco_freq = getattr(channels[0], "fnco_freq", None)
        if not isinstance(fnco_freq, (int, float)):
            return None
        return self._port_frequency_to_ghz(float(fnco_freq))

    def _resolve_loopback_modulation_frequency_ghz(
        self,
        *,
        target_or_port_id: str,
        pulse_schedule: PulseSchedule,
    ) -> float | None:
        """Resolve the modulation frequency used for software demodulation."""
        schedule_frequency = self._resolve_loopback_schedule_frequency(
            pulse_schedule=pulse_schedule,
            target=target_or_port_id,
        )
        if schedule_frequency is not None:
            nco_frequency = self._resolve_loopback_nco_frequency_ghz(
                target_or_port_id=target_or_port_id,
            )
            if nco_frequency is None:
                return schedule_frequency
            sideband: str | None = None
            try:
                sideband = self.experiment_system.get_target(target_or_port_id).sideband
            except (AttributeError, KeyError, ValueError):
                sideband = None
            if sideband == "L":
                return nco_frequency - schedule_frequency
            return schedule_frequency - nco_frequency

        try:
            return float(self.experiment_system.get_awg_frequency(target_or_port_id))
        except (AttributeError, KeyError, ValueError):
            pass

        return self._resolve_loopback_nco_frequency_ghz(
            target_or_port_id=target_or_port_id,
        )

    def _resolve_loopback_qubit_label(
        self,
        *,
        target_or_port_id: str,
    ) -> str | None:
        """Resolve a qubit label from a target label or read-in port ID."""
        resolve_qubit_label = getattr(
            self.experiment_system, "resolve_qubit_label", None
        )
        if callable(resolve_qubit_label):
            try:
                return str(resolve_qubit_label(target_or_port_id))
            except ValueError:
                pass

        if target_or_port_id.startswith("RQ"):
            return target_or_port_id[1:]
        if target_or_port_id.startswith("Q"):
            return target_or_port_id

        for cap_target in getattr(self.experiment_system, "read_in_targets", ()):
            cap_label = str(getattr(cap_target, "label", ""))
            channel = getattr(cap_target, "channel", None)
            port = None if channel is None else getattr(channel, "port", None)
            port_id = None if port is None else getattr(port, "id", None)
            if target_or_port_id not in {cap_label, port_id}:
                continue
            if callable(resolve_qubit_label):
                try:
                    return str(resolve_qubit_label(cap_label))
                except ValueError:
                    pass
            if cap_label.startswith("RQ"):
                return cap_label[1:]
            return cap_label or None
        return None

    def _resolve_loopback_result_capture_target(
        self,
        *,
        result_target: str,
        capture_targets: Sequence[str],
    ) -> str:
        """Map a result key back to the loopback capture target that produced it."""
        if result_target in capture_targets:
            return result_target

        result_qubit = self._resolve_loopback_qubit_label(
            target_or_port_id=result_target,
        )
        if result_qubit is None:
            return result_target

        for capture_target in capture_targets:
            capture_qubit = self._resolve_loopback_qubit_label(
                target_or_port_id=capture_target,
            )
            if capture_qubit == result_qubit:
                return capture_target
        return result_target

    @staticmethod
    def _resolve_loopback_output_label(
        *,
        result_target: str,
        capture_target: str,
        monitor_source_settings: Mapping[str, _LoopbackMonitorSourceSetting] | None,
    ) -> str:
        """Resolve the user-facing label for one loopback capture result."""
        if (
            monitor_source_settings is not None
            and capture_target in monitor_source_settings
        ):
            return monitor_source_settings[capture_target].label
        return result_target

    def _resolve_loopback_source_labels(
        self,
        *,
        pulse_schedule: PulseSchedule,
        capture_target: str,
    ) -> list[str]:
        """Resolve active source labels that can feed one loopback capture port."""
        capture_port = self._resolve_loopback_capture_port(
            target_or_port_id=capture_target,
        )
        if capture_port is None:
            return [label for label in pulse_schedule.labels if label == capture_target]

        source_labels: list[str] = []
        capture_qubit = self._resolve_loopback_qubit_label(
            target_or_port_id=capture_target,
        )
        for label in pulse_schedule.labels:
            target = self.targets.get(label)
            if target is None:
                continue
            source_port = target.channel.port
            if getattr(source_port, "box_id", None) != getattr(
                capture_port,
                "box_id",
                None,
            ):
                continue
            if capture_port.type == PortType.READ_IN:
                if getattr(source_port, "type", None) != PortType.READ_OUT:
                    continue
                if capture_qubit is not None:
                    source_qubit = self._resolve_loopback_qubit_label(
                        target_or_port_id=label,
                    )
                    if source_qubit != capture_qubit:
                        continue
            source_labels.append(label)
        return list(dict.fromkeys(source_labels))

    def _resolve_loopback_demodulation_frequency_ghz(
        self,
        *,
        capture_target: str,
        pulse_schedule: PulseSchedule,
        monitor_source_label: str | None = None,
    ) -> float | None:
        """Resolve the single software-demodulation frequency for a capture."""
        capture_port = self._resolve_loopback_capture_port(
            target_or_port_id=capture_target,
        )
        if (
            monitor_source_label is not None
            and capture_port is not None
            and capture_port.type == PortType.MNTR_IN
        ):
            source_frequency = self._resolve_loopback_modulation_frequency_ghz(
                target_or_port_id=monitor_source_label,
                pulse_schedule=pulse_schedule,
            )
            if source_frequency is not None:
                return source_frequency

        direct_frequency = (
            self._resolve_loopback_nco_frequency_ghz(
                target_or_port_id=capture_target,
            )
            if capture_port is not None
            else self._resolve_loopback_modulation_frequency_ghz(
                target_or_port_id=capture_target,
                pulse_schedule=pulse_schedule,
            )
        )
        if capture_port is not None and capture_port.type == PortType.MNTR_IN:
            return direct_frequency
        if direct_frequency is not None and not np.isclose(direct_frequency, 0.0):
            return direct_frequency

        source_frequencies: list[float] = []
        for label in self._resolve_loopback_source_labels(
            pulse_schedule=pulse_schedule,
            capture_target=capture_target,
        ):
            frequency = self._resolve_loopback_modulation_frequency_ghz(
                target_or_port_id=label,
                pulse_schedule=pulse_schedule,
            )
            if frequency is None:
                continue
            if any(np.isclose(frequency, known) for known in source_frequencies):
                continue
            source_frequencies.append(frequency)

        if len(source_frequencies) == 1:
            return source_frequencies[0]
        if len(source_frequencies) > 1:
            logger.warning(
                "Skip loopback software demodulation for %s because active source frequencies are ambiguous: %s",
                capture_target,
                source_frequencies,
            )
            return None
        return direct_frequency

    @staticmethod
    def _demodulate_loopback_capture(
        *,
        data: np.ndarray,
        frequency_ghz: float | None,
        sampling_period: float,
    ) -> np.ndarray:
        """Apply software demodulation along the waveform sample axis."""
        if frequency_ghz is None or data.ndim == 0:
            return data
        sample_count = data.shape[-1]
        sample_times = np.arange(sample_count, dtype=np.float64) * sampling_period
        oscillator = np.exp(-1j * 2 * np.pi * frequency_ghz * sample_times)
        return data * oscillator

    @staticmethod
    def _resolve_loopback_filter_tap_count(sample_count: int) -> int:
        """Return an odd FIR tap count suitable for one waveform length."""
        if sample_count < _LOOPBACK_DEMODULATION_FILTER_TAPS:
            return 0
        return _LOOPBACK_DEMODULATION_FILTER_TAPS

    @classmethod
    def _design_loopback_lowpass_fir(
        cls,
        *,
        frequency_ghz: float | None,
        sampling_period: float,
        sample_count: int,
    ) -> np.ndarray | None:
        """Design a low-pass FIR for software-demodulated loopback data."""
        if frequency_ghz is None or np.isclose(frequency_ghz, 0.0):
            return None
        nyquist_ghz = 0.5 / sampling_period
        cutoff_ghz = min(
            _LOOPBACK_DEMODULATION_FILTER_CUTOFF_GHZ,
            abs(frequency_ghz) * 0.45,
            nyquist_ghz * 0.8,
        )
        if cutoff_ghz <= 0.0 or cutoff_ghz >= nyquist_ghz:
            return None

        tap_count = cls._resolve_loopback_filter_tap_count(sample_count)
        if tap_count == 0:
            return None

        normalized_cutoff = cutoff_ghz / nyquist_ghz
        center = (tap_count - 1) / 2
        offsets = np.arange(tap_count, dtype=np.float64) - center
        taps = normalized_cutoff * np.sinc(normalized_cutoff * offsets)
        taps *= np.hamming(tap_count)
        tap_sum = np.sum(taps)
        if np.isclose(tap_sum, 0.0):
            return None
        return taps / tap_sum

    @classmethod
    def _filter_loopback_demodulated_capture(
        cls,
        *,
        data: np.ndarray,
        frequency_ghz: float | None,
        sampling_period: float,
    ) -> np.ndarray:
        """Apply a zero-phase low-pass FIR to demodulated loopback data."""
        if data.ndim == 0:
            return data
        taps = cls._design_loopback_lowpass_fir(
            frequency_ghz=frequency_ghz,
            sampling_period=sampling_period,
            sample_count=data.shape[-1],
        )
        if taps is None:
            return data
        return np.apply_along_axis(
            lambda values: np.convolve(values, taps, mode="same"),
            axis=-1,
            arr=data,
        )

    def _postprocess_loopback_result(
        self,
        *,
        result: MeasurementResult,
        measurement_schedule: MeasurementSchedule,
        capture_targets: Sequence[str],
        shot_averaging: bool,
        demodulation: bool,
        monitor_source_settings: Mapping[str, _LoopbackMonitorSourceSetting]
        | None = None,
    ) -> MeasurementResult:
        """Apply requested Qubex-side postprocessing to loopback captures."""
        transform_data = demodulation or shot_averaging
        processed_config = (
            self.measurement_config_factory.create(
                n_shots=result.measurement_config.n_shots,
                shot_interval=result.measurement_config.shot_interval,
                shot_averaging=shot_averaging,
                time_integration=False,
                state_classification=False,
            )
            if transform_data
            else result.measurement_config
        )
        processed_data: dict[str, list[CaptureData]] = {}
        classifier_refs = (
            None if result.classifier_refs is None else dict(result.classifier_refs)
        )

        for result_target, captures in result.data.items():
            capture_target = self._resolve_loopback_result_capture_target(
                result_target=result_target,
                capture_targets=capture_targets,
            )
            output_target = self._resolve_loopback_output_label(
                result_target=result_target,
                capture_target=capture_target,
                monitor_source_settings=monitor_source_settings,
            )
            frequency = (
                self._resolve_loopback_demodulation_frequency_ghz(
                    capture_target=capture_target,
                    pulse_schedule=measurement_schedule.pulse_schedule,
                    monitor_source_label=(
                        None
                        if monitor_source_settings is None
                        or capture_target not in monitor_source_settings
                        else monitor_source_settings[capture_target].label
                    ),
                )
                if demodulation
                else None
            )
            processed_captures: list[CaptureData] = []
            for capture in captures:
                data = np.asarray(capture.data, dtype=np.complex128)
                if demodulation:
                    data = self._demodulate_loopback_capture(
                        data=data,
                        frequency_ghz=frequency,
                        sampling_period=capture.sampling_period,
                    )
                    data = self._filter_loopback_demodulated_capture(
                        data=data,
                        frequency_ghz=frequency,
                        sampling_period=capture.sampling_period,
                    )
                if shot_averaging and data.ndim >= 2:
                    data = np.mean(data, axis=0)
                processed_captures.append(
                    CaptureData.from_primary_data(
                        target=output_target,
                        data=data,
                        config=processed_config,
                        sampling_period=capture.sampling_period,
                        classifier_ref=capture.classifier_ref,
                    )
                )
            processed_data.setdefault(output_target, []).extend(processed_captures)

            if classifier_refs is not None and result_target in classifier_refs:
                classifier_refs[output_target] = classifier_refs.pop(result_target)

        return MeasurementResult(
            data=processed_data,
            measurement_config=processed_config,
            device_config=result.device_config,
            classifier_refs=classifier_refs,
        )

    def _resolve_loopback_capture_port(
        self,
        *,
        target_or_port_id: str,
    ) -> Any | None:
        """Resolve one loopback capture port from target label or port ID."""
        control_system = self.experiment_system.control_system
        try:
            return control_system.get_port_by_id(target_or_port_id)
        except KeyError:
            pass

        get_cap_target = getattr(self.experiment_system, "get_cap_target", None)
        if not callable(get_cap_target):
            return None
        try:
            cap_target = get_cap_target(target_or_port_id)
        except KeyError:
            return None
        channel = getattr(cap_target, "channel", None)
        return None if channel is None else getattr(channel, "port", None)

    def _resolve_loopback_rfswitch_updates(
        self,
        *,
        schedule: PulseSchedule,
        capture_targets: Sequence[str],
        block_outputs: bool,
    ) -> dict[str, RFSwitchState]:
        """Build loopback RF-switch overrides keyed by port ID."""
        control_system = self.experiment_system.control_system
        updates: dict[str, RFSwitchState] = {}

        if block_outputs:
            for label in schedule.labels:
                target = self.targets.get(label)
                if target is None:
                    continue
                port = target.channel.port
                if getattr(port, "rfswitch", None) in ("pass", "block"):
                    updates[port.id] = "block"

        for capture_target in capture_targets:
            port = self._resolve_loopback_capture_port(target_or_port_id=capture_target)
            if port is None:
                continue

            if port.type == PortType.READ_IN:
                updates[port.id] = "loop"
                for box_port in control_system.get_box(port.box_id).ports:
                    if box_port.type == PortType.READ_OUT:
                        updates[box_port.id] = "block"
            elif port.type == PortType.MNTR_IN:
                updates[port.id] = "loop"

        return updates

    def _set_port_rfswitch(
        self,
        *,
        port: Any,
        rfswitch: RFSwitchState,
    ) -> None:
        """Set one port RF switch on hardware and in experiment model."""
        config_port = getattr(self.backend_controller, "config_port", None)
        if not callable(config_port):
            raise NotImplementedError(
                "Active backend does not support RF-switch configuration."
            )

        try:
            config_port(
                box_name=port.box_id,
                port=port.number,
                rfswitch=rfswitch,
            )
        except Exception as exc:
            if self._is_rfswitch_unsupported_error(exc):
                logger.warning(
                    "Skip RF-switch update for %s on %s because the port does not support RF switch configuration.",
                    port.id,
                    port.box_id,
                )
                return
            raise

        self.experiment_system.control_system.set_port_params(
            box_id=port.box_id,
            port_number=port.number,
            rfswitch=rfswitch,
        )

    @contextmanager
    def _temporary_loopback_rfswitches(
        self,
        *,
        schedule: PulseSchedule,
        capture_targets: Sequence[str],
        block_outputs: bool,
    ) -> Iterator[None]:
        """Temporarily configure RF switches for loopback capture and restore them."""
        config_port = getattr(self.backend_controller, "config_port", None)
        if not callable(config_port):
            yield
            return

        updates = self._resolve_loopback_rfswitch_updates(
            schedule=schedule,
            capture_targets=capture_targets,
            block_outputs=block_outputs,
        )
        if not updates:
            yield
            return

        control_system = self.experiment_system.control_system
        original_rfswitches: dict[str, RFSwitchState] = {}
        for port_id in sorted(updates):
            try:
                port = control_system.get_port_by_id(port_id)
            except KeyError:
                continue
            original_rfswitches[port_id] = cast(RFSwitchState, str(port.rfswitch))

        try:
            for port_id in sorted(updates):
                try:
                    port = control_system.get_port_by_id(port_id)
                except KeyError:
                    continue
                desired_rfswitch = updates[port_id]
                if str(port.rfswitch) == desired_rfswitch:
                    continue
                self._set_port_rfswitch(
                    port=port,
                    rfswitch=desired_rfswitch,
                )
            yield
        finally:
            for port_id in sorted(original_rfswitches):
                try:
                    port = control_system.get_port_by_id(port_id)
                except KeyError:
                    continue
                restore_rfswitch = original_rfswitches[port_id]
                if str(port.rfswitch) == restore_rfswitch:
                    continue
                self._set_port_rfswitch(
                    port=port,
                    rfswitch=restore_rfswitch,
                )

    async def run_measurement(
        self,
        *,
        schedule: MeasurementSchedule,
        config: MeasurementConfig,
        quel1_options: Quel1MeasurementOptions | None = None,
    ) -> MeasurementResult:
        """
        Run measurement with the given schedule and configuration.

        Parameters
        ----------
        schedule : MeasurementSchedule
            Measurement schedule.

        config : MeasurementConfig
            Measurement configuration.


        Returns
        -------
        MeasurementResult
            Measurement result.

        """
        return await self.measurement_schedule_runner.execute_async(
            schedule=schedule,
            config=config,
            quel1_options=quel1_options,
        )

    async def run_sweep_measurement(
        self,
        schedule: Callable[[SweepValue], MeasurementSchedule],
        *,
        sweep_values: ArrayLike | Sequence[SweepValue],
        config: MeasurementConfig | None = None,
        on_point: Callable[[SweepValue, MeasurementResult], None] | None = None,
    ) -> SweepMeasurementResult:
        """
        Run sweep measurement pointwise.

        Parameters
        ----------
        schedule : Callable[[SweepValue], MeasurementSchedule]
            Callback that builds one measurement schedule per sweep value.
        sweep_values : ArrayLike | Sequence[SweepValue]
            Ordered sweep values to execute.
        config : MeasurementConfig | None, optional
            Shared measurement configuration for all points.
        on_point : Callable[[SweepValue, MeasurementResult], None] | None, optional
            Callback invoked after each point measurement completes.

        Returns
        -------
        SweepMeasurementResult
            Sweep result list in the same order as input values.
        """
        resolved_config = self.create_measurement_config() if config is None else config
        normalized_values = cast(list[SweepValue], np.asarray(sweep_values).tolist())
        if self._can_use_batch_execution():
            measurement_schedules = [
                schedule(sweep_value) for sweep_value in normalized_values
            ]
            results = await self.measurement_schedule_runner.execute_many_async(
                schedules=measurement_schedules,
                config=resolved_config,
            )
        else:
            results: list[MeasurementResult] = []
            for sweep_value in normalized_values:
                measurement_schedule = schedule(sweep_value)
                result = await self.run_measurement(
                    schedule=measurement_schedule,
                    config=resolved_config,
                )
                results.append(result)
                if on_point is not None:
                    on_point(sweep_value, result)
            return SweepMeasurementResult(
                sweep_values=normalized_values,
                config=resolved_config,
                results=results,
            )
        if on_point is not None:
            for sweep_value, result in zip(normalized_values, results, strict=True):
                on_point(sweep_value, result)
        return SweepMeasurementResult(
            sweep_values=normalized_values,
            config=resolved_config,
            results=results,
        )

    async def run_ndsweep_measurement(
        self,
        schedule: Callable[[SweepPoint], MeasurementSchedule],
        *,
        sweep_points: Mapping[SweepKey, Sequence[SweepValue]],
        sweep_axes: SweepAxes | None = None,
        config: MeasurementConfig | None = None,
    ) -> NDSweepMeasurementResult:
        """
        Run N-dimensional Cartesian-product sweep measurement pointwise.

        Parameters
        ----------
        schedule : Callable[[SweepPoint], MeasurementSchedule]
            Callback that builds one measurement schedule per resolved sweep point.
        sweep_points : Mapping[SweepKey, Sequence[SweepValue]]
            Axis-value table (`axis key -> ordered values`).
        sweep_axes : SweepAxes | None, optional
            Axis order for Cartesian product. If omitted, insertion order of
            `sweep_points` is used for `dict` inputs. Other mapping inputs must
            provide this explicitly.
        config : MeasurementConfig | None, optional
            Shared measurement configuration for all points.

        Returns
        -------
        NDSweepMeasurementResult
            Cartesian sweep result with C-order flattening (last axis varies fastest).

        Raises
        ------
        ValueError
            If `sweep_axes` does not match `sweep_points` keys exactly, or if
            `sweep_axes` is omitted for a non-`dict` mapping input.
        """
        resolved_config = self.create_measurement_config() if config is None else config
        if sweep_axes is None:
            if not isinstance(sweep_points, dict):
                raise ValueError(
                    "sweep_axes must be provided when sweep_points is not a "
                    "dict-derived insertion-ordered mapping."
                )
            resolved_axes = tuple(sweep_points.keys())
        else:
            resolved_axes = tuple(sweep_axes)
        if len(set(resolved_axes)) != len(resolved_axes):
            raise ValueError("sweep_axes must not contain duplicate keys.")
        if set(resolved_axes) != set(sweep_points.keys()):
            raise ValueError(
                "sweep_axes must contain each sweep_points key exactly once."
            )

        normalized_axes_points = {axis: [*sweep_points[axis]] for axis in resolved_axes}
        shape = tuple(len(normalized_axes_points[axis]) for axis in resolved_axes)
        if self._can_use_batch_execution():
            points: list[dict[SweepKey, SweepValue]] = [
                {
                    axis: normalized_axes_points[axis][axis_index]
                    for axis, axis_index in zip(resolved_axes, ndindex, strict=True)
                }
                for ndindex in np.ndindex(shape)
            ]
            measurement_schedules = [schedule(point) for point in points]
            results = await self.measurement_schedule_runner.execute_many_async(
                schedules=measurement_schedules,
                config=resolved_config,
            )
        else:
            results: list[MeasurementResult] = []
            for ndindex in np.ndindex(shape):
                point: dict[SweepKey, SweepValue] = {
                    axis: normalized_axes_points[axis][axis_index]
                    for axis, axis_index in zip(resolved_axes, ndindex, strict=True)
                }
                measurement_schedule = schedule(point)
                result = await self.run_measurement(
                    schedule=measurement_schedule,
                    config=resolved_config,
                )
                results.append(result)

        return NDSweepMeasurementResult(
            sweep_points=normalized_axes_points,
            sweep_axes=resolved_axes,
            shape=shape,
            config=resolved_config,
            results=results,
        )

    def _can_use_batch_execution(self) -> bool:
        """Return whether backend batch execution can be used safely."""
        execute_batch_async = getattr(
            self.backend_controller, "execute_batch_async", None
        )
        if not callable(execute_batch_async):
            return False
        run_measurement_func = getattr(self.run_measurement, "__func__", None)
        if run_measurement_func is not MeasurementExecutionService.run_measurement:
            return False
        try:
            _ = self.experiment_system
        except ValueError:
            return False
        return True

    async def measure_noise(
        self,
        targets: Collection[str],
        *,
        duration: float,
    ) -> MeasurementResult:
        """
        Measure readout noise.

        Parameters
        ----------
        targets : Collection[str]
            Target labels to measure.
        duration : float
            Readout duration in ns.

        Returns
        -------
        MeasurementResult
            Measurement results.

        """
        pulse_schedule = PulseSchedule.from_waveforms(
            {target: np.zeros(0) for target in targets}
        )
        measurement_config = self.create_measurement_config(
            n_shots=1,
            shot_averaging=True,
            time_integration=False,
            state_classification=False,
        )
        measurement_schedule = self.build_measurement_schedule(
            pulse_schedule=pulse_schedule,
            readout_duration=duration,
            readout_amplitudes=dict.fromkeys(targets, 0),
            readout_amplification=False,
            final_measurement=True,
        )
        return await self.run_measurement(
            schedule=measurement_schedule,
            config=measurement_config,
        )

    def measure(
        self,
        waveforms: Mapping[str, IQArray],
        *,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        shot_averaging: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
        frequencies: dict[str, float] | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramp_time: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_amplification: bool | None = None,
        classification_line_param0: tuple[float, float, float] | None = None,
        classification_line_param1: tuple[float, float, float] | None = None,
        plot: bool | None = None,
        **deprecated_options: Any,
    ) -> MeasureResult:
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        waveforms : Mapping[str, IQArray]
            Control waveforms for each target.

        n_shots : int | None, optional
            Number of shots.
        shot_interval : float | None, optional
            Interval between shots in ns.
        shot_averaging : bool | None, optional
            Whether to average shots on hardware.
        time_integration : bool | None, optional
            Whether to integrate captured waveforms over time.
        state_classification : bool | None, optional
            Whether to enable state classification.
        frequencies : dict[str, float] | None, optional
            Channel-frequency overrides keyed by schedule label.
        readout_amplitudes : dict[str, float], optional
            Readout amplitude for each qubit.

        readout_duration : float, optional
            Readout duration in ns.

        readout_pre_margin : float, optional
            Readout pre-margin in ns.

        readout_post_margin : float, optional
            Readout post-margin in ns.

        readout_ramp_time : float, optional
            Readout ramp time in ns.

        readout_drag_coeff : float, optional
            Readout drag coefficient.

        readout_ramp_type : RampType, optional
            Readout ramp type.

        readout_amplification : bool | None, optional
            Whether to apply readout amplification pulses.
        classification_line_param0 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 0.
        classification_line_param1 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 1.

        Returns
        -------
        MeasureResult
            Measurement results.

        """
        if (
            time_integration is None
            and deprecated_options.get("enable_dsp_sum") is None
        ):
            time_integration = False

        result = self.execute(
            schedule=waveforms,
            n_shots=n_shots,
            shot_interval=shot_interval,
            shot_averaging=shot_averaging,
            time_integration=time_integration,
            state_classification=state_classification,
            frequencies=frequencies,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramp_time=readout_ramp_time,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            readout_amplification=readout_amplification,
            final_measurement=True,
            classification_line_param0=classification_line_param0,
            classification_line_param1=classification_line_param1,
            plot=plot,
            **deprecated_options,
        )
        data = {target: measures[0] for target, measures in result.data.items()}
        return MeasureResult(
            mode=result.mode,
            data=data,
            config=result.config,
        )

    def execute(
        self,
        schedule: PulseSchedule | TargetMap[IQArray],
        *,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        shot_averaging: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
        frequencies: dict[str, float] | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramp_time: float | None = None,
        readout_drag_coeff: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_amplification: bool | None = None,
        final_measurement: bool | None = None,
        classification_line_param0: tuple[float, float, float] | None = None,
        classification_line_param1: tuple[float, float, float] | None = None,
        plot: bool | None = None,
        **deprecated_options: Any,
    ) -> MultipleMeasureResult:
        """
        Measure with the given control waveforms.

        Parameters
        ----------
        schedule : PulseSchedule | TargetMap[IQArray]
            Pulse schedule or control waveforms.

        n_shots : int | None, optional
            Number of shots.
        shot_interval : float | None, optional
            Interval between shots in ns.
        shot_averaging : bool | None, optional
            Whether to average shots on hardware.
        time_integration : bool | None, optional
            Whether to integrate captured waveforms over time.
        state_classification : bool | None, optional
            Whether to enable state classification.
        frequencies : dict[str, float] | None, optional
            Channel-frequency overrides keyed by schedule label.
        readout_amplitudes : dict[str, float], optional
            Readout amplitude for each qubit.

        readout_duration : float, optional
            Readout duration in ns.

        readout_pre_margin : float, optional
            Readout pre-margin in ns.

        readout_post_margin : float, optional
            Readout post-margin in ns.

        readout_ramp_time : float, optional
            Readout ramp time in ns.

        readout_drag_coeff : float, optional
            Readout drag coefficient.

        readout_ramp_type : RampType, optional
            Readout ramp type.

        readout_amplification : bool | None, optional
            Whether to apply readout amplification pulses.
        final_measurement : bool | None, optional
            Whether to append a final readout measurement.
        classification_line_param0 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 0.
        classification_line_param1 : tuple[float, float, float] | None, optional
            Optional QuEL-1 classification line parameter 1.
        plot : bool | None, optional
            Whether to plot the results.

        Returns
        -------
        MultipleMeasureResult
            Measurement results.

        """
        legacy_options: dict[str, Any] = dict(deprecated_options)
        legacy_keys = {
            "mode",
            "shots",
            "interval",
            "readout_ramptime",
            "add_last_measurement",
            "add_pump_pulses",
            "enable_dsp_demodulation",
            "enable_dsp_sum",
            "enable_dsp_classification",
            "line_param0",
            "line_param1",
        }
        unknown_keys = sorted(set(legacy_options) - legacy_keys)
        if unknown_keys:
            joined = ", ".join(f"`{key}`" for key in unknown_keys)
            raise TypeError(f"Unexpected keyword argument(s): {joined}")

        legacy_mode = legacy_options.pop("mode", None)
        if legacy_mode is not None:
            self._warn_deprecated_alias(
                old_name="mode",
                new_name="shot_averaging",
            )
            legacy_shot_averaging = legacy_mode == "avg"
            if shot_averaging is not None and shot_averaging != legacy_shot_averaging:
                raise ValueError(
                    "`mode` conflicts with `shot_averaging`. "
                    "Provide only `shot_averaging`."
                )
            if shot_averaging is None:
                shot_averaging = legacy_shot_averaging
        if shot_averaging is None:
            shot_averaging = True

        n_shots = self._resolve_deprecated_alias(
            new_value=n_shots,
            old_value=legacy_options.pop("shots", None),
            old_name="shots",
            new_name="n_shots",
        )
        shot_interval = self._resolve_deprecated_alias(
            new_value=shot_interval,
            old_value=legacy_options.pop("interval", None),
            old_name="interval",
            new_name="shot_interval",
        )
        readout_ramp_time = self._resolve_deprecated_alias(
            new_value=readout_ramp_time,
            old_value=legacy_options.pop("readout_ramptime", None),
            old_name="readout_ramptime",
            new_name="readout_ramp_time",
        )
        final_measurement = self._resolve_deprecated_alias(
            new_value=final_measurement,
            old_value=legacy_options.pop("add_last_measurement", None),
            old_name="add_last_measurement",
            new_name="final_measurement",
        )
        if final_measurement is None:
            final_measurement = False

        readout_amplification = self._resolve_deprecated_alias(
            new_value=readout_amplification,
            old_value=legacy_options.pop("add_pump_pulses", None),
            old_name="add_pump_pulses",
            new_name="readout_amplification",
        )
        if readout_amplification is None:
            readout_amplification = False

        legacy_enable_dsp_demodulation = legacy_options.pop(
            "enable_dsp_demodulation", None
        )
        if legacy_enable_dsp_demodulation is not None:
            self._warn_deprecated_alias(
                old_name="enable_dsp_demodulation",
                message=(
                    "`enable_dsp_demodulation` is deprecated and ignored "
                    "because demodulation is always enabled."
                ),
            )
            if legacy_enable_dsp_demodulation is False:
                raise ValueError(
                    "enable_dsp_demodulation is deprecated and always enabled; "
                    "remove this argument or pass None."
                )

        time_integration = self._resolve_deprecated_alias(
            new_value=time_integration,
            old_value=legacy_options.pop("enable_dsp_sum", None),
            old_name="enable_dsp_sum",
            new_name="time_integration",
        )
        if time_integration is None:
            time_integration = True

        state_classification = self._resolve_deprecated_alias(
            new_value=state_classification,
            old_value=legacy_options.pop("enable_dsp_classification", None),
            old_name="enable_dsp_classification",
            new_name="state_classification",
        )
        if state_classification is None:
            state_classification = False
        if plot is None:
            plot = False

        classification_line_param0 = self._resolve_deprecated_alias(
            new_value=classification_line_param0,
            old_value=legacy_options.pop("line_param0", None),
            old_name="line_param0",
            new_name="classification_line_param0",
        )
        classification_line_param1 = self._resolve_deprecated_alias(
            new_value=classification_line_param1,
            old_value=legacy_options.pop("line_param1", None),
            old_name="line_param1",
            new_name="classification_line_param1",
        )

        if not isinstance(schedule, PulseSchedule):
            schedule = PulseSchedule.from_waveforms(schedule)

        measurement_config = self.measurement_config_factory.create(
            n_shots=n_shots,
            shot_interval=shot_interval,
            shot_averaging=shot_averaging,
            time_integration=time_integration,
            state_classification=state_classification,
        )

        measurement_schedule = self.build_measurement_schedule(
            pulse_schedule=schedule,
            frequencies=frequencies,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramp_time=readout_ramp_time,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            readout_amplification=readout_amplification,
            final_measurement=final_measurement,
            plot=plot,
        )

        if classification_line_param0 is None and classification_line_param1 is None:
            result = self.measurement_schedule_runner.execute_sync(
                schedule=measurement_schedule,
                config=measurement_config,
            )
        else:
            quel1_options = Quel1MeasurementOptions(
                classification_line_param0=classification_line_param0,
                classification_line_param1=classification_line_param1,
            )
            result = self.measurement_schedule_runner.execute_sync(
                schedule=measurement_schedule,
                config=measurement_config,
                quel1_options=quel1_options,
            )

        rawdata_dir = self.system_manager.rawdata_dir
        if rawdata_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            result_path = Path(rawdata_dir) / f"{timestamp}.nc"
            result.save(result_path)

        return MeasurementResultConverter.to_multiple_measure_result(
            result,
            config=self._resolve_device_config(self.backend_controller),
            classifiers=self.classifiers,
        )

    def capture_loopback(
        self,
        schedule: PulseSchedule | TargetMap[IQArray],
        *,
        n_shots: int | None = None,
        block_outputs: bool = True,
        shot_averaging: bool = True,
        demodulation: bool = True,
        include_read_in: bool = False,
        capture_targets: list[str] | None = None,
        configure_monitor_nco: bool = True,
    ) -> MeasurementResult:
        """
        Capture full-span loopback data on loopback-capable input channels.

        Parameters
        ----------
        schedule : PulseSchedule | TargetMap[IQArray]
            Pulse schedule or control waveforms to execute.
        n_shots : int | None, optional
            Number of shots.
        block_outputs : bool, optional
            Whether to block active output ports while loopback capture runs.
        shot_averaging : bool, optional
            Whether to average captured shots in Qubex after demodulation.
        demodulation : bool, optional
            Whether to demodulate captured waveforms. READ_IN captures use
            backend DSP demodulation. MNTR_IN captures are split by active
            source channel when needed, then low-pass filtered after Qubex-side
            software demodulation.
        include_read_in : bool, optional
            Whether to add matching READ_IN captures for active readout output
            targets when `capture_targets` is omitted.
        capture_targets : list[str] | None, optional
            Explicit loopback capture targets. When omitted, monitor inputs are
            resolved from active output boxes. Matching read-in inputs are added
            only when `include_read_in` is enabled.
        configure_monitor_nco : bool, optional
            Whether to configure monitor input frequency settings before
            capture. Disable only when the monitor NCO has already been primed
            and its phase origin must remain unchanged across repeated
            captures.

        Returns
        -------
        MeasurementResult
            Measurement result for loopback capture windows.
        """
        if not isinstance(schedule, PulseSchedule):
            schedule = PulseSchedule.from_waveforms(schedule)

        base_schedule = schedule.copy()
        resolved_capture_targets = (
            self._resolve_loopback_capture_targets(
                schedule=base_schedule,
                include_read_in=include_read_in,
            )
            if capture_targets is None
            else list(dict.fromkeys(capture_targets))
        )
        measurement_config = self.measurement_config_factory.create(
            n_shots=n_shots,
            shot_averaging=False,
            time_integration=False,
            state_classification=False,
        )

        def _run_once(run_spec: _LoopbackRunSpec) -> MeasurementResult:
            measurement_schedule = self.build_measurement_schedule(
                pulse_schedule=run_spec.pulse_schedule.copy(),
                capture_placement="entire_schedule",
                capture_targets=list(run_spec.capture_targets),
                final_measurement=False,
                readout_amplification=False,
                plot=False,
            )
            loopback_box_ids = self._resolve_loopback_box_ids(
                schedule=run_spec.pulse_schedule,
                capture_targets=run_spec.capture_targets,
            )
            self._initialize_loopback_capture_units(box_ids=loopback_box_ids)
            if configure_monitor_nco:
                self._configure_loopback_monitor_frequency_settings(
                    pulse_schedule=run_spec.pulse_schedule,
                    capture_targets=run_spec.capture_targets,
                    monitor_source_settings=run_spec.monitor_source_settings,
                )
            with self._temporary_loopback_rfswitches(
                schedule=run_spec.pulse_schedule,
                capture_targets=run_spec.capture_targets,
                block_outputs=block_outputs,
            ):
                result = _run_async(
                    lambda: self.run_measurement(
                        schedule=measurement_schedule,
                        config=measurement_config,
                        quel1_options=Quel1MeasurementOptions(
                            demodulation=run_spec.backend_demodulation
                        ),
                    )
                )
            return self._postprocess_loopback_result(
                result=result,
                measurement_schedule=measurement_schedule,
                capture_targets=run_spec.capture_targets,
                shot_averaging=shot_averaging,
                demodulation=run_spec.software_demodulation,
                monitor_source_settings=run_spec.monitor_source_settings,
            )

        def _run_plan_once(targets: Sequence[str]) -> MeasurementResult:
            run_specs = self._build_loopback_run_specs(
                pulse_schedule=base_schedule,
                capture_targets=targets,
                demodulation=demodulation,
            )
            result = self._merge_loopback_results(
                [_run_once(spec) for spec in run_specs]
            )
            return self._order_loopback_result_by_targets(
                result,
                target_order=base_schedule.labels,
            )

        try:
            return _run_plan_once(resolved_capture_targets)
        except Exception as exc:
            if not self._is_e7awg_capture_data_error(exc):
                raise
            logger.warning(
                "Loopback capture failed with broken-data error; retrying once after capture-unit initialization."
            )
            try:
                return _run_plan_once(resolved_capture_targets)
            except Exception as retry_exc:
                if not self._is_e7awg_capture_data_error(retry_exc):
                    raise
                monitor_only_targets = self._filter_loopback_capture_targets(
                    capture_targets=resolved_capture_targets,
                    port_type=PortType.MNTR_IN,
                )
                read_in_only_targets = self._filter_loopback_capture_targets(
                    capture_targets=resolved_capture_targets,
                    port_type=PortType.READ_IN,
                )
                fallback_targets = [
                    ("MNTR_IN", monitor_only_targets),
                    ("READ_IN", read_in_only_targets),
                ]
                for target_name, targets in fallback_targets:
                    if not targets or targets == list(resolved_capture_targets):
                        continue
                    logger.warning(
                        "Loopback capture still failed; retrying with %s targets only.",
                        target_name,
                    )
                    try:
                        return _run_plan_once(targets)
                    except Exception as fallback_exc:
                        if not self._is_e7awg_capture_data_error(fallback_exc):
                            raise
                raise

    def create_measurement_config(
        self,
        *,
        n_shots: int | None = None,
        shot_interval: float | None = None,
        shot_averaging: bool | None = None,
        time_integration: bool | None = None,
        state_classification: bool | None = None,
    ) -> MeasurementConfig:
        """
        Create a `MeasurementConfig` from optional runtime overrides.

        Parameters
        ----------
        n_shots : int | None, optional
            Number of shots.
        shot_interval : float | None, optional
            Interval between shots in ns.
        shot_averaging : bool | None, optional
            Whether to average shots on hardware.
        time_integration : bool | None, optional
            Whether to integrate captured waveforms over time.
        state_classification : bool | None, optional
            Whether to enable state classification.

        Returns
        -------
        MeasurementConfig
            Created measurement configuration.

        """
        return self.measurement_config_factory.create(
            n_shots=n_shots,
            shot_interval=shot_interval,
            shot_averaging=shot_averaging,
            time_integration=time_integration,
            state_classification=state_classification,
        )

    def build_measurement_schedule(
        self,
        pulse_schedule: PulseSchedule,
        *,
        frequencies: dict[str, float] | None = None,
        readout_amplitudes: dict[str, float] | None = None,
        readout_duration: float | None = None,
        readout_pre_margin: float | None = None,
        readout_post_margin: float | None = None,
        readout_ramp_time: float | None = None,
        readout_ramp_type: RampType | None = None,
        readout_drag_coeff: float | None = None,
        readout_amplification: bool | None = None,
        final_measurement: bool | None = None,
        capture_placement: CapturePlacement | None = None,
        capture_targets: list[str] | None = None,
        plot: bool | None = None,
    ) -> MeasurementSchedule:
        """Build a `MeasurementSchedule` from a pulse schedule and options."""
        measurement_schedule = self.schedule_builder.build(
            schedule=pulse_schedule,
            frequencies=frequencies,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramp_time=readout_ramp_time,
            readout_ramp_type=readout_ramp_type,
            readout_drag_coeff=readout_drag_coeff,
            readout_amplification=readout_amplification,
            final_measurement=final_measurement,
            capture_placement=capture_placement,
            capture_targets=capture_targets,
            plot=plot,
        )
        return measurement_schedule
