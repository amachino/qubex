"""Monitor-path loopback services for measurement workflows."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

import numpy as np
from qxpulse import PulseSchedule

from qubex.backend import BackendController
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge
from qubex.measurement.measurement_config_factory import MeasurementConfigFactory
from qubex.measurement.measurement_context import MeasurementContext
from qubex.measurement.models.capture_data import CaptureData
from qubex.measurement.models.measurement_result import MeasurementResult
from qubex.measurement.models.measurement_schedule import MeasurementSchedule
from qubex.measurement.models.quel1_measurement_options import Quel1MeasurementOptions
from qubex.measurement.services.measurement_execution_service import (
    MeasurementExecutionService,
)
from qubex.measurement.services.measurement_session_service import (
    MeasurementSessionService,
)
from qubex.system import ExperimentSystem, PortType, Target
from qubex.system.quel1.quel1_port_configurator import MixingUtil
from qubex.system.quel1.quel1_system_constants import NCO_STEP_HZ
from qubex.typing import IQArray, TargetMap

logger = logging.getLogger(__name__)

T = TypeVar("T")
RFSwitchState = Literal["pass", "block", "open", "loop"]
_QUEL1SE_R8_BOX_TYPE = "quel1se-riken8"
_QUEL1SE_R8_MONITOR_LO_HZ = 6_000_000_000
_QUEL1SE_R8_ADC_CNCO_MIN_HZ = -3_000_000_000
_QUEL1SE_R8_ADC_CNCO_MAX_HZ = 3_000_000_000
_OUTPUT_OWNED_MONITOR_LO_BOX_TYPES = frozenset(
    {
        "quel1se-fujitsu11-a",
        "quel1se-fujitsu11-b",
        "quel1-a",
        "quel1-b",
        "qube-riken-a",
        "qube-riken-b",
    }
)


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
    """
    Run one awaitable factory from a synchronous monitor API.

    Parameters
    ----------
    factory : Callable[[], Awaitable[T]]
        Factory that creates the awaitable to run on the shared measurement
        async bridge.
    timeout : float, optional
        Maximum time in seconds to wait for the awaitable.

    Returns
    -------
    T
        Result produced by the awaitable.
    """
    bridge = get_shared_async_bridge(key="measurement")
    return bridge.run(factory, timeout=timeout)


class MeasurementMonitorService:
    """
    Coordinate monitor-path loopback captures.

    Parameters
    ----------
    context : MeasurementContext
        Shared measurement context that owns the active experiment system.
    session_service : MeasurementSessionService
        Session service that provides the connected backend controller.
    execution_service : MeasurementExecutionService
        Execution service used only for ordinary schedule construction and
        measurement execution.

    Notes
    -----
    This service owns monitor-specific behavior: resolving loopback capture
    ports, preparing monitor receiver settings, applying temporary RF-switch
    states, and postprocessing monitor/read-in captures. Keeping those rules
    here leaves :class:`MeasurementExecutionService` focused on normal
    measurement execution.
    """

    def __init__(
        self,
        *,
        context: MeasurementContext,
        session_service: MeasurementSessionService,
        execution_service: MeasurementExecutionService,
    ) -> None:
        """
        Initialize the monitor service.

        Parameters
        ----------
        context : MeasurementContext
            Shared measurement context.
        session_service : MeasurementSessionService
            Session lifecycle and backend access service.
        execution_service : MeasurementExecutionService
            Service used to build and run measurement schedules.
        """
        self._context = context
        self._session_service = session_service
        self._execution_service = execution_service

    @property
    def context(self) -> MeasurementContext:
        """Return the measurement context."""
        return self._context

    @property
    def session_service(self) -> MeasurementSessionService:
        """Return the session lifecycle service."""
        return self._session_service

    @property
    def execution_service(self) -> MeasurementExecutionService:
        """Return the ordinary measurement execution service."""
        return self._execution_service

    @property
    def experiment_system(self) -> ExperimentSystem:
        """Return the active experiment system."""
        return self.context.experiment_system

    @property
    def backend_controller(self) -> BackendController:
        """Return the active backend controller."""
        return self.session_service.backend_controller

    @property
    def targets(self) -> dict[str, Target]:
        """Return available targets indexed by label."""
        return {target.label: target for target in self.experiment_system.targets}

    @property
    def measurement_config_factory(self) -> MeasurementConfigFactory:
        """Return a measurement-config factory."""
        return self.execution_service.measurement_config_factory

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

    @staticmethod
    def _normalize_loopback_box_type(box_type: object) -> str | None:
        """Return a normalized box type string."""
        if box_type is None:
            return None
        value = getattr(box_type, "value", box_type)
        if not isinstance(value, str):
            return None
        return value.lower()

    def _resolve_loopback_box_type(self, *, box_id: str) -> str | None:
        """Resolve the connected box type when available."""
        control_system = self.experiment_system.control_system
        get_box = getattr(control_system, "get_box", None)
        if callable(get_box):
            with suppress(KeyError, ValueError, AttributeError):
                box = get_box(box_id)
                for attr in ("type", "box_type", "boxtype"):
                    box_type = self._normalize_loopback_box_type(
                        getattr(box, attr, None)
                    )
                    if box_type is not None:
                        return box_type

        backend_get_box = getattr(self.backend_controller, "get_box", None)
        if callable(backend_get_box):
            with suppress(Exception):
                backend_box = backend_get_box(box_id)
                for attr in ("type", "box_type", "boxtype"):
                    box_type = self._normalize_loopback_box_type(
                        getattr(backend_box, attr, None)
                    )
                    if box_type is not None:
                        return box_type
        return None

    def _is_loopback_quel1se_r8_box(self, *, box_id: str) -> bool:
        """Return whether one box should use the quel1se-riken8 strategy."""
        return self._resolve_loopback_box_type(box_id=box_id) == _QUEL1SE_R8_BOX_TYPE

    def _uses_loopback_output_owned_monitor_lo_strategy(self, *, box_id: str) -> bool:
        """Return whether monitor LO must be preserved for one box."""
        return (
            self._resolve_loopback_box_type(box_id=box_id)
            in _OUTPUT_OWNED_MONITOR_LO_BOX_TYPES
        )

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
                source_dump.get("lo_freq"),
                getattr(source_port, "lo_freq", None),
            )
            cnco_freq_hz = self._first_loopback_frequency_hz(
                source_dump.get("cnco_freq"),
                getattr(source_port, "cnco_freq", None),
            )
            fnco_freq_hz = self._first_loopback_frequency_hz(
                self._resolve_loopback_dump_fnco_frequency_hz(
                    dump=source_dump,
                    section_name="channels",
                    channel_number=channel_number,
                ),
                getattr(channel, "fnco_freq", None),
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

    def _resolve_loopback_source_output_frequency_hz(
        self,
        source_setting: _LoopbackMonitorSourceSetting,
    ) -> int | None:
        """Resolve the RF frequency generated by one loopback source."""
        cnco_freq_hz = source_setting.cnco_freq_hz
        if cnco_freq_hz is None:
            return None
        fnco_freq_hz = source_setting.fnco_freq_hz or 0
        nco_freq_hz = cnco_freq_hz + fnco_freq_hz
        lo_freq_hz = source_setting.lo_freq_hz
        if lo_freq_hz is None:
            return nco_freq_hz

        sideband = getattr(source_setting.port, "sideband", None)
        if sideband == "U":
            return lo_freq_hz + nco_freq_hz
        if sideband == "L":
            return lo_freq_hz - nco_freq_hz
        return nco_freq_hz

    def _resolve_loopback_target_frequency_ghz(
        self,
        *,
        target_or_port_id: str,
        pulse_schedule: PulseSchedule,
    ) -> float | None:
        """Resolve the absolute RF frequency requested for one source target."""
        schedule_frequency = self._resolve_loopback_schedule_frequency(
            pulse_schedule=pulse_schedule,
            target=target_or_port_id,
        )
        if schedule_frequency is not None:
            return schedule_frequency
        try:
            target = self.experiment_system.get_target(target_or_port_id)
        except (AttributeError, KeyError, ValueError):
            return None
        frequency = getattr(target, "frequency", None)
        if isinstance(frequency, (int, float)):
            return float(frequency)
        return None

    def _resolve_loopback_observed_frequency_hz(
        self,
        *,
        source_setting: _LoopbackMonitorSourceSetting,
        pulse_schedule: PulseSchedule,
    ) -> int | None:
        """Resolve the RF frequency the monitor receiver should observe."""
        target_frequency = self._resolve_loopback_target_frequency_ghz(
            target_or_port_id=source_setting.label,
            pulse_schedule=pulse_schedule,
        )
        if target_frequency is not None:
            return round(target_frequency * 1e9)
        return self._resolve_loopback_source_output_frequency_hz(source_setting)

    @staticmethod
    def _resolve_loopback_quel1se_r8_monitor_sideband(
        source_setting: _LoopbackMonitorSourceSetting,
    ) -> Literal["U", "L"]:
        """Resolve the quel1se-riken8 monitor receiving sideband for one source."""
        if getattr(source_setting.port, "type", None) == PortType.READ_OUT:
            return "U"
        return "L"

    @staticmethod
    def _resolve_loopback_source_sideband(
        source_setting: _LoopbackMonitorSourceSetting,
    ) -> Literal["U", "L"] | None:
        """Resolve the source output sideband when it is known."""
        sideband = getattr(source_setting.port, "sideband", None)
        if sideband in ("U", "L"):
            return sideband
        return None

    @staticmethod
    def _calculate_loopback_monitor_cnco_hz(
        *,
        observed_freq_hz: int,
        lo_freq_hz: int,
        sideband: Literal["U", "L"],
    ) -> int:
        """Return monitor CNCO for a receiver LO and sideband."""
        if sideband == "U":
            return round(observed_freq_hz - lo_freq_hz)
        return round(lo_freq_hz - observed_freq_hz)

    @staticmethod
    def _calculate_loopback_monitor_nco_hz(
        *,
        observed_freq_hz: int,
        lo_freq_hz: int,
        sideband: Literal["U", "L"],
    ) -> tuple[int, int]:
        """Return monitor CNCO/FNCO settings on the standard QuEL-1 NCO grid."""
        cnco_freq_hz = (
            round(
                MeasurementMonitorService._calculate_loopback_monitor_cnco_hz(
                    observed_freq_hz=observed_freq_hz,
                    lo_freq_hz=lo_freq_hz,
                    sideband=sideband,
                )
                / NCO_STEP_HZ
            )
            * NCO_STEP_HZ
        )
        fnco_freq_hz, _ = MixingUtil.calc_fnco(
            f=observed_freq_hz,
            ssb=sideband,
            lo=lo_freq_hz,
            cnco=cnco_freq_hz,
        )
        return cnco_freq_hz, fnco_freq_hz

    def _resolve_loopback_monitor_lo_setting(
        self,
        *,
        monitor_port: Any,
        monitor_dump: Mapping[str, Any],
    ) -> tuple[int | None, bool]:
        """Resolve monitor LO and whether the strategy should set it."""
        if self._is_loopback_quel1se_r8_box(box_id=str(monitor_port.box_id)):
            return _QUEL1SE_R8_MONITOR_LO_HZ, True

        dump_lo_freq_hz = self._coerce_loopback_frequency_hz(
            monitor_dump.get("lo_freq")
        )
        if dump_lo_freq_hz is not None:
            return dump_lo_freq_hz, False

        model_lo_freq_hz = self._coerce_loopback_frequency_hz(
            getattr(monitor_port, "lo_freq", None)
        )
        return model_lo_freq_hz, False

    def _set_loopback_shared_monitor_lo_model(
        self,
        *,
        monitor_port: Any,
        lo_freq_hz: int,
    ) -> None:
        """Update model LO for monitor inputs that share the quel1se-riken8 monitor LO."""
        control_system = self.experiment_system.control_system
        updated = False
        get_box = getattr(control_system, "get_box", None)
        if callable(get_box) and self._is_loopback_quel1se_r8_box(
            box_id=str(monitor_port.box_id)
        ):
            with suppress(KeyError, ValueError, AttributeError):
                box = get_box(monitor_port.box_id)
                for candidate in getattr(box, "ports", ()):
                    if getattr(candidate, "type", None) != PortType.MNTR_IN:
                        continue
                    control_system.set_port_params(
                        box_id=candidate.box_id,
                        port_number=candidate.number,
                        lo_freq=lo_freq_hz,
                    )
                    updated = True
        if not updated:
            control_system.set_port_params(
                box_id=monitor_port.box_id,
                port_number=monitor_port.number,
                lo_freq=lo_freq_hz,
            )

    def _resolve_loopback_monitor_receiver_frequency_ghz(
        self,
        *,
        capture_target: str,
        monitor_source_label: str | None = None,
    ) -> float | None:
        """Resolve the current monitor receiver center frequency in GHz."""
        monitor_port = self._resolve_loopback_capture_port(
            target_or_port_id=capture_target,
        )
        if monitor_port is None or monitor_port.type != PortType.MNTR_IN:
            return None
        monitor_dump = self._dump_loopback_port_config(port=monitor_port)
        lo_freq_hz, _ = self._resolve_loopback_monitor_lo_setting(
            monitor_port=monitor_port,
            monitor_dump=monitor_dump,
        )
        cnco_freq_hz = self._first_loopback_frequency_hz(
            getattr(monitor_port, "cnco_freq", None),
            monitor_dump.get("cnco_freq"),
        )
        if cnco_freq_hz is None:
            return None
        channels = getattr(monitor_port, "channels", ())
        channel_number = 0
        channel = None
        if isinstance(channels, Sequence) and channels:
            channel = channels[0]
            channel_number = self._resolve_loopback_channel_number(channel)
        fnco_freq_hz = self._first_loopback_frequency_hz(
            None if channel is None else getattr(channel, "fnco_freq", None),
            self._resolve_loopback_dump_fnco_frequency_hz(
                dump=monitor_dump,
                section_name="runits",
                channel_number=channel_number,
            ),
        )
        nco_freq_hz = cnco_freq_hz + (fnco_freq_hz or 0)
        if lo_freq_hz is None:
            return nco_freq_hz * 1e-9
        if monitor_source_label is not None:
            target = self.targets.get(monitor_source_label)
            if target is not None:
                source_port = getattr(getattr(target, "channel", None), "port", None)
                if (
                    self._is_loopback_quel1se_r8_box(box_id=str(monitor_port.box_id))
                    and getattr(source_port, "type", None) == PortType.READ_OUT
                ):
                    return (lo_freq_hz + nco_freq_hz) * 1e-9
                if (
                    self._uses_loopback_output_owned_monitor_lo_strategy(
                        box_id=str(monitor_port.box_id)
                    )
                    and getattr(source_port, "sideband", None) == "U"
                ):
                    return (lo_freq_hz + nco_freq_hz) * 1e-9
        return (lo_freq_hz - nco_freq_hz) * 1e-9

    def _resolve_loopback_monitor_demodulation_frequency_ghz(
        self,
        *,
        capture_target: str,
        monitor_source_label: str,
        monitor_source_setting: _LoopbackMonitorSourceSetting | None = None,
        pulse_schedule: PulseSchedule,
    ) -> float | None:
        """Resolve software demodulation from observed RF and receiver center."""
        observed_frequency_hz = (
            None
            if monitor_source_setting is None
            else self._resolve_loopback_observed_frequency_hz(
                source_setting=monitor_source_setting,
                pulse_schedule=pulse_schedule,
            )
        )
        receiver_frequency = self._resolve_loopback_monitor_receiver_frequency_ghz(
            capture_target=capture_target,
            monitor_source_label=monitor_source_label,
        )
        if (
            observed_frequency_hz is not None
            and receiver_frequency is not None
            and monitor_source_setting is not None
        ):
            observed_frequency = observed_frequency_hz * 1e-9
            capture_port = self._resolve_loopback_capture_port(
                target_or_port_id=capture_target,
            )
            sideband = self._resolve_loopback_monitor_receiver_sideband(
                monitor_port=capture_port,
                source_setting=monitor_source_setting,
            )
            if sideband == "L":
                return receiver_frequency - observed_frequency
            return observed_frequency - receiver_frequency

        observed_frequency = self._resolve_loopback_target_frequency_ghz(
            target_or_port_id=monitor_source_label,
            pulse_schedule=pulse_schedule,
        )
        if observed_frequency is not None and receiver_frequency is not None:
            return observed_frequency - receiver_frequency
        return self._resolve_loopback_modulation_frequency_ghz(
            target_or_port_id=monitor_source_label,
            pulse_schedule=pulse_schedule,
        )

    def _resolve_loopback_monitor_receiver_sideband(
        self,
        *,
        monitor_port: Any | None,
        source_setting: _LoopbackMonitorSourceSetting,
    ) -> Literal["U", "L"] | None:
        """Resolve the monitor receiver sideband used for software residuals."""
        if monitor_port is not None and self._is_loopback_quel1se_r8_box(
            box_id=str(monitor_port.box_id),
        ):
            return self._resolve_loopback_quel1se_r8_monitor_sideband(source_setting)
        if (
            monitor_port is not None
            and self._uses_loopback_output_owned_monitor_lo_strategy(
                box_id=str(monitor_port.box_id),
            )
        ):
            return self._resolve_loopback_source_sideband(source_setting) or "L"
        return self._resolve_loopback_source_sideband(source_setting)

    def _configure_loopback_quel1se_r8_monitor_for_source(
        self,
        *,
        pulse_schedule: PulseSchedule,
        capture_target: str,
        monitor_port: Any,
        source_setting: _LoopbackMonitorSourceSetting,
    ) -> bool:
        """Configure quel1se-riken8 monitor LO/CNCO/FNCO with a fixed 6 GHz monitor LO."""
        if not self._is_loopback_quel1se_r8_box(box_id=str(monitor_port.box_id)):
            return False

        observed_freq_hz = self._resolve_loopback_observed_frequency_hz(
            source_setting=source_setting,
            pulse_schedule=pulse_schedule,
        )
        if observed_freq_hz is None:
            logger.warning(
                "Cannot configure monitor NCO for %s because source %s frequency is unavailable.",
                capture_target,
                source_setting.label,
            )
            return False

        monitor_dump = self._dump_loopback_port_config(port=monitor_port)
        lo_freq_hz, _ = self._resolve_loopback_monitor_lo_setting(
            monitor_port=monitor_port,
            monitor_dump=monitor_dump,
        )
        if lo_freq_hz is None:
            logger.warning(
                "Cannot configure monitor NCO for %s because monitor LO is unavailable.",
                capture_target,
            )
            return False

        sideband = self._resolve_loopback_quel1se_r8_monitor_sideband(source_setting)
        cnco_freq_hz, fnco_freq_hz = self._calculate_loopback_monitor_nco_hz(
            observed_freq_hz=observed_freq_hz,
            lo_freq_hz=lo_freq_hz,
            sideband=sideband,
        )
        if not (
            _QUEL1SE_R8_ADC_CNCO_MIN_HZ <= cnco_freq_hz < _QUEL1SE_R8_ADC_CNCO_MAX_HZ
        ):
            logger.warning(
                "Cannot configure monitor NCO for %s because computed CNCO %s Hz is outside the quel1se-riken8 ADC-CNCO range [%s, %s) Hz.",
                capture_target,
                cnco_freq_hz,
                _QUEL1SE_R8_ADC_CNCO_MIN_HZ,
                _QUEL1SE_R8_ADC_CNCO_MAX_HZ,
            )
            return False

        config_port = getattr(self.backend_controller, "config_port", None)
        config_runit = getattr(self.backend_controller, "config_runit", None)
        if not callable(config_port):
            return False

        config_port(
            box_name=monitor_port.box_id,
            port=monitor_port.number,
            lo_freq_hz=lo_freq_hz,
            cnco_freq_hz=cnco_freq_hz,
        )

        monitor_channels = getattr(monitor_port, "channels", ())
        if callable(config_runit):
            runit = 0
            if isinstance(monitor_channels, Sequence) and monitor_channels:
                runit = self._resolve_loopback_channel_number(monitor_channels[0])
            config_runit(
                box_name=monitor_port.box_id,
                port=monitor_port.number,
                runit=runit,
                fnco_freq_hz=fnco_freq_hz,
            )

        if lo_freq_hz is not None:
            self._set_loopback_shared_monitor_lo_model(
                monitor_port=monitor_port,
                lo_freq_hz=lo_freq_hz,
            )
        model_updates: dict[str, Any] = {"cnco_freq": cnco_freq_hz}
        if isinstance(monitor_channels, Sequence) and monitor_channels:
            model_updates["fnco_freqs"] = [fnco_freq_hz for _ in monitor_channels]
        self.experiment_system.control_system.set_port_params(
            box_id=monitor_port.box_id,
            port_number=monitor_port.number,
            **model_updates,
        )
        logger.info(
            "Configure quel1se-riken8 monitor receiver for %s from %s: set LO=%s Hz, sideband=%s, set CNCO=%s Hz, set FNCO=%s Hz.",
            capture_target,
            source_setting.label,
            lo_freq_hz,
            sideband,
            cnco_freq_hz,
            fnco_freq_hz,
        )
        return True

    def _configure_loopback_output_owned_monitor_lo_for_source(
        self,
        *,
        pulse_schedule: PulseSchedule,
        capture_target: str,
        monitor_port: Any,
        source_setting: _LoopbackMonitorSourceSetting,
    ) -> bool:
        """Configure monitor CNCO/FNCO without changing an output-owned monitor LO."""
        if not self._uses_loopback_output_owned_monitor_lo_strategy(
            box_id=str(monitor_port.box_id)
        ):
            return False

        observed_freq_hz = self._resolve_loopback_observed_frequency_hz(
            source_setting=source_setting,
            pulse_schedule=pulse_schedule,
        )
        if observed_freq_hz is None:
            logger.warning(
                "Cannot configure monitor NCO for %s because source %s frequency is unavailable.",
                capture_target,
                source_setting.label,
            )
            return False

        monitor_dump = self._dump_loopback_port_config(port=monitor_port)
        lo_freq_hz, _ = self._resolve_loopback_monitor_lo_setting(
            monitor_port=monitor_port,
            monitor_dump=monitor_dump,
        )
        if lo_freq_hz is None:
            logger.warning(
                "Cannot configure monitor NCO for %s because monitor LO is unavailable.",
                capture_target,
            )
            return False

        sideband = self._resolve_loopback_source_sideband(source_setting) or "L"
        cnco_freq_hz, fnco_freq_hz = self._calculate_loopback_monitor_nco_hz(
            observed_freq_hz=observed_freq_hz,
            lo_freq_hz=lo_freq_hz,
            sideband=sideband,
        )

        config_port = getattr(self.backend_controller, "config_port", None)
        config_runit = getattr(self.backend_controller, "config_runit", None)
        if not callable(config_port):
            return False

        config_port(
            box_name=monitor_port.box_id,
            port=monitor_port.number,
            cnco_freq_hz=cnco_freq_hz,
        )

        monitor_channels = getattr(monitor_port, "channels", ())
        if callable(config_runit):
            runit = 0
            if isinstance(monitor_channels, Sequence) and monitor_channels:
                runit = self._resolve_loopback_channel_number(monitor_channels[0])
            config_runit(
                box_name=monitor_port.box_id,
                port=monitor_port.number,
                runit=runit,
                fnco_freq_hz=fnco_freq_hz,
            )

        model_updates: dict[str, Any] = {"cnco_freq": cnco_freq_hz}
        if isinstance(monitor_channels, Sequence) and monitor_channels:
            model_updates["fnco_freqs"] = [fnco_freq_hz for _ in monitor_channels]
        self.experiment_system.control_system.set_port_params(
            box_id=monitor_port.box_id,
            port_number=monitor_port.number,
            **model_updates,
        )
        logger.info(
            "Configure monitor receiver for %s from %s without changing output-owned LO=%s Hz: sideband=%s, set CNCO=%s Hz, set FNCO=%s Hz.",
            capture_target,
            source_setting.label,
            lo_freq_hz,
            sideband,
            cnco_freq_hz,
            fnco_freq_hz,
        )
        return True

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

            if self._is_loopback_quel1se_r8_box(box_id=str(monitor_port.box_id)):
                if not self._configure_loopback_quel1se_r8_monitor_for_source(
                    pulse_schedule=pulse_schedule,
                    capture_target=capture_target,
                    monitor_port=monitor_port,
                    source_setting=source_setting,
                ):
                    logger.warning(
                        "Cannot configure quel1se-riken8 monitor receiver for %s from %s.",
                        capture_target,
                        source_setting.label,
                    )
                continue

            if self._uses_loopback_output_owned_monitor_lo_strategy(
                box_id=str(monitor_port.box_id)
            ):
                if not self._configure_loopback_output_owned_monitor_lo_for_source(
                    pulse_schedule=pulse_schedule,
                    capture_target=capture_target,
                    monitor_port=monitor_port,
                    source_setting=source_setting,
                ):
                    logger.warning(
                        "Cannot configure output-owned monitor receiver for %s from %s.",
                        capture_target,
                        source_setting.label,
                    )
                continue

            source_port = source_setting.port
            lo_freq_hz = source_setting.lo_freq_hz
            cnco_freq_hz = source_setting.cnco_freq_hz
            fnco_freq_hz = source_setting.fnco_freq_hz
            if lo_freq_hz is None and cnco_freq_hz is None and fnco_freq_hz is None:
                continue
            if lo_freq_hz is None:
                logger.warning(
                    "Cannot configure monitor NCO for %s because source %s has no LO frequency and no box-specific strategy is available.",
                    capture_target,
                    source_setting.label,
                )
                continue

            config_port(
                box_name=monitor_port.box_id,
                port=monitor_port.number,
                lo_freq_hz=lo_freq_hz,
                cnco_locked_with=source_port.number,
            )

            model_updates: dict[str, Any] = {"lo_freq": lo_freq_hz}
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
        monitor_source_setting: _LoopbackMonitorSourceSetting | None = None,
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
            source_frequency = (
                self._resolve_loopback_monitor_demodulation_frequency_ghz(
                    capture_target=capture_target,
                    monitor_source_label=monitor_source_label,
                    monitor_source_setting=monitor_source_setting,
                    pulse_schedule=pulse_schedule,
                )
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
            monitor_source_setting = (
                None
                if monitor_source_settings is None
                else monitor_source_settings.get(capture_target)
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
                        if monitor_source_setting is None
                        else monitor_source_setting.label
                    ),
                    monitor_source_setting=monitor_source_setting,
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
            source channel when needed, then software-demodulated in Qubex
            while preserving the captured pulse envelope.
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
            measurement_schedule = self.execution_service.build_measurement_schedule(
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
                    lambda: self.execution_service.run_measurement(
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
