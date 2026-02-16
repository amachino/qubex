# ruff: noqa: SLF001

"""Parallel builder for qubecalib-compatible multi-box actions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from logging import Logger
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias, TypeGuard, cast

from qubex.backend.quel1.quel1_box_adapter import adapt_quel1_box
from qubex.backend.quel1.quel1_driver_loader import load_quel1_driver
from qubex.backend.quel1.quel1_qubealib_protocols import (
    ActionProtocol,
    AwgSettingProtocol,
    MultiActionProtocol,
    PortType,
    Quel1BoxCommonProtocol,
    Quel1SystemProtocol,
    RunitSettingProtocol,
    SingleActionProtocol,
    SingleAwgIdProtocol,
    SingleAwgSettingProtocol,
    SingleRunitIdProtocol,
    SingleRunitSettingProtocol,
    SingleTriggerSettingProtocol,
    TriggerSettingProtocol,
)

PortLike: TypeAlias = PortType | str
CaptureParamMapKey: TypeAlias = tuple[str, PortType, int]
CaptureParamMap: TypeAlias = dict[CaptureParamMapKey, object]
SingleSetting: TypeAlias = (
    SingleRunitSettingProtocol | SingleAwgSettingProtocol | SingleTriggerSettingProtocol
)
SingleAwgIdFactory: TypeAlias = Callable[..., SingleAwgIdProtocol]
SingleAwgSettingFactory: TypeAlias = Callable[..., SingleAwgSettingProtocol]
SingleRunitIdFactory: TypeAlias = Callable[..., SingleRunitIdProtocol]
SingleRunitSettingFactory: TypeAlias = Callable[..., SingleRunitSettingProtocol]
SingleTriggerSettingFactory: TypeAlias = Callable[..., SingleTriggerSettingProtocol]


class _ActionBuilderProtocol(Protocol):
    """Protocol for Action.build-compatible callables."""

    def __call__(
        self,
        *,
        system: Quel1SystemProtocol,
        settings: list[object],
    ) -> ActionProtocol:
        """Build an action from common settings."""
        ...


class _WavegenTaskProtocol(Protocol):
    """Internal future-like protocol used by parallel wavegen reservation."""

    def result(self) -> object:
        """Wait for completion."""
        ...


def _is_runit_setting_shape(
    setting: object,
) -> TypeGuard[RunitSettingProtocol]:
    runit_ref = getattr(setting, "runit", None)
    return (
        runit_ref is not None
        and hasattr(setting, "cprm")
        and all(hasattr(runit_ref, attr) for attr in ("box", "port", "runit"))
    )


def _is_awg_setting_shape(setting: object) -> TypeGuard[AwgSettingProtocol]:
    awg_ref = getattr(setting, "awg", None)
    return (
        awg_ref is not None
        and hasattr(setting, "wseq")
        and all(hasattr(awg_ref, attr) for attr in ("box", "port", "channel"))
    )


def _is_trigger_setting_shape(
    setting: object,
) -> TypeGuard[TriggerSettingProtocol]:
    trigger_awg_ref = getattr(setting, "trigger_awg", None)
    return (
        trigger_awg_ref is not None
        and hasattr(setting, "triggerd_port")
        and all(hasattr(trigger_awg_ref, attr) for attr in ("box", "port", "channel"))
    )


def _resolve_clock_health_checks(
    options: ClockHealthCheckOptions | None,
) -> ClockHealthCheckOptions:
    """Return effective clock check options with default fallback."""
    return ClockHealthCheckOptions() if options is None else options


def _instantiate_awg_id(
    *,
    awg_id_class: SingleAwgIdFactory,
    box: str,
    port: PortLike,
    channel: int,
) -> SingleAwgIdProtocol:
    """Instantiate AWG ID for either common-style or single-style constructors."""
    try:
        return awg_id_class(box=box, port=port, channel=channel)
    except TypeError:
        return awg_id_class(port=port, channel=channel)


def _instantiate_runit_id(
    *,
    runit_id_class: SingleRunitIdFactory,
    box: str,
    port: PortLike,
    runit: int,
) -> SingleRunitIdProtocol:
    """Instantiate runit ID for either common-style or single-style constructors."""
    try:
        return runit_id_class(box=box, port=port, runit=runit)
    except TypeError:
        return runit_id_class(port=port, runit=runit)


@dataclass(frozen=True)
class _NormalizedRunitSetting:
    box: str
    port: PortLike
    runit: int
    cprm: object


@dataclass(frozen=True)
class _NormalizedAwgSetting:
    box: str
    port: PortLike
    channel: int
    wseq: object


@dataclass(frozen=True)
class _NormalizedTriggerSetting:
    box: str
    port: PortLike
    channel: int
    triggerd_port: PortLike


NormalizedSetting: TypeAlias = (
    _NormalizedRunitSetting | _NormalizedAwgSetting | _NormalizedTriggerSetting
)


def _normalize_common_settings(settings: Sequence[object]) -> list[NormalizedSetting]:
    """Extract known common-setting rows into normalized records."""
    normalized: list[NormalizedSetting] = []
    for setting in settings:
        if _is_runit_setting_shape(setting):
            normalized.append(
                _NormalizedRunitSetting(
                    box=setting.runit.box,
                    port=setting.runit.port,
                    runit=setting.runit.runit,
                    cprm=setting.cprm,
                )
            )
            continue
        if _is_awg_setting_shape(setting):
            normalized.append(
                _NormalizedAwgSetting(
                    box=setting.awg.box,
                    port=setting.awg.port,
                    channel=setting.awg.channel,
                    wseq=setting.wseq,
                )
            )
            continue
        if _is_trigger_setting_shape(setting):
            normalized.append(
                _NormalizedTriggerSetting(
                    box=setting.trigger_awg.box,
                    port=setting.trigger_awg.port,
                    channel=setting.trigger_awg.channel,
                    triggerd_port=setting.triggerd_port,
                )
            )
    return normalized


def _convert_to_box_setting_dict(
    *,
    settings: Sequence[object],
    awg_id_class: SingleAwgIdFactory,
    awg_setting_class: SingleAwgSettingFactory,
    runit_id_class: SingleRunitIdFactory,
    runit_setting_class: SingleRunitSettingFactory,
    trigger_setting_class: SingleTriggerSettingFactory,
) -> dict[str, list[SingleSetting]]:
    """
    Convert common driver settings into per-box single-driver settings.

    Parameters
    ----------
    settings : Sequence[object]
        Flat common settings where each entry includes a `box` location.
    awg_id_class : FactoryLike
        AWG identifier class.
    awg_setting_class : FactoryLike
        AWG setting class.
    runit_id_class : FactoryLike
        Runit identifier class.
    runit_setting_class : FactoryLike
        Runit setting class.
    trigger_setting_class : FactoryLike
        Trigger setting class.

    Returns
    -------
    dict[str, list[SingleSetting]]
        Settings grouped by box name.
    """
    settings_by_box: dict[str, list[SingleSetting]] = defaultdict(list)
    for setting in _normalize_common_settings(settings):
        if isinstance(setting, _NormalizedRunitSetting):
            settings_by_box[setting.box].append(
                runit_setting_class(
                    _instantiate_runit_id(
                        runit_id_class=runit_id_class,
                        box=setting.box,
                        port=setting.port,
                        runit=setting.runit,
                    ),
                    setting.cprm,
                )
            )
        elif isinstance(setting, _NormalizedAwgSetting):
            settings_by_box[setting.box].append(
                awg_setting_class(
                    _instantiate_awg_id(
                        awg_id_class=awg_id_class,
                        box=setting.box,
                        port=setting.port,
                        channel=setting.channel,
                    ),
                    setting.wseq,
                )
            )
        else:
            trigger_setting = setting
            settings_by_box[trigger_setting.box].append(
                trigger_setting_class(
                    _instantiate_awg_id(
                        awg_id_class=awg_id_class,
                        box=trigger_setting.box,
                        port=trigger_setting.port,
                        channel=trigger_setting.channel,
                    ),
                    trigger_setting.triggerd_port,
                )
            )
    return settings_by_box


def _collect_single_action_cprms(
    *,
    box_name: str,
    single_action: SingleActionProtocol,
) -> CaptureParamMap:
    """
    Collect capture parameters from one box-scoped single action.

    Parameters
    ----------
    box_name : str
        Box name associated with the given action.
    single_action : SingleActionProtocol
        Built single action exposing `_cprms`.

    Returns
    -------
    CaptureParamMap
        Capture parameter map keyed by `(box_name, port, runit)`.
    """
    cprms: CaptureParamMap = {}
    for runit_id, capture_param in single_action._cprms.items():
        cprms[(box_name, runit_id.port, runit_id.runit)] = capture_param
    return cprms


def _collect_multi_action_cprms(
    *,
    actions: Mapping[str, SingleActionProtocol],
) -> CaptureParamMap:
    """
    Collect capture parameters from all box-scoped single actions.

    Parameters
    ----------
    actions : Mapping[str, SingleActionProtocol]
        Per-box single actions.

    Returns
    -------
    CaptureParamMap
        Flattened capture parameter map over all boxes.
    """
    cprms: CaptureParamMap = {}
    for box_name, single_action in actions.items():
        cprms.update(
            _collect_single_action_cprms(
                box_name=box_name,
                single_action=single_action,
            )
        )
    return cprms


def _validate_box_names(
    *,
    system: Quel1SystemProtocol,
    settings_by_box: Mapping[str, list[SingleSetting]],
) -> None:
    """Validate that all settings reference existing system boxes."""
    missing_boxes = [name for name in settings_by_box if name not in system.boxes]
    if missing_boxes:
        raise ValueError(f"box {missing_boxes[0]} not found in system")


def _build_single_action_for_box(
    *,
    item: tuple[str, list[SingleSetting]],
    system: Quel1SystemProtocol,
    single_action: type[SingleActionProtocol],
    multi_action: type[MultiActionProtocol],
    logger: Logger,
    clock_health_checks: ClockHealthCheckOptions,
) -> tuple[str, SingleActionProtocol]:
    """Build one box-scoped single action."""
    box_name, box_settings = item
    box = adapt_quel1_box(system.box[box_name])
    if clock_health_checks.read_box_latched_clock_on_build:
        current_time = box.get_current_timecounter()
        last_sysref_time = box.get_latest_sysref_timecounter()
        logger.debug(
            "clock of %s, current: %s, last sysref: %s, last sysref offset: %s",
            box_name,
            current_time,
            last_sysref_time,
            multi_action._mod_by_sysref(last_sysref_time),
        )
    built_single_action = single_action.build(
        box=cast(Quel1BoxCommonProtocol, box),
        settings=cast(
            list[
                SingleRunitSettingProtocol
                | SingleAwgSettingProtocol
                | SingleTriggerSettingProtocol
            ],
            box_settings,
        ),
    )
    return box_name, built_single_action


def _build_single_actions_parallel(
    *,
    settings_by_box: Mapping[str, list[SingleSetting]],
    system: Quel1SystemProtocol,
    single_action: type[SingleActionProtocol],
    multi_action: type[MultiActionProtocol],
    logger: Logger,
    clock_health_checks: ClockHealthCheckOptions,
) -> dict[str, SingleActionProtocol]:
    """Build one single action per box in parallel."""
    max_workers = max(1, len(settings_by_box))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        actions = dict(
            executor.map(
                lambda item: _build_single_action_for_box(
                    item=item,
                    system=system,
                    single_action=single_action,
                    multi_action=multi_action,
                    logger=logger,
                    clock_health_checks=clock_health_checks,
                ),
                settings_by_box.items(),
            )
        )
    return actions


def _estimate_timediff(
    *,
    actions: Mapping[str, SingleActionProtocol],
    multi_action: type[MultiActionProtocol],
    clock_health_checks: ClockHealthCheckOptions,
) -> tuple[str, int, dict[str, int]]:
    """Estimate inter-box timing differences from SYSREF measurements."""
    reference_box_name = multi_action._get_reference_box_name(actions)
    if not clock_health_checks.measure_average_sysref_offset:
        return reference_box_name, 0, dict.fromkeys(actions, 0)

    max_workers = max(1, len(actions))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        average_offsets_at_sysref_clock = dict(
            executor.map(
                lambda item: (
                    item[0],
                    multi_action._measure_average_offset_at_sysref_clock(item[1].box),
                ),
                actions.items(),
            )
        )
    ref_sysref_time_offset = average_offsets_at_sysref_clock[reference_box_name]
    estimated_timediff = {
        box_name: average_offset - ref_sysref_time_offset
        for box_name, average_offset in average_offsets_at_sysref_clock.items()
    }
    return reference_box_name, ref_sysref_time_offset, estimated_timediff


@dataclass(frozen=True)
class ClockHealthCheckOptions:
    """
    Options for clock-related validation and diagnostics in parallel execution.

    Parameters
    ----------
    read_master_clock : bool, optional
        Read and log master clock during build.
    read_box_latched_clock_on_build : bool, optional
        Read and log each box latched clock during build.
    measure_average_sysref_offset : bool, optional
        Measure average SYSREF offsets and estimate inter-box timediff.
    validate_sysref_fluctuation_on_emit : bool, optional
        Read latest latched SYSREF at emit-time and warn on fluctuation.
    """

    read_master_clock: bool = False
    read_box_latched_clock_on_build: bool = False
    measure_average_sysref_offset: bool = False
    validate_sysref_fluctuation_on_emit: bool = False


@dataclass(frozen=True)
class QubexMultiAction:
    """Qubex-side multi-action with optional clock-health I/O."""

    _system: Quel1SystemProtocol
    _actions: MappingProxyType[str, SingleActionProtocol]
    _estimated_timediff: MappingProxyType[str, int]
    _reference_box_name: str
    _ref_sysref_time_offset: int
    _clock_options: ClockHealthCheckOptions
    _logger: Logger

    SYSREF_PERIOD: Final[int] = 2_000
    TIMING_OFFSET: Final[int] = 0
    MIN_TIME_OFFSET: Final[int] = 12_500_000

    @classmethod
    def _mod_by_sysref(cls, t: int) -> int:
        """Convert absolute counter into signed offset within SYSREF period."""
        half = cls.SYSREF_PERIOD // 2
        return (t + half) % cls.SYSREF_PERIOD - half

    @staticmethod
    def _has_capture_setting(action: SingleActionProtocol) -> bool:
        """Return True if action has runit capture settings."""
        return bool(action._cprms)

    def capture_start(self) -> dict[str, dict[PortType, object]]:
        """Start capture for boxes that include capture settings."""
        return {
            name: action.capture_start()
            for name, action in self._actions.items()
            if self._has_capture_setting(action)
        }

    def capture_stop(
        self,
        futures: dict[str, dict[PortType, object]],
    ) -> tuple[
        dict[tuple[str, PortType], object],
        dict[tuple[str, PortType, int], object],
    ]:
        """Stop capture and flatten per-box status/data maps."""
        box_results = {
            name: self._actions[name].capture_stop(future)
            for name, future in futures.items()
        }
        status: dict[tuple[str, PortType], object] = {}
        data: dict[tuple[str, PortType, int], object] = {}
        for name, (box_status, box_data) in box_results.items():
            for port, capture_return_code in box_status.items():
                status[(name, port)] = capture_return_code
            for (port, runit), runit_data in box_data.items():
                data[(name, port, runit)] = runit_data
        return status, data

    def action(
        self,
    ) -> tuple[
        dict[tuple[str, PortType], object],
        dict[tuple[str, PortType, int], object],
    ]:
        """Run capture start -> timed emission reservation -> capture stop."""
        futures = self.capture_start()
        self.emit_at(displacement=self._system.displacement)
        return self.capture_stop(futures)

    def emit_at(
        self,
        *,
        min_time_offset: int = MIN_TIME_OFFSET,
        displacement: int = 0,
    ) -> None:
        """
        Reserve synchronized emission timing across boxes.

        When clock validation is enabled, this method performs additional
        latched-clock reads and fluctuation checks before scheduling emission.
        """
        reference_box = adapt_quel1_box(self._system.box[self._reference_box_name])

        if self._clock_options.validate_sysref_fluctuation_on_emit:
            for name, action in self._actions.items():
                last_sysref = action.box.get_latest_sysref_timecounter()
                self._logger.debug(
                    f"sysref offset of {name}: latest: {self._mod_by_sysref(last_sysref)}"
                )
            current_time = reference_box.get_current_timecounter()
            last_sysref = reference_box.get_latest_sysref_timecounter()
            fluctuation = (
                self._mod_by_sysref(last_sysref) - self._ref_sysref_time_offset
            )
            if abs(fluctuation) > 4:
                self._logger.warning(
                    "large fluctuation (= %s) of sysref is detected from the previous timing measurement",
                    fluctuation,
                )
        else:
            current_time = reference_box.get_current_timecounter()

        awgs_by_box = {
            name: {(spec.port, spec.channel) for spec in action._wseqs}
            for name, action in self._actions.items()
        }
        base_time = current_time + min_time_offset
        align_offset = (16 - (base_time - self._ref_sysref_time_offset) % 16) % 16
        base_time += align_offset + displacement + self.TIMING_OFFSET

        timing_shift = self._system.timing_shift
        tasks: list[_WavegenTaskProtocol] = []
        for name, action in self._actions.items():
            if getattr(action, "_triggers", {}):
                continue
            scheduled_time = (
                base_time + self._estimated_timediff[name] + timing_shift[name]
            )
            if awgs_by_box[name]:
                tasks.append(
                    action.box.start_wavegen(
                        awgs_by_box[name],
                        timecounter=scheduled_time,
                    )
                )
            self._logger.debug(
                "reserving emission of %s at %s : base_time=%s, timediff=%s, timing_shift=%s",
                name,
                scheduled_time,
                base_time,
                self._estimated_timediff[name],
                timing_shift[name],
            )
        for task in tasks:
            task.result()


def build_parallel_multi_action(
    *,
    system: Quel1SystemProtocol,
    settings: Sequence[object],
    action_builder: _ActionBuilderProtocol,
    logger: Logger,
    clock_health_checks: ClockHealthCheckOptions | None = None,
) -> tuple[ActionProtocol | QubexMultiAction, CaptureParamMap]:
    """
    Build a multi action with per-box setup parallelized across boxes.

    For multi-box execution, this function mirrors qubecalib's
    `multi.Action.build` semantics while parallelizing the expensive per-box
    setup and SYSREF offset measurements.

    Parameters
    ----------
    system : Quel1SystemProtocol
        Quel1System-like object used by action execution.
    settings : Sequence[object]
        Common driver settings.
    action_builder : _ActionBuilderProtocol
        Builder compatible with `Action.build(system=..., settings=...)`.
    logger : Logger
        Logger used for clock/timediff diagnostics.
    clock_health_checks : ClockHealthCheckOptions | None, optional
        Clock-related validation/diagnostics options. If `None`, all checks
        are disabled to maximize execution speed.

    Returns
    -------
    tuple[ActionProtocol | QubexMultiAction, CaptureParamMap]
        Built action instance and capture-param mapping keyed by
        `(box_name, port, runit)`.

    Examples
    --------
    >>> action, cprms = build_parallel_multi_action(
    ...     system=quel1system,
    ...     settings=settings,
    ...     action_builder=Action.build,
    ...     logger=logger,
    ... )
    >>> isinstance(cprms, dict)
    True
    """
    driver = load_quel1_driver()
    multi_action = driver.MultiAction
    single_action = driver.SingleAction
    clock_health_checks = _resolve_clock_health_checks(clock_health_checks)

    settings_by_box = _convert_to_box_setting_dict(
        settings=settings,
        awg_id_class=driver.SingleAwgId,
        awg_setting_class=driver.SingleAwgSetting,
        runit_id_class=driver.SingleRunitId,
        runit_setting_class=driver.SingleRunitSetting,
        trigger_setting_class=driver.SingleTriggerSetting,
    )

    # Single-box path keeps existing builder behavior and avoids multi-action
    # setup overhead while still returning a uniform capture-parameter map.
    if len(settings_by_box) <= 1:
        action = action_builder(system=system, settings=list(settings))
        cprms: CaptureParamMap = {}
        if isinstance(action._action, tuple):
            box_name, single_action_instance = action._action
            cprms = _collect_single_action_cprms(
                box_name=box_name,
                single_action=cast(SingleActionProtocol, single_action_instance),
            )
        return action, cprms

    _validate_box_names(system=system, settings_by_box=settings_by_box)

    if clock_health_checks.read_master_clock:
        logger.debug(f"clock of master: {system._clockmaster.read_clock()}")
    actions = _build_single_actions_parallel(
        settings_by_box=settings_by_box,
        system=system,
        single_action=single_action,
        multi_action=multi_action,
        logger=logger,
        clock_health_checks=clock_health_checks,
    )
    reference_box_name, ref_sysref_time_offset, estimated_timediff = _estimate_timediff(
        actions=actions,
        multi_action=multi_action,
        clock_health_checks=clock_health_checks,
    )

    for box_name, timediff in estimated_timediff.items():
        logger.debug(f"estimated time difference of {box_name}: {timediff}")

    multi_action_instance = QubexMultiAction(
        _system=system,
        _actions=MappingProxyType(actions),
        _estimated_timediff=MappingProxyType(estimated_timediff),
        _reference_box_name=reference_box_name,
        _ref_sysref_time_offset=ref_sysref_time_offset,
        _clock_options=clock_health_checks,
        _logger=logger,
    )
    cprms = _collect_multi_action_cprms(actions=actions)
    return multi_action_instance, cprms
