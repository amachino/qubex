# ruff: noqa: SLF001

"""Parallel builder for qubecalib-compatible multi-box actions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from logging import Logger
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeAlias, cast

from qubex.backend.quel1.quel1_box_compat import adapt_quel1_box
from qubex.backend.quel1.quel1_driver_loader import load_quel1_driver

if TYPE_CHECKING:
    from qubex.backend.quel1.quel1_driver_protocols import QuelDriverModulesProtocol

PortType: TypeAlias = Any
CommonSetting: TypeAlias = Any
SingleSetting: TypeAlias = Any
CaptureParamMapKey: TypeAlias = tuple[str, PortType, int]
CaptureParamMap: TypeAlias = dict[CaptureParamMapKey, Any]


class _ClockmasterLike(Protocol):
    """Protocol for clock-master access used only for diagnostics."""

    def read_clock(self) -> Any:
        """Return current master clock state."""
        ...


class _BoxLike(Protocol):
    """Protocol for box operations required to build direct actions."""

    def get_current_timecounter(self) -> int:
        """Read current box clock counter."""
        ...

    def get_latest_sysref_timecounter(self) -> int:
        """Read latest latched SYSREF clock counter."""
        ...

    def start_wavegen(
        self,
        channels: set[tuple[PortType, int]],
        timecounter: int | None = None,
    ) -> Any:
        """Start waveform generation."""
        ...


class _SystemLike(Protocol):
    """Protocol for system object consumed by parallel action builder."""

    boxes: Mapping[str, _BoxLike]
    box: Mapping[str, _BoxLike]
    _clockmaster: _ClockmasterLike
    timing_shift: dict[str, int]
    displacement: int


class _RunitIdLike(Protocol):
    """Protocol for runit identifiers contained in single actions."""

    port: PortType
    runit: int


class _SingleActionLike(Protocol):
    """Protocol for per-box direct single action objects."""

    _cprms: Mapping[_RunitIdLike, Any]
    box: _BoxLike
    _wseqs: Mapping[Any, Any]

    def capture_start(self) -> dict[PortType, Any]:
        """Start capture and return future map."""
        ...

    def capture_stop(
        self, futures: dict[PortType, Any]
    ) -> tuple[dict[PortType, Any], dict[tuple[PortType, int], Any]]:
        """Stop capture and collect status/data."""
        ...


class _WrappedActionLike(Protocol):
    """Protocol for wrapper action returned by direct Action.build."""

    _action: Any


class _ActionBuilderLike(Protocol):
    """Protocol for direct Action.build-compatible callables."""

    def __call__(
        self, *, system: _SystemLike, settings: list[CommonSetting]
    ) -> _WrappedActionLike:
        """Build direct action from common settings."""
        ...


def _convert_to_box_setting_dict(
    *,
    settings: list[CommonSetting],
    direct_single: Any,
) -> dict[str, list[SingleSetting]]:
    """
    Convert common direct-driver settings into per-box single-driver settings.

    Parameters
    ----------
    settings : list[CommonSetting]
        Flat common settings where each entry includes a ``box`` location.
    direct_single : Any
        Imported ``driver.single`` module that provides setting/id factories.

    Returns
    -------
    dict[str, list[SingleSetting]]
        Settings grouped by box name.
    """
    settings_by_box: dict[str, list[SingleSetting]] = defaultdict(list)
    for setting in settings:
        if hasattr(setting, "runit") and hasattr(setting, "cprm"):
            settings_by_box[setting.runit.box].append(
                direct_single.RunitSetting(
                    direct_single.RunitId(
                        setting.runit.port,
                        setting.runit.runit,
                    ),
                    setting.cprm,
                )
            )
        elif hasattr(setting, "awg") and hasattr(setting, "wseq"):
            settings_by_box[setting.awg.box].append(
                direct_single.AwgSetting(
                    direct_single.AwgId(
                        setting.awg.port,
                        setting.awg.channel,
                    ),
                    setting.wseq,
                )
            )
        elif hasattr(setting, "trigger_awg") and hasattr(setting, "triggerd_port"):
            settings_by_box[setting.trigger_awg.box].append(
                direct_single.TriggerSetting(
                    direct_single.AwgId(
                        setting.trigger_awg.port,
                        setting.trigger_awg.channel,
                    ),
                    setting.triggerd_port,
                )
            )
    return settings_by_box


def _collect_single_action_cprms(
    *,
    box_name: str,
    single_action: _SingleActionLike,
) -> CaptureParamMap:
    """
    Collect capture parameters from one box-scoped single action.

    Parameters
    ----------
    box_name : str
        Box name associated with the given action.
    single_action : _SingleActionLike
        Built single action exposing ``_cprms``.

    Returns
    -------
    CaptureParamMap
        Capture parameter map keyed by ``(box_name, port, runit)``.
    """
    cprms: CaptureParamMap = {}
    for runit_id, capture_param in single_action._cprms.items():
        cprms[(box_name, runit_id.port, runit_id.runit)] = capture_param
    return cprms


def _collect_multi_action_cprms(
    *,
    actions: Mapping[str, _SingleActionLike],
) -> CaptureParamMap:
    """
    Collect capture parameters from all box-scoped single actions.

    Parameters
    ----------
    actions : Mapping[str, _SingleActionLike]
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
class _QubexMultiAction:
    """Qubex-side multi-action with optional clock-health I/O."""

    _system: _SystemLike
    _actions: MappingProxyType[str, _SingleActionLike]
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
    def _has_capture_setting(action: _SingleActionLike) -> bool:
        """Return True if action has runit capture settings."""
        return bool(action._cprms)

    def capture_start(self) -> dict[str, dict[PortType, Any]]:
        """Start capture for boxes that include capture settings."""
        return {
            name: action.capture_start()
            for name, action in self._actions.items()
            if self._has_capture_setting(action)
        }

    def capture_stop(
        self,
        futures: dict[str, dict[PortType, Any]],
    ) -> tuple[dict[tuple[str, PortType], Any], dict[tuple[str, PortType, int], Any]]:
        """Stop capture and flatten per-box status/data maps."""
        box_results = {
            name: self._actions[name].capture_stop(future)
            for name, future in futures.items()
        }
        status: dict[tuple[str, PortType], Any] = {}
        data: dict[tuple[str, PortType, int], Any] = {}
        for name, (box_status, box_data) in box_results.items():
            for port, capture_return_code in box_status.items():
                status[(name, port)] = capture_return_code
            for (port, runit), runit_data in box_data.items():
                data[(name, port, runit)] = runit_data
        return status, data

    def action(
        self,
    ) -> tuple[dict[tuple[str, PortType], Any], dict[tuple[str, PortType, int], Any]]:
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
        tasks: list[Any] = []
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
    system: _SystemLike,
    settings: list[CommonSetting],
    action_builder: _ActionBuilderLike,
    logger: Logger,
    clock_health_checks: ClockHealthCheckOptions | None = None,
) -> tuple[Any, CaptureParamMap]:
    """
    Build direct multi action with per-box setup parallelized across boxes.

    For multi-box execution, this function mirrors qubecalib's direct
    ``multi.Action.build`` semantics while parallelizing the expensive per-box
    setup and SYSREF offset measurements.

    Parameters
    ----------
    system : _SystemLike
        Quel1System-like object used by direct action execution.
    settings : list[CommonSetting]
        Common direct-driver settings.
    action_builder : _ActionBuilderLike
        Builder compatible with ``Action.build(system=..., settings=...)``.
    logger : Logger
        Logger used for clock/timediff diagnostics.
    clock_health_checks : ClockHealthCheckOptions | None, optional
        Clock-related validation/diagnostics options. If ``None``, all checks
        are disabled to maximize execution speed.

    Returns
    -------
    tuple[Any, CaptureParamMap]
        Built action instance and capture-param mapping keyed by
        ``(box_name, port, runit)``.

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
    if TYPE_CHECKING:
        driver = cast(QuelDriverModulesProtocol, driver)
    direct_multi = driver.direct_multi_module
    direct_single = driver.direct_single_module

    if clock_health_checks is None:
        clock_health_checks = ClockHealthCheckOptions()

    settings_by_box = _convert_to_box_setting_dict(
        settings=settings,
        direct_single=direct_single,
    )

    # Single-box path keeps existing builder behavior and avoids multi-action
    # setup overhead while still returning a uniform capture-parameter map.
    if len(settings_by_box) <= 1:
        action = action_builder(system=system, settings=settings)
        cprms: CaptureParamMap = {}
        if isinstance(action._action, tuple):
            box_name, single_action = action._action
            cprms = _collect_single_action_cprms(
                box_name=box_name,
                single_action=cast(_SingleActionLike, single_action),
            )
        return action, cprms

    for box_name in settings_by_box:
        if box_name not in system.boxes:
            raise ValueError(f"box {box_name} not found in system")

    if clock_health_checks.read_master_clock:
        logger.debug(f"clock of master: {system._clockmaster.read_clock()}")

    def _build_single_action(
        item: tuple[str, list[SingleSetting]],
    ) -> tuple[str, _SingleActionLike]:
        """Build one box-scoped direct single action."""
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
                direct_multi.Action._mod_by_sysref(last_sysref_time),
            )
        single_action = direct_single.Action.build(
            box=cast(Any, box),
            settings=cast(Any, box_settings),
        )
        return box_name, cast(_SingleActionLike, single_action)

    max_workers = max(1, len(settings_by_box))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        actions = dict(executor.map(_build_single_action, settings_by_box.items()))

    reference_box_name = direct_multi.Action._get_reference_box_name(cast(Any, actions))
    if clock_health_checks.measure_average_sysref_offset:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            average_offsets_at_sysref_clock = dict(
                executor.map(
                    lambda item: (
                        item[0],
                        direct_multi.Action._measure_average_offset_at_sysref_clock(
                            item[1].box
                        ),
                    ),
                    actions.items(),
                )
            )
        ref_sysref_time_offset = average_offsets_at_sysref_clock[reference_box_name]
        estimated_timediff = {
            box_name: average_offset - ref_sysref_time_offset
            for box_name, average_offset in average_offsets_at_sysref_clock.items()
        }
    else:
        ref_sysref_time_offset = 0
        estimated_timediff = dict.fromkeys(actions, 0)

    for box_name, timediff in estimated_timediff.items():
        logger.debug(f"estimated time difference of {box_name}: {timediff}")

    multi_action = _QubexMultiAction(
        _system=system,
        _actions=MappingProxyType(cast(dict[str, _SingleActionLike], actions)),
        _estimated_timediff=MappingProxyType(cast(dict[str, int], estimated_timediff)),
        _reference_box_name=reference_box_name,
        _ref_sysref_time_offset=ref_sysref_time_offset,
        _clock_options=clock_health_checks,
        _logger=logger,
    )
    cprms = _collect_multi_action_cprms(actions=actions)
    return multi_action, cprms
