# ruff: noqa: SLF001

"""Parallel builder for qubecalib-compatible multi-box actions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast

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

    def prepare_for_emission(self, awg_ids: list[tuple[PortType, int]]) -> None:
        """Prepare box AWGs for emission."""
        ...

    def read_current_and_latched_clock(self) -> tuple[int, int]:
        """Read current clock and latest latched SYSREF clock."""
        ...


class _SystemLike(Protocol):
    """Protocol for system object consumed by parallel action builder."""

    boxes: Mapping[str, _BoxLike]
    box: Mapping[str, _BoxLike]
    _clockmaster: _ClockmasterLike


class _RunitIdLike(Protocol):
    """Protocol for runit identifiers contained in single actions."""

    port: PortType
    runit: int


class _SingleActionLike(Protocol):
    """Protocol for per-box direct single action objects."""

    _cprms: Mapping[_RunitIdLike, Any]
    box: _BoxLike


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


def build_parallel_multi_action(
    *,
    system: _SystemLike,
    settings: list[CommonSetting],
    action_builder: _ActionBuilderLike,
    logger: Logger,
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
    from qubecalib.instrument.quel.quel1.driver import multi as direct_multi
    from qubecalib.instrument.quel.quel1.driver import single as direct_single

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

    logger.debug(f"clock of master: {system._clockmaster.read_clock()}")

    def _build_single_action(
        item: tuple[str, list[SingleSetting]],
    ) -> tuple[str, _SingleActionLike]:
        """Build one box-scoped direct single action."""
        box_name, box_settings = item
        box = system.box[box_name]
        awg_ids = [
            (setting.awg.port, setting.awg.channel)
            for setting in box_settings
            if isinstance(setting, direct_single.AwgSetting)
        ]
        box.prepare_for_emission(awg_ids)
        current_time, last_sysref_time = box.read_current_and_latched_clock()
        logger.debug(
            f"clock of {box_name}, current: {current_time}, last sysref: {last_sysref_time}, "
            f"last sysref offset: {direct_multi.Action._mod_by_sysref(last_sysref_time)}"
        )
        single_action = direct_single.Action.build(
            box=cast(Any, box),
            settings=cast(Any, box_settings),
        )
        return box_name, cast(_SingleActionLike, single_action)

    max_workers = max(1, len(settings_by_box))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        actions = dict(executor.map(_build_single_action, settings_by_box.items()))

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

    reference_box_name = direct_multi.Action._get_reference_box_name(cast(Any, actions))
    ref_sysref_time_offset = average_offsets_at_sysref_clock[reference_box_name]
    estimated_timediff = {
        box_name: average_offset - ref_sysref_time_offset
        for box_name, average_offset in average_offsets_at_sysref_clock.items()
    }
    for box_name, timediff in estimated_timediff.items():
        logger.debug(f"estimated time difference of {box_name}: {timediff}")

    multi_action = direct_multi.Action(
        cast(Any, system),
        MappingProxyType(cast(dict[str, Any], actions)),
        MappingProxyType(estimated_timediff),
        reference_box_name,
        ref_sysref_time_offset,
    )
    cprms = _collect_multi_action_cprms(actions=actions)
    return multi_action, cprms
