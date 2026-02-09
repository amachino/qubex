# ruff: noqa: SLF001

"""Parallel builder for qubecalib-compatible multi-box actions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from types import MappingProxyType
from typing import Any

SingleSetting = Any
CommonSetting = Any


def _convert_to_box_setting_dict(
    *,
    settings: list[CommonSetting],
    direct_single: Any,
) -> dict[str, list[SingleSetting]]:
    """Convert common settings into per-box single-driver settings."""
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


def build_parallel_multi_action(
    *,
    system: Any,
    settings: list[CommonSetting],
    action_builder: Callable[..., Any],
    logger: Logger,
) -> tuple[Any, dict[tuple[str, Any, int], Any]]:
    """
    Build an action with per-box setup parallelized for multi-box execution.

    Parameters
    ----------
    system : Any
        Quel1System-like object used by direct action execution.
    settings : list[Any]
        Common direct-driver settings.
    action_builder : Callable[..., Any]
        Builder compatible with ``Action.build(system=..., settings=...)``.
    logger : Logger
        Logger used for clock/timediff diagnostics.

    Returns
    -------
    tuple[Any, dict[tuple[str, Any, int], Any]]
        Built action instance and capture-param mapping keyed by
        ``(box_name, port, runit)``.
    """
    from qubecalib.instrument.quel.quel1.driver import multi as direct_multi
    from qubecalib.instrument.quel.quel1.driver import single as direct_single

    settings_by_box = _convert_to_box_setting_dict(
        settings=settings,
        direct_single=direct_single,
    )
    if len(settings_by_box) <= 1:
        action = action_builder(system=system, settings=settings)
        cprms: dict[tuple[str, Any, int], Any] = {}
        if isinstance(action._action, tuple):
            box_name, single_action = action._action
            for runit_id, capture_param in single_action._cprms.items():
                cprms[(box_name, runit_id.port, runit_id.runit)] = capture_param
        return action, cprms

    for box_name in settings_by_box:
        if box_name not in system.boxes:
            raise ValueError(f"box {box_name} not found in system")

    logger.info(f"clock of master: {system._clockmaster.read_clock()}")

    def _build_single_action(item: tuple[str, list[SingleSetting]]) -> tuple[str, Any]:
        box_name, box_settings = item
        box = system.box[box_name]
        awg_ids = [
            (setting.awg.port, setting.awg.channel)
            for setting in box_settings
            if isinstance(setting, direct_single.AwgSetting)
        ]
        box.prepare_for_emission(awg_ids)
        current_time, last_sysref_time = box.read_current_and_latched_clock()
        logger.info(
            f"clock of {box_name}, current: {current_time}, last sysref: {last_sysref_time}, "
            f"last sysref offset: {direct_multi.Action._mod_by_sysref(last_sysref_time)}"
        )
        single_action = direct_single.Action.build(
            box=box,
            settings=box_settings,
        )
        return box_name, single_action

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

    reference_box_name = direct_multi.Action._get_reference_box_name(actions)
    ref_sysref_time_offset = average_offsets_at_sysref_clock[reference_box_name]
    estimated_timediff = {
        box_name: average_offset - ref_sysref_time_offset
        for box_name, average_offset in average_offsets_at_sysref_clock.items()
    }
    for box_name, timediff in estimated_timediff.items():
        logger.info(f"estimated time difference of {box_name}: {timediff}")

    multi_action = direct_multi.Action(
        system,
        MappingProxyType(actions),
        MappingProxyType(estimated_timediff),
        reference_box_name,
        ref_sysref_time_offset,
    )
    cprms = {}
    for box_name, single_action in actions.items():
        for runit_id, capture_param in single_action._cprms.items():
            cprms[(box_name, runit_id.port, runit_id.runit)] = capture_param
    return multi_action, cprms
