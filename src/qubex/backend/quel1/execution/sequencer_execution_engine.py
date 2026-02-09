"""Execution engine for QuEL-1 sequencer workflows."""

from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import Any

from .capture_result_parser import parse_capture_results_with_cprms
from .parallel_action_builder import build_parallel_multi_action


class SequencerExecutionEngine:
    """Run sequencers using legacy and parallelized execution paths."""

    @staticmethod
    def set_measurement_options(
        *,
        sequencer: Any,
        repeats: int,
        integral_mode: str,
        dsp_demodulation: bool,
        software_demodulation: bool,
        enable_sum: bool,
        enable_classification: bool,
        line_param0: tuple[float, float, float] | None,
        line_param1: tuple[float, float, float] | None,
    ) -> None:
        """Set measurement options to sequencer with default classifier params."""
        if line_param0 is None:
            line_param0 = (1, 0, 0)
        if line_param1 is None:
            line_param1 = (0, 1, 0)
        sequencer.set_measurement_option(
            repeats=repeats,
            interval=sequencer.interval,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
            enable_sum=enable_sum,
            enable_classification=enable_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )

    @staticmethod
    def create_direct_settings(
        *,
        sequencer: Any,
        boxpool: Any,
        system: Any,
        runit_setting_factory: Callable[..., Any],
        runit_id_factory: Callable[..., Any],
        awg_setting_factory: Callable[..., Any],
        awg_id_factory: Callable[..., Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        """Build direct-driver settings and capture resource map from sequencer."""
        cap_settings, gen_settings, cap_resource_map = sequencer.generate_e7_settings(
            boxpool
        )
        settings: list[Any] = []
        for (name, cap_port, runit), capture_param in cap_settings.items():
            settings.append(
                runit_setting_factory(
                    runit=runit_id_factory(
                        box=name,
                        port=cap_port,
                        runit=runit,
                    ),
                    cprm=capture_param,
                )
            )
        for (name, gen_port, channel), wave_seq in gen_settings.items():
            settings.append(
                awg_setting_factory(
                    awg=awg_id_factory(
                        box=name,
                        port=gen_port,
                        channel=channel,
                    ),
                    wseq=wave_seq,
                )
            )
        settings += sequencer.select_trigger(system, settings)
        if len(settings) == 0:
            raise ValueError("no settings")
        return settings, cap_resource_map

    @classmethod
    def execute_parallel(
        cls,
        *,
        sequencer: Any,
        boxpool: Any,
        system: Any,
        action_builder: Callable[..., Any],
        runit_setting_factory: Callable[..., Any],
        runit_id_factory: Callable[..., Any],
        awg_setting_factory: Callable[..., Any],
        awg_id_factory: Callable[..., Any],
        logger: Logger,
    ) -> tuple[dict[str, Any], dict[str, Any], dict]:
        """Execute sequencer using parallelized multi-box action preparation."""
        settings, cap_resource_map = cls.create_direct_settings(
            sequencer=sequencer,
            boxpool=boxpool,
            system=system,
            runit_setting_factory=runit_setting_factory,
            runit_id_factory=runit_id_factory,
            awg_setting_factory=awg_setting_factory,
            awg_id_factory=awg_id_factory,
        )
        action, cprms = build_parallel_multi_action(
            system=system,
            settings=settings,
            action_builder=action_builder,
            logger=logger,
        )
        status, raw_results = action.action()
        return parse_capture_results_with_cprms(
            sequencer=sequencer,
            status=status,
            results=raw_results,
            cap_resource_map=cap_resource_map,
            cprms=cprms,
        )
