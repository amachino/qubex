"""Execution engine for QuEL-1 sequencer workflows."""

from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import TypeAlias

from qubex.backend.quel1.compat.qubecalib_protocols import (
    ActionProtocol,
    AwgIdProtocol,
    AwgSettingProtocol,
    BoxPoolProtocol,
    CaptureResourceMap,
    ConfigMap,
    DataMap,
    Quel1SystemProtocol,
    RunitIdProtocol,
    RunitSettingProtocol,
    SequencerProtocol,
    StatusMap,
    TriggerSettingProtocol,
)

from .capture_result_parser import parse_capture_results_with_cprms
from .parallel_action_builder import (
    ClockHealthCheckOptions,
    build_parallel_multi_action,
)

CommonSetting: TypeAlias = (
    RunitSettingProtocol | AwgSettingProtocol | TriggerSettingProtocol
)
ActionBuilder: TypeAlias = Callable[..., ActionProtocol]
RunitSettingFactory: TypeAlias = Callable[..., RunitSettingProtocol]
RunitIdFactory: TypeAlias = Callable[..., RunitIdProtocol]
AwgSettingFactory: TypeAlias = Callable[..., AwgSettingProtocol]
AwgIdFactory: TypeAlias = Callable[..., AwgIdProtocol]


class SequencerExecutionEngine:
    """
    Run QuEL-1 sequencers through typed helper steps.

    The class isolates three concerns used by the parallel execution path:
    1) measurement-option setup,
    2) conversion from sequencer output into direct-driver settings, and
    3) execution + capture-result parsing.

    Examples
    --------
    >>> SequencerExecutionEngine.set_measurement_options(
    ...     sequencer=sequencer,
    ...     repeats=1024,
    ...     integral_mode="single",
    ...     dsp_demodulation=True,
    ...     software_demodulation=False,
    ...     enable_sum=False,
    ...     enable_classification=False,
    ...     line_param0=None,
    ...     line_param1=None,
    ... )
    """

    @staticmethod
    def set_measurement_options(
        *,
        sequencer: SequencerProtocol,
        repeats: int,
        integral_mode: str,
        dsp_demodulation: bool,
        software_demodulation: bool,
        enable_sum: bool,
        enable_classification: bool,
        line_param0: tuple[float, float, float] | None,
        line_param1: tuple[float, float, float] | None,
    ) -> None:
        """
        Configure sequencer measurement options with stable defaults.

        Parameters
        ----------
        sequencer : SequencerProtocol
            Sequencer instance providing `set_measurement_option`.
        repeats : int
            Number of repetitions for one backend execution call.
        integral_mode : str
            Integration mode forwarded to direct backend.
        dsp_demodulation : bool
            Whether to enable DSP demodulation.
        software_demodulation : bool
            Whether to enable software-side demodulation.
        enable_sum : bool
            Whether to enable DSP summation.
        enable_classification : bool
            Whether to enable DSP classification.
        line_param0 : tuple[float, float, float] | None
            Classifier line parameter 0. If `None`, `(1, 0, 0)` is used.
        line_param1 : tuple[float, float, float] | None
            Classifier line parameter 1. If `None`, `(0, 1, 0)` is used.

        Examples
        --------
        >>> SequencerExecutionEngine.set_measurement_options(
        ...     sequencer=sequencer,
        ...     repeats=256,
        ...     integral_mode="integral",
        ...     dsp_demodulation=True,
        ...     software_demodulation=False,
        ...     enable_sum=False,
        ...     enable_classification=False,
        ...     line_param0=(1.0, 0.0, 0.0),
        ...     line_param1=(0.0, 1.0, 0.0),
        ... )
        """
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
        sequencer: SequencerProtocol,
        boxpool: BoxPoolProtocol,
        system: Quel1SystemProtocol,
        runit_setting_factory: RunitSettingFactory,
        runit_id_factory: RunitIdFactory,
        awg_setting_factory: AwgSettingFactory,
        awg_id_factory: AwgIdFactory,
    ) -> tuple[list[CommonSetting], CaptureResourceMap]:
        """
        Convert sequencer device settings into direct-driver setting objects.

        Parameters
        ----------
        sequencer : SequencerProtocol
            Sequencer producing capture/generator settings.
        boxpool : BoxPoolProtocol
            Backend boxpool forwarded to `generate_e7_settings`.
        system : Quel1SystemProtocol
            Quel1System-like object passed to trigger selection.
        runit_setting_factory : RunitSettingFactory
            Factory for direct runit settings.
        runit_id_factory : RunitIdFactory
            Factory for runit identifiers.
        awg_setting_factory : AwgSettingFactory
            Factory for direct AWG settings.
        awg_id_factory : AwgIdFactory
            Factory for AWG identifiers.

        Returns
        -------
        tuple[list[CommonSetting], CaptureResourceMap]
            Flat direct-driver settings and capture resource map.

        Raises
        ------
        ValueError
            If no settings are generated.

        Examples
        --------
        >>> settings, cap_resource_map = SequencerExecutionEngine.create_direct_settings(
        ...     sequencer=sequencer,
        ...     boxpool=boxpool,
        ...     system=quel1system,
        ...     runit_setting_factory=RunitSetting,
        ...     runit_id_factory=RunitId,
        ...     awg_setting_factory=AwgSetting,
        ...     awg_id_factory=AwgId,
        ... )
        >>> len(settings) > 0
        True
        """
        cap_settings, gen_settings, cap_resource_map = sequencer.generate_e7_settings(
            boxpool
        )
        settings: list[CommonSetting] = []
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
        settings.extend(sequencer.select_trigger(system, settings))
        if len(settings) == 0:
            raise ValueError("no settings")
        return settings, cap_resource_map

    @classmethod
    def execute_parallel(
        cls,
        *,
        sequencer: SequencerProtocol,
        boxpool: BoxPoolProtocol,
        system: Quel1SystemProtocol,
        action_builder: ActionBuilder,
        runit_setting_factory: RunitSettingFactory,
        runit_id_factory: RunitIdFactory,
        awg_setting_factory: AwgSettingFactory,
        awg_id_factory: AwgIdFactory,
        logger: Logger,
        clock_health_checks: ClockHealthCheckOptions | None = None,
    ) -> tuple[StatusMap, DataMap, ConfigMap]:
        """
        Execute a sequencer using parallelized multi-box action preparation.

        Parameters
        ----------
        sequencer : SequencerProtocol
            Sequencer that generates settings and parses capture results.
        boxpool : BoxPoolProtocol
            Backend boxpool for sequencer setting generation.
        system : Quel1SystemProtocol
            Quel1System-like object for direct action execution.
        action_builder : ActionBuilder
            Fallback-compatible action builder for single-box cases.
        runit_setting_factory : RunitSettingFactory
            Factory for direct runit setting objects.
        runit_id_factory : RunitIdFactory
            Factory for direct runit identifiers.
        awg_setting_factory : AwgSettingFactory
            Factory for direct AWG setting objects.
        awg_id_factory : AwgIdFactory
            Factory for direct AWG identifiers.
        logger : Logger
            Logger for parallel action diagnostics.
        clock_health_checks : ClockHealthCheckOptions | None, optional
            Clock-validation options for parallel multi-action execution.

        Returns
        -------
        tuple[StatusMap, DataMap, ConfigMap]
            Parsed status, parsed data, and backend config payload.

        Examples
        --------
        >>> status, data, config = SequencerExecutionEngine.execute_parallel(
        ...     sequencer=sequencer,
        ...     boxpool=boxpool,
        ...     system=quel1system,
        ...     action_builder=Action.build,
        ...     runit_setting_factory=RunitSetting,
        ...     runit_id_factory=RunitId,
        ...     awg_setting_factory=AwgSetting,
        ...     awg_id_factory=AwgId,
        ...     logger=logger,
        ... )
        >>> isinstance(status, dict) and isinstance(data, dict)
        True
        """
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
            clock_health_checks=clock_health_checks,
        )
        status, raw_results = action.action()
        return parse_capture_results_with_cprms(
            sequencer=sequencer,
            status=status,
            results=raw_results,
            cap_resource_map=cap_resource_map,
            cprms=cprms,
        )
