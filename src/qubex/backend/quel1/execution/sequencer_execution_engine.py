"""Execution engine for QuEL-1 sequencer workflows."""

from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import Any, Protocol, TypeAlias, cast

from .capture_result_parser import parse_capture_results_with_cprms
from .parallel_action_builder import build_parallel_multi_action

PortType: TypeAlias = Any
CommonSetting: TypeAlias = Any
CaptureStatusByTarget: TypeAlias = dict[str, Any]
CaptureDataByTarget: TypeAlias = dict[str, Any]
BackendConfigPayload: TypeAlias = dict


class _SequencerLike(Protocol):
    """Protocol for sequencer methods consumed by this execution engine."""

    interval: Any

    def set_measurement_option(self, **kwargs: Any) -> None:
        """Apply measurement options before execution."""
        ...

    def generate_e7_settings(self, boxpool: Any) -> tuple[Any, Any, dict[str, Any]]:
        """Generate direct-driver capture/generator settings."""
        ...

    def select_trigger(self, system: Any, settings: list[Any]) -> list[Any]:
        """Generate trigger settings derived from system and settings."""
        ...

    def parse_capture_result(
        self,
        status: Any,
        data: Any,
        cprm: Any,
    ) -> tuple[Any, Any]:
        """Parse one backend capture result into public result format."""
        ...


class _ActionLike(Protocol):
    """Protocol for direct action objects created by builders."""

    def action(
        self,
    ) -> tuple[dict[tuple[str, Any], Any], dict[tuple[str, Any, int], Any]]:
        """Run action and return raw capture status/data."""
        ...


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
        sequencer: _SequencerLike,
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
        sequencer : _SequencerLike
            Sequencer instance providing ``set_measurement_option``.
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
            Classifier line parameter 0. If ``None``, ``(1, 0, 0)`` is used.
        line_param1 : tuple[float, float, float] | None
            Classifier line parameter 1. If ``None``, ``(0, 1, 0)`` is used.

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
        sequencer: _SequencerLike,
        boxpool: Any,
        system: Any,
        runit_setting_factory: Callable[..., Any],
        runit_id_factory: Callable[..., Any],
        awg_setting_factory: Callable[..., Any],
        awg_id_factory: Callable[..., Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        Convert sequencer device settings into direct-driver setting objects.

        Parameters
        ----------
        sequencer : _SequencerLike
            Sequencer producing capture/generator settings.
        boxpool : Any
            Backend boxpool forwarded to ``generate_e7_settings``.
        system : Any
            Quel1System-like object passed to trigger selection.
        runit_setting_factory : Callable[..., Any]
            Factory for direct runit settings.
        runit_id_factory : Callable[..., Any]
            Factory for runit identifiers.
        awg_setting_factory : Callable[..., Any]
            Factory for direct AWG settings.
        awg_id_factory : Callable[..., Any]
            Factory for AWG identifiers.

        Returns
        -------
        tuple[list[Any], dict[str, Any]]
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
        sequencer: _SequencerLike,
        boxpool: Any,
        system: Any,
        action_builder: Callable[..., Any],
        runit_setting_factory: Callable[..., Any],
        runit_id_factory: Callable[..., Any],
        awg_setting_factory: Callable[..., Any],
        awg_id_factory: Callable[..., Any],
        logger: Logger,
    ) -> tuple[CaptureStatusByTarget, CaptureDataByTarget, BackendConfigPayload]:
        """
        Execute a sequencer using parallelized multi-box action preparation.

        Parameters
        ----------
        sequencer : _SequencerLike
            Sequencer that generates settings and parses capture results.
        boxpool : Any
            Backend boxpool for sequencer setting generation.
        system : Any
            Quel1System-like object for direct action execution.
        action_builder : Callable[..., Any]
            Fallback-compatible action builder for single-box cases.
        runit_setting_factory : Callable[..., Any]
            Factory for direct runit setting objects.
        runit_id_factory : Callable[..., Any]
            Factory for direct runit identifiers.
        awg_setting_factory : Callable[..., Any]
            Factory for direct AWG setting objects.
        awg_id_factory : Callable[..., Any]
            Factory for direct AWG identifiers.
        logger : Logger
            Logger for parallel action diagnostics.

        Returns
        -------
        tuple[CaptureStatusByTarget, CaptureDataByTarget, BackendConfigPayload]
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
        )
        action = cast(_ActionLike, action)
        status, raw_results = action.action()
        return parse_capture_results_with_cprms(
            sequencer=sequencer,
            status=status,
            results=raw_results,
            cap_resource_map=cap_resource_map,
            cprms=cprms,
        )
