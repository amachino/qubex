"""
QuEL-1 backend controller implementing measurement-facing backend contracts.

This module provides the QuEL-1 concrete `BackendController` implementation.
It exposes the required shared controller contract plus QuEL-1-specific
capabilities, while delegating concrete operations to QuEL-1 managers.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qubex.backend.backend_controller import (
    BackendController,
    BackendExecutionRequest,
    BackendExecutionResult,
)

from .managers import (
    Quel1ClockManager,
    Quel1ConfigurationManager,
    Quel1ConnectionManager,
    Quel1ContinuousWaveManager,
    Quel1ExecutionManager,
    Quel1SkewManager,
)
from .quel1_backend_constants import (
    CAPTURE_DECIMATION_FACTOR,
    ExecutionMode,
)
from .quel1_runtime_context import Quel1RuntimeContext

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .compat.qubecalib_protocols import (
        BoxPoolProtocol as BoxPool,
        PortType,
        QubeCalibProtocol as QubeCalib,
        Quel1BoxCommonProtocol as Quel1Box,
        Quel1SystemProtocol as Quel1System,
        QuelDriverClassesProtocol,
        SequencerProtocol as Sequencer,
    )
    from .managers import Quel1ContinuousWaveChannelSpec, Quel1ContinuousWaveConfig


class Quel1BackendController(BackendController):
    """
    QuEL-1 backend controller backed by qubecalib and manager delegation.

    The controller is the measurement-layer entrypoint for QuEL-1 sessions and
    execution. It implements shared `BackendController` requirements and
    delegates connection, clock, configuration, and execution details to
    backend-local manager components.
    """

    CAPTURE_DECIMATION_FACTOR: int = CAPTURE_DECIMATION_FACTOR

    def __init__(
        self,
        *,
        runtime_context: Quel1RuntimeContext | None = None,
        connection_manager: Quel1ConnectionManager | None = None,
        clock_manager: Quel1ClockManager | None = None,
        execution_manager: Quel1ExecutionManager | None = None,
        configuration_manager: Quel1ConfigurationManager | None = None,
        continuous_wave_manager: Quel1ContinuousWaveManager | None = None,
        skew_manager: Quel1SkewManager | None = None,
    ):
        """
        Initialize QuEL-1 controller and manager delegates.

        Parameters
        ----------
        runtime_context : Quel1RuntimeContext | None, optional
            Injected runtime context for testing or customization.
        connection_manager : Quel1ConnectionManager | None, optional
            Injected connection manager for testing or customization.
        clock_manager : Quel1ClockManager | None, optional
            Injected clock manager for testing or customization.
        execution_manager : Quel1ExecutionManager | None, optional
            Injected execution manager for testing or customization.
        configuration_manager : Quel1ConfigurationManager | None, optional
            Injected configuration manager for testing or customization.
        continuous_wave_manager : Quel1ContinuousWaveManager | None, optional
            Injected continuous-wave manager for testing or customization.
        skew_manager : Quel1SkewManager | None, optional
            Injected skew manager for testing or customization.
        """
        self._runtime_context = (
            runtime_context if runtime_context is not None else Quel1RuntimeContext()
        )
        self._connection_manager = (
            connection_manager
            if connection_manager is not None
            else Quel1ConnectionManager(runtime_context=self._runtime_context)
        )
        self._clock_manager = (
            clock_manager
            if clock_manager is not None
            else Quel1ClockManager(runtime_context=self._runtime_context)
        )
        self._execution_manager = (
            execution_manager
            if execution_manager is not None
            else Quel1ExecutionManager(runtime_context=self._runtime_context)
        )
        self._configuration_manager = (
            configuration_manager
            if configuration_manager is not None
            else Quel1ConfigurationManager(runtime_context=self._runtime_context)
        )
        self._continuous_wave_manager = (
            continuous_wave_manager
            if continuous_wave_manager is not None
            else Quel1ContinuousWaveManager(runtime_context=self._runtime_context)
        )
        self._skew_manager = (
            skew_manager
            if skew_manager is not None
            else Quel1SkewManager(runtime_context=self._runtime_context)
        )

    # Core Properties
    @property
    def driver(self) -> QuelDriverClassesProtocol:
        """Return loaded QuEL-1 driver class bundle."""
        return self._runtime_context.driver

    @property
    def sampling_period_ns(self) -> float:
        """Return backend sampling period in ns."""
        return self._runtime_context.sampling_period_ns

    @property
    def is_connected(self) -> bool:
        """Return whether the hardware is connected."""
        return self._connection_manager.is_connected

    @property
    def qubecalib(self) -> QubeCalib:
        """Return the QubeCalib instance or raise if unavailable."""
        return self._runtime_context.qubecalib

    @property
    def hash(self) -> int:
        """Return stable hash of the current system configuration."""
        return hash(self.qubecalib.system_config_database.asjson())

    @property
    def box_config(self) -> dict[str, Any]:
        """Return connected box configuration cache."""
        return self._connection_manager.get_box_config_cache()

    @property
    def boxpool(self) -> BoxPool:
        """Return connected box pool."""
        return self._connection_manager.boxpool

    @property
    def quel1system(self) -> Quel1System:
        """Return connected Quel1 system."""
        return self._connection_manager.quel1system

    @property
    def cap_resource_map(self) -> dict[str, dict]:
        """Return capture resource map for connected boxes."""
        return self._connection_manager.cap_resource_map

    @property
    def gen_resource_map(self) -> dict[str, dict]:
        """Return generator resource map for connected boxes."""
        return self._connection_manager.gen_resource_map

    # Connection Lifecycle
    def connect(
        self,
        box_names: str | list[str] | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Connect to the boxes.

        Parameters
        ----------
        box_names : str | list[str], optional
            List of box names to connect to. If None, connect to all available boxes.
        parallel : bool | None, optional
            Whether to reconnect boxes in parallel. If `None`, it follows
            `qubex.backend.quel1.DEFAULT_EXECUTION_MODE`.
        """
        if self._connection_manager.requires_reconnect(box_names):
            self.stop_all_continuous_waves()
        self._connection_manager.connect(box_names=box_names, parallel=parallel)

    def disconnect(self) -> None:
        """Disconnect backend resources and reset connection-related state."""
        try:
            self.stop_all_continuous_waves()
        except Exception:
            logger.exception("Failed to stop continuous waves during disconnect.")
        self._connection_manager.clear_cache()
        self._connection_manager.disconnect()

    def get_box(self, box_name: str) -> Quel1Box:
        """Return connected box instance, creating it on demand when needed."""
        return self._connection_manager.get_existing_or_create_box(
            box_name=box_name,
            reconnect=True,
        )

    def initialize_awg_and_capunits(
        self,
        box_names: str | Collection[str],
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Initialize all awg and capture units in the specified boxes.

        Parameters
        ----------
        box_names : str | list[str]
            List of box names to initialize.
        parallel : bool | None, optional
            Whether to initialize boxes in parallel. If `None`, it follows
            `qubex.backend.quel1.DEFAULT_EXECUTION_MODE`.
        """
        self._connection_manager.initialize_awg_and_capunits(
            box_names=box_names,
            parallel=parallel,
        )

    def link_status(self, box_name: str) -> dict[int, bool]:
        """Return JESD link status map for one box."""
        return self._connection_manager.link_status(box_name=box_name)

    def linkup(
        self,
        box_name: str,
        noise_threshold: float | None = None,
        **kwargs: Any,
    ) -> Quel1Box:
        """
        Linkup a box and return the box object.

        Parameters
        ----------
        box_name : str
            Name of the box to linkup.

        Returns
        -------
        Quel1Box
            Linked up box object.


        Raises
        ------
        ValueError
            If the box is not in the available boxes.
        """
        return self._connection_manager.linkup(
            box_name=box_name,
            noise_threshold=noise_threshold,
            **kwargs,
        )

    def linkup_boxes(
        self,
        box_list: list[str],
        noise_threshold: float | None = None,
        *,
        parallel: bool | None = None,
    ) -> dict[str, Quel1Box]:
        """
        Linkup all the boxes in the list.

        Parameters
        ----------
        box_list : list[str]
            List of box names.
        noise_threshold : float | None, optional
            Threshold for linkup noise checks.
        parallel : bool | None, optional
            Whether to link up boxes in parallel. If `None`, it follows
            `qubex.backend.quel1.DEFAULT_EXECUTION_MODE`.

        Returns
        -------
        dict[str, Quel1Box]
            Dictionary of linked up boxes.
        """
        return self._connection_manager.linkup_boxes(
            box_list=box_list,
            noise_threshold=noise_threshold,
            parallel=parallel,
        )

    def relinkup(self, box_name: str, noise_threshold: float | None = None) -> None:
        """
        Relink a box.

        Parameters
        ----------
        box_name : str
            Name of the box to relinkup.
        """
        self._connection_manager.relinkup(
            box_name=box_name,
            noise_threshold=noise_threshold,
        )

    def relinkup_boxes(
        self,
        box_list: list[str],
        noise_threshold: float | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Relink all the boxes in the list.

        Parameters
        ----------
        box_list : list[str]
            List of box names.
        noise_threshold : float | None, optional
            Threshold for relinkup noise checks.
        parallel : bool | None, optional
            Whether to relink boxes in parallel. If `None`, it follows
            `qubex.backend.quel1.DEFAULT_EXECUTION_MODE`.
        """
        self._connection_manager.relinkup_boxes(
            box_list=box_list,
            noise_threshold=noise_threshold,
            parallel=parallel,
        )

    # Clock Operations
    def read_clocks(self, box_list: list[str]) -> list[tuple[bool, int, int]]:
        """
        Read the clocks of the boxes.

        Parameters
        ----------
        box_list : list[str]
            List of box names.

        Returns
        -------
        list[tuple[bool, int, int]]
            List of clocks.
        """
        return self._clock_manager.read_clocks(box_list=box_list)

    def check_clocks(self, box_list: list[str]) -> bool:
        """
        Check the clock of the boxes.

        Parameters
        ----------
        box_list : list[str]
            List of box names.

        Returns
        -------
        bool
            True if the clocks are synchronized, False otherwise.
        """
        return self._clock_manager.check_clocks(box_list=box_list)

    def sync_clocks(self, box_list: list[str]) -> bool:
        """
        Sync the clocks of the boxes if not synchronized.

        Parameters
        ----------
        box_list : list[str]
            List of box names.
        """
        return self._clock_manager.sync_clocks(box_list=box_list)

    def resync_clocks(self, box_list: list[str]) -> bool:
        """
        Resync the clock of the boxes.

        Parameters
        ----------
        box_list : list[str]
            List of box names.
        """
        return self._clock_manager.resync_clocks(box_list=box_list)

    def reset_clockmaster(self, ipaddr: str) -> bool:
        """
        Reset the clock master.

        Parameters
        ----------
        ipaddr : str
            Clock master IP address.

        Returns
        -------
        bool
            True if reset succeeds.
        """
        return self._clock_manager.reset_clockmaster(ipaddr=ipaddr)

    # Configuration Operations
    def define_clockmaster(self, *, ipaddr: str) -> None:
        """
        Define the clock master in qube-calib.

        Parameters
        ----------
        ipaddr : str
            Clock master IP address.
        """
        self._configuration_manager.define_clockmaster(ipaddr=ipaddr)

    def define_box(
        self,
        *,
        box_name: str,
        ipaddr_wss: str,
        boxtype: str,
    ) -> None:
        """
        Define a box in qube-calib.

        Parameters
        ----------
        box_name : str
            Box name.
        ipaddr_wss : str
            WSS IP address.
        boxtype : str
            Box type label.
        """
        self._configuration_manager.define_box(
            box_name=box_name,
            ipaddr_wss=ipaddr_wss,
            boxtype=boxtype,
        )

    def define_port(
        self,
        *,
        port_name: str,
        box_name: str,
        port_number: int | tuple[int, int],
    ) -> None:
        """
        Define a port in qube-calib.

        Parameters
        ----------
        port_name : str
            Port name.
        box_name : str
            Box name owning the port.
        port_number : int | tuple[int, int]
            Port number.
        """
        self._configuration_manager.define_port(
            port_name=port_name,
            box_name=box_name,
            port_number=port_number,
        )

    def define_channel(
        self,
        *,
        channel_name: str,
        port_name: str,
        channel_number: int,
        ndelay_or_nwait: int = 0,
    ) -> None:
        """
        Define a channel in qube-calib.

        Parameters
        ----------
        channel_name : str
            Channel name.
        port_name : str
            Port name owning the channel.
        channel_number : int
            Channel number.
        ndelay_or_nwait : int, optional
            Capture delay or wait words.
        """
        self._configuration_manager.define_channel(
            channel_name=channel_name,
            port_name=port_name,
            channel_number=channel_number,
            ndelay_or_nwait=ndelay_or_nwait,
        )

    def add_channel_target_relation(self, channel_name: str, target_name: str) -> None:
        """
        Add a channel-target relation if it does not already exist.

        Parameters
        ----------
        channel_name : str
            Channel name.
        target_name : str
            Target name.
        """
        self._configuration_manager.add_channel_target_relation(
            channel_name=channel_name,
            target_name=target_name,
        )

    def define_target(
        self,
        target_name: str,
        channel_name: str,
        target_frequency_ghz: float | None = None,
    ) -> None:
        """
        Define a target.

        Parameters
        ----------
        target_name : str
            Name of the target.
        channel_name : str
            Name of the channel.
        target_frequency_ghz : float, optional
            Frequency of the target in GHz.
        """
        self._configuration_manager.define_target(
            target_name=target_name,
            channel_name=channel_name,
            target_frequency_ghz=target_frequency_ghz,
        )

    def modify_target_frequency(self, target: str, frequency_ghz: float) -> None:
        """
        Modify the target frequency.

        Parameters
        ----------
        target : str
            Name of the target.
        frequency_ghz : float
            Modified frequency in GHz.
        """
        self._configuration_manager.modify_target_frequency(
            target=target,
            frequency_ghz=frequency_ghz,
        )

    def modify_target_frequencies(
        self, target_frequencies_ghz: dict[str, float]
    ) -> None:
        """
        Modify the target frequencies.

        Parameters
        ----------
        target_frequencies_ghz : dict[str, float]
            Dictionary of target frequencies.
        """
        self._configuration_manager.modify_target_frequencies(
            target_frequencies_ghz=target_frequencies_ghz
        )

    def config_port(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        lo_freq_hz: int | None = None,
        cnco_freq_hz: int | None = None,
        vatt: int | None = None,
        sideband: str | None = None,
        fullscale_current: int | None = None,
        rfswitch: str | None = None,
    ) -> None:
        """
        Configure the port of a box.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port : int | tuple[int, int]
            Port number.
        lo_freq_hz : int | None, optional
            Local oscillator frequency in Hz.
        cnco_freq_hz : int | None, optional
            CNCO frequency in Hz.
        vatt : int | None, optional
            VATT value.
        sideband : str | None, optional
            Sideband value.
        fullscale_current : int | None, optional
            Fullscale current value.
        rfswitch : str | None, optional
            RF switch value.
        """
        self._configuration_manager.config_port(
            box_name=box_name,
            port=port,
            lo_freq_hz=lo_freq_hz,
            cnco_freq_hz=cnco_freq_hz,
            vatt=vatt,
            sideband=sideband,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
        )

    def config_channel(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        channel: int,
        fnco_freq_hz: int | None = None,
    ) -> None:
        """
        Configure the channel of a box.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port : int | tuple[int, int]
            Port number.
        channel : int
            Channel number.
        fnco_freq_hz : int | None, optional
            FNCO frequency in Hz.
        """
        self._configuration_manager.config_channel(
            box_name=box_name,
            port=port,
            channel=channel,
            fnco_freq_hz=fnco_freq_hz,
        )

    def config_runit(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        runit: int,
        fnco_freq_hz: int | None = None,
    ) -> None:
        """
        Configure the runit of a box.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port : int | tuple[int, int]
            Port number.
        runit : int
            Runit number.
        fnco_freq_hz : int | None, optional
            FNCO frequency in Hz.
        """
        self._configuration_manager.config_runit(
            box_name=box_name,
            port=port,
            runit=runit,
            fnco_freq_hz=fnco_freq_hz,
        )

    def dump_box(self, box_name: str) -> dict:
        """
        Dump the box configuration.

        Parameters
        ----------
        box_name : str
            Name of the box.

        Returns
        -------
        dict
            Dictionary of box configuration.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.
        """
        return self._configuration_manager.dump_box(
            box_name=box_name,
        )

    def dump_port(self, box_name: str, port_number: int | tuple[int, int]) -> dict:
        """
        Dump the port configuration.

        Parameters
        ----------
        box_name : str
            Name of the box.
        port_number : int | tuple[int, int]
            Port number.

        Returns
        -------
        dict
            Dictionary of port configuration.

        Raises
        ------
        ValueError
            If the box is not in the available boxes.
        """
        return self._configuration_manager.dump_port(
            box_name=box_name,
            port_number=port_number,
        )

    def set_box_options(self, box_options: dict[str, tuple[str, ...]]) -> None:
        """Set box option labels used for relinkup config options."""
        self._configuration_manager.set_box_options(box_options)

    def add_sequencer(self, sequencer: Sequencer) -> None:
        """
        Add a sequencer to the queue.

        Parameters
        ----------
        sequencer : Sequencer
            Sequencer to add to the queue.

        """
        self._configuration_manager.add_sequencer(sequencer=sequencer)

    def show_command_queue(self) -> None:
        """Show the current command queue."""
        logger.info(self._configuration_manager.show_command_queue())

    def clear_command_queue(self) -> None:
        """Clear the command queue."""
        self._configuration_manager.clear_command_queue()

    def clear_cache(self) -> None:
        """Clear cached box configuration data."""
        self._connection_manager.clear_cache()

    def get_box_config_cache(self) -> dict[str, Any]:
        """Return a snapshot of the box-config cache."""
        return deepcopy(self.box_config)

    def replace_box_config_cache(self, box_configs: dict[str, Any]) -> None:
        """Replace the box-config cache with the provided snapshot."""
        self._connection_manager.replace_box_config_cache(box_configs)

    def update_box_config_cache(self, box_configs: dict[str, Any]) -> None:
        """Update cached box configurations by box name."""
        self._connection_manager.update_box_config_cache(box_configs)

    def get_resource_map(self, targets: list[str]) -> dict[str, list[dict]]:
        """Build a resource map for the requested targets."""
        return self._configuration_manager.get_resource_map(targets=targets)

    # Continuous Wave Operations
    def start_continuous_wave(
        self,
        *,
        box_name: str,
        port: PortType,
        channel: int,
        amplitude: float = 1.0,
        phase_rad: float = 0.0,
        lo_freq_hz: float | None = None,
        cnco_freq_hz: float | None = None,
        fnco_freq_hz: float | None = None,
        awg_freq_hz: float | None = None,
        sideband: str | None = None,
        vatt: int | None = None,
        fullscale_current: int | None = None,
        rfswitch: str | None = None,
        configure_port: bool = False,
        blocks_per_chunk: int | None = None,
        chunk_repeats: int | None = None,
        awg_repeats: int | None = None,
    ) -> Quel1ContinuousWaveConfig:
        """
        Start one QuEL-1 continuous-wave output.

        Parameters
        ----------
        box_name : str
            Connected QuEL-1 box name.
        port : PortType
            Output port identifier.
        channel : int
            AWG channel number on the output port.
        amplitude : float, optional
            Peak amplitude normalized to the DAC code range. Valid range is
            0.0 to 1.0.
        phase_rad : float, optional
            Initial waveform phase in radians.
        lo_freq_hz : float | None, optional
            LO frequency to apply when `configure_port=True`.
        cnco_freq_hz : float | None, optional
            CNCO frequency to apply when `configure_port=True`.
        fnco_freq_hz : float | None, optional
            FNCO frequency to apply when `configure_port=True`.
        awg_freq_hz : float | None, optional
            AWG-baseband frequency in Hz. If omitted, uses 0 Hz.
        sideband : str | None, optional
            Sideband setting to apply when `configure_port=True`.
        vatt : int | None, optional
            VATT setting to apply when `configure_port=True`.
        fullscale_current : int | None, optional
            Full-scale current setting to apply when `configure_port=True`.
        rfswitch : str | None, optional
            RF switch setting to apply when `configure_port=True`.
        configure_port : bool, optional
            Whether to update output path settings before starting. The
            default keeps current hardware output and phase state.
        blocks_per_chunk : int | None, optional
            Number of 128 ns hardware blocks in one generated chunk. When
            omitted, uses one block.
        chunk_repeats : int | None, optional
            Repeat count for the generated chunk. When omitted, uses the
            hardware maximum.
        awg_repeats : int | None, optional
            Repeat count for the AWG sequence. When omitted, uses the hardware
            maximum.

        Returns
        -------
        Quel1ContinuousWaveConfig
            Resolved continuous-wave configuration.
        """
        kwargs: dict[str, Any] = {}
        if blocks_per_chunk is not None:
            kwargs["blocks_per_chunk"] = blocks_per_chunk
        if chunk_repeats is not None:
            kwargs["chunk_repeats"] = chunk_repeats
        if awg_repeats is not None:
            kwargs["awg_repeats"] = awg_repeats
        return self._continuous_wave_manager.start_continuous_wave(
            box_name=box_name,
            port=port,
            channel=channel,
            amplitude=amplitude,
            phase_rad=phase_rad,
            lo_freq_hz=lo_freq_hz,
            cnco_freq_hz=cnco_freq_hz,
            fnco_freq_hz=fnco_freq_hz,
            awg_freq_hz=awg_freq_hz,
            sideband=sideband,
            vatt=vatt,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
            configure_port=configure_port,
            **kwargs,
        )

    def start_continuous_waves(
        self,
        *,
        box_name: str,
        port: PortType,
        waves: Sequence[Quel1ContinuousWaveChannelSpec | Mapping[str, Any]],
        lo_freq_hz: float | None = None,
        cnco_freq_hz: float | None = None,
        sideband: str | None = None,
        vatt: int | None = None,
        fullscale_current: int | None = None,
        rfswitch: str | None = None,
        configure_port: bool = False,
        blocks_per_chunk: int | None = None,
        chunk_repeats: int | None = None,
        awg_repeats: int | None = None,
    ) -> tuple[Quel1ContinuousWaveConfig, ...]:
        """
        Start one port-scoped QuEL-1 continuous-wave output group.

        Parameters
        ----------
        box_name : str
            Connected QuEL-1 box name.
        port : PortType
            Output port identifier.
        waves : Sequence[Quel1ContinuousWaveChannelSpec | Mapping[str, Any]]
            Per-channel CW settings. The group starts with one `start_wavegen()`
            call after all channels are configured.
        lo_freq_hz : float | None, optional
            LO frequency to apply when `configure_port=True`.
        cnco_freq_hz : float | None, optional
            CNCO frequency to apply when `configure_port=True`.
        sideband : str | None, optional
            Sideband setting to apply when `configure_port=True`.
        vatt : int | None, optional
            VATT setting to apply when `configure_port=True`.
        fullscale_current : int | None, optional
            Full-scale current setting to apply when `configure_port=True`.
        rfswitch : str | None, optional
            RF switch setting to apply when `configure_port=True`.
        configure_port : bool, optional
            Whether to update output path settings before starting. The
            default keeps current hardware output and phase state.
        blocks_per_chunk : int | None, optional
            Number of 128 ns hardware blocks in each generated chunk. When
            omitted, uses one block.
        chunk_repeats : int | None, optional
            Repeat count for each generated chunk. When omitted, uses the
            hardware maximum.
        awg_repeats : int | None, optional
            Repeat count for each AWG sequence. When omitted, uses the hardware
            maximum.

        Returns
        -------
        tuple[Quel1ContinuousWaveConfig, ...]
            Resolved continuous-wave configurations in input order.
        """
        kwargs: dict[str, Any] = {}
        if blocks_per_chunk is not None:
            kwargs["blocks_per_chunk"] = blocks_per_chunk
        if chunk_repeats is not None:
            kwargs["chunk_repeats"] = chunk_repeats
        if awg_repeats is not None:
            kwargs["awg_repeats"] = awg_repeats
        return self._continuous_wave_manager.start_continuous_waves(
            box_name=box_name,
            port=port,
            waves=waves,
            lo_freq_hz=lo_freq_hz,
            cnco_freq_hz=cnco_freq_hz,
            sideband=sideband,
            vatt=vatt,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
            configure_port=configure_port,
            **kwargs,
        )

    def stop_continuous_wave(
        self,
        *,
        box_name: str,
        port: PortType,
        channel: int | None = None,
        timeout: float = 2.0,
        polling_period: float = 0.01,
    ) -> bool:
        """
        Stop the active QuEL-1 continuous-wave output group on a port.

        Parameters
        ----------
        box_name : str
            Connected QuEL-1 box name.
        port : PortType
            Output port identifier.
        channel : int | None, optional
            Legacy channel selector. The canonical stop target is the active
            CW group on the port.
        timeout : float, optional
            Stop timeout in seconds for cancellable task implementations.
        polling_period : float, optional
            Polling period in seconds for cancellable task implementations.

        Returns
        -------
        bool
            `True` when a remembered task was stopped or cleared, otherwise
            `False`.
        """
        return self._continuous_wave_manager.stop_continuous_wave(
            box_name=box_name,
            port=port,
            channel=channel,
            timeout=timeout,
            polling_period=polling_period,
        )

    def stop_all_continuous_waves(
        self,
        *,
        timeout: float = 2.0,
        polling_period: float = 0.01,
    ) -> None:
        """Stop all active QuEL-1 continuous-wave outputs."""
        self._continuous_wave_manager.stop_all_continuous_waves(
            timeout=timeout,
            polling_period=polling_period,
        )

    # QuEL-1 Optional Capabilities
    def load_skew_yaml(self, file_path: str | Path) -> None:
        """
        Load skew calibration YAML into the system database.

        Parameters
        ----------
        file_path : str | Path
            Path to the skew calibration YAML file.
        """
        self._skew_manager.load_skew_yaml(file_path)

    def update_skew(
        self,
        *,
        file_path: str | Path,
        wait: int,
        box_names: list[str] | None = None,
        backup: bool = False,
    ) -> dict[str, object]:
        """
        Update skew waits in one YAML file and reload backend skew settings.

        Parameters
        ----------
        file_path : str | Path
            Path to the skew calibration YAML file.
        wait : int
            Target skew index. Measured effective waits are shifted by
            `wait - measured_idx`, then normalized into `wait` and `port_wait`.
        box_names : list[str] | None, optional
            Box names to update. When omitted, all boxes in the file are
            updated.
        backup : bool, optional
            Whether to save the original file as `*.bak.YYYYMMDD_HHMMSS`
            before overwriting it.

        Returns
        -------
        dict[str, object]
            Summary of the updated skew file.
        """
        return self._skew_manager.update_skew(
            file_path=file_path,
            wait=wait,
            box_names=box_names,
            backup=backup,
        )

    def run_skew_measurement(
        self,
        *,
        skew_yaml_path: str | Path,
        box_yaml_path: str | Path,
        clockmaster_ip: str,
        box_names: list[str],
        target_box_names: list[str] | None = None,
        estimate: bool = True,
    ) -> tuple[Any, Any]:
        """Measure skew from YAML settings and return skew object and figure."""
        return self._skew_manager.run_skew_measurement(
            skew_yaml_path=skew_yaml_path,
            box_yaml_path=box_yaml_path,
            clockmaster_ip=clockmaster_ip,
            box_names=box_names,
            target_box_names=target_box_names,
            estimate=estimate,
        )

    # Execution Entry Points
    def execute_sync(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """Execute a backend request synchronously using QuEL-1 defaults."""
        return self._execution_manager.execute_sync(
            request=request,
            execution_mode=execution_mode,
            clock_health_checks=clock_health_checks,
        )

    async def execute_async(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """Execute a backend request asynchronously using QuEL-1 defaults."""
        return await self._execution_manager.execute_async(
            request=request,
            execution_mode=execution_mode,
            clock_health_checks=clock_health_checks,
        )
