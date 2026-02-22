"""
QuEL-1 backend controller implementing measurement-facing backend contracts.

This module provides the QuEL-1 concrete `BackendController` implementation.
It exposes the required shared controller contract plus QuEL-1-specific
capabilities, while delegating concrete operations to QuEL-1 managers.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qubex.backend.backend_executor import (
    BackendExecutionRequest,
    BackendExecutionResult,
)
from qubex.backend.controller_types import BackendController

from .managers import (
    Quel1ClockManager,
    Quel1ConfigurationManager,
    Quel1ConnectionManager,
    Quel1ExecutionManager,
)
from .quel1_backend_constants import ExecutionMode
from .quel1_backend_raw_result import Quel1BackendRawResult
from .quel1_runtime_context import Quel1RuntimeContext

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .compat.qubecalib_protocols import (
        BoxPoolProtocol as BoxPool,
        QubeCalibProtocol as QubeCalib,
        Quel1BoxCommonProtocol as Quel1Box,
        Quel1SystemProtocol as Quel1System,
        QuelDriverClassesProtocol,
        SequencerProtocol as Sequencer,
    )


class Quel1BackendController(BackendController):
    """
    QuEL-1 backend controller backed by qubecalib and manager delegation.

    The controller is the measurement-layer entrypoint for QuEL-1 sessions and
    execution. It implements shared `BackendController` requirements and
    delegates connection, clock, configuration, and execution details to
    backend-local manager components.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
    ):
        """Initialize QuEL-1 controller and manager delegates."""
        self._runtime_context = Quel1RuntimeContext.create(
            config_path=config_path,
        )
        self._connection_manager = Quel1ConnectionManager(
            runtime_context=self._runtime_context,
        )
        self._clock_manager = Quel1ClockManager(
            runtime_context=self._runtime_context,
        )
        self._execution_manager = Quel1ExecutionManager(
            runtime_context=self._runtime_context,
        )
        self._configuration_manager = Quel1ConfigurationManager(
            runtime_context=self._runtime_context,
        )

    # Core Properties
    @property
    def driver(self) -> QuelDriverClassesProtocol:
        """Return loaded QuEL-1 driver class bundle."""
        return self._runtime_context.driver

    @property
    def sampling_period(self) -> float:
        """Return backend sampling period in ns."""
        return self._runtime_context.sampling_period

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
        self._connection_manager.connect(box_names=box_names, parallel=parallel)

    def disconnect(self) -> None:
        """Disconnect backend resources and reset connection-related state."""
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
        noise_threshold: int | None = None,
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
            The linked up box object.

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
        noise_threshold: int | None = None,
        *,
        parallel: bool | None = None,
    ) -> dict[str, Quel1Box]:
        """
        Linkup all the boxes in the list.

        Parameters
        ----------
        box_list : list[str]
            List of box names.
        noise_threshold : int | None, optional
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

    def relinkup(self, box_name: str, noise_threshold: int | None = None) -> None:
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
        noise_threshold: int | None = None,
        *,
        parallel: bool | None = None,
    ) -> None:
        """
        Relink all the boxes in the list.

        Parameters
        ----------
        box_list : list[str]
            List of box names.
        noise_threshold : int | None, optional
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
    def define_clockmaster(self, *, ipaddr: str, reset: bool = True) -> None:
        """
        Define the clock master in qube-calib.

        Parameters
        ----------
        ipaddr : str
            Clock master IP address.
        reset : bool, optional
            Whether to reset clock master on define.
        """
        self._configuration_manager.define_clockmaster(ipaddr=ipaddr, reset=reset)

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
        target_frequency: float | None = None,
    ) -> None:
        """
        Define a target.

        Parameters
        ----------
        target_name : str
            Name of the target.
        channel_name : str
            Name of the channel.
        target_frequency : float, optional
            Frequency of the target in GHz.
        """
        self._configuration_manager.define_target(
            target_name=target_name,
            channel_name=channel_name,
            target_frequency=target_frequency,
        )

    def modify_target_frequency(self, target: str, frequency: float) -> None:
        """
        Modify the target frequency.

        Parameters
        ----------
        target : str
            Name of the target.
        frequency : float
            Modified frequency in GHz.
        """
        self._configuration_manager.modify_target_frequency(
            target=target,
            frequency=frequency,
        )

    def modify_target_frequencies(self, frequencies: dict[str, float]) -> None:
        """
        Modify the target frequencies.

        Parameters
        ----------
        frequencies : dict[str, float]
            Dictionary of target frequencies.
        """
        self._configuration_manager.modify_target_frequencies(frequencies=frequencies)

    def config_port(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        lo_freq: float | None = None,
        cnco_freq: float | None = None,
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
        lo_freq : float | None, optional
            Local oscillator frequency in GHz.
        cnco_freq : float | None, optional
            CNCO frequency in GHz.
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
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
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
        fnco_freq: float | None = None,
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
        fnco_freq : float | None, optional
            FNCO frequency in GHz.
        """
        self._configuration_manager.config_channel(
            box_name=box_name,
            port=port,
            channel=channel,
            fnco_freq=fnco_freq,
        )

    def config_runit(
        self,
        box_name: str,
        *,
        port: int | tuple[int, int],
        runit: int,
        fnco_freq: float | None = None,
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
        fnco_freq : float | None, optional
            FNCO frequency in GHz.
        """
        self._configuration_manager.config_runit(
            box_name=box_name,
            port=port,
            runit=runit,
            fnco_freq=fnco_freq,
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
            The sequencer to add to the queue.
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

    # QuEL-1 Optional Capabilities
    def load_skew_yaml(self, file_path: str | Path) -> None:
        """
        Load skew calibration YAML into the system database.

        Parameters
        ----------
        file_path : str | Path
            Path to the skew calibration YAML file.
        """
        self.qubecalib.sysdb.load_skew_yaml(str(file_path))

    def run_skew_measurement(
        self,
        *,
        skew_yaml_path: str | Path,
        box_yaml_path: str | Path,
        clockmaster_ip: str,
        box_names: list[str],
        estimate: bool = True,
    ) -> tuple[Any, Any]:
        """
        Measure skew from YAML settings and return skew object and figure.

        Parameters
        ----------
        skew_yaml_path : str | Path
            Path to skew YAML.
        box_yaml_path : str | Path
            Path to box YAML.
        clockmaster_ip : str
            Clock master IP address.
        box_names : list[str]
            Boxes to include in the measurement.
        estimate : bool, optional
            Whether to run estimation after measurement.

        Returns
        -------
        tuple[Any, Any]
            A tuple of (skew object, plotly figure).
        """
        skew = self.driver.Skew.from_yaml(
            str(skew_yaml_path),
            box_yaml=str(box_yaml_path),
            clockmaster_ip=clockmaster_ip,
            boxes=box_names,
        )
        skew.system.resync()
        skew.measure()
        if estimate:
            skew.estimate()
        fig = skew.plot()
        return skew, fig

    # Execution Entry Points
    def execute(
        self,
        *,
        request: BackendExecutionRequest,
        execution_mode: ExecutionMode | None = None,
        clock_health_checks: bool | None = None,
    ) -> BackendExecutionResult:
        """Execute a backend request using QuEL-1 execution defaults."""
        from .quel1_backend_executor import Quel1BackendExecutor

        executor = Quel1BackendExecutor(
            backend_controller=self,
            execution_manager=self._execution_manager,
            execution_mode=execution_mode,
            clock_health_checks=clock_health_checks,
        )
        return self._execution_manager.execute(
            request=request,
            executor=executor,
        )

    def execute_sequencer(
        self,
        sequencer: Sequencer,
        *,
        repeats: int,
        integral_mode: str = "integral",
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
        enable_sum: bool = False,
        enable_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
    ) -> Quel1BackendRawResult:
        """Execute one sequencer via serial path."""
        return self._execution_manager.execute_sequencer(
            sequencer=sequencer,
            repeats=repeats,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
            enable_sum=enable_sum,
            enable_classification=enable_classification,
            line_param0=line_param0,
            line_param1=line_param1,
        )

    def execute_sequencer_parallel(
        self,
        sequencer: Sequencer,
        *,
        repeats: int,
        integral_mode: str = "integral",
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
        enable_sum: bool = False,
        enable_classification: bool = False,
        line_param0: tuple[float, float, float] | None = None,
        line_param1: tuple[float, float, float] | None = None,
        clock_health_checks: bool = False,
    ) -> Quel1BackendRawResult:
        """Execute one sequencer via parallel action path."""
        return self._execution_manager.execute_sequencer_parallel(
            sequencer=sequencer,
            repeats=repeats,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
            enable_sum=enable_sum,
            enable_classification=enable_classification,
            line_param0=line_param0,
            line_param1=line_param1,
            clock_health_checks=clock_health_checks,
        )
