from __future__ import annotations

from pathlib import Path
from typing import Final

from rich.prompt import Confirm

try:
    from qubecalib import QubeCalib
except ImportError:
    pass

from .control_system import PortType
from .experiment_system import ExperimentSystem


class QubeCalibManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(
        self,
        experiment_system: ExperimentSystem,
    ):
        self._experiment_system: Final = experiment_system
        self._qubecalib: Final = QubeCalib()

    @property
    def experiment_system(self) -> ExperimentSystem:
        return self._experiment_system

    @property
    def qubecalib(self) -> QubeCalib:
        return self._qubecalib

    def configure(
        self,
        chip_id: str,
        *,
        path_to_save: str = ".config/system_settings.json",
    ) -> Path:
        experiment_system = self.experiment_system
        control_system = experiment_system.control_system
        control_params = experiment_system.control_params

        # qubecalib object
        qc = self.qubecalib

        # define clock master
        qc.define_clockmaster(
            ipaddr=control_system.clock_master_address,
            reset=False,  # TODO: check if this should be True
        )

        # define boxes, ports, and channels
        for box in control_system.boxes:
            # define box
            qc.define_box(
                box_name=box.id,
                ipaddr_wss=box.address,
                boxtype=box.type.value,
            )

            # define ports
            for port in box.ports:
                if port.type == PortType.NOT_AVAILABLE:
                    continue
                qc.define_port(
                    port_name=port.id,
                    box_name=box.id,
                    port_number=port.number,
                )

                # define channels
                for channel in port.channels:
                    if port.type == PortType.READ_IN:
                        mux = experiment_system.get_mux_by_readout_port(port)
                        if mux is None:
                            raise ValueError(
                                f"No mux found for readout port: {port.id}"
                            )
                        ndelay_or_nwait = control_params.capture_delay[mux.index]
                    else:
                        ndelay_or_nwait = 0
                    qc.define_channel(
                        channel_name=channel.id,
                        port_name=port.id,
                        channel_number=channel.number,
                        ndelay_or_nwait=ndelay_or_nwait,
                    )

        # define gen targets
        target_gen_channel_map = experiment_system.target_gen_channel_map
        for target, gen_channel in target_gen_channel_map.items():
            qc.define_target(
                target_name=target.label,
                channel_name=gen_channel.id,
                target_frequency=target.frequency,
            )
        # define cap targets
        target_cap_channel_map = experiment_system.target_cap_channel_map
        for target, cap_channel in target_cap_channel_map.items():
            qc.define_target(
                target_name=target.label,
                channel_name=cap_channel.id,
                target_frequency=target.frequency,
            )

        # save the system settings to a JSON file
        system_settings_path = Path(path_to_save)
        system_settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(system_settings_path, "w") as f:
            f.write(qc.system_config_database.asjson())

        return system_settings_path

    def configure_box_settings(
        self,
        chip_id: str,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        loopback: bool = False,
    ):
        """
        Configures the box settings for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").
        include : list[str] | None, optional
            The list of box IDs to include, by default None.
        exclude : list[str] | None, optional
            The list of box IDs to exclude, by default None.
        loopback : bool, optional
            Whether to enable loopback mode, by default False.

        Examples
        --------
        >>> config = ConfigLoader()
        >>> config.configure_box_settings("64Q")

        >>> config.configure_box_settings(
        ...     chip_id="64Q",
        ...     include=["Q2A", "Q73A"],
        ...     loopback=True,
        ... )

        >>> config.configure_box_settings(
        ...     chip_id="64Q",
        ...     exclude=["Q2A", "Q73A"],
        ...     loopback=True,
        ... )
        """

        try:
            system_settings_path = self.get_system_settings_path(chip_id)
            qc = QubeCalib(str(system_settings_path))
        except FileNotFoundError:
            print(f"System settings file not found for chip ID: {chip_id}")
            return

        control_system = self.get_control_system(chip_id)
        control_system = self.configure_control_system(chip_id, loopback=loopback)
        qube_system = control_system.qube_system
        boxes = list(qube_system.boxes.values())

        if include is not None:
            boxes = [box for box in boxes if box.id in include]
        if exclude is not None:
            boxes = [box for box in boxes if box.id not in exclude]

        box_list_str = "\n".join([f"{box.id} ({box.name})" for box in boxes])
        confirmed = Confirm.ask(
            f"""
You are going to configure the following boxes:

[bold bright_green]{box_list_str}[/bold bright_green]

This operation will overwrite the existing device settings. Do you want to continue?
"""
        )
        if not confirmed:
            print("Operation cancelled.")
            return

        for box in boxes:
            quel1_box = qc.create_box(box.id, reconnect=False)
            quel1_box.reconnect()
            for port in box.ports:
                if port.type in [PortType.NOT_AVAILABLE, PortType.CTRL]:
                    quel1_box.config_port(
                        port=port.number,
                        lo_freq=port.lo_freq,
                        cnco_freq=port.cnco_freq,
                        vatt=port.vatt,
                        sideband=port.sideband,
                        fullscale_current=port.fullscale_current,
                        rfswitch=port.rfswitch,
                    )
                    for channel in port.channels:
                        quel1_box.config_channel(
                            port=port.number,
                            channel=channel.number,
                            fnco_freq=channel.fnco_freq,
                        )
                elif port.type == PortType.READ_IN:
                    quel1_box.config_port(
                        port=port.number,
                        lo_freq=port.lo_freq,
                        cnco_locked_with=qube_system.get_readout_pair(port).number,
                        rfswitch=port.rfswitch,
                    )
                    for channel in port.channels:
                        quel1_box.config_runit(
                            port=port.number,
                            runit=channel.number,
                            fnco_freq=channel.fnco_freq,
                        )
        print("Box settings configured.")

    def save_box_settings(
        self,
        *,
        chip_id: str,
        path_to_save: str | None = None,
    ):
        """
        Saves the box settings for the given chip ID.

        Parameters
        ----------
        chip_id : str
            The quantum chip ID (e.g., "64Q").

        Examples
        --------
        >>> config = ConfigLoader()
        >>> config.save_box_settings("64Q")
        """
        try:
            system_settings_path = self.get_system_settings_path(chip_id)
            qc = QubeCalib(str(system_settings_path))
        except FileNotFoundError:
            print(f"System settings file not found for chip ID: {chip_id}")
            return
        if path_to_save is None:
            box_settings_path = self.get_box_settings_path(chip_id)
        else:
            box_settings_path = Path(path_to_save)
        box_settings_path.parent.mkdir(parents=True, exist_ok=True)
        qc.store_all_box_configs(box_settings_path)

        print(f"Box settings saved to: {box_settings_path}")
