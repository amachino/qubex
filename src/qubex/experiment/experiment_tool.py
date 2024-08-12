from __future__ import annotations

import subprocess
from typing import Final, Sequence

from qubecalib import QubeCalib
from quel_ic_config import Quel1Box
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from ..backend import ConfigLoader, ExperimentSystem
from ..measurement.measurement import Measurement

console = Console()


class ExperimentTool:
    def __init__(
        self,
        chip_id: str,
        qubits: list[str],
        config: ConfigLoader,
        control_system: ExperimentSystem,
        measurement: Measurement,
    ):
        self._chip_id: Final = chip_id
        self._qubits: Final = qubits
        self._config: Final = config
        self._control_system: Final = control_system
        self._measurement: Final = measurement
        self._backend: Final = measurement._backend

    def get_qubecalib(self) -> QubeCalib:
        """Get the QubeCalib instance."""
        return self._backend.qubecalib

    def reboot_fpga(self, box_id: str) -> None:
        """Reboot the FPGA."""
        # Run the following commands in the terminal.
        # $ source /tools/Xilinx/Vivado/2020.1/settings64.sh
        # $ quel_reboot_fpga --port 3121 --adapter xxx
        box = self._config.get_box(box_id)
        adapter = box.adapter
        reboot_command = f"quel_reboot_fpga --port 3121 --adapter {adapter}"
        subprocess.run(reboot_command, shell=True)

    def get_quel1_box(self, box_id: str) -> Quel1Box:
        """Get the Quel1Box instance."""
        box = self._backend.qubecalib.create_box(box_id, reconnect=False)
        box.reconnect()
        return box

    def dump_box(self, box_id: str) -> dict:
        """Dump the information of a box."""
        return self._backend.dump_box(box_id)

    def configure_box(self, box_id: str) -> None:
        """
        Configure the box.

        Parameters
        ----------

        Examples
        --------
        >>> from qubex import Experiment
        >>> ex = Experiment(chip_id="64Q")
        >>> ex.tool.configure_box("Q73A")
        """
        self.configure_boxes([box_id])

    def configure_boxes(self, box_ids: list[str]) -> None:
        """
        Configure the boxes.

        Parameters
        ----------
        box_ids : list[str]
            List of box identifiers.

        Examples
        --------
        >>> from qubex import Experiment
        >>> ex = Experiment(chip_id="64Q")
        >>> ex.tool.configure_boxes(["Q73A", "U10B"])
        """
        self._config.configure_box_settings(self._chip_id, include=box_ids)

    def relinkup_box(self, box_id: str) -> None:
        """
        Relink up the box.

        Parameters
        ----------
        box_id : str
            Identifier of the box.

        Examples
        --------
        >>> from qubex import Experiment
        >>> ex = Experiment(chip_id="64Q")
        >>> ex.tool.relinkup_box("Q73A")
        """
        self.relinkup_boxes([box_id])

    def relinkup_boxes(self, box_ids: list[str]) -> None:
        """
        Relink up the boxes.

        Parameters
        ----------
        box_ids : list[str]
            List of box identifiers.

        Examples
        --------
        >>> ex.tool.relinkup()
        """
        confirmed = Confirm.ask(
            f"""
You are going to relinkup the following boxes:

[bold bright_green]{box_ids}[/bold bright_green]

This operation will reset LO/NCO settings. Do you want to continue?
"""
        )
        if not confirmed:
            print("Operation cancelled.")
            return

        print("Relinking up the boxes...")
        self._measurement.relinkup(box_ids)
        print("Operation completed.")

    def print_wiring_info(
        self,
        qubits: list[str] | None = None,
    ) -> None:
        """
        Print the wiring information of the chip.

        Examples
        --------
        >>> from qubex import Experiment
        >>> ex = Experiment(chip_id="64Q")
        >>> ex.tool.print_wiring_info()
        """

        control_system = self._control_system
        quantum_system = control_system.quantum_system

        table = Table(
            show_header=True,
            header_style="bold",
            title=f"WIRING INFO ({quantum_system.chip.id})",
        )
        table.add_column("QUBIT", justify="center", width=7)
        table.add_column("CTRL", justify="center", width=11)
        table.add_column("READ.OUT", justify="center", width=11)
        table.add_column("READ.IN", justify="center", width=11)

        control_system.port_qubit_map

        if qubits is None:
            qubits = list(quantum_system.qubits.keys())

        for qubit in qubits:
            port_set = control_system.qubit_port_set_map[qubit]
            ctrl_port = port_set.ctrl_port
            read_out_port = port_set.read_out_port
            read_in_port = port_set.read_in_port
            if ctrl_port is None or read_out_port is None or read_in_port is None:
                table.add_row(qubit, "-", "-", "-")
                continue
            ctrl_box = ctrl_port.box_id
            read_out_box = read_out_port.box_id
            read_in_box = read_in_port.box_id
            ctrl = f"{ctrl_box}-{ctrl_port.number}"
            read_out = f"{read_out_box}-{read_out_port.number}"
            read_in = f"{read_in_box}-{read_in_port.number}"
            table.add_row(qubit, ctrl, read_out, read_in)

        console.print(table)

    def print_box_info(self, box_id: str) -> None:
        """
        Print the information of a box.

        Parameters
        ----------
        box_id : str
            Identifier of the box.

        Examples
        --------
        >>> from qubex import Experiment
        >>> ex = Experiment(chip_id="64Q")
        >>> ex.tool.print_box_info("Q73A")
        """

        control_system = self._control_system
        qube_system = control_system.qube_system

        if box_id not in qube_system.boxes:
            print(f"Box {box_id} is not found.")
            return

        box = qube_system.boxes[box_id]

        table1 = Table(
            show_header=True,
            header_style="bold",
            title=f"BOX INFO ({box.id})",
        )
        table2 = Table(
            show_header=True,
            header_style="bold",
        )
        table1.add_column("PORT", justify="right")
        table1.add_column("TYPE", justify="right")
        table1.add_column("SSB", justify="right")
        table1.add_column("LO", justify="right")
        table1.add_column("CNCO", justify="right")
        table1.add_column("VATT", justify="right")
        table1.add_column("FSC", justify="right")
        table2.add_column("PORT", justify="right")
        table2.add_column("TYPE", justify="right")
        table2.add_column("SSB", justify="right")
        table2.add_column("FNCO-0", justify="right")
        table2.add_column("FNCO-1", justify="right")
        table2.add_column("FNCO-2", justify="right")

        ssb_map = {"U": "USB", "L": "LSB"}

        for port in box.ports:
            number = str(port.number)
            type = port.type.value
            ssb = ssb_map[port.sideband] if port.sideband else ""
            lo = f"{port.lo_freq:_}" if port.lo_freq else ""
            cnco = f"{port.cnco_freq:_}" if port.cnco_freq else ""
            vatt = str(port.vatt) if port.vatt else ""
            fsc = str(port.fullscale_current) if port.fullscale_current else ""

            table1.add_row(
                number,
                type,
                ssb,
                lo,
                cnco,
                vatt,
                fsc,
            )
            table2.add_row(
                number,
                type,
                ssb,
                *[f"{ch.fnco_freq:_}" for ch in port.channels],
            )
        console.print(table1)
        console.print(table2)

    def configure_port(
        self,
        box_id: str,
        port_number: int,
        channel_number: int,
        *,
        lo_freq: int | None = None,
        cnco_freq: int | None = None,
        fnco_freq: int | None = None,
        vatt: int | None = None,
        sideband: str | None = None,
        fullscale_current: int | None = None,
        rfswitch: str | None = None,
    ) -> None:
        confirmed = Confirm.ask(
            f"""
You are going to configure the following port settings:

[bold bright_green]{box_id}-{port_number}[/bold bright_green]

Do you want to continue?
"""
        )
        if not confirmed:
            print("Operation cancelled.")
            return
        quel1_box = self.get_quel1_box(box_id)
        quel1_box.config_port(
            port=port_number,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            vatt=vatt,
            sideband=sideband,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
        )
        quel1_box.config_channel(
            port=port_number,
            channel=channel_number,
            fnco_freq=fnco_freq,
        )

    def get_base_frequencies(self, box_id: str, port_number: int) -> list[int]:
        settings = self.dump_box(box_id)["ports"][port_number]
        ssb = settings["sideband"]
        lo = int(settings["lo_freq"])
        cnco = int(settings["cnco_freq"])
        channels = settings["channels"]
        fnco_list = [int(ch["fnco_freq"]) for ch in channels.values()]
        base_frequencies = []
        for fnco in fnco_list:
            if ssb == "U":
                f = lo + cnco + fnco
            else:
                f = lo - cnco - fnco
            base_frequencies.append(f)
        return base_frequencies

    def print_base_frequencies(self, targets: Sequence[str] | str) -> None:
        control_system = self._control_system
        port_set = control_system.qubit_port_set_map[target]
        control = port_set.ctrl_port
        readout = port_set.read_out_port

        if control is None or readout is None:
            print(f"Ports for qubit {target} are not found.")
            return

        control_base_frequencies = self.get_base_frequencies(
            box_id=control.box_id,
            port_number=control.number,
        )
        readout_base_frequency = self.get_base_frequencies(
            box_id=readout.box_id,
            port_number=readout.number,
        )[0]

        table = Table(
            show_header=True,
            header_style="bold",
            title=f"BASE FREQUENCIES ({target})",
        )
        for i in range(len(control_base_frequencies)):
            table.add_column(f"CTRL_{i}", justify="center")
        table.add_column("READ", justify="center")
        table.add_row(
            *[f"{f * 1e-9:.3f} GHz" for f in control_base_frequencies],
            f"{readout_base_frequency * 1e-9:.3f} GHz",
        )
        console.print(table)
