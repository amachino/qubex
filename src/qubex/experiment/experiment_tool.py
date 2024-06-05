from __future__ import annotations

from typing import Final

from qubecalib import QubeCalib
from rich.console import Console
from rich.table import Table

from ..config import Config
from ..measurement import DEFAULT_CONFIG_DIR
from ..qube_backend import QubeBackend

console = Console()


class ExperimentTool:
    def __init__(
        self,
        chip_id: str,
        config_dir: str = DEFAULT_CONFIG_DIR,
    ):
        config = Config(config_dir)
        config.configure_system_settings(chip_id)
        config_path = config.get_system_settings_path(chip_id)
        self._config: Final = config
        self._system: Final = config.get_quantum_system(chip_id)
        self._backend: Final = QubeBackend(config_path)

    def get_qubecalib(self) -> QubeCalib:
        """Get the QubeCalib instance."""
        return self._backend.qubecalib

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
        chip_id = self._system.chip.id
        self._config.configure_box_settings(chip_id, include=[box_id])

    def print_wiring_info(self):
        """
        Print the wiring information of the chip.

        Examples
        --------
        >>> from qubex import Experiment
        >>> ex = Experiment(chip_id="64Q")
        >>> ex.tool.print_wiring_info()
        """

        table = Table(
            show_header=True,
            header_style="bold",
            title=f"WIRING INFO ({self._system.chip.id})",
        )
        table.add_column("QUBIT", justify="center", width=7)
        table.add_column("CTRL", justify="center", width=11)
        table.add_column("READ.OUT", justify="center", width=11)
        table.add_column("READ.IN", justify="center", width=11)

        for qubit in self._system.qubits:
            ports = self._config.get_ports_by_qubit(
                chip_id=self._system.chip.id,
                qubit=qubit.label,
            )
            ctrl_port = ports[0]
            read_out_port = ports[1]
            read_in_port = ports[2]
            if ctrl_port is None or read_out_port is None or read_in_port is None:
                table.add_row(qubit.label, "-", "-", "-")
                continue
            ctrl_box = ctrl_port.box
            read_out_box = read_out_port.box
            read_in_box = read_in_port.box
            ctrl = f"{ctrl_box.id}-{ctrl_port.number}"
            read_out = f"{read_out_box.id}-{read_out_port.number}"
            read_in = f"{read_in_box.id}-{read_in_port.number}"
            table.add_row(qubit.label, ctrl, read_out, read_in)

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
        box_ids = [box.id for box in self._config.get_all_boxes()]
        if box_id not in box_ids:
            console.print(f"Box {box_id} is not found.")
            return

        table1 = Table(
            show_header=True,
            header_style="bold",
            title=f"BOX INFO ({box_id})",
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
        table2.add_column("FNCO-0", justify="right")
        table2.add_column("FNCO-1", justify="right")
        table2.add_column("FNCO-2", justify="right")
        table2.add_column("FNCO-3", justify="right")

        port_map = self._config.get_port_map(box_id)
        ssb_map = {"U": "[cyan]USB[/cyan]", "L": "[green]LSB[/green]"}

        ports = self.dump_box(box_id)["ports"]
        for number, port in ports.items():
            direction = port["direction"]
            lo = int(port["lo_freq"])
            cnco = int(port["cnco_freq"])
            type = port_map[number].value
            if direction == "in":
                ssb = ""
                vatt = ""
                fsc = ""
                fncos = [str(int(ch["fnco_freq"])) for ch in port["runits"].values()]
            elif direction == "out":
                ssb = ssb_map[port["sideband"]]
                vatt = port.get("vatt", "")
                fsc = port["fullscale_current"]
                fncos = [str(int(ch["fnco_freq"])) for ch in port["channels"].values()]
            table1.add_row(
                str(number),
                type,
                ssb,
                str(lo),
                str(cnco),
                str(vatt),
                str(fsc),
            )
            table2.add_row(
                str(number),
                type,
                *fncos,
            )
        console.print(table1)
        console.print(table2)
