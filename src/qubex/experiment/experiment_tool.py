from __future__ import annotations

import subprocess
from typing import Sequence

from qubecalib import QubeCalib
from quel_ic_config import Quel1Box
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from ..backend import StateManager

console = Console()
state_manager = StateManager.shared()


def get_qubecalib() -> QubeCalib:
    """Get the QubeCalib instance."""
    return state_manager.device_controller.qubecalib


def get_quel1_box(box_id: str) -> Quel1Box:
    """Get the Quel1Box instance."""
    qc = state_manager.device_controller.qubecalib
    box = qc.create_box(box_id, reconnect=False)
    box.reconnect()
    return box


def dump_box(box_id: str) -> dict:
    """Dump the information of a box."""
    return state_manager.device_controller.dump_box(box_id)


def reboot_fpga(box_id: str) -> None:
    """Reboot the FPGA."""
    # Run the following commands in the terminal.
    # $ source /tools/Xilinx/Vivado/2020.1/settings64.sh
    # $ quel_reboot_fpga --port 3121 --adapter xxx
    experiment_system = state_manager.experiment_system
    box = experiment_system.get_box(box_id)
    adapter = box.adapter
    reboot_command = f"quel_reboot_fpga --port 3121 --adapter {adapter}"
    subprocess.run(reboot_command, shell=True)


def configure_box(box_id: str) -> None:
    """Configure the box."""
    state_manager.push([box_id])


def configure_boxes(box_ids: Sequence[str]) -> None:
    """Configure the boxes."""
    state_manager.push(box_ids)


def relinkup_box(box_id: str) -> None:
    """Relink up the box."""
    relinkup_boxes([box_id])


def relinkup_boxes(box_ids: list[str]) -> None:
    """Relink up the boxes."""
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
    state_manager.device_controller.relinkup_boxes(box_ids)
    state_manager.device_controller.sync_clocks(box_ids)
    print("Operation completed.")


def print_wiring_info(qubits: list[str] | None = None) -> None:
    """Print the wiring information of the chip."""

    experiment_system = state_manager.experiment_system

    table = Table(
        show_header=True,
        header_style="bold",
        title=f"WIRING INFO ({experiment_system.chip.id})",
    )
    table.add_column("MUX", justify="center")
    table.add_column("QUBIT", justify="center")
    table.add_column("CTRL", justify="center")
    table.add_column("READ.OUT", justify="center")
    table.add_column("READ.IN", justify="center")

    if qubits is None:
        qubits = [qubit.label for qubit in experiment_system.ge_targets]

    for qubit in qubits:
        port_set = experiment_system.get_qubit_port_set(qubit)
        if port_set is None:
            table.add_row(qubit, "-", "-", "-", "-")
            continue
        ctrl_port = port_set.ctrl_port
        read_out_port = port_set.read_out_port
        read_in_port = port_set.read_in_port
        mux = experiment_system.get_mux_by_readout_port(read_out_port)
        if ctrl_port is None or read_out_port is None or read_in_port is None:
            table.add_row(qubit, "-", "-", "-", "-")
            continue
        mux_number = str(mux.index) if mux is not None else ""
        ctrl_box = ctrl_port.box_id
        read_out_box = read_out_port.box_id
        read_in_box = read_in_port.box_id
        ctrl = f"{ctrl_box}-{ctrl_port.number}"
        read_out = f"{read_out_box}-{read_out_port.number}"
        read_in = f"{read_in_box}-{read_in_port.number}"
        table.add_row(mux_number, qubit, ctrl, read_out, read_in)

    console.print(table)


def print_box_info(box_id: str, fetch: bool = False) -> None:
    """Print the information of a box."""
    state_manager.print_box_info(box_id, fetch=fetch)


def print_base_frequencies(qubits: Sequence[str] | str) -> None:
    """Print the base frequencies of the qubits."""
    if isinstance(qubits, str):
        qubits = [qubits]

    table = Table(
        show_header=True,
        header_style="bold",
        title="BASE FREQUENCIES",
    )
    table.add_column("QUBIT", justify="center")
    table.add_column("READ", justify="center")
    table.add_column("CTRL_0", justify="center")
    table.add_column("CTRL_1", justify="center")
    table.add_column("CTRL_2", justify="center")

    for qubit in qubits:
        port_set = state_manager.experiment_system.get_qubit_port_set(qubit)
        if port_set is None:
            continue
        control = port_set.ctrl_port.base_frequencies
        readout = port_set.read_out_port.base_frequencies

        table.add_row(
            qubit,
            f"{readout[0] * 1e-9:.3f} GHz",
            *[f"{f * 1e-9:.3f} GHz" for f in control],
        )
    console.print(table)


def print_frequency_diffs(qubits: Sequence[str] | str) -> None:
    """Print the frequency differences of the target and base frequencies."""
    if isinstance(qubits, str):
        qubits = [qubits]

    targets = [
        target
        for target in state_manager.experiment_system.targets
        if target.qubit in qubits
    ]

    table = Table(
        show_header=True,
        header_style="bold",
        title="BASE FREQUENCY DIFFS",
    )
    table.add_column("TARGET", justify="left")
    table.add_column("FREQ (GHz)", justify="right")
    table.add_column("BASE (GHz)", justify="right")
    table.add_column("DIFF (MHz)", justify="right")

    experiment_system = state_manager.experiment_system
    rows = []
    for target in targets:
        freq = target.frequency
        base_freq = experiment_system.get_base_frequency(target.label)
        diff = freq - base_freq
        rows.append(
            [
                target.label,
                f"{freq:.3f}",
                f"{base_freq:.3f}",
                f"{diff * 1e3:+.3f}",
            ]
        )
    rows.sort(key=lambda x: x[0])
    for row in rows:
        abs_diff = abs(float(row[-1]))
        if abs_diff >= 250:
            style = "bold red"
        elif abs_diff >= 200:
            style = "bold yellow"
        else:
            style = None
        table.add_row(
            *row,
            style=style,
        )
    console.print(table)
