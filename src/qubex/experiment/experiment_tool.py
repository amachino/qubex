from __future__ import annotations

import math
import subprocess
from typing import Collection

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

try:
    import quel_clock_master as qcm
    from qubecalib import QubeCalib
    from quel_ic_config import Quel1Box
except ImportError:
    pass


from ..backend import LatticeGraph, StateManager

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


def dump_port(box_id: str, port_number: int) -> dict:
    """Dump the information of a port."""
    return state_manager.device_controller.dump_port(box_id, port_number)


def reboot_fpga(box_id: str) -> None:
    """Reboot the FPGA."""
    # Run the following commands in the terminal.
    # $ source /tools/Xilinx/Vivado/2020.1/settings64.sh
    # $ quel_reboot_fpga --port 3121 --adapter xxx
    experiment_system = state_manager.experiment_system
    box = experiment_system.get_box(box_id)
    adapter = box.adapter
    reboot_command = f"quel_reboot_fpga --port 3121 --adapter {adapter}"
    subprocess.run(reboot_command.split())


def relinkup_box(box_id: str) -> None:
    """Relink up the box."""
    relinkup_boxes([box_id])


def relinkup_boxes(box_ids: list[str], noise_threshold: int = 500) -> None:
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
    state_manager.device_controller.relinkup_boxes(
        box_ids,
        noise_threshold=noise_threshold,
    )
    state_manager.device_controller.sync_clocks(box_ids)
    print("Operation completed.")


def reset_clockmaster(ipaddr: str = "10.3.0.255") -> bool:
    """Reset the clock master."""
    return qcm.QuBEMasterClient(ipaddr).reset()


def resync_clocks(box_ids: Collection[str]) -> bool:
    """Resync the clocks of the boxes."""
    return state_manager.device_controller.resync_clocks(list(box_ids))


def print_chip_info(
    save_image: bool = False,
) -> None:
    """Print the information of the chip."""
    chip = state_manager.experiment_system.chip
    graph = LatticeGraph(chip.n_qubits)
    graph.plot_graph(
        save_image=save_image,
        image_name="chip_layout",
    )

    resonator_frequency = [
        f"{resonator.frequency:.3f}" if not math.isnan(resonator.frequency) else "N/A"
        for resonator in chip.resonators
    ]
    graph.plot_data(
        title="Resonator frequency (GHz)",
        value=resonator_frequency,
        text=resonator_frequency,
        save_image=save_image,
        image_name="resonator_frequency",
    )

    qubit_frequency = [
        f"{qubit.frequency:.3f}" if not math.isnan(qubit.frequency) else "N/A"
        for qubit in chip.qubits
    ]
    graph.plot_data(
        title="Qubit frequency (GHz)",
        value=qubit_frequency,
        text=qubit_frequency,
        save_image=save_image,
        image_name="qubit_frequency",
    )

    qubit_anharmonicity = [
        f"{qubit.anharmonicity * 1e3:.1f}"
        if not math.isnan(qubit.anharmonicity)
        else "N/A"
        for qubit in chip.qubits
    ]
    graph.plot_data(
        title="Qubit anharmonicity (MHz)",
        value=qubit_anharmonicity,
        text=qubit_anharmonicity,
        save_image=save_image,
        image_name="qubit_anharmonicity",
    )

    props = state_manager.config_loader._props_dict[chip.id]

    external_loss_rate = [
        f"{v * 1e3:.2f}" if v is not None else "N/A"
        for v in props["external_loss_rate"].values()
    ]
    graph.plot_data(
        title="External loss rate (MHz)",
        value=external_loss_rate,
        text=external_loss_rate,
        save_image=save_image,
        image_name="external_loss_rate",
    )

    internal_loss_rate = [
        f"{v * 1e3:.2f}" if v is not None else "N/A"
        for v in props["internal_loss_rate"].values()
    ]
    graph.plot_data(
        title="Internal loss rate (MHz)",
        value=internal_loss_rate,
        text=internal_loss_rate,
        save_image=save_image,
        image_name="internal_loss_rate",
    )

    t1 = [f"{v * 1e-3:.2f}" if v is not None else "N/A" for v in props["t1"].values()]
    graph.plot_data(
        title="T1 (μs)",
        value=t1,
        text=t1,
        save_image=save_image,
        image_name="t1",
    )

    t2_star = [
        f"{v * 1e-3:.2f}" if v is not None else "N/A" for v in props["t2_star"].values()
    ]
    graph.plot_data(
        title="T2* (μs)",
        value=t2_star,
        text=t2_star,
        save_image=save_image,
        image_name="t2_star",
    )

    t2_echo = [
        f"{v * 1e-3:.2f}" if v is not None else "N/A" for v in props["t2_echo"].values()
    ]
    graph.plot_data(
        title="T2 echo (μs)",
        value=t2_echo,
        text=t2_echo,
        save_image=save_image,
        image_name="t2_echo",
    )


def print_wiring_info(qubits: Collection[str] | None = None) -> None:
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


def print_box_info(box_id: str | None = None, fetch: bool = False) -> None:
    """Print the information of a box."""
    if box_id is None:
        box_ids = [box.id for box in state_manager.experiment_system.boxes]
    else:
        box_ids = [box_id]
    for box_id in box_ids:
        state_manager.print_box_info(box_id, fetch=fetch)


def print_target_frequencies(qubits: Collection[str] | str | None = None) -> None:
    """Print the target frequencies of the qubits."""
    if qubits is None:
        qubits = [qubit.label for qubit in state_manager.experiment_system.qubits]
    elif isinstance(qubits, str):
        qubits = [qubits]

    targets = [
        target
        for target in state_manager.experiment_system.targets
        if target.qubit in qubits
    ]

    table = Table(
        show_header=True,
        header_style="bold",
        title="TARGET FREQUENCIES",
    )
    table.add_column("LABEL", justify="left")
    table.add_column("TARGET", justify="right")
    table.add_column("COARSE", justify="right")
    table.add_column("FINE", justify="right")
    # table.add_column("LO", justify="right")
    table.add_column("NCO", justify="right")
    table.add_column("FNCO", justify="right")
    table.add_column("DIFF", justify="right")

    rows = []
    for target in targets:
        qubit = target.qubit
        tfreq = target.frequency
        cfreq = target.coarse_frequency
        ffreq = target.fine_frequency
        # lo = target.channel.lo_freq
        nco = target.channel.nco_freq
        fnco = target.channel.fnco_freq
        diff = tfreq - ffreq
        rows.append(
            (
                qubit,
                [
                    target.label,
                    f"{tfreq * 1e3:.3f}",
                    f"{cfreq * 1e3:.3f}",
                    f"{ffreq * 1e3:.3f}",
                    # f"{lo * 1e-6:.0f}",
                    f"{nco * 1e-6:.3f}",
                    f"{fnco * 1e-6:+.3f}",
                    f"{diff * 1e3:+.3f}",
                ],
            )
        )
    rows.sort(key=lambda x: (x[0]))

    current_qubit = None
    for qubit, row in rows:
        if qubit != current_qubit:
            if current_qubit is not None:
                table.add_section()
            current_qubit = qubit

        abs_diff = abs(float(row[-1]))
        if abs_diff >= 250 or math.isnan(abs_diff):
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


def print_cr_targets(qubits: Collection[str] | str | None = None) -> None:
    """Print the target frequencies of the qubits."""
    if qubits is None:
        qubits = [qubit.label for qubit in state_manager.experiment_system.qubits]
    elif isinstance(qubits, str):
        qubits = [qubits]

    targets = [
        target
        for target in state_manager.experiment_system.cr_targets
        if target.qubit in qubits and not target.label.endswith("-CR")
    ]

    table = Table(
        show_header=True,
        header_style="bold",
        title="CROSS-RESONANCE TARGETS",
    )
    table.add_column("LABEL", justify="left")
    table.add_column("TARGET", justify="right")
    table.add_column("COARSE", justify="right")
    table.add_column("FINE", justify="right")
    # table.add_column("LO", justify="right")
    table.add_column("NCO", justify="right")
    table.add_column("FNCO", justify="right")
    table.add_column("DIFF", justify="right")

    rows = []
    for target in targets:
        qubit = target.qubit
        tfreq = target.frequency
        cfreq = target.coarse_frequency
        ffreq = target.fine_frequency
        # lo = target.channel.lo_freq
        nco = target.channel.nco_freq
        fnco = target.channel.fnco_freq
        diff = tfreq - ffreq
        rows.append(
            (
                qubit,
                [
                    target.label,
                    f"{tfreq * 1e3:.3f}",
                    f"{cfreq * 1e3:.3f}",
                    f"{ffreq * 1e3:.3f}",
                    # f"{lo * 1e-6:.0f}",
                    f"{nco * 1e-6:.3f}",
                    f"{fnco * 1e-6:+.3f}",
                    f"{diff * 1e3:+.3f}",
                ],
            )
        )
    rows.sort(key=lambda x: (x[0]))

    current_qubit = None
    for qubit, row in rows:
        if qubit != current_qubit:
            if current_qubit is not None:
                table.add_section()
            current_qubit = qubit

        abs_diff = abs(float(row[-1]))
        if abs_diff >= 250 or math.isnan(abs_diff):
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
