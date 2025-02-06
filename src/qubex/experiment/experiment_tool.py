from __future__ import annotations

import math
import subprocess
from typing import Collection, Literal

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
    *info_type: Literal[
        "resonator_frequency",
        "qubit_frequency",
        "qubit_anharmonicity",
        "external_loss_rate",
        "internal_loss_rate",
        "t1",
        "t2_star",
        "t2_echo",
        "static_zz_interaction",
        "qubit_qubit_coupling_strength",
    ],
    save_image: bool = False,
) -> None:
    """Print the information of the chip."""
    chip = state_manager.experiment_system.chip
    props = state_manager.config_loader._props_dict[chip.id]

    graph = LatticeGraph(chip.n_qubits)

    if len(info_type) == 0:
        info_type = (
            "resonator_frequency",
            "qubit_frequency",
            "qubit_anharmonicity",
            "external_loss_rate",
            "internal_loss_rate",
            "t1",
            "t2_star",
            "t2_echo",
            "qubit_qubit_coupling_strength",
            "static_zz_interaction",
        )

    if "resonator_frequency" in info_type:
        graph.plot_lattice_data(
            title="Resonator frequency",
            values=[resonator.frequency for resonator in chip.resonators],
            texts=[
                f"{resonator.label}<br>{resonator.frequency:.3f}<br>GHz"
                if not math.isnan(resonator.frequency)
                else "N/A"
                for resonator in chip.resonators
            ],
            hovertexts=[
                f"{resonator.label}: {resonator.frequency * 1e3:.3f} MHz"
                if not math.isnan(resonator.frequency)
                else f"{resonator.label}: N/A"
                for resonator in chip.resonators
            ],
            save_image=save_image,
            image_name="resonator_frequency",
        )

    if "qubit_frequency" in info_type:
        graph.plot_lattice_data(
            title="Qubit frequency",
            values=[qubit.frequency for qubit in chip.qubits],
            texts=[
                f"{qubit.label}<br>{qubit.frequency:.3f}<br>GHz"
                if not math.isnan(qubit.frequency)
                else "N/A"
                for qubit in chip.qubits
            ],
            hovertexts=[
                f"{qubit.label}: {qubit.frequency * 1e3:.3f} MHz"
                if not math.isnan(qubit.frequency)
                else f"{qubit.label}: N/A"
                for qubit in chip.qubits
            ],
            save_image=save_image,
            image_name="qubit_frequency",
        )

    if "qubit_anharmonicity" in info_type:
        graph.plot_lattice_data(
            title="Qubit anharmonicity",
            values=[qubit.anharmonicity * 1e3 for qubit in chip.qubits],
            texts=[
                f"{qubit.label}<br>{qubit.anharmonicity * 1e3:.1f}<br>MHz"
                if not math.isnan(qubit.anharmonicity)
                else "N/A"
                for qubit in chip.qubits
            ],
            hovertexts=[
                f"{qubit.label}: {qubit.anharmonicity * 1e3:.3f} MHz"
                if not math.isnan(qubit.anharmonicity)
                else f"{qubit.label}: N/A"
                for qubit in chip.qubits
            ],
            save_image=save_image,
            image_name="qubit_anharmonicity",
        )

    if "external_loss_rate" in info_type:
        graph.plot_lattice_data(
            title="External loss rate",
            values=props["external_loss_rate"].values(),
            texts=[
                f"{qubit}<br>{value * 1e3:.2f}<br>MHz" if value is not None else "N/A"
                for qubit, value in props["external_loss_rate"].items()
            ],
            hovertexts=[
                f"{qubit}: {value * 1e3:.3f} MHz"
                if value is not None
                else f"{qubit}: N/A"
                for qubit, value in props["external_loss_rate"].items()
            ],
            save_image=save_image,
            image_name="external_loss_rate",
        )

    if "internal_loss_rate" in info_type:
        graph.plot_lattice_data(
            title="Internal loss rate",
            values=props["internal_loss_rate"].values(),
            texts=[
                f"{qubit}<br>{value * 1e3:.2f}<br>MHz" if value is not None else "N/A"
                for qubit, value in props["internal_loss_rate"].items()
            ],
            hovertexts=[
                f"{qubit}: {value * 1e3:.3f} MHz"
                if value is not None
                else f"{qubit}: N/A"
                for qubit, value in props["internal_loss_rate"].items()
            ],
            save_image=save_image,
            image_name="internal_loss_rate",
        )

    if "t1" in info_type:
        graph.plot_lattice_data(
            title="T1",
            values=props["t1"].values(),
            texts=[
                f"{qubit}<br>{value * 1e-3:.2f}<br>μs" if value is not None else "N/A"
                for qubit, value in props["t1"].items()
            ],
            hovertexts=[
                f"{qubit}: {value * 1e-3:.3f} μs"
                if value is not None
                else f"{qubit}: N/A"
                for qubit, value in props["t1"].items()
            ],
            save_image=save_image,
            image_name="t1",
        )

    if "t2_star" in info_type:
        graph.plot_lattice_data(
            title="T2*",
            values=props["t2_star"].values(),
            texts=[
                f"{qubit}<br>{value * 1e-3:.2f}<br>μs" if value is not None else "N/A"
                for qubit, value in props["t2_star"].items()
            ],
            hovertexts=[
                f"{qubit}: {value * 1e-3:.3f} μs"
                if value is not None
                else f"{qubit}: N/A"
                for qubit, value in props["t2_star"].items()
            ],
            save_image=save_image,
            image_name="t2_star",
        )

    if "t2_echo" in info_type:
        graph.plot_lattice_data(
            title="T2 echo",
            values=props["t2_echo"].values(),
            texts=[
                f"{qubit}<br>{value * 1e-3:.2f}<br>μs" if value is not None else "N/A"
                for qubit, value in props["t2_echo"].items()
            ],
            hovertexts=[
                f"{qubit}: {value * 1e-3:.3f} μs"
                if value is not None
                else f"{qubit}: N/A"
                for qubit, value in props["t2_echo"].items()
            ],
            save_image=save_image,
            image_name="t2_echo",
        )

    if "static_zz_interaction" in info_type:
        graph.plot_graph_data(
            title="Static ZZ interaction",
            edge_values={
                key: value for key, value in props["static_zz_interaction"].items()
            },
            edge_hovertexts={
                key: f"{key}: {value * 1e6:.1f} kHz" if value is not None else "N/A"
                for key, value in props["static_zz_interaction"].items()
            },
            save_image=save_image,
            image_name="static_zz_interaction",
        )

    if "qubit_qubit_coupling_strength" in info_type:
        graph.plot_graph_data(
            title="Qubit-Qubit coupling strength",
            edge_values={
                key: value
                for key, value in props["qubit_qubit_coupling_strength"].items()
            },
            edge_hovertexts={
                key: f"{key}: {value * 1e3:.1f} MHz" if value is not None else "N/A"
                for key, value in props["qubit_qubit_coupling_strength"].items()
            },
            save_image=save_image,
            image_name="qubit_qubit_coupling_strength",
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
