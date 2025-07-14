from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Collection, Literal

import yaml
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

try:
    import quel_clock_master as qcm
    from qubecalib import QubeCalib
    from qubecalib.instrument.quel.quel1.tool.skew import Skew, SkewSetting
    from quel_ic_config import Quel1Box
except ImportError:
    pass


from ..backend import LatticeGraph, SystemManager
from ..diagnostics import ChipInspector

console = Console()
system_manager = SystemManager.shared()


def check_skew(
    box_ids: Collection[str],
    estimate: bool = True,
    config_dir: str = "/home/shared/config/",
    skew_file: str = "skew.yaml",
    box_file: str = "box.yaml",
) -> dict:
    """Check the skew of the boxes."""
    box_ids = list(box_ids)

    box_file_path = Path(config_dir) / box_file
    skew_file_path = Path(config_dir) / skew_file

    with open(skew_file_path, "r") as file:
        config = yaml.safe_load(file)
    ref_port = config["reference_port"].split("-")[0]

    confirmed = Confirm.ask(
        f"""
You are going to check the skew of the following boxes using [bold bright_green]'{ref_port}'[/bold bright_green] as the reference.

[bold bright_green]{box_ids}[/bold bright_green]

Do you want to continue?
"""
    )
    if not confirmed:
        print("Operation cancelled.")
        return {}

    all_box_ids = list(set(list(box_ids) + [ref_port]))
    qc = get_qubecalib()
    qc.sysdb.load_box_yaml(str(box_file_path))
    setting = SkewSetting.from_yaml(str(skew_file_path))
    system = qc.sysdb.create_quel1system(*all_box_ids)
    system.resync(*all_box_ids)
    skew = Skew.create(setting=setting, system=system, sysdb=qc.sysdb)
    skew.measure()
    if estimate:
        skew.estimate()
    fig = skew.plot()
    fig.update_layout(
        title=f"Skew : {str(', '.join(box_ids))} (Ref. {ref_port})",
        width=800,
    )
    fig.show()
    return {
        "skew": skew,
        "fig": fig,
    }


def get_qubecalib() -> QubeCalib:
    """Get the QubeCalib instance."""
    return system_manager.device_controller.qubecalib


def get_quel1_box(box_id: str) -> Quel1Box:
    """Get the Quel1Box instance."""
    qc = system_manager.device_controller.qubecalib
    box = qc.create_box(box_id, reconnect=False)
    box.reconnect()
    return box


def dump_box(box_id: str) -> dict:
    """Dump the information of a box."""
    return system_manager.device_controller.dump_box(box_id)


def dump_port(box_id: str, port_number: int) -> dict:
    """Dump the information of a port."""
    return system_manager.device_controller.dump_port(box_id, port_number)


def reboot_fpga(box_id: str) -> None:
    """Reboot the FPGA."""
    # Run the following commands in the terminal.
    # $ source /tools/Xilinx/Vivado/2020.1/settings64.sh
    # $ quel_reboot_fpga --port 3121 --adapter xxx
    experiment_system = system_manager.experiment_system
    box = experiment_system.get_box(box_id)
    adapter = box.adapter
    reboot_command = f"quel_reboot_fpga --port 3121 --adapter {adapter}"
    subprocess.run(reboot_command.split())


def relinkup_box(box_id: str, noise_threshold: int | None = None) -> None:
    """Relink up the box."""
    relinkup_boxes([box_id], noise_threshold=noise_threshold)


def relinkup_boxes(box_ids: list[str], noise_threshold: int | None) -> None:
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
    system_manager.device_controller.relinkup_boxes(
        box_ids,
        noise_threshold=noise_threshold,
    )
    system_manager.device_controller.sync_clocks(box_ids)
    print("Operation completed.")


def reset_clockmaster(ipaddr: str = "10.3.0.255") -> bool:
    """Reset the clock master."""
    return qcm.QuBEMasterClient(ipaddr).reset()


def resync_clocks(box_ids: Collection[str]) -> bool:
    """Resync the clocks of the boxes."""
    return system_manager.device_controller.resync_clocks(list(box_ids))


def print_chip_info(
    *info_type: Literal[
        "all",
        "chip_summary",
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
        "average_readout_fidelity",
        "average_gate_fidelity",
        "x90_gate_fidelity",
        "x180_gate_fidelity",
        "zx90_gate_fidelity",
    ],
    directed: bool = False,
    save_image: bool = False,
) -> None:
    """Print the information of the chip."""
    chip = system_manager.experiment_system.chip

    props: dict[str, dict[str, float]] = {
        key: {
            qubit: value if value is not None else math.nan
            for qubit, value in values.items()
        }
        for key, values in system_manager.config_loader._props_dict[chip.id].items()
    }

    graph = LatticeGraph(chip.n_qubits)

    draw_individual_results = False

    if len(info_type) == 0:
        info_type = (
            "chip_summary",
            "qubit_frequency",
            "qubit_anharmonicity",
            "t1",
            "t2_echo",
            "average_readout_fidelity",
            "x90_gate_fidelity",
            "zx90_gate_fidelity",
        )
    elif "all" in info_type:
        draw_individual_results = True
        info_type = (
            "chip_summary",
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
            "average_readout_fidelity",
            "average_gate_fidelity",
            "x90_gate_fidelity",
            "x180_gate_fidelity",
            "zx90_gate_fidelity",
        )

    if "chip_summary" in info_type:
        inspector = ChipInspector(
            chip_id=chip.id,
            config_dir=system_manager.config_loader._config_dir,
            props_dir=system_manager.config_loader._params_dir,
        )
        if chip.n_qubits == 144:
            inspection_params = {
                "max_frequency": 5.8,
                "min_frequency": 2,
            }
        else:
            inspection_params = None
        summary = inspector.execute(
            params=inspection_params,
        )
        summary.draw(
            draw_individual_results=draw_individual_results,
            save_image=save_image,
        )

    if "resonator_frequency" in info_type:
        if values := props.get("resonator_frequency"):
            graph.plot_lattice_data(
                title="Resonator frequency (GHz)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value:.3f}<br>GHz" if not math.isnan(value) else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value * 1e3:.3f} MHz"
                    if not math.isnan(value)
                    else f"{qubit}: N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="resonator_frequency",
            )

    if "qubit_frequency" in info_type:
        if values := props.get("qubit_frequency"):
            graph.plot_lattice_data(
                title="Qubit frequency (GHz)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value:.3f}<br>GHz" if not math.isnan(value) else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value * 1e3:.3f} MHz"
                    if not math.isnan(value)
                    else f"{qubit}: N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="qubit_frequency",
            )

    if "qubit_anharmonicity" in info_type:
        if values := props.get("anharmonicity"):
            graph.plot_lattice_data(
                title="Qubit anharmonicity (MHz)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value * 1e3:.1f}<br>MHz"
                    if not math.isnan(value)
                    else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value * 1e3:.3f} MHz"
                    if not math.isnan(value)
                    else f"{qubit}: N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="qubit_anharmonicity",
            )

    if "external_loss_rate" in info_type:
        if values := props.get("external_loss_rate"):
            graph.plot_lattice_data(
                title="External loss rate (MHz)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value * 1e3:.2f}<br>MHz"
                    if not math.isnan(value)
                    else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value * 1e3:.3f} MHz"
                    if not math.isnan(value)
                    else f"{qubit}: N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="external_loss_rate",
            )

    if "internal_loss_rate" in info_type:
        if values := props.get("internal_loss_rate"):
            graph.plot_lattice_data(
                title="Internal loss rate (MHz)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value * 1e3:.2f}<br>MHz"
                    if not math.isnan(value)
                    else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value * 1e3:.3f} MHz"
                    if not math.isnan(value)
                    else f"{qubit}: N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="internal_loss_rate",
            )

    if "t1" in info_type:
        if values := props.get("t1"):
            graph.plot_lattice_data(
                title="T1 (μs)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value * 1e-3:.2f}<br>μs"
                    if not math.isnan(value)
                    else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value * 1e-3:.3f} μs"
                    if not math.isnan(value)
                    else f"{qubit}: N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="t1",
            )

    if "t2_star" in info_type:
        if values := props.get("t2_star"):
            graph.plot_lattice_data(
                title="T2* (μs)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value * 1e-3:.2f}<br>μs"
                    if not math.isnan(value)
                    else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value * 1e-3:.3f} μs"
                    if not math.isnan(value)
                    else f"{qubit}: N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="t2_star",
            )

    if "t2_echo" in info_type:
        if values := props.get("t2_echo"):
            graph.plot_lattice_data(
                title="T2 echo (μs)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value * 1e-3:.2f}<br>μs"
                    if not math.isnan(value)
                    else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value * 1e-3:.3f} μs"
                    if not math.isnan(value)
                    else f"{qubit}: N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="t2_echo",
            )

    def create_undirected_data(
        data: dict[str, float],
        method: Literal["avg", "max", "min"],
    ) -> dict[str, float]:
        result = {}
        for key, value in data.items():
            if value is None or math.isnan(value):
                continue
            pair = key.split("-")
            inv_key = f"{pair[1]}-{pair[0]}"
            if (
                inv_key in result
                and result[inv_key] is not None
                and not math.isnan(result[inv_key])
            ):
                if method == "avg":
                    result[inv_key] = (result[inv_key] + value) / 2
                elif method == "max":
                    result[inv_key] = max(result[inv_key], value)
                elif method == "min":
                    result[inv_key] = min(result[inv_key], value)
                else:
                    raise ValueError(f"Unknown method: {method}")
            else:
                result[key] = float(value)
        return result

    if "static_zz_interaction" in info_type:
        if static_zz_interaction := props.get("static_zz_interaction"):
            values = (
                static_zz_interaction
                if directed
                else create_undirected_data(
                    data=static_zz_interaction,
                    method="avg",
                )
            )
            graph.plot_graph_data(
                directed=directed,
                title="Static ZZ interaction (kHz)",
                edge_values={key: value for key, value in values.items()},
                edge_texts={
                    key: f"{value * 1e6:.0f}" if not math.isnan(value) else None
                    for key, value in values.items()
                },
                edge_hovertexts={
                    key: f"{key}: {value * 1e6:.1f} kHz"
                    if not math.isnan(value)
                    else "N/A"
                    for key, value in values.items()
                },
                save_image=save_image,
                image_name="static_zz_interaction",
            )

    if "qubit_qubit_coupling_strength" in info_type:
        if qubit_qubit_coupling_strength := props.get("qubit_qubit_coupling_strength"):
            values = (
                qubit_qubit_coupling_strength
                if directed
                else create_undirected_data(
                    data=qubit_qubit_coupling_strength,
                    method="avg",
                )
            )
            graph.plot_graph_data(
                directed=directed,
                title="Qubit-qubit coupling strength (MHz)",
                edge_values={key: value for key, value in values.items()},
                edge_texts={
                    key: f"{value * 1e3:.1f}" if not math.isnan(value) else None
                    for key, value in values.items()
                },
                edge_hovertexts={
                    key: f"{key}: {value * 1e3:.1f} MHz"
                    if not math.isnan(value)
                    else "N/A"
                    for key, value in values.items()
                },
                save_image=save_image,
                image_name="qubit_qubit_coupling_strength",
            )

    if "average_readout_fidelity" in info_type:
        if values := props.get("average_readout_fidelity"):
            graph.plot_lattice_data(
                title="Average readout fidelity (%)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value:.2%}" if not math.isnan(value) else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value:.2%}"
                    if not math.isnan(value)
                    else f"{qubit}: N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="average_readout_fidelity",
            )

    if "average_gate_fidelity" in info_type:
        if values := props.get("average_gate_fidelity"):
            graph.plot_lattice_data(
                title="Average gate fidelity (%)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value:.2%}" if not math.isnan(value) else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value:.2%}"
                    if not math.isnan(value)
                    else f"{qubit}: N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="average_gate_fidelity",
            )

    if "x90_gate_fidelity" in info_type:
        if values := props.get("x90_gate_fidelity"):
            graph.plot_lattice_data(
                title="X90 gate fidelity (%)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value:.2%}" if not math.isnan(value) else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value:.2%}"
                    if not math.isnan(value)
                    else f"{qubit}: N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="x90_gate_fidelity",
            )

    if "x180_gate_fidelity" in info_type:
        if values := props.get("x180_gate_fidelity"):
            graph.plot_lattice_data(
                title="X180 gate fidelity (%)",
                values=list(values.values()),
                texts=[
                    f"{qubit}<br>{value:.2%}" if not math.isnan(value) else "N/A"
                    for qubit, value in values.items()
                ],
                hovertexts=[
                    f"{qubit}: {value:.2%}" if not math.isnan(value) else "N/A"
                    for qubit, value in values.items()
                ],
                save_image=save_image,
                image_name="x180_gate_fidelity",
            )

    if "zx90_gate_fidelity" in info_type:
        if values := props.get("zx90_gate_fidelity"):
            graph.plot_graph_data(
                directed=True,
                title="ZX90 gate fidelity (%)",
                edge_values={key: value for key, value in values.items()},
                edge_texts={
                    key: f"{value:.2%}" if not math.isnan(value) else None
                    for key, value in values.items()
                },
                edge_hovertexts={
                    key: f"{key}: {value:.2%}" if not math.isnan(value) else "N/A"
                    for key, value in values.items()
                },
                save_image=save_image,
                image_name="zx90_gate_fidelity",
            )

            values = create_undirected_data(
                data=values,
                method="max",
            )
            graph.plot_graph_data(
                directed=False,
                title="ZX90 gate fidelity (%)",
                edge_values={key: value for key, value in values.items()},
                edge_texts={
                    key: f"{value * 1e2:.1f}" if not math.isnan(value) else None
                    for key, value in values.items()
                },
                edge_hovertexts={
                    key: f"{key}: {value:.2%}" if not math.isnan(value) else "N/A"
                    for key, value in values.items()
                },
                save_image=save_image,
                image_name="zx90_gate_fidelity",
            )


def print_wiring_info(qubits: Collection[str] | None = None) -> None:
    """Print the wiring information of the chip."""

    experiment_system = system_manager.experiment_system

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

    ctrl_ports = {}
    read_out_ports = {}
    read_in_ports = {}

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

        ctrl_ports[qubit] = ctrl
        read_out_ports[qubit] = read_out
        read_in_ports[qubit] = read_in

    console.print(table)

    graph = LatticeGraph(experiment_system.chip.n_qubits)
    ctrl_port_labels = [
        ctrl_ports.get(qubit.label, "-") for qubit in experiment_system.qubits
    ]
    read_out_port_labels = [
        read_out_ports.get(qubit.label, "-") for qubit in experiment_system.qubits
    ]
    read_in_port_labels = [
        read_in_ports.get(qubit.label, "-") for qubit in experiment_system.qubits
    ]
    texts = [
        f"[{qubit.label}]<br>{ctrl_port}<br>{read_out_port}<br>{read_in_port}"
        for qubit, ctrl_port, read_out_port, read_in_port in zip(
            experiment_system.qubits,
            ctrl_port_labels,
            read_out_port_labels,
            read_in_port_labels,
        )
    ]
    graph.plot_lattice_data(
        title="Wiring Info",
        values=[0] * experiment_system.chip.n_qubits,
        colorscale="RdBu",
        texts=texts,
    )


def print_box_info(box_id: str, fetch: bool = True) -> None:
    """Print the information of a box."""
    system_manager.print_box_info(box_id, fetch=fetch)


def print_target_frequencies(qubits: Collection[str] | str | None = None) -> None:
    """Print the target frequencies of the qubits."""
    experiment_system = system_manager.experiment_system

    if qubits is None:
        qubits = [qubit.label for qubit in experiment_system.qubits]
    elif isinstance(qubits, str):
        qubits = [qubits]

    targets = [
        target
        for target in experiment_system.targets
        if target.is_related_to_qubits(qubits)
    ]

    table = Table(
        show_header=True,
        header_style="bold",
        title="TARGET FREQUENCIES",
    )
    table.add_column("LABEL", justify="left")
    table.add_column("LO", justify="right")
    table.add_column("NCO", justify="right")
    table.add_column("CNCO", justify="right")
    table.add_column("FNCO", justify="right")
    table.add_column("F_FINE", justify="right")
    table.add_column("F_TARGET", justify="right")
    table.add_column("F_DIFF", justify="right")

    rows = []
    for target in targets:
        qubit = target.qubit
        tfreq = target.frequency
        ffreq = target.fine_frequency
        diff = tfreq - ffreq
        lo = target.channel.lo_freq
        nco = target.channel.nco_freq
        cnco = target.channel.cnco_freq
        fnco = target.channel.fnco_freq
        rows.append(
            (
                qubit,
                [
                    target.label,
                    f"{lo * 1e-6:.0f}",
                    f"{nco * 1e-6:.3f}",
                    f"{cnco * 1e-6:.3f}",
                    f"{fnco * 1e-6:+.3f}",
                    f"{ffreq * 1e3:.3f}",
                    f"{tfreq * 1e3:.3f}",
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
        qubits = [qubit.label for qubit in system_manager.experiment_system.qubits]
    elif isinstance(qubits, str):
        qubits = [qubits]

    targets = [
        target
        for target in system_manager.experiment_system.cr_targets
        if target.qubit in qubits and not target.label.endswith("-CR")
    ]

    table = Table(
        show_header=True,
        header_style="bold",
        title="CROSS-RESONANCE TARGETS",
    )
    table.add_column("LABEL", justify="left")
    table.add_column("LO", justify="right")
    table.add_column("NCO", justify="right")
    table.add_column("CNCO", justify="right")
    table.add_column("FNCO", justify="right")
    table.add_column("F_FINE", justify="right")
    table.add_column("F_TARGET", justify="right")
    table.add_column("F_DIFF", justify="right")

    rows = []
    for target in targets:
        qubit = target.qubit
        tfreq = target.frequency
        ffreq = target.fine_frequency
        lo = target.channel.lo_freq
        nco = target.channel.nco_freq
        cnco = target.channel.cnco_freq
        fnco = target.channel.fnco_freq
        diff = tfreq - ffreq
        rows.append(
            (
                qubit,
                [
                    target.label,
                    f"{lo * 1e-6:.0f}",
                    f"{nco * 1e-6:.3f}",
                    f"{cnco * 1e-6:.3f}",
                    f"{fnco * 1e-6:+.3f}",
                    f"{ffreq * 1e3:.3f}",
                    f"{tfreq * 1e3:.3f}",
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
