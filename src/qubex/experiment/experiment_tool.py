from __future__ import annotations

import contextlib
import logging
import math
import subprocess
from collections import defaultdict
from collections.abc import Collection
from pathlib import Path
from typing import Literal

import yaml

with contextlib.suppress(ImportError):
    import quel_clock_master as qcm
    from qubecalib import QubeCalib
    from qubecalib.instrument.quel.quel1.tool.skew import Skew
    from quel_ic_config import Quel1Box

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from qubex.backend import LatticeGraph, SystemManager
from qubex.diagnostics import ChipInspector

logger = logging.getLogger(__name__)


console = Console()
system_manager = SystemManager.shared()


def check_skew(
    box_ids: Collection[str],
    estimate: bool | None = None,
    config_dir: str | None = None,
    skew_file: str | None = None,
    box_file: str | None = None,
) -> dict:
    """Check the skew of the boxes."""
    if estimate is None:
        estimate = True
    if skew_file is None:
        skew_file = "skew.yaml"
    if box_file is None:
        box_file = "box.yaml"
    clock_master_address = (
        system_manager.experiment_system.control_system.clock_master_address
    )

    if config_dir is not None:
        config_path = Path(config_dir)
    else:
        config_path = system_manager.config_loader.config_path

    box_ids = list(box_ids)

    box_file_path = config_path / box_file
    skew_file_path = config_path / skew_file

    with open(skew_file_path) as file:
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
        logger.info("Operation cancelled.")
        return {}

    all_box_ids = list({*box_ids, ref_port})
    skew = Skew.from_yaml(
        str(skew_file_path),
        box_yaml=str(box_file_path),
        clockmaster_ip=clock_master_address,
        boxes=all_box_ids,
    )
    skew.system.resync()
    skew.measure()
    if estimate:
        skew.estimate()
    fig = skew.plot()
    fig.update_layout(
        title=f"Skew : {', '.join(box_ids)!s} (Ref. {ref_port})",
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
    # TODO: use appropriate noise threshold
    box.reconnect(background_noise_threshold=10000)
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
        logger.info("Operation cancelled.")
        return

    logger.info("Relinking up the boxes...")
    system_manager.device_controller.relinkup_boxes(
        box_ids,
        noise_threshold=noise_threshold,
    )
    system_manager.device_controller.sync_clocks(box_ids)
    logger.info("Operation completed.")


def reset_clockmaster(
    ipaddr: str | None = None,
) -> bool:
    """Reset the clock master."""
    if ipaddr is None:
        ipaddr = system_manager.experiment_system.control_system.clock_master_address

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
    directed: bool | None = None,
    save_image: bool | None = None,
) -> None:
    """Print the information of the chip."""
    if directed is None:
        directed = False
    if save_image is None:
        save_image = False
    chip = system_manager.experiment_system.chip
    loader = system_manager.config_loader
    graph = LatticeGraph(chip.n_qubits)

    def _is_valid(value: float | None) -> bool:
        return value is not None and not math.isnan(value)

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
            "t1",
            "t2_star",
            "t2_echo",
            "static_zz_interaction",
            "qubit_qubit_coupling_strength",
            "average_readout_fidelity",
            "x90_gate_fidelity",
            "x180_gate_fidelity",
            "zx90_gate_fidelity",
        )

    try:
        if "chip_summary" in info_type:
            inspector = ChipInspector(
                chip_id=chip.id,
                config_dir=system_manager.config_loader.config_path,
                props_dir=system_manager.config_loader.params_path,
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
            if values := loader.load_param_data("resonator_frequency"):
                graph.plot_lattice_data(
                    title="Resonator frequency (GHz)",
                    values=list(values.values()),
                    texts=[
                        f"{qubit}<br>{value:.3f}<br>GHz" if _is_valid(value) else "N/A"
                        for qubit, value in values.items()
                    ],
                    hovertexts=[
                        f"{qubit}: {value * 1e3:.3f} MHz"
                        if _is_valid(value)
                        else f"{qubit}: N/A"
                        for qubit, value in values.items()
                    ],
                    save_image=save_image,
                    image_name="resonator_frequency",
                )

        if "qubit_frequency" in info_type:
            if values := loader.load_param_data("qubit_frequency"):
                graph.plot_lattice_data(
                    title="Qubit frequency (GHz)",
                    values=list(values.values()),
                    texts=[
                        f"{qubit}<br>{value:.3f}<br>GHz" if _is_valid(value) else "N/A"
                        for qubit, value in values.items()
                    ],
                    hovertexts=[
                        f"{qubit}: {value * 1e3:.3f} MHz"
                        if _is_valid(value)
                        else f"{qubit}: N/A"
                        for qubit, value in values.items()
                    ],
                    save_image=save_image,
                    image_name="qubit_frequency",
                )

        if "qubit_anharmonicity" in info_type:
            if values := loader.load_param_data("qubit_anharmonicity"):
                graph.plot_lattice_data(
                    title="Qubit anharmonicity (MHz)",
                    values=list(values.values()),
                    texts=[
                        f"{qubit}<br>{value * 1e3:.1f}<br>MHz"
                        if _is_valid(value)
                        else "N/A"
                        for qubit, value in values.items()
                    ],
                    hovertexts=[
                        f"{qubit}: {value * 1e3:.3f} MHz"
                        if _is_valid(value)
                        else f"{qubit}: N/A"
                        for qubit, value in values.items()
                    ],
                    save_image=save_image,
                    image_name="qubit_anharmonicity",
                )

        if "t1" in info_type:
            if values := loader.load_param_data("t1"):
                graph.plot_lattice_data(
                    title="T1 (μs)",
                    values=list(values.values()),
                    texts=[
                        f"{qubit}<br>{value * 1e-3:.2f}<br>μs"
                        if _is_valid(value)
                        else "N/A"
                        for qubit, value in values.items()
                    ],
                    hovertexts=[
                        f"{qubit}: {value * 1e-3:.3f} μs"
                        if _is_valid(value)
                        else f"{qubit}: N/A"
                        for qubit, value in values.items()
                    ],
                    save_image=save_image,
                    image_name="t1",
                )

        if "t2_star" in info_type:
            if values := loader.load_param_data("t2_star"):
                graph.plot_lattice_data(
                    title="T2* (μs)",
                    values=list(values.values()),
                    texts=[
                        f"{qubit}<br>{value * 1e-3:.2f}<br>μs"
                        if _is_valid(value)
                        else "N/A"
                        for qubit, value in values.items()
                    ],
                    hovertexts=[
                        f"{qubit}: {value * 1e-3:.3f} μs"
                        if _is_valid(value)
                        else f"{qubit}: N/A"
                        for qubit, value in values.items()
                    ],
                    save_image=save_image,
                    image_name="t2_star",
                )

        if "t2_echo" in info_type:
            if values := loader.load_param_data("t2_echo"):
                graph.plot_lattice_data(
                    title="T2 echo (μs)",
                    values=list(values.values()),
                    texts=[
                        f"{qubit}<br>{value * 1e-3:.2f}<br>μs"
                        if _is_valid(value)
                        else "N/A"
                        for qubit, value in values.items()
                    ],
                    hovertexts=[
                        f"{qubit}: {value * 1e-3:.3f} μs"
                        if _is_valid(value)
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
            if static_zz_interaction := loader.load_param_data("static_zz_interaction"):
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
                    edge_values=dict(values.items()),
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
            if qubit_qubit_coupling_strength := loader.load_param_data(
                "qubit_qubit_coupling_strength"
            ):
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
                    edge_values=dict(values.items()),
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
            if values := loader.load_param_data("average_readout_fidelity"):
                graph.plot_lattice_data(
                    title="Average readout fidelity (%)",
                    values=list(values.values()),
                    texts=[
                        f"{qubit}<br>{value:.2%}" if _is_valid(value) else "N/A"
                        for qubit, value in values.items()
                    ],
                    hovertexts=[
                        f"{qubit}: {value:.2%}" if _is_valid(value) else f"{qubit}: N/A"
                        for qubit, value in values.items()
                    ],
                    save_image=save_image,
                    image_name="average_readout_fidelity",
                )

        if "x90_gate_fidelity" in info_type:
            if values := loader.load_param_data("x90_gate_fidelity"):
                graph.plot_lattice_data(
                    title="X90 gate fidelity (%)",
                    values=list(values.values()),
                    texts=[
                        f"{qubit}<br>{value:.2%}" if _is_valid(value) else "N/A"
                        for qubit, value in values.items()
                    ],
                    hovertexts=[
                        f"{qubit}: {value:.2%}" if _is_valid(value) else f"{qubit}: N/A"
                        for qubit, value in values.items()
                    ],
                    save_image=save_image,
                    image_name="x90_gate_fidelity",
                )

        if "x180_gate_fidelity" in info_type:
            if values := loader.load_param_data("x180_gate_fidelity"):
                graph.plot_lattice_data(
                    title="X180 gate fidelity (%)",
                    values=list(values.values()),
                    texts=[
                        f"{qubit}<br>{value:.2%}" if _is_valid(value) else "N/A"
                        for qubit, value in values.items()
                    ],
                    hovertexts=[
                        f"{qubit}: {value:.2%}" if _is_valid(value) else "N/A"
                        for qubit, value in values.items()
                    ],
                    save_image=save_image,
                    image_name="x180_gate_fidelity",
                )

        if "zx90_gate_fidelity" in info_type:
            if values := loader.load_param_data("zx90_gate_fidelity"):
                graph.plot_graph_data(
                    directed=True,
                    title="ZX90 gate fidelity (%)",
                    edge_values=dict(values.items()),
                    edge_texts={
                        key: f"{value:.2%}" if _is_valid(value) else None
                        for key, value in values.items()
                    },
                    edge_hovertexts={
                        key: f"{key}: {value:.2%}" if _is_valid(value) else "N/A"
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
                    edge_values=dict(values.items()),
                    edge_texts={
                        key: f"{value * 1e2:.1f}" if _is_valid(value) else None
                        for key, value in values.items()
                    },
                    edge_hovertexts={
                        key: f"{key}: {value:.2%}" if _is_valid(value) else "N/A"
                        for key, value in values.items()
                    },
                    save_image=save_image,
                    image_name="zx90_gate_fidelity",
                )

    except Exception as e:
        logger.error("Error occurred while printing chip info", exc_info=e)


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
            strict=True,
        )
    ]
    graph.plot_lattice_data(
        title="Wiring Info",
        values=[0] * experiment_system.chip.n_qubits,
        colorscale="RdBu",
        texts=texts,
    )


def print_box_info(box_id: str, fetch: bool | None = None) -> None:
    """Print the information of a box."""
    if fetch is None:
        fetch = True
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
    table.add_column("F_AWG", justify="right")

    rows = []
    for target in targets:
        qubit = target.qubit
        tfreq = target.frequency
        ffreq = target.fine_frequency
        diff = tfreq - ffreq
        awg = target.awg_frequency

        if target.channel.port.lo_freq is None:
            lo = None
        else:
            lo = target.channel.lo_freq
        nco = target.channel.nco_freq
        cnco = target.channel.cnco_freq
        fnco = target.channel.fnco_freq
        rows.append(
            (
                qubit,
                [
                    target.label,
                    f"{lo * 1e-6:.0f}" if lo is not None else "",
                    f"{nco * 1e-6:.3f}",
                    f"{cnco * 1e-6:.3f}",
                    f"{fnco * 1e-6:+.3f}",
                    f"{ffreq * 1e3:.3f}",
                    f"{tfreq * 1e3:.3f}",
                    f"{diff * 1e3:+.3f}",
                    f"{awg * 1e3:+.3f}",
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


def _configure_loopback(mux: str | int, *, enable: bool) -> None:
    """
    Configure RF switches for all qubits in the given MUX.

    Mapping when enable is True:
        read_in  -> loop
        read_out -> block
        ctrl     -> block

    Mapping when enable is False:
        read_in  -> open
        read_out -> pass
        ctrl     -> pass
    """
    qubits = system_manager.experiment_system.quantum_system.get_qubits_in_mux(mux)

    boxes: dict[str, Quel1Box] = {}
    box_confs: dict[str, dict] = defaultdict(dict)

    # Switch configuration pattern
    if enable:
        read_in_conf, read_out_conf, ctrl_conf = "loop", "block", "block"
    else:
        read_in_conf, read_out_conf, ctrl_conf = "open", "pass", "pass"

    for qubit in qubits:
        port_set = system_manager.experiment_system.get_qubit_port_set(qubit.label)
        if port_set is None:
            raise ValueError("Qubit port set not found")

        read_in_port = port_set.read_in_port
        ctrl_port = port_set.ctrl_port
        read_out_port = port_set.read_out_port

        # Fetch / cache boxes
        if read_in_port.box_id not in boxes:
            boxes[read_in_port.box_id] = get_quel1_box(read_in_port.box_id)
        if read_out_port.box_id not in boxes:
            boxes[read_out_port.box_id] = get_quel1_box(read_out_port.box_id)
        if ctrl_port.box_id not in boxes:
            boxes[ctrl_port.box_id] = get_quel1_box(ctrl_port.box_id)

        # Store configuration entries
        box_confs[read_in_port.box_id][read_in_port.number] = read_in_conf
        box_confs[read_out_port.box_id][read_out_port.number] = read_out_conf
        box_confs[ctrl_port.box_id][ctrl_port.number] = ctrl_conf

    action = "enabled" if enable else "disabled"
    try:
        for box_id, confs in box_confs.items():
            boxes[box_id].config_rfswitches(confs)
        logger.info(f"Loopback {action} for MUX#{mux} {[q.label for q in qubits]}.")
        logger.info(dict(box_confs))
    except Exception as e:
        logger.error(f"Error {action} loopback: {e}")


def enable_loopback(*, mux: str | int):
    """Enable loopback for the specified MUX (backward-compatible API)."""
    _configure_loopback(mux, enable=True)


def disable_loopback(*, mux: str | int):
    """Disable loopback for the specified MUX (backward-compatible API)."""
    _configure_loopback(mux, enable=False)
