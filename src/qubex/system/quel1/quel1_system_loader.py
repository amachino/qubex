"""QuEL-1 system assembly helpers used by ConfigLoader."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from qubex.backend.backend_controller import (
    BACKEND_KIND_QUEL1,
    normalize_backend_kind,
)
from qubex.system.wiring import split_box_port_specifier

if TYPE_CHECKING:
    from qubex.system.control_system import ControlSystem
    from qubex.system.experiment_system import WiringInfo
    from qubex.system.quantum_system import QuantumSystem

logger = logging.getLogger(__name__)


class Quel1SystemLoader:
    """Assemble QuEL-1 control-system and wiring models from normalized config rows."""

    def resolve_clock_master_address(
        self,
        *,
        chip_id: str,
        chip_dict: dict[str, Any],
        system_dict: dict[str, Any],
    ) -> str | None:
        """Resolve clock-master address from system/chip configuration."""
        chip_info = chip_dict.get(chip_id)
        chip_clock_master = None
        if isinstance(chip_info, dict):
            raw_clock_master = chip_info.get("clock_master")
            if raw_clock_master is not None:
                chip_clock_master = str(raw_clock_master)

        backend = system_dict.get("backend")
        if normalize_backend_kind(backend) == BACKEND_KIND_QUEL1:
            quel1_section = system_dict.get(BACKEND_KIND_QUEL1)
            if isinstance(quel1_section, dict):
                raw_clock_master = quel1_section.get("clock_master")
                if raw_clock_master is not None:
                    return str(raw_clock_master)

        return chip_clock_master

    def load_control_system(
        self,
        *,
        chip_id: str,
        box_dict: dict[str, Any],
        wiring_rows: list[dict[str, Any]] | None,
        wiring_file: str,
        clock_master_address: str | None,
    ) -> ControlSystem | None:
        """Build ControlSystem from legacy-shaped wiring rows."""
        from qubex.system.control_system import Box, ControlSystem

        box_ports = defaultdict(list)
        if wiring_rows is None:
            logger.warning(f"Chip `{chip_id}` is missing in `{wiring_file}`. ")
            return None
        for wiring in wiring_rows:
            box, port = split_box_port_specifier(wiring["read_out"])
            box_ports[box].append(port)
            box, port = split_box_port_specifier(wiring["read_in"])
            box_ports[box].append(port)
            for ctrl in wiring["ctrl"]:
                box, port = split_box_port_specifier(ctrl)
                box_ports[box].append(port)
            if (pump_specifier := wiring.get("pump")) is not None:
                box, port = split_box_port_specifier(pump_specifier)
                box_ports[box].append(port)
        boxes = [
            Box.new(
                id=id,
                name=box["name"],
                type=box["type"],
                address=box["address"],
                adapter=box["adapter"],
                port_numbers=box_ports[id],
                options=box.get("options"),
            )
            for id, box in box_dict.items()
            if id in box_ports
        ]
        return ControlSystem(
            boxes=boxes,
            clock_master_address=clock_master_address,
        )

    def load_wiring_info(
        self,
        *,
        wiring_rows: list[dict[str, Any]] | None,
        quantum_system: QuantumSystem | None,
        control_system: ControlSystem | None,
    ) -> WiringInfo | None:
        """Build wiring information from normalized rows and control-system ports."""
        from qubex.system.experiment_system import WiringInfo

        if wiring_rows is None:
            return None
        if quantum_system is None or control_system is None:
            return None

        def get_gen_port(specifier: str | None):
            if specifier is None:
                return None
            box_id, port_num = split_box_port_specifier(specifier)
            port = control_system.get_gen_port(box_id, port_num)
            return port

        def get_cap_port(specifier: str | None):
            if specifier is None:
                return None
            box_id, port_num = split_box_port_specifier(specifier)
            port = control_system.get_cap_port(box_id, port_num)
            return port

        ctrl = []
        read_out = []
        read_in = []
        pump = []
        for wiring in wiring_rows:
            mux_num = int(wiring["mux"])
            mux = quantum_system.get_mux(mux_num)
            qubits = quantum_system.get_qubits_in_mux(mux_num)
            for identifier, qubit in zip(wiring["ctrl"], qubits, strict=True):
                ctrl_port = get_gen_port(identifier)
                if ctrl_port is not None:
                    ctrl.append((qubit, ctrl_port))
            read_out_port = get_gen_port(wiring["read_out"])
            if read_out_port is not None:
                read_out.append((mux, read_out_port))
            read_in_port = get_cap_port(wiring["read_in"])
            if read_in_port is not None:
                read_in.append((mux, read_in_port))
            pump_port = get_gen_port(wiring.get("pump"))
            if pump_port is not None:
                pump.append((mux, pump_port))

        wiring_info = WiringInfo(
            ctrl=ctrl,
            read_out=read_out,
            read_in=read_in,
            pump=pump,
        )
        return wiring_info
