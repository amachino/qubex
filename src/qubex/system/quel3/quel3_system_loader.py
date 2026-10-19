"""QuEL-3 system assembly helpers used by ConfigLoader."""

from __future__ import annotations

from typing import Any

from qubex.system.control_system import BoxType, ControlSystem, PortType
from qubex.system.quel1.quel1_system_loader import Quel1SystemLoader
from qubex.system.wiring import split_box_port_specifier


class Quel3SystemLoader(Quel1SystemLoader):
    """
    Assemble QuEL-3 control-system and wiring models.

    Notes
    -----
    Ports 3 and 10 retain their default pump role unless assigned to `ctrl`
    in wiring. Assigning either port to both `ctrl` and `pump` is invalid.
    """

    def load_control_system(
        self,
        *,
        chip_id: str,
        box_dict: dict[str, Any],
        wiring_rows: list[dict[str, Any]] | None,
        wiring_file: str,
        clock_master_address: str | None,
    ) -> ControlSystem | None:
        """Build the control system and resolve pump-capable ports from wiring."""
        system = super().load_control_system(
            chip_id=chip_id,
            box_dict=box_dict,
            wiring_rows=wiring_rows,
            wiring_file=wiring_file,
            clock_master_address=clock_master_address,
        )
        if system is None or wiring_rows is None:
            return system

        control_ports = {
            split_box_port_specifier(specifier)
            for row in wiring_rows
            for specifier in row["ctrl"]
        }
        pump_ports = {
            split_box_port_specifier(specifier)
            for row in wiring_rows
            if (specifier := row.get("pump")) is not None
        }
        for box in system.boxes:
            if box.type != BoxType.QUEL3:
                continue
            for port in box.output_ports:
                if port.number not in (3, 10):
                    continue
                key = (box.id, port.number)
                if key not in control_ports:
                    continue
                if key in pump_ports:
                    raise ValueError(
                        f"QuEL-3 port {box.id}:{port.number} cannot be assigned "
                        "to both ctrl and pump in wiring."
                    )
                port.id = f"{box.id}.CTRL{len(box.control_ports)}"
                port.type = PortType.CTRL
                for channel in port.channels:
                    channel.id = f"{port.id}.CH{channel.number}"
        return system
