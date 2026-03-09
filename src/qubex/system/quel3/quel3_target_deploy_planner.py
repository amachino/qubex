"""Target deploy planning for QuEL-3 push-time configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from qubex.backend.quel3.models import InstrumentDeployRequest, RoleName
from qubex.system.target import TargetType

if TYPE_CHECKING:
    from qubex.system.control_system import GenPort
    from qubex.system.experiment_system import ExperimentSystem
    from qubex.system.target import Target

FIXED_TIMELINE_SAMPLING_RATE_HZ = 2.5e9


class Quel3TargetDeployPlanner:
    """Build QuEL-3 deploy requests from logical target registry metadata."""

    def build_deploy_requests(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
        target_labels: Sequence[str] | None = None,
    ) -> tuple[InstrumentDeployRequest, ...]:
        """Build deterministic one-target-per-instrument deploy requests."""
        selected_box_ids = set(box_ids)
        selected_target_labels = (
            set(target_labels) if target_labels is not None else None
        )

        requests: list[InstrumentDeployRequest] = []
        for _label, target in sorted(experiment_system.gen_targets.items()):
            port = target.channel.port
            if port.box_id not in selected_box_ids:
                continue
            if (
                selected_target_labels is not None
                and target.label not in selected_target_labels
            ):
                continue
            role = self._resolve_instrument_role(target.type)
            port_id = self._resolve_port_id(
                experiment_system=experiment_system,
                target=target,
            )
            frequency_hz = self._resolve_target_frequency_hz(target=target)
            frequency_margin_hz = self._resolve_target_frequency_margin_hz(
                experiment_system=experiment_system,
                target=target,
            )
            freq_min = frequency_hz - frequency_margin_hz
            freq_max = frequency_hz + frequency_margin_hz
            alias = self._build_alias(
                port_id=port_id,
                role=role,
                target_label=target.label,
            )
            requests.append(
                InstrumentDeployRequest(
                    port_id=port_id,
                    role=role,
                    frequency_range_min_hz=freq_min,
                    frequency_range_max_hz=freq_max,
                    alias=alias,
                    target_labels=(target.label,),
                )
            )
        return tuple(requests)

    def _resolve_port_id(
        self,
        *,
        experiment_system: ExperimentSystem,
        target: Target,
    ) -> str:
        """Resolve quelware port ID from one logical generator target."""
        port = target.channel.port
        unit_label = self._resolve_unit_label(
            experiment_system=experiment_system,
            box_id=port.box_id,
        )
        port_number = self._resolve_port_number(port=port)

        if target.type == TargetType.READ:
            read_out_port_number = port_number
            read_in_port_number = self._resolve_read_in_port_number(
                experiment_system=experiment_system,
                read_out_port=port,
            )
            return f"{unit_label}:trx_p{read_in_port_number:02d}p{read_out_port_number:02d}"
        return f"{unit_label}:tx_p{port_number:02d}"

    @staticmethod
    def _resolve_unit_label(
        *,
        experiment_system: ExperimentSystem,
        box_id: str,
    ) -> str:
        """Resolve quelware unit label from one control box."""
        return experiment_system.get_box(box_id).name

    @staticmethod
    def _resolve_port_number(*, port: GenPort) -> int:
        """Resolve validated integer port number from one generator port."""
        if not isinstance(port.number, int):
            raise TypeError(f"Port number must be int for QuEL-3 deployment: {port}")
        return port.number

    def _resolve_read_in_port_number(
        self,
        *,
        experiment_system: ExperimentSystem,
        read_out_port: GenPort,
    ) -> int:
        """Resolve paired read-in port number for one read-out generator port."""
        mux = experiment_system.get_mux_by_readout_port(read_out_port)
        if mux is None:
            raise ValueError(f"Readout mux is not found for port `{read_out_port.id}`.")

        for read_in_mux, cap_port in experiment_system.wiring_info.read_in:
            if read_in_mux.index != mux.index:
                continue
            if not isinstance(cap_port.number, int):
                raise TypeError(
                    "Capture port number must be int for QuEL-3 readout deployment."
                )
            return cap_port.number
        raise ValueError(f"Read-in pair is not found for readout mux `{mux.index}`.")

    @staticmethod
    def _resolve_instrument_role(target_type: TargetType) -> RoleName:
        """Resolve instrument role name from logical target type."""
        if target_type == TargetType.READ:
            return "TRANSCEIVER"
        if target_type in (
            TargetType.CTRL_GE,
            TargetType.CTRL_EF,
            TargetType.CTRL_CR,
            TargetType.PUMP,
        ):
            return "TRANSMITTER"
        raise ValueError(f"Unsupported target type for deployment: {target_type}.")

    @staticmethod
    def _resolve_target_frequency_hz(*, target: Target) -> float:
        """Resolve validated target frequency in Hz from GHz value."""
        frequency_hz = float(target.frequency) * 1e9
        if not math.isfinite(frequency_hz):
            raise ValueError(
                f"Target frequency must be finite: label={target.label} frequency={target.frequency}"
            )
        return frequency_hz

    @staticmethod
    def _resolve_target_frequency_margin_hz(
        *,
        experiment_system: ExperimentSystem,
        target: Target,
    ) -> float:
        """Resolve validated QuEL-3 deploy margin in Hz for one target."""
        frequency_margin = float(
            experiment_system.control_params.get_frequency_margin(target.type)
        )
        if not math.isfinite(frequency_margin):
            raise ValueError(
                f"frequency_margin must be finite: label={target.label} value={frequency_margin}"
            )
        if frequency_margin < 0:
            raise ValueError(
                f"frequency_margin must be non-negative: label={target.label} value={frequency_margin}"
            )
        nyquist_hz = FIXED_TIMELINE_SAMPLING_RATE_HZ / 2
        frequency_margin_hz = frequency_margin * 1e9
        if frequency_margin_hz >= nyquist_hz:
            raise ValueError(
                "frequency_margin must be smaller than Nyquist: "
                f"label={target.label} value={frequency_margin} nyquist_hz={nyquist_hz}"
            )
        return frequency_margin_hz

    @staticmethod
    def _build_alias(*, port_id: str, role: RoleName, target_label: str) -> str:
        """Build deterministic instrument alias from port, role, and target label."""
        normalized_port_id = port_id.replace(":", "_")
        normalized_target_label = re.sub(r"[^0-9A-Za-z]+", "_", target_label).lower()
        return f"inst_{role.lower()}_{normalized_port_id}_{normalized_target_label}"
