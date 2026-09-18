"""Target deploy planning for QuEL-3 push-time configuration."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from qubex.backend.quel3.models import (
    InstrumentConfiguration,
    InstrumentRoleName,
    InstrumentSpec,
)
from qubex.system.target_type import TargetType

if TYPE_CHECKING:
    from qubex.system.control_system import GenPort
    from qubex.system.experiment_system import ExperimentSystem
    from qubex.system.target import Target

FIXED_TIMELINE_SAMPLING_RATE_HZ = 2.5e9
logger = logging.getLogger(__name__)


class Quel3TargetDeployPlanner:
    """Build QuEL-3 instrument configuration from logical target metadata."""

    def build_configuration(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
        target_labels: Sequence[str] | None = None,
    ) -> InstrumentConfiguration:
        """Build a deterministic configuration with one instrument per target."""
        selected_box_ids = set(box_ids)
        selected_target_labels = (
            set(target_labels) if target_labels is not None else None
        )

        instruments: list[InstrumentSpec] = []
        for _label, target in sorted(experiment_system.gen_targets.items()):
            port = target.channel.port
            if port.box_id not in selected_box_ids:
                continue
            if (
                selected_target_labels is not None
                and target.label not in selected_target_labels
            ):
                continue
            if not self._has_finite_target_frequency(target=target):
                logger.warning(
                    "Skipping QuEL-3 deploy target with non-finite frequency: "
                    "label=%s frequency=%s",
                    target.label,
                    target.frequency,
                )
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
            instruments.append(
                InstrumentSpec(
                    port_id=port_id,
                    role=role,
                    frequency_range_min_hz=freq_min,
                    frequency_range_max_hz=freq_max,
                    alias=target.label,
                )
            )
        return InstrumentConfiguration(instruments=tuple(instruments))

    def _resolve_port_id(
        self,
        *,
        experiment_system: ExperimentSystem,
        target: Target,
    ) -> str:
        """Resolve quelware port ID from one logical generator target."""
        port = target.channel.port
        unit_label = port.box_id
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
    def _resolve_instrument_role(target_type: TargetType) -> InstrumentRoleName:
        """Resolve instrument role name from logical target type."""
        if target_type == TargetType.READ:
            return "TRANSCEIVER"
        if target_type in (
            TargetType.CTRL_GE,
            TargetType.CTRL_EF,
            TargetType.CTRL_FH,
            TargetType.CTRL_2Q,
            TargetType.CTRL_CR,
            TargetType.PUMP,
        ):
            return "TRANSMITTER"
        raise ValueError(f"Unsupported target type for deployment: {target_type}.")

    @staticmethod
    def _has_finite_target_frequency(*, target: Target) -> bool:
        """Return whether the target frequency can be deployed to hardware."""
        return math.isfinite(float(target.frequency) * 1e9)

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
