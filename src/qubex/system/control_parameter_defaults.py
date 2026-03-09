"""Backend-specific defaults for resolved control parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from qubex.system.control_parameters import ControlParameters, JPAParameters
from qubex.system.target_type import TargetType

if TYPE_CHECKING:
    from .quantum_system import QuantumSystem


_SUPPORTED_TARGET_TYPES: Final[tuple[TargetType, ...]] = (
    TargetType.READ,
    TargetType.CTRL_GE,
    TargetType.CTRL_EF,
    TargetType.CTRL_CR,
    TargetType.PUMP,
)


@dataclass(frozen=True)
class ControlParameterDefaults:
    """
    Backend-specific defaults used to materialize `ControlParameters`.

    This object is used only while loading configuration. It merges per-file
    parameter mappings with backend defaults and expands the maps used by the
    active backend into complete qubit- and mux-indexed values.
    """

    control_amplitude: float
    readout_amplitude: float
    control_vatt: int | None
    readout_vatt: int | None
    pump_vatt: int | None
    control_fsc: int | None
    readout_fsc: int | None
    pump_fsc: int | None
    capture_delay: int | None
    capture_delay_word: int | None
    pump_frequency: float
    pump_amplitude: float
    dc_voltage: float
    frequency_margin_by_type: dict[str, float] | None = None

    def create_control_parameters(
        self,
        *,
        quantum_system: QuantumSystem,
        frequency_margin: Mapping[str, float],
        control_amplitude: Mapping[str, float],
        readout_amplitude: Mapping[str, float],
        control_vatt: Mapping[str, int | None],
        readout_vatt: Mapping[int, int | None],
        pump_vatt: Mapping[int, int | None],
        control_fsc: Mapping[str, int | None],
        readout_fsc: Mapping[int, int | None],
        pump_fsc: Mapping[int, int | None],
        capture_delay: Mapping[int, int | None],
        capture_delay_word: Mapping[int, int | None],
        jpa_params: Mapping[int, Mapping[str, Any] | None],
        pump_frequency_by_mux: Mapping[int, float] | None = None,
    ) -> ControlParameters:
        """
        Return fully materialized control parameters for the active backend.

        Parameters
        ----------
        quantum_system : QuantumSystem
            Quantum-system topology used to enumerate all qubits and muxes.
        frequency_margin : Mapping[str, float]
            Explicit frequency-margin overrides by target type.
        control_amplitude : Mapping[str, float]
            Explicit control-amplitude overrides by qubit label.
        readout_amplitude : Mapping[str, float]
            Explicit readout-amplitude overrides by qubit label.
        control_vatt : Mapping[str, int | None]
            Explicit control VATT overrides by qubit label.
        readout_vatt : Mapping[int, int | None]
            Explicit readout VATT overrides by mux index.
        pump_vatt : Mapping[int, int | None]
            Explicit pump VATT overrides by mux index.
        control_fsc : Mapping[str, int | None]
            Explicit control FSC overrides by qubit label.
        readout_fsc : Mapping[int, int | None]
            Explicit readout FSC overrides by mux index.
        pump_fsc : Mapping[int, int | None]
            Explicit pump FSC overrides by mux index.
        capture_delay : Mapping[int, int | None]
            Explicit capture-delay overrides by mux index.
        capture_delay_word : Mapping[int, int | None]
            Explicit capture-delay-word overrides by mux index.
        jpa_params : Mapping[int, Mapping[str, Any] | None]
            Explicit JPA overrides by mux index.
        pump_frequency_by_mux : Mapping[int, float] | None, optional
            Default pump frequencies by mux index derived from hardware traits.

        Returns
        -------
        ControlParameters
            Effective control parameters with all expected keys populated for
            the active backend.

        Notes
        -----
        If a backend does not use a setting, the materialized field value can
        still be `None`. That `None` is the resolved effective value kept for
        serialization, while typed getters on `ControlParameters` raise when
        such unsupported settings are accessed.
        """
        return ControlParameters(
            frequency_margin=self._resolve_frequency_margin(frequency_margin),
            control_amplitude=self._resolve_qubit_values(
                quantum_system=quantum_system,
                values=control_amplitude,
                default=self.control_amplitude,
            ),
            readout_amplitude=self._resolve_qubit_values(
                quantum_system=quantum_system,
                values=readout_amplitude,
                default=self.readout_amplitude,
            ),
            control_vatt=self._resolve_qubit_values(
                quantum_system=quantum_system,
                values=control_vatt,
                default=self.control_vatt,
            ),
            readout_vatt=self._resolve_mux_values(
                quantum_system=quantum_system,
                values=readout_vatt,
                default=self.readout_vatt,
            ),
            pump_vatt=self._resolve_mux_values(
                quantum_system=quantum_system,
                values=pump_vatt,
                default=self.pump_vatt,
            ),
            control_fsc=self._resolve_qubit_values(
                quantum_system=quantum_system,
                values=control_fsc,
                default=self.control_fsc,
            ),
            readout_fsc=self._resolve_mux_values(
                quantum_system=quantum_system,
                values=readout_fsc,
                default=self.readout_fsc,
            ),
            pump_fsc=self._resolve_mux_values(
                quantum_system=quantum_system,
                values=pump_fsc,
                default=self.pump_fsc,
            ),
            capture_delay=self._resolve_mux_values(
                quantum_system=quantum_system,
                values=capture_delay,
                default=self.capture_delay,
            ),
            capture_delay_word=self._resolve_mux_values(
                quantum_system=quantum_system,
                values=capture_delay_word,
                default=self.capture_delay_word,
            ),
            jpa_params=self._resolve_jpa_parameters(
                quantum_system=quantum_system,
                values=jpa_params,
                pump_frequency_by_mux=pump_frequency_by_mux,
            ),
        )

    def _resolve_frequency_margin(
        self,
        values: Mapping[str, float],
    ) -> dict[str, float]:
        resolved = dict(values)
        default_by_type = self.frequency_margin_by_type
        if default_by_type is None:
            return resolved
        for target_type in _SUPPORTED_TARGET_TYPES:
            resolved.setdefault(
                target_type.value,
                default_by_type.get(target_type.value, 0.1),
            )
        return resolved

    def _resolve_qubit_values(
        self,
        *,
        quantum_system: QuantumSystem,
        values: Mapping[str, Any],
        default: Any,
    ) -> dict[str, Any]:
        resolved = dict(values)
        for qubit in quantum_system.qubits:
            resolved.setdefault(qubit.label, default)
        return resolved

    def _resolve_mux_values(
        self,
        *,
        quantum_system: QuantumSystem,
        values: Mapping[int, Any],
        default: Any,
    ) -> dict[int, Any]:
        resolved = dict(values)
        for mux in quantum_system.muxes:
            resolved.setdefault(mux.index, default)
        return resolved

    def _resolve_jpa_parameters(
        self,
        *,
        quantum_system: QuantumSystem,
        values: Mapping[int, Mapping[str, Any] | None],
        pump_frequency_by_mux: Mapping[int, float] | None = None,
    ) -> dict[int, JPAParameters]:
        resolved: dict[int, JPAParameters] = {
            int(index): self._resolve_one_jpa_parameters(
                raw_value,
                pump_frequency=(
                    pump_frequency_by_mux.get(int(index))
                    if pump_frequency_by_mux is not None
                    else None
                ),
            )
            for index, raw_value in values.items()
        }
        for mux in quantum_system.muxes:
            resolved.setdefault(
                mux.index,
                self._resolve_one_jpa_parameters(
                    None,
                    pump_frequency=(
                        pump_frequency_by_mux.get(mux.index)
                        if pump_frequency_by_mux is not None
                        else None
                    ),
                ),
            )
        return resolved

    def _resolve_one_jpa_parameters(
        self,
        value: Mapping[str, Any] | None,
        *,
        pump_frequency: float | None = None,
    ) -> JPAParameters:
        default_pump_frequency = (
            pump_frequency if pump_frequency is not None else self.pump_frequency
        )
        if value is None:
            return {
                "pump_frequency": default_pump_frequency,
                "pump_amplitude": self.pump_amplitude,
                "dc_voltage": self.dc_voltage,
            }
        raw_pump_frequency = value.get("pump_frequency")
        raw_pump_amplitude = value.get("pump_amplitude")
        raw_dc_voltage = value.get("dc_voltage")
        return {
            "pump_frequency": (
                float(raw_pump_frequency)
                if raw_pump_frequency is not None
                else default_pump_frequency
            ),
            "pump_amplitude": (
                float(raw_pump_amplitude)
                if raw_pump_amplitude is not None
                else self.pump_amplitude
            ),
            "dc_voltage": (
                float(raw_dc_voltage) if raw_dc_voltage is not None else self.dc_voltage
            ),
        }
