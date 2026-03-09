"""Resolved control-parameter models and access helpers."""

from __future__ import annotations

from typing import TypeVar

from typing_extensions import TypedDict

from qubex.core import MutableModel

from .target_type import TargetType

_ValueT = TypeVar("_ValueT")


class JPAParameters(TypedDict):
    """Resolved JPA parameters for one mux."""

    dc_voltage: float
    pump_frequency: float
    pump_amplitude: float


class ControlParameters(MutableModel):
    """
    Fully materialized control parameters for the active backend.

    Maps used by the active backend are expanded during config loading, so this
    model can be serialized as-is without applying runtime fallback logic.

    Notes
    -----
    Serialized fields such as VATT, FSC, and capture delay may still contain
    `None`. In that case, `None` is the effective value for the active backend
    and means that the corresponding hardware knob is not used. Typed getter
    methods raise `ValueError` when such unsupported settings are accessed.
    """

    frequency_margin: dict[str, float]
    control_amplitude: dict[str, float]
    readout_amplitude: dict[str, float]
    control_vatt: dict[str, int | None]
    readout_vatt: dict[int, int | None]
    pump_vatt: dict[int, int | None]
    control_fsc: dict[str, int | None]
    readout_fsc: dict[int, int | None]
    pump_fsc: dict[int, int | None]
    capture_delay: dict[int, int | None]
    capture_delay_word: dict[int, int | None]
    jpa_params: dict[int, JPAParameters]

    def get_frequency_margin(self, target_type: TargetType | str) -> float:
        """Return the materialized frequency margin for a target type."""
        key = target_type.value if isinstance(target_type, TargetType) else target_type
        return self.frequency_margin.get(key, 0.1)

    def get_control_amplitude(self, qubit: str) -> float:
        """Return the materialized control amplitude for a qubit."""
        return self.control_amplitude[qubit]

    def get_ef_control_amplitude(self, qubit: str) -> float:
        """Return the derived ef control amplitude for a qubit."""
        return self.get_control_amplitude(qubit) / (2**0.5)

    def get_readout_amplitude(self, qubit: str) -> float:
        """Return the materialized readout amplitude for a qubit."""
        return self.readout_amplitude[qubit]

    def get_control_vatt(self, qubit: str) -> int:
        """Return the effective control VATT for a qubit."""
        return self._require_qubit_value(
            values=self.control_vatt,
            qubit=qubit,
            field_name="Control VATT",
        )

    def get_readout_vatt(self, mux: int) -> int:
        """Return the effective readout VATT for a mux."""
        return self._require_mux_value(
            values=self.readout_vatt,
            mux=mux,
            field_name="Readout VATT",
        )

    def get_pump_vatt(self, mux: int) -> int:
        """Return the effective pump VATT for a mux."""
        return self._require_mux_value(
            values=self.pump_vatt,
            mux=mux,
            field_name="Pump VATT",
        )

    def get_control_fsc(self, qubit: str) -> int:
        """Return the effective control FSC for a qubit."""
        return self._require_qubit_value(
            values=self.control_fsc,
            qubit=qubit,
            field_name="Control FSC",
        )

    def get_readout_fsc(self, mux: int) -> int:
        """Return the effective readout FSC for a mux."""
        return self._require_mux_value(
            values=self.readout_fsc,
            mux=mux,
            field_name="Readout FSC",
        )

    def get_pump_fsc(self, mux: int) -> int:
        """Return the effective pump FSC for a mux."""
        return self._require_mux_value(
            values=self.pump_fsc,
            mux=mux,
            field_name="Pump FSC",
        )

    def get_capture_delay(self, mux: int) -> int:
        """Return the effective capture delay for a mux."""
        return self._require_mux_value(
            values=self.capture_delay,
            mux=mux,
            field_name="Capture delay",
        )

    def get_capture_delay_word(self, mux: int) -> int:
        """Return the effective capture-delay word for a mux."""
        return self._require_mux_value(
            values=self.capture_delay_word,
            mux=mux,
            field_name="Capture delay word",
        )

    def get_pump_frequency(self, mux: int) -> float:
        """Return the materialized pump frequency for a mux."""
        return self.jpa_params[mux]["pump_frequency"]

    def get_pump_amplitude(self, mux: int) -> float:
        """Return the materialized pump amplitude for a mux."""
        return self.jpa_params[mux]["pump_amplitude"]

    def get_dc_voltage(self, mux: int) -> float:
        """Return the materialized DC voltage for a mux."""
        return self.jpa_params[mux]["dc_voltage"]

    def _require_qubit_value(
        self,
        *,
        values: dict[str, _ValueT | None],
        qubit: str,
        field_name: str,
    ) -> _ValueT:
        value = values[qubit]
        if value is None:
            raise ValueError(
                f"{field_name} is not supported for qubit `{qubit}` on the active backend."
            )
        return value

    def _require_mux_value(
        self,
        *,
        values: dict[int, _ValueT | None],
        mux: int,
        field_name: str,
    ) -> _ValueT:
        value = values[mux]
        if value is None:
            raise ValueError(
                f"{field_name} is not supported for mux {mux} on the active backend."
            )
        return value
