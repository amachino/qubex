"""QuEL-3 instrument roles and deployable definitions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

InstrumentRoleName = Literal[
    "TRANSMITTER", "TRANSCEIVER", "TRANSCEIVER_LOOPBACK", "RECEIVER"
]


class InstrumentSpec(BaseModel):
    """Define one deployable instrument independently of its runtime resource ID."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )

    port_id: str
    alias: str
    role: InstrumentRoleName
    frequency_range_min_hz: float
    frequency_range_max_hz: float

    @model_validator(mode="after")
    def _validate_deployment(self) -> Self:
        """Require a qualified port, local alias, and ordered frequency range."""
        unit_label, separator, local_port_id = self.port_id.partition(":")
        if (
            not separator
            or not unit_label.strip()
            or not local_port_id.strip()
            or ":" in local_port_id
        ):
            raise ValueError("Instrument port ID must include a unit label and port.")
        if not self.alias or ":" in self.alias:
            raise ValueError("Instrument alias must be nonempty and contain no ':'.")
        if self.frequency_range_min_hz > self.frequency_range_max_hz:
            raise ValueError(
                "Instrument frequency range minimum must not exceed maximum."
            )
        return self
