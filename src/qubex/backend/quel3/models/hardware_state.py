"""Hardware-state models for QuEL-3 runtime inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

Quel3HardwareStateSeverity: TypeAlias = Literal["info", "warning", "error"]
Quel3HardwareStateView: TypeAlias = Literal[
    "summary",
    "units",
    "ports",
    "instruments",
    "diagnostics",
    "all",
]


@dataclass(frozen=True)
class Quel3UnitState:
    """One discovered QuEL-3 unit."""

    label: str
    status: str | None = None


@dataclass(frozen=True)
class Quel3PortState:
    """One QuEL-3 port resource."""

    id: str
    unit_label: str
    role: str | None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Quel3InstrumentState:
    """One QuEL-3 instrument resource."""

    id: str
    unit_label: str
    port_id: str
    alias: str | None
    normalized_alias: str | None
    role: str | None
    mode: str | None
    frequency_range_min_hz: float | None = None
    frequency_range_max_hz: float | None = None
    sampling_period_fs: int | None = None
    bitdepth: int | None = None
    timeline_step_samples: int | None = None
    samples_per_tick: int | None = None


@dataclass(frozen=True)
class Quel3PortDiagnostic:
    """Diagnostic dump for one QuEL-3 port."""

    port_id: str
    unit_label: str
    text: str


@dataclass(frozen=True)
class Quel3HardwareStateIssue:
    """One issue found while collecting or evaluating QuEL-3 hardware state."""

    severity: Quel3HardwareStateSeverity
    code: str
    message: str
    detail: str | None = None
    resource_id: str | None = None


@dataclass(frozen=True)
class Quel3HardwareState:
    """Structured QuEL-3 hardware state snapshot."""

    generated_at: str
    endpoint: str
    port: int
    selected_unit_labels: tuple[str, ...]
    units: tuple[Quel3UnitState, ...]
    ports: tuple[Quel3PortState, ...]
    instruments: tuple[Quel3InstrumentState, ...]
    diagnostics: tuple[Quel3PortDiagnostic, ...] = ()
    issues: tuple[Quel3HardwareStateIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)
