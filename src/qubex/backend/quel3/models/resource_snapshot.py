"""Resource snapshot models for QuEL-3 runtime inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

Quel3ResourceSeverity: TypeAlias = Literal["info", "warning", "error"]
Quel3ResourceLevel: TypeAlias = Literal["unit", "port", "instrument", "diagnosis"]


@dataclass(frozen=True)
class Quel3UnitControlState:
    """One supported QuEL-3 unit control."""

    key: str
    allowed_values: tuple[str, ...]
    current_value: str


@dataclass(frozen=True)
class Quel3UnitState:
    """One discovered QuEL-3 unit and its supported controls."""

    label: str
    controls: tuple[Quel3UnitControlState, ...] = ()


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
class Quel3ResourceIssue:
    """One issue found while collecting or evaluating QuEL-3 resources."""

    severity: Quel3ResourceSeverity
    code: str
    message: str
    detail: str | None = None
    resource_id: str | None = None


@dataclass(frozen=True)
class Quel3ResourceSnapshot:
    """
    Report observed QuEL-3 resources and diagnostic issues.

    Each acquisition level includes the preceding levels: unit, port, instrument,
    then diagnosis. Acquisition errors can leave partial results. A snapshot does
    not populate the execution cache or represent a deployable
    `InstrumentConfiguration`.
    """

    generated_at: str
    endpoint: str
    port: int | None
    selected_unit_labels: tuple[str, ...]
    units: tuple[Quel3UnitState, ...]
    ports: tuple[Quel3PortState, ...]
    instruments: tuple[Quel3InstrumentState, ...]
    diagnostics: tuple[Quel3PortDiagnostic, ...] = ()
    issues: tuple[Quel3ResourceIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)
