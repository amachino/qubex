"""Hardware-state models for QuEL-3 runtime inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

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
    port: int | None
    selected_unit_labels: tuple[str, ...]
    units: tuple[Quel3UnitState, ...]
    ports: tuple[Quel3PortState, ...]
    instruments: tuple[Quel3InstrumentState, ...]
    diagnostics: tuple[Quel3PortDiagnostic, ...] = ()
    issues: tuple[Quel3HardwareStateIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)

    def print(
        self,
        *,
        view: Quel3HardwareStateView = "summary",
        console: Console | None = None,
    ) -> None:
        """
        Print one Rich hardware-state view.

        Parameters
        ----------
        view : Quel3HardwareStateView, optional
            Hardware-state view to render.
        console : Console | None, optional
            Rich console receiving the rendered view. A default console is
            created when omitted.
        """
        output_console = Console(highlight=False) if console is None else console
        output_console.print(_format_quel3_hardware_state(self, view=view))


_RIGHT_HEADERS = {"Units", "Ports", "Instruments", "Diagnostics", "Min GHz", "Max GHz"}
_NOWRAP_HEADERS = {"Severity", "Unit", "Role", "Mode"}


def _format_quel3_hardware_state(
    state: Quel3HardwareState,
    *,
    view: Quel3HardwareStateView = "summary",
) -> RenderableType:
    """Return a Rich renderable for one QuEL-3 hardware-state view."""
    if view == "summary":
        return Group(_summary_panel(state), _issues_table(state.issues))
    if view == "units":
        return _units_table(state)
    if view == "ports":
        return _ports_table(state)
    if view == "instruments":
        return _instruments_table(state)
    if view == "diagnostics":
        return _diagnostics_group(state)
    if view == "all":
        return Group(
            _summary_panel(state),
            _units_table(state),
            _ports_table(state),
            _instruments_table(state),
            _diagnostics_group(state),
            _issues_table(state.issues),
        )
    raise ValueError(f"Unsupported QuEL-3 hardware-state view: {view!r}")


def _summary_panel(state: Quel3HardwareState) -> Panel:
    """Return a summary panel for one hardware state."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()
    endpoint = (
        state.endpoint if state.port is None else f"{state.endpoint}:{state.port}"
    )
    grid.add_row("Endpoint", endpoint)
    grid.add_row("Generated", state.generated_at)
    grid.add_row(
        "Selected units",
        ", ".join(state.selected_unit_labels) or "all",
    )
    grid.add_row("Units", str(len(state.units)))
    grid.add_row("Ports", str(len(state.ports)))
    grid.add_row("Instruments", str(len(state.instruments)))
    grid.add_row("Diagnostics", str(len(state.diagnostics)))
    return Panel(
        grid,
        title="QuEL-3 hardware state",
        border_style=_summary_border_style(state),
        box=box.ROUNDED,
    )


def _units_table(state: Quel3HardwareState) -> Table:
    """Return a table of unit states."""
    rows = [[unit.label, unit.status or "unknown"] for unit in state.units]
    return _table("Units", ["Unit", "Status"], rows)


def _ports_table(state: Quel3HardwareState) -> Table:
    """Return a table of port states."""
    rows = [
        [
            _resource_label(port.id, port.unit_label),
            port.unit_label,
            port.role or "",
            ", ".join(_resource_label(dep, port.unit_label) for dep in port.depends_on),
        ]
        for port in sorted(state.ports, key=_port_sort_key)
    ]
    return _table("Ports", ["Port", "Unit", "Role", "Depends on"], rows)


def _instruments_table(state: Quel3HardwareState) -> Table:
    """Return a table of instrument states."""
    instruments = sorted(state.instruments, key=_instrument_sort_key)
    show_unit = len({instrument.unit_label for instrument in instruments}) > 1
    headers = [
        "Port",
        "Alias",
        "Role",
        "Mode",
        "Min GHz",
        "Max GHz",
        "Sampling fs",
    ]
    if show_unit:
        headers.insert(0, "Unit")
    rows = [
        _instrument_row(instrument, show_unit=show_unit) for instrument in instruments
    ]
    return _table("Instruments", headers, rows)


def _diagnostics_group(state: Quel3HardwareState) -> RenderableType:
    """Return diagnostic panels for one hardware state."""
    if not state.diagnostics:
        return Text("(no diagnostics)", style="dim")
    panels: list[RenderableType] = [
        Panel(
            Syntax(
                diagnostic.text.rstrip() or "(empty diagnostic dump)",
                "yaml",
                word_wrap=True,
                background_color="default",
            ),
            title=f"Port {diagnostic.port_id}",
            border_style="cyan",
            box=box.ROUNDED,
        )
        for diagnostic in state.diagnostics
    ]
    return Group(*panels)


def _issues_table(issues: tuple[Quel3HardwareStateIssue, ...]) -> Table:
    """Return a table of hardware-state issues."""
    rows = [
        [
            issue.severity,
            issue.code,
            issue.message,
            issue.detail or "",
            issue.resource_id or "",
        ]
        for issue in issues
    ]
    return _table(
        "Issues",
        ["Severity", "Code", "Message", "Detail", "Resource"],
        rows,
    )


def _table(title: str, headers: list[str], rows: list[list[Any]]) -> Table:
    """Return a styled Rich table."""
    result = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="bright_black",
        row_styles=("", "dim"),
    )
    for header in headers:
        result.add_column(
            header,
            justify="right" if header in _RIGHT_HEADERS else "left",
            no_wrap=header in _NOWRAP_HEADERS,
            overflow="fold",
        )
    if not rows:
        result.add_row(
            Text(f"No {title.lower()}", style="dim"),
            *[""] * (len(headers) - 1),
        )
        return result
    for row in rows:
        result.add_row(
            *(_cell(value, header) for value, header in zip(row, headers, strict=True))
        )
    return result


def _cell(value: Any, header: str) -> Text:
    """Return a styled table cell."""
    text = str(value)
    if not text:
        return Text("-", style="dim")
    if header == "Severity":
        return _severity_text(text)
    if header in _RIGHT_HEADERS:
        return Text(text, style="cyan")
    return Text(text)


def _severity_text(severity: str) -> Text:
    """Return styled severity text."""
    style = {
        "info": "cyan",
        "warning": "bold yellow",
        "error": "bold red",
    }.get(severity, "bold")
    return Text(severity.upper(), style=style)


def _summary_border_style(state: Quel3HardwareState) -> str:
    """Return summary panel border style from issues."""
    severities = {issue.severity for issue in state.issues}
    if "error" in severities:
        return "red"
    if "warning" in severities:
        return "yellow"
    return "green"


def _instrument_row(
    instrument: Quel3InstrumentState,
    *,
    show_unit: bool,
) -> list[Any]:
    """Return a display row for one instrument."""
    row: list[Any] = [
        _resource_label(instrument.port_id, instrument.unit_label),
        instrument.normalized_alias or instrument.alias or "",
        instrument.role or "",
        instrument.mode or "",
        _frequency_ghz(instrument.frequency_range_min_hz),
        _frequency_ghz(instrument.frequency_range_max_hz),
        instrument.sampling_period_fs or "",
    ]
    if show_unit:
        row.insert(0, instrument.unit_label)
    return row


def _frequency_ghz(value: float | None) -> str:
    """Format one frequency in GHz."""
    if value is None:
        return ""
    return f"{value / 1.0e9:.4f}"


def _resource_label(resource_id: str, unit_label: str) -> str:
    """Return a compact resource label when the unit prefix matches."""
    prefix = f"{unit_label}:"
    if resource_id.startswith(prefix):
        return resource_id.removeprefix(prefix)
    return resource_id


def _resource_suffix(resource_id: str) -> str:
    """Return resource ID suffix after the first unit separator."""
    return resource_id.split(":", maxsplit=1)[-1]


def _port_sort_key(port: Quel3PortState) -> tuple[str, str]:
    """Return stable sort key for one port."""
    return port.unit_label, _resource_suffix(port.id)


def _instrument_sort_key(instrument: Quel3InstrumentState) -> tuple[str, str, str, str]:
    """Return stable sort key for one instrument."""
    return (
        instrument.unit_label,
        _resource_suffix(instrument.port_id),
        instrument.normalized_alias or instrument.alias or "",
        instrument.id,
    )
