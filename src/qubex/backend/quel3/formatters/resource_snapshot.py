"""Rich formatting for collected QuEL-3 resources."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from qubex.backend.quel3.models import (
    Quel3InstrumentState,
    Quel3PortState,
    Quel3ResourceIssue,
    Quel3ResourceSnapshot,
)

Quel3ResourceView: TypeAlias = Literal[
    "summary",
    "units",
    "ports",
    "instruments",
    "diagnostics",
    "all",
]


def print_resource_snapshot(
    snapshot: Quel3ResourceSnapshot,
    *,
    view: Quel3ResourceView = "summary",
    console: Console | None = None,
) -> None:
    """
    Print an existing resource snapshot without reading quelware resources.

    Parameters
    ----------
    snapshot : Quel3ResourceSnapshot
        Collected resources and diagnostic issues.
    view : Quel3ResourceView, optional
        Section or summary to render. Defaults to `summary`.
    console : Console | None, optional
        Rich console receiving the rendered view. A default console is created
        when omitted.

    Raises
    ------
    ValueError
        If the view is unsupported.
    """
    output_console = Console(highlight=False) if console is None else console
    output_console.print(format_resource_snapshot(snapshot, view=view))


_RIGHT_HEADERS = {"Units", "Ports", "Instruments", "Diagnostics", "Min GHz", "Max GHz"}
_NOWRAP_HEADERS = {"Severity", "Unit", "Role", "Mode"}


def format_resource_snapshot(
    snapshot: Quel3ResourceSnapshot,
    *,
    view: Quel3ResourceView = "summary",
) -> RenderableType:
    """
    Format an existing resource snapshot without reading quelware resources.

    Parameters
    ----------
    snapshot : Quel3ResourceSnapshot
        Collected resources and diagnostic issues.
    view : Quel3ResourceView, optional
        Section or summary to render. Defaults to `summary`.

    Returns
    -------
    RenderableType
        Rich renderable suitable for a console or a larger layout.

    Raises
    ------
    ValueError
        If the view is unsupported.
    """
    if view == "summary":
        return Group(_summary_panel(snapshot), _issues_table(snapshot.issues))
    if view == "units":
        return _units_table(snapshot)
    if view == "ports":
        return _ports_table(snapshot)
    if view == "instruments":
        return _instruments_table(snapshot)
    if view == "diagnostics":
        return _diagnostics_group(snapshot)
    if view == "all":
        return Group(
            _summary_panel(snapshot),
            _units_table(snapshot),
            _ports_table(snapshot),
            _instruments_table(snapshot),
            _diagnostics_group(snapshot),
            _issues_table(snapshot.issues),
        )
    raise ValueError(f"Unsupported QuEL-3 resource snapshot view: {view!r}")


def _summary_panel(snapshot: Quel3ResourceSnapshot) -> Panel:
    """Return a summary panel for one resource snapshot."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()
    endpoint = (
        snapshot.endpoint
        if snapshot.port is None
        else f"{snapshot.endpoint}:{snapshot.port}"
    )
    grid.add_row("Endpoint", endpoint)
    grid.add_row("Generated", snapshot.generated_at)
    grid.add_row(
        "Selected units",
        ", ".join(snapshot.selected_unit_labels) or "all",
    )
    grid.add_row("Units", str(len(snapshot.units)))
    grid.add_row("Ports", str(len(snapshot.ports)))
    grid.add_row("Instruments", str(len(snapshot.instruments)))
    grid.add_row("Diagnostics", str(len(snapshot.diagnostics)))
    return Panel(
        grid,
        title="QuEL-3 resource snapshot",
        border_style=_summary_border_style(snapshot),
        box=box.ROUNDED,
    )


def _units_table(snapshot: Quel3ResourceSnapshot) -> Table:
    """Return a table of unit states."""
    rows = [
        [
            unit.label,
            control.key,
            control.current_value,
            ", ".join(control.allowed_values),
        ]
        for unit in snapshot.units
        for control in unit.controls
    ]
    rows.extend(
        [unit.label, "", "", ""] for unit in snapshot.units if not unit.controls
    )
    return _table("Units", ["Unit", "Control", "Current", "Allowed"], rows)


def _ports_table(snapshot: Quel3ResourceSnapshot) -> Table:
    """Return a table of port states."""
    rows = [
        [
            _resource_label(port.id, port.unit_label),
            port.unit_label,
            port.role or "",
            ", ".join(_resource_label(dep, port.unit_label) for dep in port.depends_on),
        ]
        for port in sorted(snapshot.ports, key=_port_sort_key)
    ]
    return _table("Ports", ["Port", "Unit", "Role", "Depends on"], rows)


def _instruments_table(snapshot: Quel3ResourceSnapshot) -> Table:
    """Return a table of instrument states."""
    instruments = sorted(snapshot.instruments, key=_instrument_sort_key)
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


def _diagnostics_group(snapshot: Quel3ResourceSnapshot) -> RenderableType:
    """Return raw diagnostic YAML for one resource snapshot."""
    if not snapshot.diagnostics:
        return Text("(no diagnostics)", style="dim")
    diagnostics: list[RenderableType] = [
        Text(
            diagnostic.text.rstrip() or "(empty diagnostic dump)",
        )
        for diagnostic in snapshot.diagnostics
    ]
    return Group(*diagnostics)


def _issues_table(issues: tuple[Quel3ResourceIssue, ...]) -> Table:
    """Return a table of resource snapshot issues."""
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


def _summary_border_style(snapshot: Quel3ResourceSnapshot) -> str:
    """Return summary panel border style from issues."""
    severities = {issue.severity for issue in snapshot.issues}
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
