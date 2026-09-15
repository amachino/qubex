"""Preview models for `configure()`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from rich.console import Console
from rich.table import Table

from qubex.backend.backend_controller import BackendKind
from qubex.typing import ConfigurationMode

if TYPE_CHECKING:
    from qubex.system.experiment_system import ExperimentSystem


@dataclass(frozen=True)
class ConfigureStateChange:
    """One field comparison between current and post-`configure()` device state."""

    box_id: str
    component: str
    field: str
    before: object
    after: object
    unit: str | None = None
    is_frequency: bool = False

    @property
    def has_change(self) -> bool:
        """Return whether the compared field would change."""
        return self.before != self.after


@dataclass(frozen=True)
class ConfigurePreview:
    """Structured preview of post-`configure()` device state changes."""

    backend_kind: BackendKind
    box_ids: tuple[str, ...]
    mode: ConfigurationMode | None
    entries: tuple[ConfigureStateChange, ...] = ()
    missing_box_ids: tuple[str, ...] = ()

    @property
    def changes(self) -> tuple[ConfigureStateChange, ...]:
        """Return field-level comparisons that would change."""
        return tuple(entry for entry in self.entries if entry.has_change)

    @property
    def has_changes(self) -> bool:
        """Return whether `configure()` would change any tracked fields."""
        return len(self.changes) > 0

    @property
    def has_frequency_changes(self) -> bool:
        """Return whether `configure()` would change frequency-related fields."""
        return any(change.is_frequency for change in self.changes)

    @property
    def is_complete(self) -> bool:
        """Return whether all requested boxes were fetched for comparison."""
        return len(self.missing_box_ids) == 0

    def print_summary(self, console: Console | None = None) -> None:
        """Print field-level changes that `configure()` would apply."""
        if console is None:
            console = Console()

        table = Table(
            show_header=True,
            header_style="bold",
            title="Configure Preview Changes",
        )
        table.add_column("BOX", justify="left")
        table.add_column("COMPONENT", justify="left")
        table.add_column("FIELD", justify="left")
        table.add_column("BEFORE", justify="right")
        table.add_column("AFTER", justify="right")
        table.add_column("UNIT", justify="left")
        table.add_column("FREQ", justify="center")

        self._add_rows(table, self.changes, include_change=False)
        console.print(table)

    def print_full(self, console: Console | None = None) -> None:
        """Print all previewed field-level comparisons."""
        if console is None:
            console = Console()

        table = Table(
            show_header=True,
            header_style="bold",
            title="Configure Preview Full",
        )
        table.add_column("BOX", justify="left")
        table.add_column("COMPONENT", justify="left")
        table.add_column("FIELD", justify="left")
        table.add_column("BEFORE", justify="right")
        table.add_column("AFTER", justify="right")
        table.add_column("UNIT", justify="left")
        table.add_column("FREQ", justify="center")
        table.add_column("CHANGE", justify="center")

        self._add_rows(table, self.entries, include_change=True)
        console.print(table)

    def _add_rows(
        self,
        table: Table,
        entries: Sequence[ConfigureStateChange],
        *,
        include_change: bool,
    ) -> None:
        """Add preview rows to `table`."""
        for entry in entries:
            row = [
                entry.box_id,
                entry.component,
                entry.field,
                _format_value(entry.before),
                _format_value(entry.after),
                entry.unit or "",
                "yes" if entry.is_frequency else "",
            ]
            if include_change:
                row.append("yes" if entry.has_change else "no")
            table.add_row(*row)
        for box_id in self.missing_box_ids:
            row = [box_id, "box", "fetch", "failed", "", "", ""]
            if include_change:
                row.append("")
            table.add_row(*row)
        if not entries and not self.missing_box_ids:
            row = ["-", "-", "-", "no changes", "", "", ""]
            if include_change:
                row.append("")
            table.add_row(*row)


class ConfigurePreviewSynchronizer(Protocol):
    """Synchronizer capability for backends that support configure previews."""

    def preview_configure(
        self,
        *,
        experiment_system: ExperimentSystem,
        box_ids: Sequence[str],
        mode: ConfigurationMode | None,
        parallel: bool | None = None,
        target_labels: Sequence[str] | None = None,
    ) -> ConfigurePreview:
        """Preview backend-specific hardware changes for `configure()`."""
        ...


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:_}"
    return str(value)
