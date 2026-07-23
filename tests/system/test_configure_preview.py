"""Tests for configure preview models."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from qubex.backend.backend_controller import BACKEND_KIND_QUEL1
from qubex.system import ConfigurePreview, ConfigureStateChange


def test_configure_preview_prints_summary_and_full_tables() -> None:
    """Given preview entries, summary should show changes while full shows all entries."""
    preview = ConfigurePreview(
        backend_kind=BACKEND_KIND_QUEL1,
        box_ids=("A",),
        mode="ge-cr-cr",
        entries=(
            ConfigureStateChange(
                box_id="A",
                component="port 1",
                field="lo_freq",
                before=10_000_000_000,
                after=11_000_000_000,
                unit="Hz",
                is_frequency=True,
            ),
            ConfigureStateChange(
                box_id="A",
                component="port 1",
                field="rfswitch",
                before="pass",
                after="pass",
            ),
        ),
    )
    summary_io = StringIO()
    full_io = StringIO()

    preview.print_summary(Console(file=summary_io, force_terminal=False, width=120))
    preview.print_full(Console(file=full_io, force_terminal=False, width=120))

    summary = summary_io.getvalue()
    full = full_io.getvalue()
    assert "Configure Preview Changes" in summary
    assert "A" in summary
    assert "lo_freq" in summary
    assert "rfswitch" not in summary
    assert "Configure Preview Full" in full
    assert "CHANGE" in full
    assert "lo_freq" in full
    assert "rfswitch" in full
    assert "yes" in full
    assert "no" in full
