"""Tests for QuEL-3 hardware state Rich formatting."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from qubex.backend.quel3.formatting import format_quel3_hardware_state
from qubex.backend.quel3.models import (
    Quel3HardwareState,
    Quel3HardwareStateIssue,
    Quel3InstrumentState,
    Quel3PortDiagnostic,
    Quel3PortState,
    Quel3UnitState,
)


def _render_text(renderable: object) -> str:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    console.print(renderable)
    return output.getvalue()


def test_summary_view_returns_rich_renderable() -> None:
    """Given hardware state, summary view should render key counts."""
    state = Quel3HardwareState(
        generated_at="2026-07-07T00:00:00+00:00",
        endpoint="localhost",
        port=50051,
        selected_unit_labels=("unit-a",),
        units=(Quel3UnitState(label="unit-a"),),
        ports=(Quel3PortState(id="unit-a:tx_p01", unit_label="unit-a", role="TX"),),
        instruments=(
            Quel3InstrumentState(
                id="unit-a:inst-q00",
                unit_label="unit-a",
                port_id="unit-a:tx_p01",
                alias="unit-a:Q00",
                normalized_alias="Q00",
                role="TRANSMITTER",
                mode="FIXED_TIMELINE",
                frequency_range_min_hz=4.1e9,
                frequency_range_max_hz=4.3e9,
            ),
        ),
        diagnostics=(),
        issues=(
            Quel3HardwareStateIssue(
                severity="warning",
                code="UNKNOWN_PORT_DEPENDENCY",
                message="Port references an unknown dependency.",
                resource_id="unit-a:tx_p01",
            ),
        ),
    )

    text = _render_text(format_quel3_hardware_state(state, view="summary"))

    assert "QuEL-3 hardware state" in text
    assert "Units" in text
    assert "Instruments" in text
    assert "UNKNOWN_PORT_DEPENDENCY" in text


def test_summary_view_omits_absent_endpoint_port() -> None:
    """An absent endpoint port should not render as a literal None suffix."""
    state = Quel3HardwareState(
        generated_at="2026-07-07T00:00:00+00:00",
        endpoint="api.example.com",
        port=None,
        selected_unit_labels=(),
        units=(),
        ports=(),
        instruments=(),
    )

    text = _render_text(format_quel3_hardware_state(state, view="summary"))

    assert "api.example.com" in text
    assert "api.example.com:None" not in text


def test_diagnostics_view_renders_port_dumps() -> None:
    """Given diagnostic data, diagnostics view should render port dump text."""
    state = Quel3HardwareState(
        generated_at="2026-07-07T00:00:00+00:00",
        endpoint="localhost",
        port=50051,
        selected_unit_labels=(),
        units=(),
        ports=(),
        instruments=(),
        diagnostics=(
            Quel3PortDiagnostic(
                port_id="unit-a:tx_p01",
                unit_label="unit-a",
                text="line: value",
            ),
        ),
        issues=(),
    )

    text = _render_text(format_quel3_hardware_state(state, view="diagnostics"))

    assert "unit-a:tx_p01" in text
    assert "line: value" in text
