"""Tests for rendering QuEL-3 resource snapshots without acquisition."""

import json
from io import StringIO
from typing import Any, cast

import pytest
from rich.console import Console

from qubex.backend.quel3.formatters import (
    Quel3ResourceView,
    format_resource_snapshot,
    print_resource_snapshot,
)
from qubex.backend.quel3.models import (
    Quel3InstrumentState,
    Quel3PortDiagnostic,
    Quel3PortState,
    Quel3ResourceIssue,
    Quel3ResourceSnapshot,
    Quel3UnitControlState,
    Quel3UnitState,
)


def test_resource_snapshot_print_omits_absent_endpoint_port() -> None:
    """An absent endpoint port should not render as a literal None suffix."""
    state = Quel3ResourceSnapshot(
        generated_at="2026-07-07T00:00:00+00:00",
        endpoint="api.example.com",
        port=None,
        selected_unit_labels=(),
        units=(),
        ports=(),
        instruments=(),
    )
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)

    print_resource_snapshot(state, console=console)

    assert "api.example.com" in output.getvalue()
    assert "api.example.com:None" not in output.getvalue()


def test_resource_snapshot_prints_diagnostics_as_raw_yaml() -> None:
    """Diagnostic view should print YAML without Rich panels or syntax framing."""
    diagnostic_yaml = "port:\n  id: unit-a:tx_p01\n  state: ready\n"
    state = Quel3ResourceSnapshot(
        generated_at="2026-07-07T00:00:00+00:00",
        endpoint="localhost",
        port=50051,
        selected_unit_labels=("unit-a",),
        units=(),
        ports=(),
        instruments=(),
        diagnostics=(
            Quel3PortDiagnostic(
                port_id="unit-a:tx_p01",
                unit_label="unit-a",
                text=diagnostic_yaml,
            ),
        ),
    )
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)

    print_resource_snapshot(state, view="diagnostics", console=console)

    assert output.getvalue() == diagnostic_yaml


def test_resource_snapshot_prints_unit_configuration_controls() -> None:
    """Units view should print each control's current and allowed values."""
    state = Quel3ResourceSnapshot(
        generated_at="2026-07-07T00:00:00+00:00",
        endpoint="localhost",
        port=50051,
        selected_unit_labels=("unit-a",),
        units=(
            Quel3UnitState(
                label="unit-a",
                controls=(
                    Quel3UnitControlState(
                        key="quel3.monitor.mode",
                        allowed_values=("disabled", "loopback"),
                        current_value="loopback",
                    ),
                ),
            ),
        ),
        ports=(),
        instruments=(),
    )
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)

    print_resource_snapshot(state, view="units", console=console)

    rendered = output.getvalue()
    assert "unit-a" in rendered
    assert "quel3.monitor.mode" in rendered
    assert "loopback" in rendered
    assert "disabled, loopback" in rendered


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        ("summary", ("QuEL-3 resource snapshot", "Read failed", "unit-a:tx_p01")),
        ("units", ("unit-a", "unit-b")),
        ("ports", ("tx_p01", "TX")),
        ("instruments", ("Q00", "4.1000", "4.3000", "400000")),
        ("diagnostics", ("state: ready",)),
        (
            "all",
            (
                "QuEL-3 resource snapshot",
                "unit-b",
                "Q00",
                "Read failed",
                "state: ready",
            ),
        ),
    ],
)
def test_format_resource_snapshot_renders_selected_view(
    view: Quel3ResourceView,
    expected: tuple[str, ...],
) -> None:
    """Each view should render collected resources and leave the snapshot unchanged."""
    snapshot = _snapshot()
    original = snapshot.to_dict()
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=160)

    console.print(format_resource_snapshot(snapshot, view=view))

    assert all(text in output.getvalue() for text in expected)
    assert snapshot.to_dict() == original


def test_format_resource_snapshot_rejects_invalid_view() -> None:
    """An unsupported view should fail even when formatting an existing snapshot."""
    with pytest.raises(ValueError, match="view"):
        format_resource_snapshot(_snapshot(), view=cast(Any, "invalid"))


def test_resource_snapshot_serializes_nested_resources_and_issues() -> None:
    """Snapshot serialization should preserve field names and recursively serialize resources."""
    data = json.loads(json.dumps(_snapshot().to_dict()))

    assert set(data) == {
        "generated_at",
        "endpoint",
        "port",
        "selected_unit_labels",
        "units",
        "ports",
        "instruments",
        "diagnostics",
        "issues",
    }
    assert data["selected_unit_labels"] == ["unit-a", "unit-b"]
    assert data["units"] == [
        {"label": "unit-a", "controls": []},
        {"label": "unit-b", "controls": []},
    ]
    assert data["ports"] == [
        {"id": "unit-a:tx_p01", "unit_label": "unit-a", "role": "TX", "depends_on": []},
    ]
    assert data["instruments"][0]["frequency_range_min_hz"] == 4.1e9
    assert data["diagnostics"] == [
        {"port_id": "unit-a:tx_p01", "unit_label": "unit-a", "text": "state: ready\n"},
    ]
    assert data["issues"] == [
        {
            "severity": "error",
            "code": "READ_ERROR",
            "message": "Read failed",
            "detail": None,
            "resource_id": "unit-a:tx_p01",
        },
    ]


def _snapshot() -> Quel3ResourceSnapshot:
    return Quel3ResourceSnapshot(
        generated_at="2026-09-29T00:00:00+00:00",
        endpoint="localhost",
        port=50051,
        selected_unit_labels=("unit-a", "unit-b"),
        units=(Quel3UnitState("unit-a"), Quel3UnitState("unit-b")),
        ports=(Quel3PortState("unit-a:tx_p01", "unit-a", "TX"),),
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
                sampling_period_fs=400_000,
            ),
        ),
        diagnostics=(Quel3PortDiagnostic("unit-a:tx_p01", "unit-a", "state: ready\n"),),
        issues=(
            Quel3ResourceIssue(
                "error", "READ_ERROR", "Read failed", resource_id="unit-a:tx_p01"
            ),
        ),
    )
