"""Tests for the vendored ONS61797 client."""

from __future__ import annotations

import pytest

from qubex.third_party.ons61797 import ONS61797


def test_get_output_mode_queries_omd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Output-mode readback should use the documented OMD query."""
    commands: list[str] = []
    client = object.__new__(ONS61797)
    client.instrument = None

    def query(_: ONS61797, cmd: str) -> str:
        commands.append(cmd)
        return "0"

    monkeypatch.setattr(ONS61797, "query", query)

    assert client.get_output_mode() == 0
    assert commands == ["OMD?"]
