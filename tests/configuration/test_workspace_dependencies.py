"""Tests for workspace dependency wiring."""

from __future__ import annotations

import re
from pathlib import Path


def test_backend_extra_includes_quelware_client() -> None:
    """Given the root project config, backend extra should install quelware-client."""
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject_text = pyproject_path.read_text()
    backend_match = re.search(
        r"^backend\s*=\s*\[(?P<deps>[^\]]*)\]", pyproject_text, re.MULTILINE
    )

    assert backend_match is not None

    assert '"quelware-client"' in backend_match.group("deps")
