"""Tests for project optional dependency declarations."""

from __future__ import annotations

import re
from pathlib import Path


def test_project_optional_dependencies_split_backend_quel1_quel3() -> None:
    """Given project metadata, optional dependencies should expose backend, quel1, and quel3 extras."""
    text = Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_text()
    version = (
        Path(__file__).resolve().parent.parent.joinpath("VERSION").read_text().strip()
    )

    assert re.search(
        rf'^backend\s*=\s*\["qxdriver-quel1 == {re.escape(version)}"\]',
        text,
        re.MULTILINE,
    )
    assert re.search(
        rf'^quel1\s*=\s*\["qxdriver-quel1 == {re.escape(version)}"\]',
        text,
        re.MULTILINE,
    )
    assert re.search(
        r'^quel3\s*=\s*\["quelware-client"\]',
        text,
        re.MULTILINE,
    )
