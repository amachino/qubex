"""Tests for project optional dependency declarations."""

from __future__ import annotations

import re
from pathlib import Path


def test_project_optional_dependencies_split_backend_quel1_quel3() -> None:
    """Given project metadata, backend should include both explicit hardware extras."""
    text = Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_text()
    version = (
        Path(__file__).resolve().parent.parent.joinpath("VERSION").read_text().strip()
    )

    assert re.search(
        r'^backend\s*=\s*\["qubex\[quel1,quel3\]"\]',
        text,
        re.MULTILINE,
    )
    assert re.search(
        rf'^quel1\s*=\s*\["qxdriver-quel1 == {re.escape(version)}"\]',
        text,
        re.MULTILINE,
    )
    assert re.search(
        r'^quel3\s*=\s*\["quelware-client >= 0\.7\.3", "quelware-core >= 0\.9\.0"\]',
        text,
        re.MULTILINE,
    )
