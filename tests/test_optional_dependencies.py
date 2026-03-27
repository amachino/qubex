"""Tests for project optional dependency declarations."""

from __future__ import annotations

import re
from pathlib import Path


def _read_project_text() -> str:
    """Load the project pyproject text."""
    return Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_text()


def test_project_optional_dependencies_split_backend_quel1_quel3() -> None:
    """Given project metadata, optional dependencies should expose backend, quel1, and quel3 extras."""
    text = _read_project_text()
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


def test_project_uv_sources_use_workspace_members_for_driver_packages() -> None:
    """Given project metadata, uv sources should resolve driver packages from the workspace."""
    text = _read_project_text()

    assert "quelware_client = { workspace = true }" in text
    assert "qxdriver_quel1 = { workspace = true }" in text


def test_project_uv_workspace_members_include_local_driver_packages() -> None:
    """Given project metadata, uv workspace members should include local driver packages."""
    text = _read_project_text()

    assert '"packages/quelware-client/quelware-client",' in text
    assert '"packages/qxdriver-quel1",' in text
