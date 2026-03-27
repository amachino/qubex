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


def test_project_uv_sources_use_repo_local_paths_for_external_driver_packages() -> None:
    """Given project metadata, uv sources should resolve external driver packages from repo-local submodule paths."""
    text = _read_project_text()

    assert (
        'quelware_client = { path = "packages/quelware-client/quelware-client" }'
        in text
    )
    assert 'qxdriver_quel1 = { path = "packages/qxdriver-quel1" }' in text


def test_project_uv_workspace_members_exclude_external_driver_packages() -> None:
    """Given project metadata, uv workspace should only include in-repository packages."""
    text = _read_project_text()
    workspace = text.split("[tool.uv.workspace]", 1)[1].split(
        "[tool.pytest.ini_options]",
        1,
    )[0]

    assert '"packages/qxcore",' in workspace
    assert '"packages/qxvisualizer",' in workspace
    assert '"packages/quelware-client/quelware-client",' not in workspace
    assert '"packages/qxdriver-quel1",' not in workspace
