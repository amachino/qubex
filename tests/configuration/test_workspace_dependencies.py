"""Tests for workspace dependency wiring."""

from __future__ import annotations

import re
from pathlib import Path


def _extract_extra_deps(pyproject_text: str, extra_name: str) -> str:
    """Return the dependency list text for one optional-dependency extra."""
    match = re.search(
        rf"^{extra_name}\s*=\s*\[(?P<deps>[^\]]*)\]",
        pyproject_text,
        re.MULTILINE,
    )
    assert match is not None
    return match.group("deps")


def test_backend_extra_includes_quelware_client_and_qxdriver() -> None:
    """Backend extra should install both packaged hardware backends."""
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject_text = pyproject_path.read_text()
    deps = _extract_extra_deps(pyproject_text, "backend")

    assert '"quelware-client"' in deps
    assert '"qxdriver-quel1"' in deps


def test_backend_specific_extras_are_exposed() -> None:
    """Backend-specific extras should map to the expected packages."""
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject_text = pyproject_path.read_text()

    assert _extract_extra_deps(pyproject_text, "qubecalib").strip() == '"qubecalib"'
    assert _extract_extra_deps(pyproject_text, "quel1").strip() == '"qxdriver-quel1"'
    assert _extract_extra_deps(pyproject_text, "quel3").strip() == '"quelware-client"'


def test_make_sync_uses_backend_extra_only() -> None:
    """Make sync should use the locked backend-only environment."""
    makefile_path = Path(__file__).resolve().parents[2] / "Makefile"
    makefile_text = makefile_path.read_text()

    assert "uv sync --all-groups --extra backend --locked" in makefile_text
    assert "--all-extras" not in makefile_text


def test_makefile_exposes_lock_target() -> None:
    """Makefile should provide a command to refresh the lockfile."""
    makefile_path = Path(__file__).resolve().parents[2] / "Makefile"
    makefile_text = makefile_path.read_text()

    assert ".PHONY: lock sync" in makefile_text
    assert "\nlock:\n\tgit submodule update --init --recursive\n\tuv lock\n" in makefile_text


def test_uv_config_limits_optional_backend_resolution() -> None:
    """Uv config should conflict qubecalib with qxdriver-based extras."""
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject_text = pyproject_path.read_text()
    workspace_members = pyproject_text.split("[tool.uv.workspace]", maxsplit=1)[1]

    assert 'qubecalib = { path = "packages/qube-calib", extra = "qubecalib" }' in pyproject_text
    assert '{ path = "packages/qxdriver-quel1", extra = "backend" }' in pyproject_text
    assert '{ path = "packages/qxdriver-quel1", extra = "quel1" }' in pyproject_text
    assert '{ extra = "backend" }' in pyproject_text
    assert '{ extra = "quel1" }' in pyproject_text
    assert '{ extra = "qubecalib" }' in pyproject_text
    assert '"packages/qxdriver-quel1",' not in workspace_members


def test_qubecalib_is_managed_as_packages_submodule() -> None:
    """Qubecalib checkout should be managed as a packages submodule."""
    gitmodules_path = Path(__file__).resolve().parents[2] / ".gitmodules"
    gitmodules_text = gitmodules_path.read_text()

    assert '[submodule "packages/qube-calib"]' in gitmodules_text
    assert "\tpath = packages/qube-calib" in gitmodules_text
    assert "\turl = https://github.com/qiqb-osaka/qube-calib.git" in gitmodules_text


def test_ci_workflows_install_expected_dependency_sets() -> None:
    """CI workflows should install only the groups each workflow needs."""
    root = Path(__file__).resolve().parents[2]
    python_workflow = (root / ".github/workflows/python-package.yml").read_text()
    docs_workflow = (root / ".github/workflows/docs.yml").read_text()

    assert "uv sync --locked --dev --extra backend" in python_workflow
    assert "--all-extras" not in python_workflow
    assert "uv sync --locked --all-groups --extra backend" in docs_workflow
    assert "--all-extras" not in docs_workflow
