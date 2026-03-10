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
    """Make sync should avoid all-extras so qubecalib is not installed by default."""
    makefile_path = Path(__file__).resolve().parents[2] / "Makefile"
    makefile_text = makefile_path.read_text()

    assert "uv sync --all-groups --extra backend" in makefile_text
    assert "--all-extras" not in makefile_text
