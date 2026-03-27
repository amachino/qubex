"""Tests for release-version synchronization helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_sync_release_version_module():
    """Load the sync_release_version script as a module."""
    module_path = (
        Path(__file__).resolve().parent.parent / "scripts/sync_release_version.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sync_release_version",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pin_dependency_replaces_only_version_text() -> None:
    """Given an exact pin, when syncing, then only the version text changes."""
    module = _load_sync_release_version_module()
    original = '"qxcore == 0.0.0.dev0",'

    updated = module.pin_dependency_version(  # type: ignore[attr-defined]
        original,
        package="qxcore",
        version="1.5.0b4",
        path=Path("pyproject.toml"),
    )

    assert updated == '"qxcore == 1.5.0b4",'


def test_pin_dependency_preserves_existing_spacing() -> None:
    """Given nonstandard spacing, when syncing, then spacing is preserved."""
    module = _load_sync_release_version_module()
    original = '"qxcore==0.0.0.dev0",'

    updated = module.pin_dependency_version(  # type: ignore[attr-defined]
        original,
        package="qxcore",
        version="1.5.0b4",
        path=Path("pyproject.toml"),
    )

    assert updated == '"qxcore==1.5.0b4",'


def test_pin_dependency_raises_for_missing_entry() -> None:
    """Given a missing pin, when syncing, then an error is raised."""
    module = _load_sync_release_version_module()

    with pytest.raises(ValueError, match="Dependency entry"):
        module.pin_dependency_version(  # type: ignore[attr-defined]
            '"qxpulse >= 1.0",',
            package="qxcore",
            version="1.5.0b4",
            path=Path("pyproject.toml"),
        )


def test_set_pyproject_version_replaces_static_version_line(tmp_path: Path) -> None:
    """Given a static project version, when syncing, then the version line is updated."""
    module = _load_sync_release_version_module()
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "qxdriver-quel1"\nversion = "0.0.0.dev0"\n')

    module.set_pyproject_version(  # type: ignore[attr-defined]
        pyproject,
        version="1.5.0b4",
    )

    assert 'version = "1.5.0b4"' in pyproject.read_text()


def test_set_pyproject_version_raises_for_missing_version(tmp_path: Path) -> None:
    """Given missing static version metadata, when syncing, then an error is raised."""
    module = _load_sync_release_version_module()
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "qxdriver-quel1"\n')

    with pytest.raises(ValueError, match="Static project.version"):
        module.set_pyproject_version(  # type: ignore[attr-defined]
            pyproject,
            version="1.5.0b4",
        )
