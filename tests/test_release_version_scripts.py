"""Tests for release-version synchronization helpers."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("directory", "distribution"),
    [
        ("qxcore", "qubex-core"),
        ("qxpulse", "qubex-pulse"),
        ("qxschema", "qubex-schema"),
        ("qxsimulator", "qubex-simulator"),
        ("qxvisualizer", "qubex-visualizer"),
        ("qxdriver-quel1", "qubex-driver-quel1"),
    ],
)
def test_companion_distribution_names(directory, distribution) -> None:
    """Companion distributions use the qubex prefix while source paths stay stable."""
    root = Path(__file__).resolve().parent.parent
    metadata = (root / "packages" / directory / "pyproject.toml").read_text()
    assert re.search(rf'^name = "{distribution}"$', metadata, re.MULTILINE)


def test_driver_version_uses_published_distribution_name(monkeypatch) -> None:
    """The preserved driver import reports the renamed distribution's version."""

    def installed_version(name):
        assert name == "qubex-driver-quel1"
        return "1.5.0rc4"

    monkeypatch.setattr(importlib.metadata, "version", installed_version)
    root = Path(__file__).resolve().parent.parent
    module = runpy.run_path(
        str(root / "packages/qxdriver-quel1/src/qxdriver_quel1/__init__.py")
    )
    assert module["__version__"] == "1.5.0rc4"


def test_placeholder_fitting_is_not_a_release_dependency() -> None:
    """The unimplemented fitting distribution is neither required nor published."""
    module = _load_sync_release_version_module()
    root = Path(__file__).resolve().parent.parent
    assert '"qubex-fitting ==' not in (root / "pyproject.toml").read_text()
    assert "qubex-fitting:" not in (root / "Makefile").read_text()
    assert "qubex-fitting" not in module.PACKAGE_PYPROJECTS


def test_all_workspace_dependency_pins_are_synchronized() -> None:
    """Every exact workspace dependency is covered by release synchronization."""
    module = _load_sync_release_version_module()
    for path in module.PACKAGE_PYPROJECTS.values():
        dependencies = set(re.findall(r'"([\w-]+)\s*==\s*[^\"]+"', path.read_text()))
        workspace_dependencies = dependencies & set(module.WORKSPACE_PACKAGES)
        assert workspace_dependencies == set(module.PINNED_DEPENDENCIES.get(path, ()))


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
