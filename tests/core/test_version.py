"""Tests for version helper utilities."""

from __future__ import annotations

import importlib.metadata

from qubex import version as version_module


def test_get_optional_version_returns_none_for_missing_distribution(
    monkeypatch,
) -> None:
    """Given missing distribution, when optional version is requested, then None is returned."""

    def _raise_not_found(package_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(package_name)

    monkeypatch.setattr(version_module, "_get_installed_version", _raise_not_found)

    assert version_module.get_optional_version("qxdriver-quel") is None


def test_resolve_first_available_version_uses_preferred_installed_package(
    monkeypatch,
) -> None:
    """Given multiple package names, when resolving, then first installed candidate is selected."""
    versions = {
        "qxdriver-quel": "0.1.0",
        "qubecalib": "3.1.16",
    }

    def _get_optional_version(package_name: str) -> str | None:
        return versions.get(package_name)

    monkeypatch.setattr(version_module, "get_optional_version", _get_optional_version)

    assert version_module.resolve_first_available_version(
        ("qxdriver-quel", "qubecalib")
    ) == ("qxdriver-quel", "0.1.0")


def test_resolve_first_available_version_returns_none_when_no_candidate_installed(
    monkeypatch,
) -> None:
    """Given no installed candidate, when resolving, then None is returned."""
    monkeypatch.setattr(version_module, "get_optional_version", lambda _: None)

    assert (
        version_module.resolve_first_available_version(("qxdriver-quel", "qubecalib"))
        is None
    )
