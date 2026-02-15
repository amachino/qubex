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
