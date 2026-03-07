# ruff: noqa: SLF001

"""Tests for quelware import helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from qubex.backend.quel3.infra import quelware_imports


def test_candidate_local_quelware_paths_include_packages_and_lib_layouts(
    tmp_path: Path,
) -> None:
    """Yields packages and lib quelware source directories."""
    candidates = quelware_imports._candidate_local_quelware_paths(tmp_path)

    expected = (
        tmp_path / "lib" / "quelware-client-internal" / "quelware-client" / "src",
        tmp_path
        / "lib"
        / "quelware-client-internal"
        / "quelware-core"
        / "python"
        / "src",
        tmp_path / "packages" / "quelware-client" / "quelware-client" / "src",
        tmp_path / "packages" / "quelware-client" / "quelware-core" / "python" / "src",
    )

    assert candidates == expected


def test_append_local_quelware_paths_adds_only_existing_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Given existing candidates, helper prepends each path without duplicates."""
    existing = tmp_path / "existing"
    existing.mkdir()
    already_present = tmp_path / "already-present"
    already_present.mkdir()
    missing = tmp_path / "missing"

    monkeypatch.setattr(
        quelware_imports,
        "_candidate_local_quelware_paths",
        lambda _: (existing, missing, already_present),
    )
    monkeypatch.setattr(sys, "path", [str(already_present)])

    quelware_imports._append_local_quelware_paths()

    assert str(existing) in sys.path
    assert str(missing) not in sys.path
    assert sys.path.count(str(existing)) == 1
    assert sys.path.count(str(already_present)) == 1
