"""Tests for quelware import helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qubex.backend.quel3.infra import quelware_imports


def test_import_module_with_workspace_fallback_uses_standard_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given one module name, helper should delegate to the standard importer."""
    imported = SimpleNamespace(name="quelware_client.client")
    captured: dict[str, object] = {}

    def _import_module(module_name: str) -> object:
        captured["module_name"] = module_name
        return imported

    monkeypatch.setattr(quelware_imports.importlib, "import_module", _import_module)

    module = quelware_imports.import_module_with_workspace_fallback(
        "quelware_client.client"
    )

    assert module is imported
    assert captured == {"module_name": "quelware_client.client"}


def test_import_module_with_workspace_fallback_propagates_import_errors() -> None:
    """Given missing quelware module, helper should propagate ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError, match="quelware_client"):
        quelware_imports.import_module_with_workspace_fallback(
            "quelware_client.missing"
        )
