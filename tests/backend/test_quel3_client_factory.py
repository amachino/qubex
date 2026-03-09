"""Tests for QuEL-3 quelware client factory selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qubex.backend.quel3.infra import quelware_imports as quelware_imports_module


def test_validate_client_runtime_rejects_missing_unit_label() -> None:
    """Given standalone mode without unit label, validation should fail fast."""
    with pytest.raises(ValueError, match="standalone_unit_label"):
        quelware_imports_module.validate_quelware_client_runtime(
            client_mode="standalone",
            standalone_unit_label=None,
        )


def test_validate_client_runtime_normalizes_string_input() -> None:
    """Given mixed-case client-mode input, validation should normalize it."""
    client_mode = quelware_imports_module.validate_quelware_client_runtime(
        client_mode=" Standalone ",
        standalone_unit_label="quel3-02-a01",
    )

    assert client_mode == "standalone"


def test_load_client_factory_returns_server_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given server mode, loading the client factory should return the quelware server client."""
    create_quelware_client = object()
    monkeypatch.setattr(
        quelware_imports_module,
        "import_module_with_workspace_fallback",
        lambda _: SimpleNamespace(
            create_quelware_client=create_quelware_client,
            create_standalone_client=object(),
        ),
    )

    client_factory = quelware_imports_module.load_quelware_client_factory(
        client_mode="server",
        standalone_unit_label=None,
    )

    assert client_factory is create_quelware_client


def test_load_client_factory_binds_unit_label_for_standalone_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given standalone mode, loading the client factory should bind the configured unit label."""
    captured: dict[str, object] = {}

    def _create_standalone_client(
        endpoint: str,
        port: int,
        *,
        unit_label: str,
    ) -> tuple[str, int, str]:
        captured["endpoint"] = endpoint
        captured["port"] = port
        captured["unit_label"] = unit_label
        return (endpoint, port, unit_label)

    monkeypatch.setattr(
        quelware_imports_module,
        "import_module_with_workspace_fallback",
        lambda _: SimpleNamespace(
            create_quelware_client=object(),
            create_standalone_client=_create_standalone_client,
        ),
    )

    client_factory = quelware_imports_module.load_quelware_client_factory(
        client_mode="standalone",
        standalone_unit_label="quel3-02-a01",
    )
    context_manager = client_factory("worker-host", 61000)

    assert context_manager == ("worker-host", 61000, "quel3-02-a01")
    assert captured == {
        "endpoint": "worker-host",
        "port": 61000,
        "unit_label": "quel3-02-a01",
    }
