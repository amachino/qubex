"""Tests for QuEL-3 quelware client factory selection."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest

from qubex.backend.quel3.infra import quelware_imports as quelware_imports_module
from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
    Quel3ConnectionManager,
    Quel3ExecutionManager,
    Quel3RuntimeConfig,
    Quel3SessionManager,
)


def test_validate_client_runtime_rejects_standalone_mode() -> None:
    """Given standalone mode, validation should reject the unsupported runtime."""
    with pytest.raises(ValueError, match="Unsupported QuEL-3 client mode"):
        quelware_imports_module.validate_quelware_client_runtime(
            client_mode="standalone",
        )


def test_validate_client_runtime_normalizes_string_input() -> None:
    """Given mixed-case client-mode input, validation should normalize it."""
    client_mode = quelware_imports_module.validate_quelware_client_runtime(
        client_mode=" Server ",
    )

    assert client_mode == "server"


def test_load_client_factory_returns_server_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given server mode, loading the client factory should return the quelware server client."""
    create_quelware_client = object()
    monkeypatch.setattr(
        quelware_imports_module.importlib,
        "import_module",
        lambda _: SimpleNamespace(
            create_quelware_client=create_quelware_client,
        ),
    )

    client_factory = quelware_imports_module.load_quelware_client_factory(
        client_mode="server",
    )

    assert client_factory is create_quelware_client


def test_load_client_factory_binds_pat_for_server_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Given PAT path, loading the server client factory should pass a file-backed provider."""
    captured: dict[str, object] = {}
    pat_path = tmp_path / "pat.txt"
    pat_path.write_text("dummy-token\n", encoding="utf-8")

    def _create_quelware_client(
        endpoint: str,
        port: int,
        *,
        pat: Callable[[], str],
    ) -> tuple[str, int]:
        captured["endpoint"] = endpoint
        captured["port"] = port
        captured["pat"] = pat
        return (endpoint, port)

    monkeypatch.setattr(
        quelware_imports_module.importlib,
        "import_module",
        lambda _: SimpleNamespace(
            create_quelware_client=_create_quelware_client,
        ),
    )

    client_factory = quelware_imports_module.load_quelware_client_factory(
        client_mode="server",
        pat_path=str(pat_path),
    )
    context_manager = client_factory("worker-host", 61000)

    assert context_manager == ("worker-host", 61000)
    pat_provider = captured.pop("pat")
    assert callable(pat_provider)
    assert pat_provider() == "dummy-token"
    assert captured == {"endpoint": "worker-host", "port": 61000}


def test_runtime_config_normalizes_client_runtime() -> None:
    """Given mixed-case runtime input, runtime config should expose canonical values."""
    config = Quel3RuntimeConfig(
        endpoint="worker-host",
        port=61000,
        client_mode=" Server ",
        pat_path="/run/secrets/quelware-pat",
    )

    assert config.endpoint == "worker-host"
    assert config.port == 61000
    assert config.client_mode == "server"
    assert config.pat_path == "/run/secrets/quelware-pat"


def test_managers_accept_shared_runtime_config() -> None:
    """Given one runtime config, all QuEL-3 managers should expose that shared config."""
    config = Quel3RuntimeConfig(endpoint="worker-host", port=61000)
    session_manager = Quel3SessionManager(runtime_config=config)

    managers = (
        Quel3ConnectionManager(runtime_config=config),
        session_manager,
        Quel3ConfigurationManager(runtime_config=config),
        Quel3ExecutionManager(
            runtime_config=config,
            sampling_period_ns=0.4,
            capture_decimation_factor=4,
            session_manager=session_manager,
        ),
    )

    assert all(manager.runtime_config is config for manager in managers)
