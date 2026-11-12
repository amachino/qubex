"""Tests for QuEL-3 quelware client factory selection."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast

import pytest

from qubex.backend.quel3.infra import (
    Quel3HttpTransportConfig,
    Quel3RuntimeConfig,
    QuelwareCredentials,
    quelware_imports as quelware_imports_module,
)
from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
    Quel3ConnectionManager,
    Quel3ExecutionManager,
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
    captured: dict[str, object] = {}
    original_channel = object()
    channel_factory = object()
    grpc_module = SimpleNamespace(Channel=original_channel)

    def _create_quelware_client(endpoint: str, port: int) -> tuple[str, int]:
        captured["channel"] = grpc_module.Channel
        return endpoint, port

    client_module = SimpleNamespace(
        create_quelware_client=_create_quelware_client,
    )
    real_import_module = quelware_imports_module.importlib.import_module

    def _import_module(name: str):
        if name == "quelware_client.client":
            return client_module
        if name == "quelware_client.client._grpc":
            return grpc_module
        if name == "grpclib.client":
            return SimpleNamespace(Channel=channel_factory)
        return real_import_module(name)

    monkeypatch.setattr(
        quelware_imports_module.importlib,
        "import_module",
        _import_module,
    )

    client_factory = quelware_imports_module.load_quelware_client_factory(
        client_mode="server",
    )
    result = client_factory("worker-host", 61000)

    assert result == ("worker-host", 61000)
    assert captured["channel"] is channel_factory
    assert grpc_module.Channel is original_channel


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

    original_channel = object()
    grpc_module = SimpleNamespace(Channel=original_channel)
    client_module = SimpleNamespace(create_quelware_client=_create_quelware_client)
    real_import_module = quelware_imports_module.importlib.import_module

    def _import_module(name: str):
        if name == "quelware_client.client":
            return client_module
        if name == "quelware_client.client._grpc":
            return grpc_module
        if name == "grpclib.client":
            return SimpleNamespace(Channel=object())
        return real_import_module(name)

    monkeypatch.setattr(
        quelware_imports_module.importlib, "import_module", _import_module
    )

    client_factory = Quel3RuntimeConfig(
        credentials=QuelwareCredentials(pat_path=str(pat_path))
    ).load_client_factory()
    context_manager = client_factory("worker-host", 61000)

    assert context_manager == ("worker-host", 61000)
    pat_provider = captured.pop("pat")
    assert callable(pat_provider)
    assert pat_provider() == "dummy-token"
    assert captured == {"endpoint": "worker-host", "port": 61000}


def test_load_client_factory_scopes_https_channel_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    quel3_http_transport,
) -> None:
    """HTTPS transport should restore the upstream channel after client creation."""
    client_id_path = tmp_path / "client-id"
    client_secret_path = tmp_path / "client-secret"
    client_id_path.write_text("configured-id\n", encoding="utf-8")
    client_secret_path.write_text("configured-secret\n", encoding="utf-8")
    captured: dict[str, object] = {}
    original_channel = object()
    grpc_module = SimpleNamespace(Channel=original_channel)

    def _create_quelware_client(endpoint: str, port: int):
        captured["channel"] = grpc_module.Channel(endpoint, port)
        return endpoint, port

    client_module = SimpleNamespace(create_quelware_client=_create_quelware_client)
    real_import_module = quelware_imports_module.importlib.import_module

    def _import_module(name: str):
        if name == "quelware_client.client":
            return client_module
        if name == "quelware_client.client._grpc":
            return grpc_module
        return real_import_module(name)

    monkeypatch.setattr(
        quelware_imports_module.importlib, "import_module", _import_module
    )
    http_transport = Quel3HttpTransportConfig(
        base_path="api",
        default_timeout_seconds=12.5,
        secret_header_paths={
            "CF-Access-Client-Id": str(client_id_path),
            "CF-Access-Client-Secret": str(client_secret_path),
        },
    )

    client_factory = quelware_imports_module.load_quelware_client_factory(
        client_mode="server",
        transport="https",
        http_transport=http_transport,
    )
    result = client_factory("api.example.com", 443)

    assert result == ("api.example.com", 443)
    assert isinstance(captured["channel"], quel3_http_transport.ProtobufHttpChannel)
    assert grpc_module.Channel is original_channel


def test_load_client_factory_restores_channel_after_creation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client creation error should not leave the upstream channel patched."""
    original_channel = object()
    grpc_module = SimpleNamespace(Channel=original_channel)

    def _create_quelware_client(endpoint: str, port: int):
        del endpoint, port
        raise RuntimeError("create failed")

    client_module = SimpleNamespace(create_quelware_client=_create_quelware_client)
    real_import_module = quelware_imports_module.importlib.import_module

    def _import_module(name: str):
        if name == "quelware_client.client":
            return client_module
        if name == "quelware_client.client._grpc":
            return grpc_module
        return real_import_module(name)

    monkeypatch.setattr(
        quelware_imports_module.importlib, "import_module", _import_module
    )
    channel_factory = object()
    monkeypatch.setattr(
        quelware_imports_module, "_load_channel_factory", lambda **_: channel_factory
    )
    client_factory = quelware_imports_module.load_quelware_client_factory(
        client_mode="server",
        transport="http",
    )

    with pytest.raises(RuntimeError, match="create failed"):
        client_factory("localhost", 8080)

    assert grpc_module.Channel is original_channel


def test_load_client_factory_reads_file_backed_proxy_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    quel3_http_transport,
) -> None:
    """An explicit proxy should be read from its secret file without trailing newlines."""
    proxy_url_path = tmp_path / "proxy-url"
    proxy_url_path.write_text(
        "https://proxy-user:proxy-password@proxy.example.com:3128\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    original_channel = object()
    grpc_module = SimpleNamespace(Channel=original_channel)

    class _Channel:
        def __init__(self, endpoint: str, port: int, **kwargs: object) -> None:
            captured.update(endpoint=endpoint, port=port, **kwargs)

    def _create_quelware_client(endpoint: str, port: int):
        return grpc_module.Channel(endpoint, port)

    client_module = SimpleNamespace(create_quelware_client=_create_quelware_client)
    real_import_module = quelware_imports_module.importlib.import_module

    def _import_module(name: str):
        if name == "quelware_client.client":
            return client_module
        if name == "quelware_client.client._grpc":
            return grpc_module
        return real_import_module(name)

    monkeypatch.setattr(
        quelware_imports_module.importlib,
        "import_module",
        _import_module,
    )
    monkeypatch.setattr(
        quel3_http_transport,
        "ProtobufHttpChannel",
        _Channel,
    )
    client_factory = quelware_imports_module.load_quelware_client_factory(
        client_mode="server",
        transport="http",
        http_transport=Quel3HttpTransportConfig(
            proxy_url_path=str(proxy_url_path),
        ),
    )

    client_factory("api.example.com", 443)

    assert captured["endpoint"] == "api.example.com"
    assert captured["port"] == 443
    assert captured["proxy_url"] == (
        "https://proxy-user:proxy-password@proxy.example.com:3128"
    )
    assert captured["scheme"] == "http"
    assert "ssl_context" not in captured
    assert grpc_module.Channel is original_channel


def test_runtime_config_normalizes_client_runtime() -> None:
    """Given mixed-case runtime input, runtime config should expose canonical values."""
    config = Quel3RuntimeConfig(
        endpoint="worker-host",
        port=61000,
        client_mode=" Server ",
        transport=" HTTPS ",
        http_transport=Quel3HttpTransportConfig(),
        credentials=QuelwareCredentials(pat_path="/run/secrets/quelware-pat"),
    )

    assert config.endpoint == "worker-host"
    assert config.port == 61000
    assert config.client_mode == "server"
    assert config.credentials.pat_path == "/run/secrets/quelware-pat"
    assert config.transport == "https"


def test_runtime_config_leaves_https_port_unspecified() -> None:
    """HTTPS runtime config should preserve an unspecified port for URL construction."""
    config = Quel3RuntimeConfig.from_mapping(
        {
            "endpoint": "api.example.com",
            "transport": "https",
        }
    )

    assert config.port is None


def test_runtime_config_with_secret_headers_is_hashable() -> None:
    """HTTPS runtime config with secret headers should remain hashable."""
    config = Quel3RuntimeConfig(
        transport="https",
        http_transport=Quel3HttpTransportConfig(
            secret_header_paths={"X-API-Key": "/run/secrets/api-key"},
        ),
    )

    assert isinstance(hash(config), int)


def test_runtime_config_rejects_http_options_for_grpc() -> None:
    """Native gRPC should reject HTTP-specific transport options."""
    with pytest.raises(ValueError, match="http_transport"):
        Quel3RuntimeConfig(http_transport=Quel3HttpTransportConfig(base_path="api"))


def test_http_transport_config_rejects_empty_proxy_secret_path() -> None:
    """An empty proxy URL secret path should be rejected before client creation."""
    with pytest.raises(ValueError, match="must not be empty"):
        Quel3HttpTransportConfig(proxy_url_path="")


def test_runtime_config_rejects_secret_headers_for_http() -> None:
    """HTTP transport should reject file-backed secret headers."""
    with pytest.raises(ValueError, match="HTTPS"):
        Quel3RuntimeConfig(
            transport="http",
            http_transport=Quel3HttpTransportConfig(
                secret_header_paths={
                    "X-API-Key": "/run/secrets/api-key",
                },
            ),
        )


def test_runtime_config_from_mapping_parses_nested_http_options() -> None:
    """Nested QuEL-3 YAML options should become one flat HTTP transport config."""
    config = Quel3RuntimeConfig.from_mapping(
        {
            "quelware_endpoint": "api.example.com",
            "quelware_port": 443,
            "client_mode": "server",
            "quelware_pat_path": "/run/secrets/quelware-pat",
            "transport": "https",
            "http_transport": {
                "base_path": "/quelware",
                "default_timeout_seconds": 30.0,
                "proxy": {
                    "url_path": "/run/secrets/quelware-proxy-url",
                },
                "secret_header_paths": {
                    "CF-Access-Client-Id": "/run/secrets/cf-access-client-id",
                    "CF-Access-Client-Secret": "/run/secrets/cf-access-client-secret",
                },
            },
        }
    )

    assert config == Quel3RuntimeConfig(
        endpoint="api.example.com",
        port=443,
        client_mode="server",
        transport="https",
        http_transport=Quel3HttpTransportConfig(
            base_path="/quelware",
            default_timeout_seconds=30.0,
            proxy_url_path="/run/secrets/quelware-proxy-url",
            secret_header_paths={
                "CF-Access-Client-Id": "/run/secrets/cf-access-client-id",
                "CF-Access-Client-Secret": "/run/secrets/cf-access-client-secret",
            },
        ),
        credentials=QuelwareCredentials(pat_path="/run/secrets/quelware-pat"),
    )


@pytest.mark.parametrize(
    ("http_transport", "exception_type", "message"),
    [
        ({"cafile": "/run/secrets/ca.pem"}, ValueError, "Unsupported"),
        ({"proxy": "http://proxy.example.com:3128"}, TypeError, "mapping"),
        ({"proxy": {}}, ValueError, "url_path.*required"),
        (
            {"secret_header_paths": {"X-API-Key": 123}},
            TypeError,
            "secret_header_paths",
        ),
    ],
)
def test_runtime_config_from_mapping_rejects_invalid_http_options(
    http_transport: object,
    exception_type: type[Exception],
    message: str,
) -> None:
    """Invalid nested HTTP options should fail inside the QuEL-3 config owner."""
    with pytest.raises(exception_type, match=message):
        Quel3RuntimeConfig.from_mapping(
            {
                "transport": "https",
                "http_transport": http_transport,
            }
        )


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


def test_connection_probe_lists_units_without_scanning_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QuEL-3 connect should inspect local unit labels without listing resources."""
    calls: list[str] = []

    @asynccontextmanager
    async def _client_factory(
        endpoint: str,
        port: int,
    ) -> AsyncIterator[SimpleNamespace]:
        calls.append(f"client:{endpoint}:{port}")
        yield SimpleNamespace(
            list_unit_labels=lambda: calls.append("list-units") or [],
        )

    manager = Quel3ConnectionManager(
        runtime_config=Quel3RuntimeConfig(endpoint="worker-host", port=61000)
    )
    monkeypatch.setattr(
        manager,
        "load_quelware_client_factory",
        lambda: _client_factory,
    )

    manager.connect()

    assert manager.is_connected is True
    assert calls == ["client:worker-host:61000", "list-units"]


@pytest.mark.parametrize("initial", [None, "initial-secret"])
def test_pat_provider_is_shared_and_rotatable(monkeypatch, initial) -> None:
    """Direct PAT updates should reach every shared runtime without exposing secrets."""
    from qubex.backend.quel3 import Quel3BackendController
    from qubex.backend.quel3.infra import runtime_config as runtime_module

    providers = []
    monkeypatch.setattr(
        runtime_module,
        "load_quelware_client_factory",
        lambda **kw: providers.append(kw["pat_provider"]),
    )
    controller = Quel3BackendController(
        runtime_config=Quel3RuntimeConfig(
            credentials=QuelwareCredentials(
                pat_provider=None if initial is None else lambda: initial,
                pat_path="missing-file",
            )
        ),
    )
    if initial is not None:
        controller.runtime_config.load_client_factory()
        assert providers.pop()() == initial
    before = hash(controller.runtime_config)
    controller.set_quelware_pat_provider(lambda: "updated-secret")
    for owner in (
        controller.connection_manager,
        controller.session_manager,
        controller.configuration_manager,
        controller.execution_manager,
        controller.resource_reader,
    ):
        owner.runtime_config.load_client_factory()
    assert [provider() for provider in providers] == ["updated-secret"] * 5
    controller.set_quelware_pat_provider(lambda: "rotated-secret")
    assert providers[0]() == "rotated-secret"
    assert hash(controller.runtime_config) == before
    assert "secret" not in repr(controller.runtime_config)
    assert "secret" not in repr(controller)


def test_pat_path_provider_reads_updated_file(monkeypatch, tmp_path) -> None:
    """Runtime config should read and normalize the PAT file on each provider call."""
    from qubex.backend.quel3.infra import runtime_config as runtime_module

    path = tmp_path / "pat"
    monkeypatch.setenv("HOME", str(tmp_path))
    providers = []
    monkeypatch.setattr(
        runtime_module,
        "load_quelware_client_factory",
        lambda **kw: providers.append(kw["pat_provider"]),
    )
    config = Quel3RuntimeConfig(credentials=QuelwareCredentials(pat_path="~/pat"))
    config.load_client_factory()
    path.write_text("first\n")
    assert providers[0]() == "first"
    path.write_text("second\r\n")
    assert providers[0]() == "second"


@pytest.mark.parametrize(
    ("value", "error"), [("secret", TypeError), (None, TypeError), (123, TypeError)]
)
def test_pat_provider_rejects_invalid_values(monkeypatch, value, error) -> None:
    """Invalid PAT updates should preserve the previous credentials."""
    from qubex.backend.quel3.infra import runtime_config as runtime_module

    providers = []
    monkeypatch.setattr(
        runtime_module,
        "load_quelware_client_factory",
        lambda **kw: providers.append(kw["pat_provider"]),
    )
    config = Quel3RuntimeConfig()
    config.credentials.set_pat_provider(lambda: "previous-secret")
    config.load_client_factory()
    with pytest.raises(error) as exc:
        config.credentials.set_pat_provider(value)
    assert "previous-secret" not in str(exc.value)
    assert providers[0]() == "previous-secret"


@pytest.mark.parametrize("connected", [True, False])
def test_controller_rejects_provider_change_while_in_use(connected) -> None:
    """A connected controller or open session should reject PAT changes."""
    from qubex.backend.quel3 import Quel3BackendController

    controller = Quel3BackendController(
        connection_manager=cast(
            Quel3ConnectionManager, SimpleNamespace(is_connected=connected)
        ),
        session_manager=cast(
            Quel3SessionManager, SimpleNamespace(is_open=not connected)
        ),
    )
    with pytest.raises(RuntimeError, match="Disconnect"):
        controller.set_quelware_pat_provider(lambda: "new-secret")


def test_pat_provider_rejects_combined_runtime_options() -> None:
    """Explicit runtime configuration cannot be combined with endpoint overrides."""
    from qubex.backend.quel3 import Quel3BackendController

    with pytest.raises(ValueError, match="cannot be combined"):
        Quel3BackendController(
            runtime_config=Quel3RuntimeConfig(
                credentials=QuelwareCredentials(pat_provider=lambda: "secret")
            ),
            quelware_endpoint="conflicting-host",
        )


def test_pat_provider_is_lazy_and_reads_current_value(monkeypatch) -> None:
    """Registering and binding a provider should defer evaluation until token access."""
    from qubex.backend.quel3 import Quel3BackendController
    from qubex.backend.quel3.infra import runtime_config as runtime_module

    calls = []
    state = {"pat": "first-secret"}
    providers = []

    def provider() -> str:
        calls.append("read")
        return state["pat"]

    monkeypatch.setattr(
        runtime_module,
        "load_quelware_client_factory",
        lambda **kw: providers.append(kw["pat_provider"]),
    )
    controller = Quel3BackendController(
        runtime_config=Quel3RuntimeConfig(
            credentials=QuelwareCredentials(pat_provider=provider)
        )
    )
    controller.set_quelware_pat_provider(provider)
    controller.runtime_config.load_client_factory()
    assert calls == []
    assert providers[0]() == "first-secret"
    state["pat"] = "second-secret"
    assert providers[0]() == "second-secret"
    assert calls == ["read", "read"]


def test_pat_provider_failure_is_deferred(monkeypatch) -> None:
    """A failing provider should raise only when the token is requested."""
    from qubex.backend.quel3 import Quel3BackendController
    from qubex.backend.quel3.infra import runtime_config as runtime_module

    providers = []
    error = RuntimeError("credentials unavailable")

    def provider() -> str:
        raise error

    monkeypatch.setattr(
        runtime_module,
        "load_quelware_client_factory",
        lambda **kw: providers.append(kw["pat_provider"]),
    )
    controller = Quel3BackendController(
        runtime_config=Quel3RuntimeConfig(
            credentials=QuelwareCredentials(pat_provider=provider)
        )
    )
    controller.runtime_config.load_client_factory()
    with pytest.raises(RuntimeError) as exc:
        providers[0]()
    assert exc.value is error


@pytest.mark.parametrize("provider", ["secret", 123])
def test_runtime_constructor_rejects_noncallable_provider(provider) -> None:
    """Runtime construction should reject a noncallable provider without exposing it."""
    with pytest.raises(TypeError, match="must be callable") as exc:
        Quel3RuntimeConfig(credentials=QuelwareCredentials(pat_provider=provider))
    assert str(provider) not in str(exc.value)


def test_runtime_replacement_preserves_credentials(monkeypatch) -> None:
    """Replacing transport settings should retain the shared credentials object."""
    from dataclasses import replace

    from qubex.backend.quel3.infra import runtime_config as runtime_module

    providers = []
    monkeypatch.setattr(
        runtime_module,
        "load_quelware_client_factory",
        lambda **kw: providers.append(kw["pat_provider"]),
    )
    credentials = QuelwareCredentials(pat_provider=lambda: "original-secret")
    config = Quel3RuntimeConfig(credentials=credentials)
    replaced = replace(config, endpoint="other-host")
    assert replaced.credentials is credentials
    replaced.load_client_factory()
    credentials.set_pat_provider(lambda: "updated-secret")
    assert providers[0]() == "updated-secret"


def test_default_runtime_credentials_are_independent(monkeypatch) -> None:
    """Default credentials should not leak providers between runtime instances."""
    from qubex.backend.quel3.infra import runtime_config as runtime_module

    providers = []
    monkeypatch.setattr(
        runtime_module,
        "load_quelware_client_factory",
        lambda **kw: providers.append(kw["pat_provider"]),
    )
    first, second = Quel3RuntimeConfig(), Quel3RuntimeConfig()
    first.credentials.set_pat_provider(lambda: "private-secret")
    first.load_client_factory()
    second.load_client_factory()
    assert providers[0]() == "private-secret"
    assert providers[1] is None


@pytest.mark.parametrize(
    "component",
    [
        "connection_manager",
        "session_manager",
        "configuration_manager",
        "execution_manager",
        "resource_reader",
    ],
)
def test_provider_update_rejects_unshared_injected_credentials(component) -> None:
    """Provider updates must reject components whose credentials would remain unchanged."""
    from qubex.backend.quel3 import Quel3BackendController

    injected = Quel3BackendController()
    controller = Quel3BackendController(**{component: getattr(injected, component)})
    with pytest.raises(ValueError, match=r"share.*credentials"):
        controller.set_quelware_pat_provider(lambda: "new-secret")
    assert controller.runtime_config.credentials.resolve_pat_provider() is None
    assert injected.runtime_config.credentials.resolve_pat_provider() is None


def test_provider_update_accepts_injected_shared_credentials() -> None:
    """Components with distinct runtime configs may share the same credentials."""
    from qubex.backend.quel3 import Quel3BackendController

    credentials = QuelwareCredentials()
    injected = Quel3BackendController(
        runtime_config=Quel3RuntimeConfig(credentials=credentials)
    )
    controller = Quel3BackendController(
        runtime_config=Quel3RuntimeConfig(credentials=credentials),
        connection_manager=injected.connection_manager,
        session_manager=injected.session_manager,
        configuration_manager=injected.configuration_manager,
        execution_manager=injected.execution_manager,
        resource_reader=injected.resource_reader,
    )
    controller.set_quelware_pat_provider(lambda: "shared-secret")
    provider = injected.runtime_config.credentials.resolve_pat_provider()
    assert provider is not None
    assert provider() == "shared-secret"


@pytest.mark.parametrize("transport", ["grpc", "http", "https"])
def test_real_upstream_client_evaluates_provider_at_creation(
    monkeypatch, transport
) -> None:
    """The upstream client should snapshot one PAT per creation without connecting."""
    from unittest.mock import Mock

    upstream = pytest.importorskip("quelware_client.client._grpc")

    channel = Mock()
    monkeypatch.setattr(
        quelware_imports_module,
        "_load_channel_factory",
        lambda **kwargs: lambda *args: channel,
    )
    state = {"pat": "first-secret"}
    calls = []
    metadata = []

    def provider() -> str:
        calls.append("read")
        return state["pat"]

    monkeypatch.setattr(
        upstream,
        "SessionAgentGrpc",
        lambda channel, *, metadata: metadata_log(metadata),
    )

    def metadata_log(value):
        metadata.append(value["x-pat"])
        return Mock()

    credentials = QuelwareCredentials(pat_provider=provider)
    factory = Quel3RuntimeConfig(
        transport=transport, credentials=credentials
    ).load_client_factory()
    assert calls == []
    first = factory("localhost", 50051)
    assert calls == ["read"]
    state["pat"] = "second-secret"
    assert metadata == ["first-secret"]
    second = factory("localhost", 50051)
    assert calls == ["read", "read"]
    assert metadata == ["first-secret", "second-secret"]
    assert first is not second
