# ruff: noqa: SLF001

"""Tests for quelware duplicated-proxy suppression patch."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import ClassVar

from qubex.patches.quel_ic_config import suppress_duplicated_proxy_patch as patch


class _FakeLockKeeper:
    _clients: ClassVar[set[_FakeLockKeeper]] = set()
    _clients_lock = threading.RLock()

    def __init__(self, *, target: tuple[str, int]) -> None:
        self._target = target
        self._locked = True
        self.deactivate_calls = 0

    @property
    def has_lock(self) -> bool:
        return self._locked

    def deactivate(self, timeout: float = 0.0) -> bool:
        _ = timeout
        self.deactivate_calls += 1
        self._locked = False
        return True

    def _register_self(self) -> None:
        raise AssertionError("must be patched")


class _FakeSyncAsyncCoapClient:
    _clients: ClassVar[set[_FakeSyncAsyncCoapClient]] = set()
    _create_lock = threading.Lock()

    def __init__(self, *, target: tuple[str, int]) -> None:
        self._target = target
        self._locked = True
        self.cleanup_calls = 0

    @property
    def has_lock(self) -> bool:
        return self._locked

    def _cleanup(self) -> None:
        self.cleanup_calls += 1
        self._locked = False

    def _register_self(self) -> None:
        raise AssertionError("must be patched")


def test_apply_patch_does_not_fail_when_quelware_missing(monkeypatch) -> None:
    """Given missing quelware modules, when patch applies, then no exception is raised."""
    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: True)
    import_calls: list[str] = []

    def _raise_import_error(name: str):
        import_calls.append(name)
        raise ImportError(name)

    monkeypatch.setattr(patch.importlib, "import_module", _raise_import_error)

    patch.apply_quelware_duplicated_proxy_patch()

    assert import_calls == [
        "quel_ic_config.exstickge_sock_client",
        "quel_ic_config.exstickge_coap_client",
    ]


def test_apply_patch_skips_for_quelware_0_8(monkeypatch) -> None:
    """Given quelware 0.8.x, when patch applies, then target modules are not imported."""
    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: False)
    import_calls: list[str] = []

    def _import_module(name: str):
        import_calls.append(name)
        raise AssertionError("import must not be called for quelware 0.8.x")

    monkeypatch.setattr(patch.importlib, "import_module", _import_module)

    patch.apply_quelware_duplicated_proxy_patch()

    assert import_calls == []


def test_apply_patch_releases_existing_proxy_before_registering(monkeypatch) -> None:
    """Given duplicated proxies, when registering, then stale proxy is released and no RuntimeError is raised."""
    _FakeLockKeeper._clients = set()
    _FakeSyncAsyncCoapClient._clients = set()

    sock_mod = SimpleNamespace(AbstractLockKeeper=_FakeLockKeeper)
    coap_mod = SimpleNamespace(AbstractSyncAsyncCoapClient=_FakeSyncAsyncCoapClient)
    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: True)

    def _import_module(name: str):
        if name == "quel_ic_config.exstickge_sock_client":
            return sock_mod
        if name == "quel_ic_config.exstickge_coap_client":
            return coap_mod
        raise ImportError(name)

    monkeypatch.setattr(patch.importlib, "import_module", _import_module)

    patch.apply_quelware_duplicated_proxy_patch()

    existing_sock_proxy = _FakeLockKeeper(target=("10.5.0.2", 16384))
    existing_sock_proxy._register_self()
    new_sock_proxy = _FakeLockKeeper(target=("10.5.0.2", 16384))
    new_sock_proxy._register_self()

    assert existing_sock_proxy.deactivate_calls == 1
    assert existing_sock_proxy not in _FakeLockKeeper._clients
    assert new_sock_proxy in _FakeLockKeeper._clients

    existing_coap_proxy = _FakeSyncAsyncCoapClient(target=("10.5.0.2", 5683))
    existing_coap_proxy._register_self()
    new_coap_proxy = _FakeSyncAsyncCoapClient(target=("10.5.0.2", 5683))
    new_coap_proxy._register_self()

    assert existing_coap_proxy.cleanup_calls == 1
    assert existing_coap_proxy not in _FakeSyncAsyncCoapClient._clients
    assert new_coap_proxy in _FakeSyncAsyncCoapClient._clients
