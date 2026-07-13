# ruff: noqa: SLF001

"""Tests for quelware file-lock disabling patch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from qubex.patches.quel_ic_config import disable_quelware_filelock_patch as patch


class _FakeAbstractLockKeeper:
    _DEFAULT_LOOP_WAIT = 0.25

    def __init__(self, *, target: tuple[str, int], loop_wait: float = 0.25) -> None:
        self._target = target
        self._loop_wait = loop_wait


class _FakeFileLockKeeper(_FakeAbstractLockKeeper):
    def __init__(
        self,
        *,
        target: tuple[str, int],
        loop_wait: float = 0.25,
        lock_directory: object = None,
    ) -> None:
        raise RuntimeError(f"lock directory '{lock_directory}' is unavailable")


class _FakeDummyLockKeeper(_FakeAbstractLockKeeper):
    def __init__(self, *, target: tuple[str, int], loop_wait: float = 0.25) -> None:
        super().__init__(target=target, loop_wait=loop_wait)
        self._locked = False

    @property
    def has_lock(self) -> bool:
        return self._locked

    def _take_lock(self) -> bool:
        self._locked = True
        return True

    def _keep_lock(self) -> bool:
        return True

    def _release_lock(self) -> None:
        self._locked = False


class _FakeAbstractSyncAsyncCoapClient:
    _DEFAULT_LOOPING_TIMEOUT = 0.25

    def __init__(self, target: tuple[str, int], looping_timeout: float = 0.25) -> None:
        self._target = target
        self._looping_timeout = looping_timeout
        self._locked = False
        self._terminated = False

    def terminate(self) -> None:
        self._terminated = True


class _FakeSyncAsyncCoapClientWithFileLock(_FakeAbstractSyncAsyncCoapClient):
    def __init__(
        self,
        target: tuple[str, int],
        lock_directory: object = None,
        looping_timeout: float = 0.25,
    ) -> None:
        raise RuntimeError(f"lock directory '{lock_directory}' is unavailable")


class _FakeSyncAsyncCoapClientWithDummyLock(_FakeAbstractSyncAsyncCoapClient):
    async def _take_lock(self, context, with_token: bool) -> None:
        _ = context
        _ = with_token
        self._locked = True

    async def _keep_lock(self, context) -> bool:
        _ = context
        return True

    async def _release_lock(self, context, key=None) -> bool:
        _ = context
        _ = key
        self._locked = False
        return True

    def _release_lock_body(self) -> None:
        self._locked = False

    def _check_lock_at_host(self, data: object) -> bool:
        _ = data
        return True

    def _cleanup(self) -> None:
        self._locked = False

    def _cleanup_at_exit(self) -> None:
        return None

    def terminate(self) -> None:
        self._locked = False


def test_apply_patch_does_not_fail_when_quelware_missing(monkeypatch) -> None:
    """Given missing quelware modules, when patch applies, then no exception is raised."""
    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: True)
    import_calls: list[str] = []

    def _raise_import_error(name: str):
        import_calls.append(name)
        raise ImportError(name)

    monkeypatch.setattr(patch.importlib, "import_module", _raise_import_error)

    patch.apply_quelware_filelock_patch()
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

    patch.apply_quelware_filelock_patch()

    assert import_calls == []


def test_apply_patch_replaces_filelock_classes(monkeypatch) -> None:
    """Given quelware modules, when patch applies, then file-lock classes reuse dummy lock."""
    sock_mod = SimpleNamespace(
        AbstractLockKeeper=_FakeAbstractLockKeeper,
        DummyLockKeeper=_FakeDummyLockKeeper,
        FileLockKeeper=_FakeFileLockKeeper,
        _DEFAULT_LOCK_DIRECTORY=object(),
    )
    coap_mod = SimpleNamespace(
        AbstractSyncAsyncCoapClient=_FakeAbstractSyncAsyncCoapClient,
        SyncAsyncCoapClientWithDummyLock=_FakeSyncAsyncCoapClientWithDummyLock,
        SyncAsyncCoapClientWithFileLock=_FakeSyncAsyncCoapClientWithFileLock,
        _DEFAULT_LOCK_DIRECTORY=object(),
    )

    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: True)

    def _import_module(name: str):
        if name == "quel_ic_config.exstickge_sock_client":
            return sock_mod
        if name == "quel_ic_config.exstickge_coap_client":
            return coap_mod
        raise ImportError(name)

    monkeypatch.setattr(patch.importlib, "import_module", _import_module)

    patch.apply_quelware_filelock_patch()

    assert (
        sock_mod.FileLockKeeper.has_lock.fget is sock_mod.DummyLockKeeper.has_lock.fget
    )
    assert sock_mod.FileLockKeeper._take_lock is sock_mod.DummyLockKeeper._take_lock
    assert sock_mod.FileLockKeeper._keep_lock is sock_mod.DummyLockKeeper._keep_lock
    assert (
        sock_mod.FileLockKeeper._release_lock is sock_mod.DummyLockKeeper._release_lock
    )

    sock_client = sock_mod.FileLockKeeper(
        target=("192.168.0.10", 1234), lock_directory="/missing"
    )
    assert sock_client.has_lock is False
    assert sock_client._take_lock() is True
    assert sock_client.has_lock is True
    sock_client._release_lock()
    assert sock_client.has_lock is False

    assert (
        coap_mod.SyncAsyncCoapClientWithFileLock._take_lock
        is coap_mod.SyncAsyncCoapClientWithDummyLock._take_lock
    )
    assert (
        coap_mod.SyncAsyncCoapClientWithFileLock._keep_lock
        is coap_mod.SyncAsyncCoapClientWithDummyLock._keep_lock
    )
    assert (
        coap_mod.SyncAsyncCoapClientWithFileLock._release_lock
        is coap_mod.SyncAsyncCoapClientWithDummyLock._release_lock
    )
    assert (
        coap_mod.SyncAsyncCoapClientWithFileLock._release_lock_body
        is coap_mod.SyncAsyncCoapClientWithDummyLock._release_lock_body
    )
    assert (
        coap_mod.SyncAsyncCoapClientWithFileLock._check_lock_at_host
        is coap_mod.SyncAsyncCoapClientWithDummyLock._check_lock_at_host
    )
    assert (
        coap_mod.SyncAsyncCoapClientWithFileLock._cleanup
        is coap_mod.SyncAsyncCoapClientWithDummyLock._cleanup
    )
    assert (
        coap_mod.SyncAsyncCoapClientWithFileLock._cleanup_at_exit
        is coap_mod.SyncAsyncCoapClientWithDummyLock._cleanup_at_exit
    )

    coap_client = coap_mod.SyncAsyncCoapClientWithFileLock(
        target=("192.168.0.11", 1234),
        lock_directory="/missing",
    )
    assert coap_client._locked is False
    asyncio.run(coap_client._take_lock(context=None, with_token=False))
    assert coap_client._locked is True
    assert asyncio.run(coap_client._keep_lock(context=None)) is True
    assert asyncio.run(coap_client._release_lock(context=None)) is True
    assert coap_client._locked is False
    coap_client.terminate()
    assert coap_client._terminated is True
    assert coap_client._locked is False
