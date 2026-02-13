# ruff: noqa: ANN001, SLF001

"""Disable quelware file-lock implementations that create lock files."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def _is_quelware_0_10_or_later() -> bool:
    """Return whether installed `quel_ic_config` is 0.10 or later."""
    try:
        v = version("quel_ic_config")
    except PackageNotFoundError:
        return False

    parts = v.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return False

    major, minor = int(parts[0]), int(parts[1])
    return major > 0 or (major == 0 and minor >= 10)


def _patch_sock_file_lockkeeper(sock_module: Any) -> None:
    """Patch socket file-lock keeper to in-memory dummy lock behavior."""
    file_lock_keeper_cls = getattr(sock_module, "FileLockKeeper", None)
    abstract_lock_keeper_cls = getattr(sock_module, "AbstractLockKeeper", None)
    if file_lock_keeper_cls is None or abstract_lock_keeper_cls is None:
        return
    if getattr(file_lock_keeper_cls, "__qubex_lockfile_patch_applied__", False):
        return

    default_lock_dir = getattr(sock_module, "_DEFAULT_LOCK_DIRECTORY", None)

    def _init(
        self,
        *,
        target: tuple[str, int],
        loop_wait: float = abstract_lock_keeper_cls._DEFAULT_LOOP_WAIT,
        lock_directory: Any = default_lock_dir,
    ) -> None:
        _ = lock_directory
        abstract_lock_keeper_cls.__init__(self, target=target, loop_wait=loop_wait)
        self._locked = False

    def _has_lock(self) -> bool:
        return bool(getattr(self, "_locked", False))

    def _take_lock(self) -> bool:
        self._locked = True
        return True

    def _keep_lock(self) -> bool:
        return True

    def _release_lock(self) -> None:
        self._locked = False

    file_lock_keeper_cls.__init__ = _init
    file_lock_keeper_cls.has_lock = property(_has_lock)
    file_lock_keeper_cls._take_lock = _take_lock
    file_lock_keeper_cls._keep_lock = _keep_lock
    file_lock_keeper_cls._release_lock = _release_lock
    file_lock_keeper_cls.__qubex_lockfile_patch_applied__ = True


def _patch_coap_file_lock_client(coap_module: Any) -> None:
    """Patch CoAP file-lock client to in-memory dummy lock behavior."""
    file_lock_client_cls = getattr(coap_module, "SyncAsyncCoapClientWithFileLock", None)
    abstract_coap_client_cls = getattr(coap_module, "AbstractSyncAsyncCoapClient", None)
    if file_lock_client_cls is None or abstract_coap_client_cls is None:
        return
    if getattr(file_lock_client_cls, "__qubex_lockfile_patch_applied__", False):
        return

    default_lock_dir = getattr(coap_module, "_DEFAULT_LOCK_DIRECTORY", None)

    def _init(
        self,
        target: tuple[str, int],
        lock_directory: Any = default_lock_dir,
        looping_timeout: float = abstract_coap_client_cls._DEFAULT_LOOPING_TIMEOUT,
    ) -> None:
        _ = lock_directory
        abstract_coap_client_cls.__init__(
            self, target=target, looping_timeout=looping_timeout
        )

    async def _take_lock(self, context: Any, with_token: bool) -> None:
        _ = context
        _ = with_token
        self._locked = True

    async def _keep_lock(self, context: Any) -> bool:
        _ = context
        if not self._locked:
            await _take_lock(self, context, False)
        return True

    async def _release_lock(self, context: Any, key: Any = None) -> bool:
        _ = context
        _ = key
        self._release_lock_body()
        return True

    def _release_lock_body(self) -> None:
        self._locked = False

    def _check_lock_at_host(self, data: Any) -> bool:
        _ = data
        return True

    def _cleanup(self) -> None:
        self._release_lock_body()

    def _cleanup_at_exit(self) -> None:
        return None

    def _terminate(self) -> None:
        abstract_coap_client_cls.terminate(self)
        self._release_lock_body()

    file_lock_client_cls.__init__ = _init
    file_lock_client_cls._take_lock = _take_lock
    file_lock_client_cls._keep_lock = _keep_lock
    file_lock_client_cls._release_lock = _release_lock
    file_lock_client_cls._release_lock_body = _release_lock_body
    file_lock_client_cls._check_lock_at_host = _check_lock_at_host
    file_lock_client_cls._cleanup = _cleanup
    file_lock_client_cls._cleanup_at_exit = _cleanup_at_exit
    file_lock_client_cls.terminate = _terminate
    file_lock_client_cls.__qubex_lockfile_patch_applied__ = True


def apply_quelware_filelock_patch() -> None:
    """Apply lockfile-disabling monkey patch for quelware 0.10+ when available."""
    if not _is_quelware_0_10_or_later():
        return

    try:
        sock_module = importlib.import_module("quel_ic_config.exstickge_sock_client")
    except ImportError:
        sock_module = None
    if sock_module is not None:
        _patch_sock_file_lockkeeper(sock_module)

    try:
        coap_module = importlib.import_module("quel_ic_config.exstickge_coap_client")
    except ImportError:
        coap_module = None
    if coap_module is not None:
        _patch_coap_file_lock_client(coap_module)


apply_quelware_filelock_patch()
