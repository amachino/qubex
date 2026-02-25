# ruff: noqa: ANN001, SLF001

"""Suppress duplicated quelware proxy errors caused by stale in-process clients."""

from __future__ import annotations

import importlib
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any

logger = logging.getLogger(__name__)


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


def _get_target_address(proxy: Any) -> str | None:
    """Return target host address from one proxy object when available."""
    target = getattr(proxy, "_target", None)
    if isinstance(target, tuple) and len(target) > 0 and isinstance(target[0], str):
        return target[0]
    return None


def _proxy_has_lock(proxy: Any) -> bool:
    """Return whether one proxy currently reports active lock ownership."""
    try:
        return bool(getattr(proxy, "has_lock", False))
    except Exception:
        return False


def _proxy_is_active(proxy: Any) -> bool:
    """Return whether one proxy appears to be actively running."""
    is_alive = getattr(proxy, "is_alive", None)
    if not callable(is_alive):
        # Be conservative when liveness cannot be determined.
        return True
    try:
        return bool(is_alive())
    except Exception:
        return True


def _collect_duplicate_proxies(*, proxies: Any, current: Any) -> list[Any]:
    """Collect locked proxies targeting the same host except the current one."""
    target_address = _get_target_address(current)
    if target_address is None:
        return []

    duplicates: list[Any] = []
    for proxy in tuple(proxies):
        if proxy is current:
            continue
        if _get_target_address(proxy) != target_address:
            continue
        if _proxy_has_lock(proxy):
            duplicates.append(proxy)
    return duplicates


def _release_stale_lockkeeper_proxy(proxy: Any) -> None:
    """Release one stale `AbstractLockKeeper` proxy if possible."""
    if hasattr(proxy, "_to_release"):
        proxy._to_release = True

    deactivate = getattr(proxy, "deactivate", None)
    if callable(deactivate):
        try:
            deactivate()
        except Exception:
            logger.debug("Failed to deactivate stale lockkeeper proxy.", exc_info=True)

    if _proxy_has_lock(proxy):
        release_lock = getattr(proxy, "_release_lock", None)
        if callable(release_lock):
            try:
                release_lock()
            except Exception:
                logger.debug(
                    "Failed to release stale lockkeeper proxy lock.",
                    exc_info=True,
                )


def _release_stale_coap_proxy(proxy: Any) -> None:
    """Release one stale `AbstractSyncAsyncCoapClient` proxy if possible."""
    cleanup = getattr(proxy, "_cleanup", None)
    if callable(cleanup):
        try:
            cleanup()
        except Exception:
            logger.debug("Failed to cleanup stale CoAP proxy.", exc_info=True)

    if hasattr(proxy, "_locked"):
        proxy._locked = False


def _collect_active_duplicate_proxies(*, proxies: Any, current: Any) -> list[Any]:
    """Collect locked and active duplicates for one proxy."""
    return [
        proxy
        for proxy in _collect_duplicate_proxies(proxies=proxies, current=current)
        if _proxy_is_active(proxy)
    ]


def _collect_stale_duplicate_proxies(*, proxies: Any, current: Any) -> list[Any]:
    """Collect locked but inactive duplicates for one proxy."""
    return [
        proxy
        for proxy in _collect_duplicate_proxies(proxies=proxies, current=current)
        if not _proxy_is_active(proxy)
    ]


def _patch_sock_register_self(sock_module: Any) -> None:
    """Patch socket lock keeper duplicate-registration handling."""
    lock_keeper_cls = getattr(sock_module, "AbstractLockKeeper", None)
    if lock_keeper_cls is None:
        return
    if getattr(lock_keeper_cls, "__qubex_duplicate_proxy_patch_applied__", False):
        return

    def _register_self(self) -> None:
        if not self.has_lock:
            raise RuntimeError(
                f"try to register proxy object for {self._target[0]} which doesn't have lock"
            )

        with lock_keeper_cls._clients_lock:
            stale_duplicated = _collect_stale_duplicate_proxies(
                proxies=lock_keeper_cls._clients,
                current=self,
            )

        for proxy in stale_duplicated:
            logger.warning(
                "Detected stale duplicated proxy object for %s; releasing it.",
                self._target[0],
            )
            _release_stale_lockkeeper_proxy(proxy)

        with lock_keeper_cls._clients_lock:
            for proxy in stale_duplicated:
                lock_keeper_cls._clients.discard(proxy)
            active_duplicated = _collect_active_duplicate_proxies(
                proxies=lock_keeper_cls._clients,
                current=self,
            )
            if active_duplicated:
                raise RuntimeError(f"duplicated proxy object for {self._target[0]}")
            lock_keeper_cls._clients.add(self)

    lock_keeper_cls._register_self = _register_self
    lock_keeper_cls.__qubex_duplicate_proxy_patch_applied__ = True


def _patch_coap_register_self(coap_module: Any) -> None:
    """Patch CoAP client duplicate-registration handling."""
    coap_client_cls = getattr(coap_module, "AbstractSyncAsyncCoapClient", None)
    if coap_client_cls is None:
        return
    if getattr(coap_client_cls, "__qubex_duplicate_proxy_patch_applied__", False):
        return

    def _register_self(self) -> None:
        with coap_client_cls._create_lock:
            stale_duplicated = _collect_stale_duplicate_proxies(
                proxies=coap_client_cls._clients,
                current=self,
            )

        for proxy in stale_duplicated:
            logger.warning(
                "Detected stale duplicated proxy object for %s; releasing it.",
                self._target[0],
            )
            _release_stale_coap_proxy(proxy)

        with coap_client_cls._create_lock:
            for proxy in stale_duplicated:
                coap_client_cls._clients.discard(proxy)
            active_duplicated = _collect_active_duplicate_proxies(
                proxies=coap_client_cls._clients,
                current=self,
            )
            if active_duplicated:
                raise RuntimeError(f"duplicated proxy object for {self._target[0]}")
            coap_client_cls._clients.add(self)

    coap_client_cls._register_self = _register_self
    coap_client_cls.__qubex_duplicate_proxy_patch_applied__ = True


def apply_quelware_duplicated_proxy_patch() -> None:
    """Apply duplicate-proxy suppression patch for quelware 0.10+."""
    if not _is_quelware_0_10_or_later():
        return

    try:
        sock_module = importlib.import_module("quel_ic_config.exstickge_sock_client")
    except ImportError:
        sock_module = None
    if sock_module is not None:
        _patch_sock_register_self(sock_module)

    try:
        coap_module = importlib.import_module("quel_ic_config.exstickge_coap_client")
    except ImportError:
        coap_module = None
    if coap_module is not None:
        _patch_coap_register_self(coap_module)


apply_quelware_duplicated_proxy_patch()
