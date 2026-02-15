# ruff: noqa: SLF001

"""Tests for box-cache synchronization in Quel1BackendController."""

from __future__ import annotations

import datetime
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

from qubex.backend.quel1.quel1_backend_controller import Quel1BackendController


class _FakeBoxPool:
    def __init__(self) -> None:
        self._box_config_cache: dict[str, dict[str, Any]] = {}


def _make_controller() -> Quel1BackendController:
    controller = Quel1BackendController()
    cast(Any, controller)._boxpool = _FakeBoxPool()
    return controller


def test_update_box_config_cache_syncs_to_quel1system_config_cache() -> None:
    """Given updates, when cache is updated, then both boxpool and system config_cache are updated."""
    controller = _make_controller()
    system = SimpleNamespace(
        config_cache={"stale": {"ports": {99: {}}}},
        config_fetched_at=None,
    )
    cast(Any, controller)._quel1system = system

    controller.update_box_config_cache({"A": {"ports": {1: {"cnco_freq": 1_500}}}})

    assert controller.boxpool._box_config_cache == {
        "A": {"ports": {1: {"cnco_freq": 1_500}}}
    }
    assert system.config_cache == {
        "stale": {"ports": {99: {}}},
        "A": {"ports": {1: {"cnco_freq": 1_500}}},
    }
    assert isinstance(system.config_fetched_at, datetime.datetime)


def test_replace_box_config_cache_syncs_to_quel1system_config_cache() -> None:
    """Given replacement data, when replacing cache, then both boxpool and system config_cache are replaced."""
    controller = _make_controller()
    system = SimpleNamespace(
        config_cache={"stale": {"ports": {99: {}}}},
        config_fetched_at=None,
    )
    cast(Any, controller)._quel1system = system
    controller.boxpool._box_config_cache = {"old": {"ports": {0: {}}}}
    replacement = {"B": {"ports": {2: {"fnco_freq": 200}}}}

    controller.replace_box_config_cache(replacement)

    assert controller.boxpool._box_config_cache == replacement
    assert system.config_cache == replacement
    assert controller.boxpool._box_config_cache is not replacement
    assert system.config_cache is not replacement
    assert isinstance(system.config_fetched_at, datetime.datetime)


def test_clear_cache_clears_boxpool_and_quel1system_cache() -> None:
    """Given populated caches, when clearing, then boxpool and system caches become empty."""
    controller = _make_controller()
    system = SimpleNamespace(
        config_cache={"A": {"ports": {1: {}}}},
        config_fetched_at=datetime.datetime.now(),
    )
    cast(Any, controller)._quel1system = system
    controller.boxpool._box_config_cache = {"A": {"ports": {1: {}}}}

    controller.clear_cache()

    assert controller.boxpool._box_config_cache == {}
    assert system.config_cache == {}
    assert system.config_fetched_at is None


def test_cache_update_keeps_input_immutable() -> None:
    """Given nested config input, when updating cache, then cache stores deep-copied data."""
    controller = _make_controller()
    system = SimpleNamespace(config_cache={}, config_fetched_at=None)
    cast(Any, controller)._quel1system = system
    incoming = {"A": {"ports": {1: {"cnco_freq": 1_000}}}}
    original = deepcopy(incoming)

    controller.update_box_config_cache(incoming)
    incoming["A"]["ports"][1]["cnco_freq"] = 9_999

    assert controller.boxpool._box_config_cache == original
    assert system.config_cache == original
