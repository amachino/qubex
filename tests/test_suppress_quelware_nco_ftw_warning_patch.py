"""Tests for quelware NCO FTW warning suppression patch."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from qubex.patches.quel_ic_config import (
    suppress_quelware_nco_ftw_warning_patch as patch,
)


class _FakeValidator:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], object]] = []

    def validate_python(self, data: dict[str, Any], self_instance: object) -> None:
        self.calls.append((data, self_instance))
        for key, value in data.items():
            setattr(self_instance, key, value)


class _FakeAbstractNcoFtw:
    __pydantic_validator__ = _FakeValidator()
    ftw: int
    delta_b: int

    def __init__(self, /, **data: Any) -> None:
        warnings.warn(
            "A custom validator is returning a value other than `self`.",
            UserWarning,
            stacklevel=2,
        )
        for key, value in data.items():
            setattr(self, key, value)


def test_apply_patch_does_not_fail_when_quelware_missing(monkeypatch) -> None:
    """Given missing quelware modules, when patch applies, then no exception is raised."""
    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: True)
    import_calls: list[str] = []

    def _raise_import_error(name: str):
        import_calls.append(name)
        raise ImportError(name)

    monkeypatch.setattr(patch.importlib, "import_module", _raise_import_error)

    patch.apply_quelware_nco_ftw_warning_patch()

    assert import_calls == ["quel_ic_config.ad9082_nco"]


def test_apply_patch_skips_for_quelware_0_8(monkeypatch) -> None:
    """Given quelware 0.8.x, when patch applies, then target module is not imported."""
    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: False)
    import_calls: list[str] = []

    def _import_module(name: str):
        import_calls.append(name)
        raise AssertionError("import must not be called for quelware 0.8.x")

    monkeypatch.setattr(patch.importlib, "import_module", _import_module)

    patch.apply_quelware_nco_ftw_warning_patch()

    assert import_calls == []


def test_apply_patch_replaces_nco_ftw_init(monkeypatch) -> None:
    """Given quelware NCO FTW model, when patch applies, then construction should not warn."""
    ad9082_nco_module = SimpleNamespace(AbstractNcoFtw=_FakeAbstractNcoFtw)
    monkeypatch.setattr(patch, "_is_quelware_0_10_or_later", lambda: True)

    def _import_module(name: str):
        if name == "quel_ic_config.ad9082_nco":
            return ad9082_nco_module
        raise ImportError(name)

    monkeypatch.setattr(patch.importlib, "import_module", _import_module)

    patch.apply_quelware_nco_ftw_warning_patch()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = _FakeAbstractNcoFtw(ftw=1, delta_b=0)

    assert caught == []
    assert obj.ftw == 1
    assert obj.delta_b == 0
    assert _FakeAbstractNcoFtw.__pydantic_validator__.calls == [
        ({"ftw": 1, "delta_b": 0}, obj)
    ]


def test_apply_patch_suppresses_real_quelware_warning() -> None:
    """Given real quelware NCO FTW creation, the patch should suppress validator warnings."""
    pytest.importorskip("quel_ic_config.ad9082")

    patch.apply_quelware_nco_ftw_warning_patch()

    from quel_ic_config.ad9082 import NcoFtw

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = NcoFtw.from_frequency(1.0, 1000)

    assert caught == []
    assert isinstance(obj, NcoFtw)
