"""Tests for Model hash behavior."""

from __future__ import annotations

from enum import Enum
from typing import ClassVar

from qubex.core import Model, MutableModel


class _FrozenHashModel(Model):
    """Frozen model for hash tests."""

    calls: ClassVar[int] = 0
    value: int

    def to_dict(self) -> dict:
        """Count serialization calls for hash computation checks."""
        type(self).calls += 1
        return super().to_dict()


class _MutableHashModel(MutableModel):
    """Mutable model for hash tests."""

    calls: ClassVar[int] = 0
    value: int

    def to_dict(self) -> dict:
        """Count serialization calls for hash computation checks."""
        type(self).calls += 1
        return super().to_dict()


class _Mode(Enum):
    """Simple enum for serialization tests."""

    A = "A"
    B = "B"


class _EnumModel(Model):
    """Model with enum field for serialization tests."""

    mode: _Mode


def test_model_hash_is_recomputed() -> None:
    """Given frozen models, when reading hash twice, then it is recomputed."""
    _FrozenHashModel.calls = 0
    model = _FrozenHashModel(value=1)

    first = model.hash
    second = model.hash

    assert first == second
    assert _FrozenHashModel.calls == 2


def test_mutable_model_hash_is_not_cached() -> None:
    """Given mutable models, when reading hash twice, then it is recomputed."""
    _MutableHashModel.calls = 0
    model = _MutableHashModel(value=1)

    first = model.hash
    second = model.hash

    assert first == second
    assert _MutableHashModel.calls == 2


def test_mutable_model_hash_changes_after_assignment() -> None:
    """Given mutable models, when fields change, then hash value changes."""
    model = _MutableHashModel(value=1)
    before = model.hash

    model.value = 2
    after = model.hash

    assert before != after


def test_model_json_roundtrip_supports_enum_fields() -> None:
    """Given enum fields, when JSON round-tripped, then values are preserved."""
    original = _EnumModel(mode=_Mode.B)

    payload = original.to_json()
    restored = _EnumModel.from_json(payload)

    assert restored.mode is _Mode.B
