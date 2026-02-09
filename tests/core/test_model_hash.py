"""Tests for Model hash behavior."""

from __future__ import annotations

from typing import ClassVar

from qubex.core import Model, MutableModel


class _FrozenHashModel(Model):
    """Frozen model for hash tests."""

    calls: ClassVar[int] = 0
    value: int

    def _canonical_hash_bytes(self) -> bytes:
        """Count canonical serialization calls for cache checks."""
        type(self).calls += 1
        return super()._canonical_hash_bytes()


class _MutableHashModel(MutableModel):
    """Mutable model for hash tests."""

    calls: ClassVar[int] = 0
    value: int

    def _canonical_hash_bytes(self) -> bytes:
        """Count canonical serialization calls for cache checks."""
        type(self).calls += 1
        return super()._canonical_hash_bytes()


def test_model_hash_is_cached() -> None:
    """Given frozen models, when reading hash twice, then it is computed once."""
    _FrozenHashModel.calls = 0
    model = _FrozenHashModel(value=1)

    first = model.hash
    second = model.hash

    assert first == second
    assert _FrozenHashModel.calls == 1


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
