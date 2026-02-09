"""Tests for JSON serializer helpers."""

from __future__ import annotations

from qxcore.serialization import to_canonical_json


def test_to_canonical_json_sorts_keys_and_compacts_output() -> None:
    """Canonical JSON should be stable and compact."""
    payload = {"b": 1, "a": {"d": 2, "c": 3}}

    assert to_canonical_json(payload) == '{"a":{"c":3,"d":2},"b":1}'
