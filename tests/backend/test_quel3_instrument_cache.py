"""Behavior tests for the shared QuEL-3 instrument cache."""

from types import SimpleNamespace
from typing import cast

import pytest

from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol


def _info(alias: str, resource_id: str, port_id: str) -> InstrumentInfoProtocol:
    return cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id=resource_id,
            port_id=port_id,
            definition=SimpleNamespace(alias=alias, role="TRANSMITTER"),
            config=SimpleNamespace(sampling_period_fs=400_000),
        ),
    )


@pytest.mark.parametrize("alias", ["unit-a:Q00", "unit-a: Q00 ", " Q00 "])
def test_cache_keeps_complete_hardware_objects(alias: str) -> None:
    """The cache should retain the original hardware info under its local alias."""
    info = _info(alias, "unit-a:inst-1", "unit-a:tx_p01")
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(info,))

    assert cache.get("Q00") is info
    assert cache.get("Q00").id == "unit-a:inst-1"
    snapshot = cache.snapshot()
    snapshot.clear()
    assert cache.get("Q00") is info


def test_replace_ports_removes_old_instruments_and_keeps_other_ports() -> None:
    """Port replacement should remove obsolete aliases and preserve other ports."""
    old = _info("Q00", "unit-a:old", "unit-a:tx_p01")
    other = _info("Q01", "unit-b:other", "unit-b:tx_p02")
    new = _info("Q02", "unit-a:new", "unit-a:tx_p01")
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(old, other))

    cache.replace_ports(port_ids=(old.port_id,), instrument_infos=(new,))

    assert cache.snapshot() == {"Q01": other, "Q02": new}
    cache.replace_ports(port_ids=(new.port_id,), instrument_infos=())
    assert cache.snapshot() == {"Q01": other}


def test_replace_units_removes_missing_ports() -> None:
    """Unit replacement should also remove instruments on vanished ports."""
    first = _info("Q00", "unit-a:1", "unit-a:tx_p01")
    second = _info("Q01", "unit-a:2", "unit-a:tx_p02")
    other = _info("Q02", "unit-b:3", "unit-b:tx_p01")
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(first, second, other))

    cache.replace_units(unit_labels=("unit-a",), instrument_infos=(first,))

    assert cache.snapshot() == {"Q00": first, "Q02": other}


def test_duplicate_alias_rejects_replacement_atomically() -> None:
    """A duplicate local alias should fail without overwriting unrelated entries."""
    first = _info("Q00", "unit-a:1", "unit-a:tx_p01")
    duplicate = _info("unit-b:Q00", "unit-b:2", "unit-b:tx_p01")
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(first,))

    with pytest.raises(ValueError, match=r"Duplicate.*Q00"):
        cache.replace_ports(
            port_ids=(duplicate.port_id,), instrument_infos=(duplicate,)
        )

    assert cache.snapshot() == {"Q00": first}


@pytest.mark.parametrize("allow_duplicate_aliases", [False, True])
def test_alias_overwrite_does_not_hide_invalid_resource_ids(
    allow_duplicate_aliases: bool,
) -> None:
    """Alias overwrite should still reject repeated resource IDs atomically."""
    old = _info("old", "unit-a:old", "unit-a:p0")
    first = _info("Q00", "unit-a:1", "unit-a:p0")
    second = _info("Q01", "unit-a:1", "unit-a:p1")
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(old,))

    with pytest.raises(ValueError, match="Duplicate instrument resource ID"):
        cache.replace_all(
            instrument_infos=(first, second),
            allow_duplicate_aliases=allow_duplicate_aliases,
        )

    assert cache.snapshot() == {"old": old}


@pytest.mark.parametrize(
    ("alias", "resource_id", "port_id"),
    [
        ("", "unit-a:1", "unit-a:tx_p01"),
        ("unit-a:   ", "unit-a:1", "unit-a:tx_p01"),
        ("Q00", "", "unit-a:tx_p01"),
        ("Q00", "unit-a:1", ""),
    ],
)
def test_invalid_hardware_info_does_not_replace_cache(
    alias: str, resource_id: str, port_id: str
) -> None:
    """Incomplete hardware identities should fail before publishing a replacement."""
    valid = _info("Q01", "unit-b:1", "unit-b:tx_p01")
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(valid,))

    with pytest.raises(ValueError, match="must not be empty"):
        cache.replace_all(instrument_infos=(_info(alias, resource_id, port_id),))

    assert cache.snapshot() == {"Q01": valid}


def test_port_replacement_rejects_out_of_scope_info() -> None:
    """A scoped replacement should reject hardware info from another port."""
    cache = InstrumentCache()
    with pytest.raises(ValueError, match="outside"):
        cache.replace_ports(
            port_ids=("unit-a:tx_p01",),
            instrument_infos=(_info("Q00", "unit-b:1", "unit-b:tx_p01"),),
        )
    assert cache.snapshot() == {}


def test_missing_alias_explains_explicit_refresh() -> None:
    """An uncached alias should explain how to populate the cache before execution."""
    with pytest.raises(ValueError, match=r"Q00.*deploy.*refresh"):
        InstrumentCache().get("Q00")


def test_hash_changes_when_same_alias_gets_new_resource_id() -> None:
    """Redeploying an alias with a new resource ID should change the cache hash."""
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(_info("Q00", "unit-a:old", "unit-a:tx_p01"),))
    previous_hash = cache.hash
    cache.replace_all(instrument_infos=(_info("Q00", "unit-a:new", "unit-a:tx_p01"),))
    assert cache.hash != previous_hash
    cache.clear()
    assert cache.snapshot() == {}


def test_export_uses_observed_definitions_without_runtime_identity() -> None:
    """Export should preserve deployable values while omitting hardware identities."""
    info = SimpleNamespace(
        id="unit-a:inst-1",
        port_id="unit-a:tx_p01",
        definition=SimpleNamespace(
            alias="unit-a:Q00",
            role=SimpleNamespace(name="TRANSMITTER"),
            mode=SimpleNamespace(name="FIXED_TIMELINE"),
            profile=SimpleNamespace(
                frequency_range_min=4.1e9, frequency_range_max=4.3e9
            ),
        ),
        config=SimpleNamespace(sampling_period_fs=400_000),
    )
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(cast(InstrumentInfoProtocol, info),))

    configuration = cache.export_configuration()

    assert configuration.model_dump(mode="json") == {
        "instruments": [
            {
                "port_id": "unit-a:tx_p01",
                "alias": "Q00",
                "role": "TRANSMITTER",
                "frequency_range_min_hz": 4.1e9,
                "frequency_range_max_hz": 4.3e9,
            }
        ]
    }
    info.definition.profile.frequency_range_min = 4.2e9
    assert configuration.instruments[0].frequency_range_min_hz == 4.1e9
    assert cache.export_configuration().instruments[0].frequency_range_min_hz == 4.2e9
    assert cache.get("Q00") is info


@pytest.mark.parametrize(
    ("mode", "role", "profile"),
    [
        (
            None,
            "TRANSMITTER",
            SimpleNamespace(frequency_range_min=4e9, frequency_range_max=6e9),
        ),
        (
            "UNSPECIFIED",
            "TRANSMITTER",
            SimpleNamespace(frequency_range_min=4e9, frequency_range_max=6e9),
        ),
        (
            "FIXED_TIMELINE",
            "UNSPECIFIED",
            SimpleNamespace(frequency_range_min=4e9, frequency_range_max=6e9),
        ),
        ("FIXED_TIMELINE", "TRANSMITTER", None),
        (
            "FIXED_TIMELINE",
            "TRANSMITTER",
            SimpleNamespace(frequency_range_min=None, frequency_range_max=6e9),
        ),
    ],
)
def test_export_rejects_unrepresentable_information_without_changing_cache(
    mode: object, role: object, profile: object
) -> None:
    """Export should reject incomplete definitions without changing cached information."""
    info = cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id="unit-a:inst-1",
            port_id="unit-a:tx_p01",
            definition=SimpleNamespace(
                alias="Q00", mode=mode, role=role, profile=profile
            ),
        ),
    )
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(info,))

    with pytest.raises(ValueError, match="Q00"):
        cache.export_configuration()

    assert cache.get("Q00") is info
