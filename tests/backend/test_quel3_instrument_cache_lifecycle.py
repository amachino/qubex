"""Tests for QuEL-3 instrument deployment and explicit cache refresh."""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol
from qubex.backend.quel3.managers import (
    Quel3ConfigurationManager,
    Quel3HardwareStateReader,
)
from qubex.backend.quel3.models import InstrumentConfiguration, InstrumentSpec


def _info(alias: str, resource_id: str, port_id: str) -> InstrumentInfoProtocol:
    return cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id=resource_id,
            port_id=port_id,
            definition=SimpleNamespace(
                alias=alias,
                role="TRANSMITTER",
                mode="FIXED_TIMELINE",
                profile=SimpleNamespace(
                    frequency_range_min=4.0e9,
                    frequency_range_max=6.0e9,
                ),
            ),
            config=SimpleNamespace(sampling_period_fs=400_000),
        ),
    )


def _specification(
    alias: str = "Q00", port_id: str = "unit-a:tx_p01"
) -> InstrumentSpec:
    return InstrumentSpec(
        port_id=port_id,
        alias=alias,
        role="TRANSMITTER",
        frequency_range_min_hz=4.0e9,
        frequency_range_max_hz=6.0e9,
    )


class _ConfigurationManager(Quel3ConfigurationManager):
    def __init__(self, calls: list[object], *, fail: bool = False) -> None:
        super().__init__()
        self.calls = calls
        self.fail = fail

    async def _deploy_instruments(
        self,
        *,
        specifications: tuple[InstrumentSpec, ...],
        append: bool = False,
        parallel: bool = True,
    ) -> None:
        assert append is False
        self.calls.append(("deploy", specifications, parallel))
        if self.fail:
            raise RuntimeError("deploy failed")


class _Reader:
    def __init__(
        self,
        calls: list[object],
        infos: tuple[InstrumentInfoProtocol, ...],
        *,
        fail: bool = False,
    ) -> None:
        self.calls = calls
        self.infos = infos
        self.fail = fail

    def read_instrument_infos(
        self,
        *,
        unit_labels: Sequence[str] = (),
        port_ids: Sequence[str] = (),
        parallel: bool = True,
    ) -> tuple[InstrumentInfoProtocol, ...]:
        self.calls.append(("read", tuple(unit_labels), tuple(port_ids), parallel))
        if self.fail:
            raise RuntimeError("read failed")
        return self.infos


def _reader(
    calls: list[object],
    infos: tuple[InstrumentInfoProtocol, ...],
    *,
    fail: bool = False,
) -> Quel3HardwareStateReader:
    return cast(Quel3HardwareStateReader, _Reader(calls, infos, fail=fail))


@pytest.mark.parametrize("source", ["memory", "yaml"])
def test_deploy_reads_hardware_after_write_and_preserves_other_ports(
    source: str, tmp_path: Path
) -> None:
    """Deploy should replace listed ports from memory or YAML using fresh hardware infos."""
    calls: list[object] = []
    specification = _specification()
    configuration = InstrumentConfiguration(instruments=(specification,))
    manager = _ConfigurationManager(calls)
    if source == "yaml":
        path = configuration.save_yaml(tmp_path / "instruments.yaml")
        configuration = manager.load_instrument_configuration(path)
    actual = _info("unit-a:Q00", "unit-a:new", specification.port_id)
    other = _info("Q01", "unit-b:other", "unit-b:tx_p01")
    cache = InstrumentCache()
    cache.replace_all(
        instrument_infos=(
            _info("obsolete", "unit-a:old", specification.port_id),
            other,
        )
    )

    result = manager.deploy_instruments(
        configuration=configuration,
        instrument_cache=cache,
        hardware_state_reader=_reader(calls, (actual,)),
        parallel=False,
    )

    assert calls == [
        ("deploy", (specification,), False),
        ("read", (), (specification.port_id,), False),
    ]
    assert result == {"Q00": actual}
    assert cache.get("Q00") is actual
    assert cache.snapshot() == {"Q00": actual, "Q01": other}


@pytest.mark.parametrize(
    "failure", ["deploy", "read", "missing", "duplicate", "wrong-port", "conflict"]
)
def test_failed_deploy_does_not_retain_stale_resource_ids(failure: str) -> None:
    """Write or readback failures should leave touched ports absent and others intact."""
    calls: list[object] = []
    specification = _specification()
    other = _info("Q01", "unit-b:other", "unit-b:tx_p01")
    actual = _info("Q00", "unit-a:new", specification.port_id)
    infos = {
        "missing": (),
        "duplicate": (actual, _info("Q00", "unit-a:duplicate", specification.port_id)),
        "wrong-port": (_info("Q00", "unit-a:new", "unit-a:tx_p02"),),
        "conflict": (actual, _info("Q01", "unit-a:conflict", specification.port_id)),
    }.get(failure, (actual,))
    manager = _ConfigurationManager(calls, fail=failure == "deploy")
    cache = InstrumentCache()
    cache.replace_all(
        instrument_infos=(_info("Q00", "unit-a:old", specification.port_id), other)
    )

    with pytest.raises((RuntimeError, ValueError)):
        manager.deploy_instruments(
            configuration=InstrumentConfiguration(instruments=(specification,)),
            instrument_cache=cache,
            hardware_state_reader=_reader(calls, infos, fail=failure == "read"),
        )

    assert cache.snapshot() == {"Q01": other}


def test_empty_deploy_does_not_clear_cache_or_contact_hardware() -> None:
    """An empty configuration should leave hardware and cache unchanged."""
    calls: list[object] = []
    info = _info("Q00", "unit-a:old", "unit-a:tx_p01")
    manager = _ConfigurationManager(calls)
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(info,))

    result = manager.deploy_instruments(
        configuration=InstrumentConfiguration(instruments=()),
        instrument_cache=cache,
        hardware_state_reader=_reader(calls, ()),
    )

    assert result == {}
    assert cache.snapshot() == {"Q00": info}
    assert calls == []


def test_refresh_selected_unit_preserves_other_units() -> None:
    """Refreshing a selected unit should remove absent instruments only in that unit."""
    calls: list[object] = []
    other = _info("Q01", "unit-b:other", "unit-b:tx_p01")
    cache = InstrumentCache()
    cache.replace_all(
        instrument_infos=(_info("Q00", "unit-a:old", "unit-a:tx_p01"), other)
    )

    result = _ConfigurationManager(calls).refresh_instrument_cache(
        unit_labels=("unit-a",),
        instrument_cache=cache,
        hardware_state_reader=_reader(calls, ()),
    )

    assert result == {}
    assert cache.snapshot() == {"Q01": other}
    assert calls == [("read", ("unit-a",), (), True)]


@pytest.mark.parametrize("include_instrument", [False, True])
def test_refresh_all_replaces_complete_cache(include_instrument: bool) -> None:
    """An all-unit refresh should replace the entire cache with the acquired scope."""
    calls: list[object] = []
    old = _info("Q00", "unit-a:old", "unit-a:tx_p01")
    new = _info("unit-b:Q01", "unit-b:new", "unit-b:tx_p02")
    infos = (new,) if include_instrument else ()
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(old,))

    result = _ConfigurationManager(calls).refresh_instrument_cache(
        instrument_cache=cache,
        hardware_state_reader=_reader(calls, infos),
        parallel=False,
    )

    assert result == ({"Q01": new} if include_instrument else {})
    assert cache.snapshot() == result
    assert calls == [("read", (), (), False)]


@pytest.mark.parametrize("failure", ["read", "duplicate", "out-of-scope"])
def test_failed_refresh_retains_previous_complete_snapshot(failure: str) -> None:
    """Read or validation failure should preserve the previous complete cache."""
    calls: list[object] = []
    info = _info("Q00", "unit-a:old", "unit-a:tx_p01")
    other = _info("Q01", "unit-b:other", "unit-b:tx_p01")
    infos = {
        "read": (),
        "duplicate": (_info("Q01", "unit-a:new", "unit-a:tx_p01"),),
        "out-of-scope": (_info("Q02", "unit-c:new", "unit-c:tx_p01"),),
    }[failure]
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(info, other))

    with pytest.raises((RuntimeError, ValueError)):
        _ConfigurationManager(calls).refresh_instrument_cache(
            unit_labels=("unit-a",),
            instrument_cache=cache,
            hardware_state_reader=_reader(calls, infos, fail=failure == "read"),
        )

    assert cache.snapshot() == {"Q00": info, "Q01": other}


def test_empty_unit_selection_does_not_refresh_all_units() -> None:
    """An explicitly empty unit selection should not become an unscoped refresh."""
    calls: list[object] = []
    cache = InstrumentCache()

    result = _ConfigurationManager(calls).refresh_instrument_cache(
        unit_labels=(),
        instrument_cache=cache,
        hardware_state_reader=_reader(calls, ()),
    )

    assert result == {}
    assert calls == []


def test_configuration_export_and_yaml_round_trip_do_not_refresh_cache(
    tmp_path: Path,
) -> None:
    """Export and YAML round trips should use cached specifications without hardware reads."""
    calls: list[object] = []
    info = _info("unit-a:Q00", "unit-a:live", "unit-a:tx_p01")
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(info,))
    manager = _ConfigurationManager(calls)

    configuration = manager.get_instrument_configuration(instrument_cache=cache)
    path = manager.save_instrument_configuration(
        tmp_path / "instruments.yaml", instrument_cache=cache
    )
    loaded = manager.load_instrument_configuration(path)

    assert configuration == InstrumentConfiguration(instruments=(_specification(),))
    assert loaded == configuration
    assert path == tmp_path / "instruments.yaml"
    assert cache.get("Q00") is info
    assert calls == []


def test_loading_configuration_does_not_publish_instrument_state(
    tmp_path: Path,
) -> None:
    """Loading a deployable configuration should leave existing runtime infos unchanged."""
    calls: list[object] = []
    info = _info("Q00", "unit-a:live", "unit-a:tx_p01")
    cache = InstrumentCache()
    cache.replace_all(instrument_infos=(info,))
    expected = InstrumentConfiguration(
        instruments=(_specification(alias="Q02", port_id="unit-b:tx_p02"),)
    )
    path = expected.save_yaml(tmp_path / "instruments.yaml")

    loaded = _ConfigurationManager(calls).load_instrument_configuration(path)

    assert loaded == expected
    assert cache.snapshot() == {"Q00": info}
    assert calls == []
