"""Tests for QuEL-3 hardware state collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from qubex.backend.quel3.interfaces import QuelwareClientFactory
from qubex.backend.quel3.managers import Quel3HardwareStateReader


@dataclass(frozen=True)
class _Category:
    name: str


@dataclass(frozen=True)
class _Role:
    name: str


@dataclass(frozen=True)
class _Mode:
    name: str


@dataclass(frozen=True)
class _Profile:
    frequency_range_min: float | None = None
    frequency_range_max: float | None = None


@dataclass(frozen=True)
class _Config:
    sampling_period_fs: int | None = None
    bitdepth: int | None = None
    timeline_step_samples: int | None = None
    samples_per_tick: int | None = None


@dataclass(frozen=True)
class _Definition:
    alias: str
    role: object
    mode: object | None = None
    profile: object | None = None


@dataclass(frozen=True)
class _ResourceInfo:
    id: str
    category: object


@dataclass(frozen=True)
class _PortInfo:
    id: str
    role: object
    depends_on: list[str]


@dataclass(frozen=True)
class _UnitControlSpec:
    key: str
    allowed_values: tuple[str, ...]
    current_value: str


@dataclass(frozen=True)
class _UnitConfiguration:
    supported: tuple[_UnitControlSpec, ...]

    def values(self) -> dict[str, str]:
        return {spec.key: spec.current_value for spec in self.supported}


@dataclass(frozen=True)
class _InstrumentInfo:
    id: str
    port_id: str
    definition: _Definition
    config: _Config


class _FakeClient:
    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        _ = (exc_type, exc, tb)

    def list_unit_labels(self) -> list[str]:
        return ["unit-a", "unit-b"]

    async def list_resource_infos(self) -> list[_ResourceInfo]:
        return [
            _ResourceInfo("unit-a:tx_p01", _Category("PORT")),
            _ResourceInfo("unit-a:inst-q00", _Category("INSTRUMENT")),
            _ResourceInfo("unit-b:inst-q01", _Category("INSTRUMENT")),
        ]

    async def get_port_info(self, resource_id: str) -> _PortInfo:
        if resource_id != "unit-a:tx_p01":
            raise KeyError(resource_id)
        return _PortInfo(
            id="unit-a:tx_p01",
            role=_Role("TX"),
            depends_on=["unit-a:clock"],
        )

    async def get_instrument_info(self, resource_id: str) -> _InstrumentInfo:
        if resource_id == "unit-a:inst-q00":
            return _InstrumentInfo(
                id="unit-a:inst-q00",
                port_id="unit-a:tx_p01",
                definition=_Definition(
                    alias="unit-a:Q00",
                    role=_Role("TRANSMITTER"),
                    mode=_Mode("FIXED_TIMELINE"),
                    profile=_Profile(
                        frequency_range_min=4.1e9,
                        frequency_range_max=4.3e9,
                    ),
                ),
                config=_Config(
                    sampling_period_fs=400_000,
                    bitdepth=16,
                    timeline_step_samples=4,
                    samples_per_tick=2,
                ),
            )
        return _InstrumentInfo(
            id="unit-b:inst-q01",
            port_id="unit-b:tx_p02",
            definition=_Definition(
                alias="Q01",
                role=_Role("TRANSMITTER"),
                mode=_Mode("FIXED_TIMELINE"),
                profile=_Profile(
                    frequency_range_min=5.0e9,
                    frequency_range_max=5.2e9,
                ),
            ),
            config=_Config(),
        )

    async def dump_port_state(self, port_id: str) -> str:
        return f"state: {port_id}"

    async def get_unit_configuration(self, unit_label: str) -> _UnitConfiguration:
        return _UnitConfiguration(
            supported=(
                _UnitControlSpec(
                    key="quel3.monitor.mode",
                    allowed_values=("disabled", "loopback"),
                    current_value=(
                        "loopback" if unit_label == "unit-a" else "disabled"
                    ),
                ),
            )
        )


class _UnqualifiedInstrumentResourceClient(_FakeClient):
    async def list_resource_infos(self) -> list[_ResourceInfo]:
        return [
            _ResourceInfo("unit-a:tx_p01", _Category("PORT")),
            _ResourceInfo("inst-q00", _Category("INSTRUMENT")),
        ]

    async def get_instrument_info(self, resource_id: str) -> _InstrumentInfo:
        return _InstrumentInfo(
            id="inst-q00",
            port_id="unit-a:tx_p01",
            definition=_Definition(
                alias="Q00",
                role=_Role("TRANSMITTER"),
                mode=_Mode("FIXED_TIMELINE"),
                profile=_Profile(
                    frequency_range_min=4.1e9,
                    frequency_range_max=4.3e9,
                ),
            ),
            config=_Config(),
        )


class _UnqualifiedOtherUnitResourceClient(_FakeClient):
    async def list_resource_infos(self) -> list[_ResourceInfo]:
        return [
            _ResourceInfo("unit-a:tx_p01", _Category("PORT")),
            _ResourceInfo("port-other", _Category("PORT")),
            _ResourceInfo("inst-other", _Category("INSTRUMENT")),
        ]

    async def get_port_info(self, resource_id: str) -> _PortInfo:
        if resource_id == "unit-a:tx_p01":
            return await super().get_port_info(resource_id)
        return _PortInfo(
            id="unit-b:tx_p02",
            role=_Role("TX"),
            depends_on=[],
        )

    async def get_instrument_info(self, resource_id: str) -> _InstrumentInfo:
        return _InstrumentInfo(
            id=resource_id,
            port_id="unit-b:tx_p02",
            definition=_Definition(
                alias="Q00",
                role=_Role("TRANSMITTER"),
                mode=_Mode("FIXED_TIMELINE"),
                profile=_Profile(
                    frequency_range_min=5.0e9,
                    frequency_range_max=5.2e9,
                ),
            ),
            config=_Config(),
        )


class _BoxLocalAliasClient(_FakeClient):
    async def list_resource_infos(self) -> list[_ResourceInfo]:
        return [
            _ResourceInfo("unit-a:tx_p01", _Category("PORT")),
            _ResourceInfo("unit-b:tx_p01", _Category("PORT")),
            _ResourceInfo("unit-a:inst-q00", _Category("INSTRUMENT")),
            _ResourceInfo("unit-b:inst-q00", _Category("INSTRUMENT")),
        ]

    async def get_port_info(self, resource_id: str) -> _PortInfo:
        return _PortInfo(
            id=resource_id,
            role=_Role("TX"),
            depends_on=[],
        )

    async def get_instrument_info(self, resource_id: str) -> _InstrumentInfo:
        unit_label = resource_id.split(":", maxsplit=1)[0]
        return _InstrumentInfo(
            id=resource_id,
            port_id=f"{unit_label}:tx_p01",
            definition=_Definition(
                alias="Q00",
                role=_Role("TRANSMITTER"),
                mode=_Mode("FIXED_TIMELINE"),
                profile=_Profile(
                    frequency_range_min=4.1e9,
                    frequency_range_max=4.3e9,
                ),
            ),
            config=_Config(),
        )


class _MultiInstrumentClient(_FakeClient):
    async def list_resource_infos(self) -> list[_ResourceInfo]:
        return [
            _ResourceInfo("unit-a:tx_p01", _Category("PORT")),
            _ResourceInfo("unit-a:rx_p02", _Category("PORT")),
            _ResourceInfo("unit-b:tx_p01", _Category("PORT")),
            _ResourceInfo("unit-a:inst-q00", _Category("INSTRUMENT")),
            _ResourceInfo("unit-a:inst-q01", _Category("INSTRUMENT")),
            _ResourceInfo("unit-b:inst-q00", _Category("INSTRUMENT")),
        ]

    async def get_port_info(self, resource_id: str) -> _PortInfo:
        return _PortInfo(
            id=resource_id,
            role=_Role("TX") if "tx_" in resource_id else _Role("RX"),
            depends_on=[],
        )

    async def get_instrument_info(self, resource_id: str) -> _InstrumentInfo:
        definitions = {
            "unit-a:inst-q00": ("unit-a:tx_p01", "unit-a:Q00"),
            "unit-a:inst-q01": ("unit-a:rx_p02", "Q01"),
            "unit-b:inst-q00": ("unit-b:tx_p01", "Q00"),
        }
        port_id, alias = definitions[resource_id]
        return _InstrumentInfo(
            id=resource_id,
            port_id=port_id,
            definition=_Definition(
                alias=alias,
                role=_Role("TRANSMITTER"),
                mode=_Mode("FIXED_TIMELINE"),
                profile=_Profile(
                    frequency_range_min=4.1e9,
                    frequency_range_max=4.3e9,
                ),
            ),
            config=_Config(),
        )


class _MonitorPortClient(_FakeClient):
    def __init__(self) -> None:
        self.get_port_info_calls: list[str] = []

    async def list_resource_infos(self) -> list[_ResourceInfo]:
        return [
            _ResourceInfo("unit-a:tx_p01", _Category("PORT")),
            _ResourceInfo("unit-a:mon", _Category("PORT")),
        ]

    async def get_port_info(self, resource_id: str) -> _PortInfo:
        self.get_port_info_calls.append(resource_id)
        if resource_id == "unit-a:mon":
            raise RuntimeError("monitor port does not support get_port_info")
        return await super().get_port_info(resource_id)


class _FakeHardwareStateReader(Quel3HardwareStateReader):
    def __init__(self, client: _FakeClient) -> None:
        super().__init__()
        self._client = client

    def _load_quelware_client_factory(self) -> QuelwareClientFactory:
        """Return a fake quelware client factory for tests."""
        return cast(QuelwareClientFactory, lambda endpoint, port: self._client)


def _make_reader(client: _FakeClient) -> Quel3HardwareStateReader:
    return _FakeHardwareStateReader(client)


def test_collect_state_normalizes_units_ports_and_instruments() -> None:
    """Given quelware resources, hardware state should expose normalized Qubex data."""
    client = _FakeClient()
    reader = _make_reader(client)

    state = reader.collect_state(unit_labels=("unit-a",), parallel=False)

    assert [unit.label for unit in state.units] == ["unit-a"]
    assert [port.id for port in state.ports] == ["unit-a:tx_p01"]
    assert state.ports[0].role == "TX"
    assert state.ports[0].depends_on == ("unit-a:clock",)
    assert [instrument.id for instrument in state.instruments] == ["unit-a:inst-q00"]
    instrument = state.instruments[0]
    assert instrument.alias == "unit-a:Q00"
    assert instrument.normalized_alias == "Q00"
    assert instrument.role == "TRANSMITTER"
    assert instrument.mode == "FIXED_TIMELINE"
    assert instrument.frequency_range_min_hz == pytest.approx(4.1e9)
    assert instrument.frequency_range_max_hz == pytest.approx(4.3e9)
    assert instrument.sampling_period_fs == 400_000
    assert [issue.code for issue in state.issues].count("UNKNOWN_PORT_DEPENDENCY") == 1


def test_collect_state_normalizes_unit_configuration_controls() -> None:
    """Given unit configuration, hardware state should expose supported controls."""
    reader = _make_reader(_FakeClient())

    state = reader.collect_state(unit_labels=("unit-a",), parallel=False)

    assert [unit.label for unit in state.units] == ["unit-a"]
    assert [
        (control.key, control.current_value, control.allowed_values)
        for control in state.units[0].controls
    ] == [
        (
            "quel3.monitor.mode",
            "loopback",
            ("disabled", "loopback"),
        )
    ]


def test_collect_state_records_unit_configuration_fetch_errors() -> None:
    """Given failed unit configuration fetch, hardware state should keep an issue."""
    client = _FakeClient()

    async def _fail_unit_configuration(unit_label: str) -> _UnitConfiguration:
        raise RuntimeError(f"failed {unit_label}")

    client.get_unit_configuration = _fail_unit_configuration  # type: ignore[method-assign]
    reader = _make_reader(client)

    state = reader.collect_state(unit_labels=("unit-a",), parallel=False)

    assert state.units[0].controls == ()
    assert any(
        issue.code == "RESOURCE_FETCH_ERROR"
        and issue.message == "get_unit_configuration failed."
        and issue.resource_id == "unit-a"
        for issue in state.issues
    )


def test_collect_state_keeps_monitor_port_without_fetching_port_info() -> None:
    """Given a monitor port, hardware state should not request unsupported port info."""
    client = _MonitorPortClient()
    reader = _make_reader(client)

    state = reader.collect_state(unit_labels=("unit-a",), parallel=False)

    assert [port.id for port in state.ports] == ["unit-a:mon", "unit-a:tx_p01"]
    assert state.ports[0].role is None
    assert client.get_port_info_calls == ["unit-a:tx_p01"]
    assert not any(
        issue.code == "RESOURCE_FETCH_ERROR" and issue.resource_id == "unit-a:mon"
        for issue in state.issues
    )


def test_collect_state_records_fetch_errors_without_raising() -> None:
    """Given a failed resource fetch, hardware state should keep an issue."""
    client = _FakeClient()

    async def _fail_instrument(resource_id: str) -> _InstrumentInfo:
        raise RuntimeError(f"failed {resource_id}")

    client.get_instrument_info = _fail_instrument  # type: ignore[method-assign]
    reader = _make_reader(client)

    state = reader.collect_state(unit_labels=("unit-a",), parallel=True)

    assert state.instruments == ()
    assert any(issue.code == "RESOURCE_FETCH_ERROR" for issue in state.issues)
    assert any(issue.resource_id == "unit-a:inst-q00" for issue in state.issues)


def test_collect_state_diagnostics_are_opt_in() -> None:
    """Given diagnostics disabled, hardware state should omit diagnostics."""
    client = _FakeClient()
    reader = _make_reader(client)

    state = reader.collect_state(unit_labels=("unit-a",), include_diagnostics=False)

    assert state.diagnostics == ()

    state_with_diagnostics = reader.collect_state(
        unit_labels=("unit-a",),
        include_diagnostics=True,
    )

    assert state_with_diagnostics.diagnostics[0].text == "state: unit-a:tx_p01"


def test_collect_state_filters_local_port_after_unit_scoping() -> None:
    """Given selected unit and local port ID, state should keep that unit port."""
    client = _BoxLocalAliasClient()
    reader = _make_reader(client)

    state = reader.collect_state(
        unit_labels=("unit-a",),
        port_ids=("tx_p01",),
        parallel=False,
    )

    assert [port.id for port in state.ports] == ["unit-a:tx_p01"]
    assert [instrument.id for instrument in state.instruments] == ["unit-a:inst-q00"]


def test_collect_state_matches_local_port_across_selected_units() -> None:
    """Given local port ID without unit selection, state should keep all matches."""
    client = _BoxLocalAliasClient()
    reader = _make_reader(client)

    state = reader.collect_state(port_ids=("tx_p01",), parallel=False)

    assert [port.id for port in state.ports] == ["unit-a:tx_p01", "unit-b:tx_p01"]
    assert [instrument.id for instrument in state.instruments] == [
        "unit-a:inst-q00",
        "unit-b:inst-q00",
    ]


def test_collect_state_filters_alias_related_ports_and_diagnostics() -> None:
    """Given local alias filter, state should keep matched instruments and ports."""
    client = _MultiInstrumentClient()
    reader = _make_reader(client)

    state = reader.collect_state(
        unit_labels=("unit-a",),
        instrument_aliases=("Q00",),
        include_diagnostics=True,
        parallel=False,
    )

    assert [port.id for port in state.ports] == ["unit-a:tx_p01"]
    assert [instrument.id for instrument in state.instruments] == ["unit-a:inst-q00"]
    assert [diagnostic.port_id for diagnostic in state.diagnostics] == ["unit-a:tx_p01"]


def test_collect_state_matches_unit_qualified_alias() -> None:
    """Given unit-qualified alias, state should keep only that unit match."""
    client = _MultiInstrumentClient()
    reader = _make_reader(client)

    state = reader.collect_state(
        instrument_aliases=("unit-b:Q00",),
        parallel=False,
    )

    assert [port.id for port in state.ports] == ["unit-b:tx_p01"]
    assert [instrument.id for instrument in state.instruments] == ["unit-b:inst-q00"]


def test_collect_state_intersects_port_and_alias_filters() -> None:
    """Given port and alias filters, state should keep only their intersection."""
    client = _MultiInstrumentClient()
    reader = _make_reader(client)

    state = reader.collect_state(
        port_ids=("unit-a:rx_p02",),
        instrument_aliases=("Q00",),
        include_diagnostics=True,
        parallel=False,
    )

    assert state.ports == ()
    assert state.instruments == ()
    assert state.diagnostics == ()


def test_collect_state_units_view_returns_only_units() -> None:
    """Units view should return only unit state."""
    client = _FakeClient()
    reader = _make_reader(client)

    state = reader.collect_state(unit_labels=("unit-a",), view="units")

    assert [unit.label for unit in state.units] == ["unit-a"]
    assert state.ports == ()
    assert state.instruments == ()


def test_collect_state_ports_view_returns_only_selected_ports() -> None:
    """Ports view should return only selected port state."""
    client = _MultiInstrumentClient()
    reader = _make_reader(client)

    state = reader.collect_state(
        unit_labels=("unit-a",),
        port_ids=("rx_p02",),
        view="ports",
    )

    assert [port.id for port in state.ports] == ["unit-a:rx_p02"]
    assert state.instruments == ()


def test_collect_state_instruments_view_returns_only_selected_instruments() -> None:
    """Instruments view should return only selected instrument state."""
    client = _MultiInstrumentClient()
    reader = _make_reader(client)

    state = reader.collect_state(
        unit_labels=("unit-a",),
        instrument_aliases=("Q00",),
        view="instruments",
    )

    assert state.ports == ()
    assert [instrument.id for instrument in state.instruments] == ["unit-a:inst-q00"]
    assert not any(issue.code == "ORPHAN_INSTRUMENT" for issue in state.issues)


def test_collect_state_diagnostics_view_returns_selected_diagnostics() -> None:
    """Diagnostics view should return diagnostics for selected ports."""
    client = _MultiInstrumentClient()
    reader = _make_reader(client)

    state = reader.collect_state(
        unit_labels=("unit-a",),
        port_ids=("rx_p02",),
        include_diagnostics=True,
        view="diagnostics",
    )

    assert [diagnostic.port_id for diagnostic in state.diagnostics] == ["unit-a:rx_p02"]
    assert state.instruments == ()


def test_collect_state_rejects_old_filter_kwargs() -> None:
    """Given removed hardware-state filter kwargs, reader raises TypeError."""
    reader = _make_reader(_FakeClient())

    with pytest.raises(TypeError, match="instrument_port_ids"):
        cast(Any, reader).collect_state(instrument_port_ids=("unit-a:tx_p01",))
    with pytest.raises(TypeError, match="diagnostic_port_ids"):
        cast(Any, reader).collect_state(diagnostic_port_ids=("unit-a:tx_p01",))


def test_backend_settings_projection_uses_hardware_state_instruments() -> None:
    """Given hardware state, backend settings projection should keep deploy cache fields."""
    client = _FakeClient()
    configuration_calls: list[str] = []
    get_unit_configuration = client.get_unit_configuration

    async def _track_unit_configuration(unit_label: str) -> _UnitConfiguration:
        configuration_calls.append(unit_label)
        return await get_unit_configuration(unit_label)

    client.get_unit_configuration = _track_unit_configuration  # type: ignore[method-assign]
    reader = _make_reader(client)

    settings = reader.fetch_backend_settings_from_hardware(
        unit_labels_by_box_id={
            "BOX1": "unit-a",
            "BOX2": "unit-c",
        },
        parallel=False,
    )

    assert settings == {
        "BOX1": {
            "instruments": {
                "Q00": {
                    "resource_id": "unit-a:inst-q00",
                    "port_id": "unit-a:tx_p01",
                    "role": "TRANSMITTER",
                    "definition": {
                        "alias": "unit-a:Q00",
                        "role": "TRANSMITTER",
                        "mode": "FIXED_TIMELINE",
                        "profile": {
                            "frequency_range_min": 4.1e9,
                            "frequency_range_max": 4.3e9,
                        },
                    },
                }
            }
        },
        "BOX2": {"instruments": {}},
    }
    assert configuration_calls == []


def test_backend_settings_fetch_keeps_unqualified_instrument_resources() -> None:
    """Backend settings fetch should keep unqualified instrument resources."""
    client = _UnqualifiedInstrumentResourceClient()
    reader = _make_reader(client)

    settings = reader.fetch_backend_settings_from_hardware(
        unit_labels_by_box_id={"BOX1": "unit-a"},
        parallel=False,
    )

    assert settings["BOX1"]["instruments"]["Q00"]["resource_id"] == "inst-q00"
    assert settings["BOX1"]["instruments"]["Q00"]["port_id"] == "unit-a:tx_p01"


def test_collect_state_filters_unqualified_resources_by_resolved_unit() -> None:
    """Selected-unit collection should filter unqualified resources after fetch."""
    client = _UnqualifiedOtherUnitResourceClient()
    reader = _make_reader(client)

    state = reader.collect_state(unit_labels=("unit-a",), parallel=False)

    assert [port.id for port in state.ports] == ["unit-a:tx_p01"]
    assert state.instruments == ()


def test_collect_state_scopes_duplicate_aliases_by_unit() -> None:
    """Multi-unit collection should allow box-local normalized aliases."""
    client = _BoxLocalAliasClient()
    reader = _make_reader(client)

    state = reader.collect_state(parallel=False)

    assert [instrument.normalized_alias for instrument in state.instruments] == [
        "Q00",
        "Q00",
    ]
    assert not any(issue.code == "DUPLICATE_INSTRUMENT_ALIAS" for issue in state.issues)
