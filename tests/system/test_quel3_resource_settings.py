"""Tests for projecting QuEL-3 observations into system backend settings."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel3 import Quel3BackendController
from qubex.backend.quel3.models import Quel3InstrumentState, Quel3ResourceSnapshot
from qubex.system.quel3 import Quel3SystemSynchronizer


@pytest.mark.parametrize("parallel", [None, False, True])
def test_settings_projection_preserves_scopes_aliases_and_optional_fields(
    parallel: bool | None,
) -> None:
    """Settings should retain selected units, normalized aliases, and available fields."""
    instrument = Quel3InstrumentState(
        id="instrument-a",
        unit_label="unit-a",
        port_id="unit-a:tx_p01",
        alias="unit-a:Q00",
        normalized_alias="Q00",
        role="TRANSMITTER",
        mode="FIXED_TIMELINE",
        frequency_range_min_hz=4e9,
        frequency_range_max_hz=6e9,
    )
    snapshot = Quel3ResourceSnapshot(
        generated_at="2026-09-29T00:00:00+00:00",
        endpoint="localhost",
        port=None,
        selected_unit_labels=("unit-a", "unit-b", "unit-c"),
        units=(),
        ports=(),
        instruments=(
            instrument,
            replace(
                instrument,
                id="instrument-b",
                unit_label="unit-b",
                port_id="unit-b:tx_p01",
                alias="Q00",
                normalized_alias=None,
                mode=None,
                frequency_range_min_hz=None,
                frequency_range_max_hz=None,
            ),
            replace(instrument, id="anonymous", alias=None, normalized_alias=None),
            replace(instrument, id="outside", unit_label="unit-d"),
        ),
    )
    calls: list[dict[str, object]] = []

    def collect_snapshot(**kwargs: object) -> Quel3ResourceSnapshot:
        calls.append(kwargs)
        return snapshot

    controller = Quel3BackendController(
        resource_reader=cast(Any, SimpleNamespace(collect_snapshot=collect_snapshot))
    )
    synchronizer = Quel3SystemSynchronizer(backend_controller=controller)

    settings = synchronizer.fetch_backend_settings_from_hardware(
        experiment_system=cast(Any, None),
        box_ids=("unit-a", "unit-b", "unit-c"),
        parallel=parallel,
    )

    assert settings == {
        "unit-a": {
            "instruments": {
                "Q00": {
                    "resource_id": "instrument-a",
                    "port_id": "unit-a:tx_p01",
                    "role": "TRANSMITTER",
                    "definition": {
                        "alias": "unit-a:Q00",
                        "role": "TRANSMITTER",
                        "mode": "FIXED_TIMELINE",
                        "profile": {
                            "frequency_range_min": 4e9,
                            "frequency_range_max": 6e9,
                        },
                    },
                }
            }
        },
        "unit-b": {
            "instruments": {
                "Q00": {
                    "resource_id": "instrument-b",
                    "port_id": "unit-b:tx_p01",
                    "role": "TRANSMITTER",
                    "definition": {"alias": "Q00", "role": "TRANSMITTER"},
                }
            }
        },
        "unit-c": {"instruments": {}},
    }
    assert calls == [
        {
            "unit_labels": ("unit-a", "unit-b", "unit-c"),
            "parallel": parallel is not False,
            "level": "instrument",
            "port_ids": (),
            "instrument_aliases": (),
            "timeout_seconds": None,
        }
    ]
    assert controller.get_instrument_configuration().instruments == ()


def test_empty_settings_selection_skips_collection() -> None:
    """An empty box selection should return no settings without contacting the reader."""

    def reject_collection(**kwargs: object) -> None:
        pytest.fail("Empty selections must not collect resources.")

    controller = Quel3BackendController(
        resource_reader=cast(Any, SimpleNamespace(collect_snapshot=reject_collection))
    )
    synchronizer = Quel3SystemSynchronizer(backend_controller=controller)

    assert (
        synchronizer.fetch_backend_settings_from_hardware(
            experiment_system=cast(Any, None), box_ids=()
        )
        == {}
    )
