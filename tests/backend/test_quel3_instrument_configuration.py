"""Tests for portable QuEL-3 instrument configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError

from qubex.backend.quel3.models import InstrumentConfiguration
from qubex.backend.quel3.models.instrument import InstrumentRoleName, InstrumentSpec


def _spec_data(**overrides: object) -> dict[str, Any]:
    """Return a complete editable instrument specification."""
    return {
        "port_id": "unit-a:tx_p02",
        "alias": "Q00",
        "role": "TRANSMITTER",
        "frequency_range_min_hz": 4.1e9,
        "frequency_range_max_hz": 4.3e9,
        **overrides,
    }


@pytest.mark.parametrize(
    "role", ["TRANSMITTER", "TRANSCEIVER", "TRANSCEIVER_LOOPBACK", "RECEIVER"]
)
def test_instrument_spec_preserves_each_supported_role(
    role: InstrumentRoleName,
) -> None:
    """Each supported role should remain unchanged in a deployable specification."""
    spec = InstrumentSpec.model_validate(_spec_data(role=role))

    assert spec.role == role
    assert spec.port_id == "unit-a:tx_p02"


def test_instrument_spec_normalizes_alias_and_port_whitespace() -> None:
    """Surrounding whitespace should be removed before instrument lookup and deployment."""
    spec = InstrumentSpec.model_validate(
        _spec_data(alias=" Q00 ", port_id=" unit-a:tx_p02 ")
    )

    assert spec.alias == "Q00"
    assert spec.port_id == "unit-a:tx_p02"


@pytest.mark.parametrize(
    "overrides",
    [
        {"alias": " "},
        {"alias": "unit-a:Q00"},
        {"port_id": "tx_p02"},
        {"port_id": ":tx_p02"},
        {"port_id": "unit-a:"},
        {"role": "UNSPECIFIED"},
        {"role": "unknown"},
        {"frequency_range_min_hz": float("nan")},
        {"frequency_range_min_hz": float("-inf")},
        {"frequency_range_max_hz": float("inf")},
        {"frequency_range_min_hz": 4.5e9},
        {"resource_id": "unit-a:stale"},
        {"config": {}},
        {"metadata": {}},
    ],
)
def test_instrument_spec_rejects_invalid_or_runtime_fields(
    overrides: dict[str, object],
) -> None:
    """Invalid definitions and runtime-only fields should be rejected at the model boundary."""
    with pytest.raises(ValidationError):
        InstrumentSpec.model_validate(_spec_data(**overrides))


def test_instrument_spec_accepts_zero_width_frequency_range() -> None:
    """Equal finite frequency bounds should permit a zero-margin deployment."""
    spec = InstrumentSpec.model_validate(_spec_data(frequency_range_max_hz=4.1e9))

    assert spec.frequency_range_min_hz == spec.frequency_range_max_hz == 4.1e9


def test_configuration_rejects_duplicate_normalized_aliases_across_units() -> None:
    """Aliases should be unique after normalization even when instruments use different units."""
    with pytest.raises(ValidationError, match="Duplicate instrument alias"):
        InstrumentConfiguration.model_validate(
            {
                "instruments": [
                    _spec_data(),
                    _spec_data(alias=" Q00 ", port_id="unit-b:tx_p03"),
                ]
            }
        )


def test_configuration_and_specs_are_immutable() -> None:
    """Loaded definitions and their configuration should reject in-place reassignment."""
    spec = InstrumentSpec.model_validate(_spec_data())
    configuration = InstrumentConfiguration(instruments=(spec,))

    with pytest.raises(ValidationError, match="frozen"):
        cast(Any, spec).alias = "Q01"
    with pytest.raises(ValidationError, match="frozen"):
        cast(Any, configuration).instruments = ()

    assert configuration.instruments == (spec,)
    assert spec.alias == "Q00"


def test_configuration_yaml_roundtrip_keeps_only_deployable_fields(
    tmp_path: Path,
) -> None:
    """YAML should round-trip plain deployable fields without runtime identity or metadata."""
    spec = InstrumentSpec.model_validate(_spec_data())
    configuration = InstrumentConfiguration(instruments=(spec,))
    destination = tmp_path / "settings" / "instruments.yaml"

    saved = configuration.save_yaml(destination)
    restored = InstrumentConfiguration.load_yaml(saved)

    assert saved == destination
    assert restored == configuration
    assert yaml.safe_load(saved.read_text()) == {"instruments": [_spec_data()]}


def test_configuration_load_accepts_handwritten_scientific_notation(
    tmp_path: Path,
) -> None:
    """Handwritten scientific notation should be validated as numeric Hz values."""
    path = tmp_path / "instruments.yaml"
    path.write_text("""instruments:
  - port_id: unit-a:tx_p02
    alias: Q00
    role: TRANSMITTER
    frequency_range_min_hz: 4.1e9
    frequency_range_max_hz: 4.3e+9
""")

    configuration = InstrumentConfiguration.load_yaml(path)

    assert configuration.instruments == (InstrumentSpec.model_validate(_spec_data()),)


@pytest.mark.parametrize(
    "document",
    [
        "",
        "[]",
        "instruments: null",
        "instruments: []\nresource_id: old",
        "!!python/tuple [1, 2]",
    ],
)
def test_configuration_load_rejects_invalid_documents(
    tmp_path: Path, document: str
) -> None:
    """Malformed configuration documents and Python-specific YAML tags should fail offline."""
    path = tmp_path / "instruments.yaml"
    path.write_text(document)

    with pytest.raises((ValidationError, yaml.YAMLError)):
        InstrumentConfiguration.load_yaml(path)


def test_empty_configuration_roundtrips_without_metadata(tmp_path: Path) -> None:
    """An explicit empty configuration should preserve its empty instrument tuple."""
    configuration = InstrumentConfiguration()
    path = configuration.save_yaml(tmp_path / "empty.yaml")

    assert InstrumentConfiguration.load_yaml(path).instruments == ()
    assert yaml.safe_load(path.read_text()) == {"instruments": []}
