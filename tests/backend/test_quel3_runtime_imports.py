"""Tests for lazy quelware execution and deployment dependencies."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel3.infra import quelware_imports
from qubex.backend.quel3.models import Quel3CaptureMode


def test_load_execution_api_resolves_runtime_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution dependencies should load lazily and construct the requested directives."""
    capture_mode = object()
    modules = {
        "quelware_client.client.helpers.sequencer": SimpleNamespace(
            Sequencer=SimpleNamespace
        ),
        "quelware_core.entities.directives": SimpleNamespace(
            CaptureMode=SimpleNamespace(AVERAGED_VALUE=capture_mode),
            SetFrequency=SimpleNamespace,
            SetCaptureMode=SimpleNamespace,
        ),
        "quelware_client.core.instrument_driver": SimpleNamespace(
            create_instrument_driver_fixed_timeline=SimpleNamespace
        ),
    }
    monkeypatch.setattr(
        quelware_imports.importlib, "import_module", modules.__getitem__
    )

    api = quelware_imports.load_quelware_execution_api(
        client_factory=cast(Any, SimpleNamespace)
    )

    assert api.client_factory is SimpleNamespace
    assert api.sequencer_factory is SimpleNamespace
    assert api.fixed_timeline_driver_factory is SimpleNamespace
    assert cast(Any, api.set_frequency_directive_factory(hz=6e9)).hz == 6e9
    directive = api.build_capture_mode_directive(Quel3CaptureMode.AVERAGED_VALUE)
    assert cast(Any, directive).mode is capture_mode


def test_load_instrument_entities_resolves_deployment_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deployment dependencies should construct definitions with the runtime's role values."""
    role = object()
    mode = object()
    modules = {
        "quelware_core.entities.instrument": SimpleNamespace(
            FixedTimelineProfile=SimpleNamespace,
            InstrumentDefinition=SimpleNamespace,
            InstrumentMode=SimpleNamespace(FIXED_TIMELINE=mode),
            InstrumentRole=SimpleNamespace(TRANSMITTER=role),
        )
    }
    monkeypatch.setattr(
        quelware_imports.importlib, "import_module", modules.__getitem__
    )

    entities = quelware_imports.load_quelware_instrument_entities()
    profile = entities.fixed_timeline_profile_factory(
        frequency_range_min=4e9, frequency_range_max=6e9
    )
    definition = entities.instrument_definition_factory(
        alias="Q00",
        role=entities.role_value("TRANSMITTER"),
        mode=entities.instrument_mode_namespace.FIXED_TIMELINE,
        profile=profile,
    )

    assert definition.alias == "Q00"
    assert definition.role is role
    assert definition.mode is mode
    assert definition.profile is profile
    assert profile.frequency_range_min == 4e9
    assert profile.frequency_range_max == 6e9
