"""Tests for Controller-only QuEL-3 configuration and hardware-state workflows."""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel3 import (
    InstrumentConfiguration,
    InstrumentSpec,
    Quel3BackendController,
    Quel3HardwareState,
)
from qubex.backend.quel3.instrument_cache import InstrumentCache
from qubex.backend.quel3.interfaces.client import InstrumentInfoProtocol
from qubex.backend.quel3.managers import Quel3ConfigurationManager
from qubex.backend.quel3.models import Quel3HardwareStateIssue


def _info(resource_id: str = "unit-a:old") -> InstrumentInfoProtocol:
    return cast(
        InstrumentInfoProtocol,
        SimpleNamespace(
            id=resource_id,
            port_id="unit-a:tx_p01",
            definition=SimpleNamespace(
                alias="unit-a:Q00",
                role="TRANSMITTER",
                mode="FIXED_TIMELINE",
                profile=SimpleNamespace(
                    frequency_range_min=4e9, frequency_range_max=6e9
                ),
            ),
            config=SimpleNamespace(sampling_period_fs=400_000, samples_per_tick=4),
        ),
    )


class _Reader:
    def __init__(self, infos: tuple[InstrumentInfoProtocol, ...]) -> None:
        self.infos = infos
        self.reads = 0
        self.state = Quel3HardwareState(
            generated_at="2026-09-10T00:00:00Z",
            endpoint="localhost",
            port=50051,
            selected_unit_labels=("unit-a",),
            units=(),
            ports=(),
            instruments=(),
            issues=(
                Quel3HardwareStateIssue(
                    severity="error",
                    code="RESOURCE_FETCH_ERROR",
                    message="Read failed.",
                ),
            ),
        )

    def read_instrument_infos(
        self,
        *,
        unit_labels: Sequence[str] = (),
        port_ids: Sequence[str] = (),
        parallel: bool = True,
    ) -> tuple[InstrumentInfoProtocol, ...]:
        self.reads += 1
        return self.infos

    def collect_state(self, **kwargs: object) -> Quel3HardwareState:
        return self.state


def test_controller_saves_last_confirmed_configuration_without_reading_hardware(
    tmp_path: Path,
) -> None:
    """Controller get and save should use the last refresh without another hardware read."""
    reader = _Reader((_info(),))
    controller = Quel3BackendController(hardware_state_reader=cast(Any, reader))
    controller.refresh_instrument_cache()
    configuration = controller.get_instrument_configuration()
    previous_hash = controller.hash
    reader.infos = ()

    path = controller.save_instrument_configuration(tmp_path / "instruments.yaml")
    loaded = controller.load_instrument_configuration(path)

    assert path == tmp_path / "instruments.yaml"
    assert loaded == configuration
    assert loaded.instruments[0].alias == "Q00"
    assert reader.reads == 1
    assert controller.hash == previous_hash


def test_loading_configuration_does_not_populate_execution_cache(
    tmp_path: Path,
) -> None:
    """Loading a file should return configuration while leaving the execution cache empty."""
    reader = _Reader(())
    controller = Quel3BackendController(hardware_state_reader=cast(Any, reader))
    configuration = InstrumentConfiguration(
        instruments=(
            InstrumentSpec(
                port_id="unit-a:tx_p01",
                alias="Q00",
                role="TRANSMITTER",
                frequency_range_min_hz=4e9,
                frequency_range_max_hz=6e9,
            ),
        )
    )
    path = configuration.save_yaml(tmp_path / "instruments.yaml")
    previous_hash = controller.hash

    loaded = controller.load_instrument_configuration(path)

    assert loaded == configuration
    assert controller.get_instrument_configuration().instruments == ()
    assert reader.reads == 0
    assert controller.hash == previous_hash


def test_controller_deploys_loaded_configuration_and_passes_readback_to_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """File deployment should execute with the newly read hardware identity and driver config."""
    reader = _Reader((_info(),))
    manager = Quel3ConfigurationManager()
    deployed: list[object] = []
    actual = _info("unit-a:new")

    async def deploy(**kwargs: object) -> None:
        deployed.append(kwargs["specifications"])
        reader.infos = (actual,)

    monkeypatch.setattr(manager, "_deploy_instruments", deploy)
    observed: list[InstrumentInfoProtocol] = []

    def execute_sync(
        *,
        request: object,
        instrument_cache: InstrumentCache,
        parallel: bool,
    ) -> object:
        observed.append(instrument_cache.get("Q00"))
        return "executed"

    controller = Quel3BackendController(
        configuration_manager=manager,
        hardware_state_reader=cast(Any, reader),
        execution_manager=cast(
            Any,
            SimpleNamespace(
                sampling_period_ns=0.4,
                execute_sync=execute_sync,
            ),
        ),
    )
    controller.refresh_instrument_cache()
    path = controller.save_instrument_configuration(tmp_path / "instruments.yaml")
    configuration = controller.load_instrument_configuration(path)
    old_hash = controller.hash

    result = controller.deploy_instruments(configuration=configuration)
    executed = controller.execute_sync(
        request=BackendExecutionRequest(payload=object())
    )

    assert deployed == [configuration.instruments]
    assert result == {"Q00": actual}
    assert reader.reads == 2
    assert observed[0] is actual
    assert executed == "executed"
    assert controller.hash != old_hash


def test_diagnostic_hardware_state_does_not_change_cached_configuration() -> None:
    """A partial diagnostic snapshot should leave confirmed instrument configuration intact."""
    reader = _Reader((_info(),))
    controller = Quel3BackendController(hardware_state_reader=cast(Any, reader))
    controller.refresh_instrument_cache()
    configuration = controller.get_instrument_configuration()
    previous_hash = controller.hash

    state = controller.get_hardware_state()

    assert state is reader.state
    assert state.issues[0].severity == "error"
    assert controller.get_instrument_configuration() == configuration
    assert controller.hash == previous_hash
    assert reader.reads == 1


def test_disconnect_discards_configuration_from_the_previous_connection() -> None:
    """Disconnect should make previously confirmed instrument configuration unavailable."""
    controller = Quel3BackendController(
        hardware_state_reader=cast(Any, _Reader((_info(),)))
    )
    controller.refresh_instrument_cache()

    controller.disconnect()

    assert controller.get_instrument_configuration().instruments == ()
