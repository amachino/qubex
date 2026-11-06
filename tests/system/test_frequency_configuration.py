"""System sweep integration contracts."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast, get_type_hints

import pytest

from qubex.backend.frequency_sweep import SweepConfiguration, SweepSegment
from qubex.backend.quel1.sweep_configuration import Quel1SweepConfiguration
from qubex.backend.quel3.models import InstrumentConfiguration
from qubex.backend.quel3.sweep_configuration import Quel3SweepConfiguration
from qubex.system.quel1.quel1_system_synchronizer import Quel1SystemSynchronizer
from qubex.system.quel3.quel3_system_synchronizer import Quel3SystemSynchronizer
from qubex.system.system_manager import SystemManager
from qubex.system.system_synchronizer import SystemSynchronizer


def test_frequency_override_keeps_backend_calls_inside_sweep_scope():
    """Legacy frequency overrides still update and restore both model and backend."""
    calls = []
    target = SimpleNamespace(label="R00", frequency=6.0)

    def modify(values):
        target.frequency = values["R00"]

    manager = SimpleNamespace(
        experiment_system=SimpleNamespace(
            targets=[target], modify_target_frequencies=modify
        ),
        backend_controller=SimpleNamespace(
            modify_target_frequencies=lambda values: calls.append(values)
        ),
    )
    with SystemManager.modified_frequencies(cast(Any, manager), {"R00": 6.1}):
        assert target.frequency == 6.1
        assert calls == [{"R00": 6.1}]
    assert target.frequency == 6.0
    assert calls == [{"R00": 6.1}, {"R00": 6.0}]


def test_system_delegates_scope_and_restores_on_exception():
    """System delegates native configuration scopes without monitoring measurement calls."""
    calls = []
    configuration = SweepConfiguration()

    @contextmanager
    def modified(system, configuration):
        calls.append(configuration)
        try:
            yield
        finally:
            calls.append("restored")

    manager = SimpleNamespace(
        experiment_system=object(),
        _resolve_system_synchronizer=lambda: SimpleNamespace(
            modified_sweep_configuration=modified
        ),
    )
    with (
        pytest.raises(RuntimeError, match="acquisition"),
        SystemManager.modified_sweep_configuration(cast(Any, manager), configuration),
    ):
        raise RuntimeError("acquisition")
    assert calls == [configuration, "restored"]


def test_system_preserves_an_additional_backends_native_configuration():
    """Common sweep APIs carry additional configuration subclasses through planning and restoration."""

    class AdditionalSweepConfiguration(SweepConfiguration):
        pass

    configuration = AdditionalSweepConfiguration()
    system = object()
    calls = []
    segments = (SweepSegment((0, 1), (6.0, 6.1), configuration),)

    def plan(experiment_system, target, frequencies, **options):
        assert experiment_system is system
        assert target == "custom"
        assert frequencies == [6.0, 6.1]
        return segments

    @contextmanager
    def modified(experiment_system, settings):
        assert experiment_system is system
        assert settings is configuration
        calls.append("applied")
        try:
            yield
        finally:
            calls.append("restored")

    manager = SimpleNamespace(
        experiment_system=system,
        _resolve_system_synchronizer=lambda: SimpleNamespace(
            plan_frequency_sweep=plan,
            modified_sweep_configuration=modified,
        ),
    )

    result = SystemManager.plan_frequency_sweep(
        cast(Any, manager), "custom", [6.0, 6.1]
    )
    assert result is segments
    with SystemManager.modified_sweep_configuration(
        cast(Any, manager), result[0].configuration
    ):
        assert calls == ["applied"]
    assert calls == ["applied", "restored"]


@pytest.mark.parametrize(
    "owner",
    [
        SystemManager,
        SystemSynchronizer,
        Quel1SystemSynchronizer,
        Quel3SystemSynchronizer,
    ],
)
def test_sweep_configuration_contract_uses_common_base_type(owner):
    """Scope contracts require a sweep configuration without enumerating native types."""
    annotations = get_type_hints(owner.modified_sweep_configuration)
    assert annotations["configuration"] is SweepConfiguration


@pytest.mark.parametrize(
    ("synchronizer_type", "configuration"),
    [
        (Quel1SystemSynchronizer, Quel3SweepConfiguration(InstrumentConfiguration())),
        (Quel3SystemSynchronizer, Quel1SweepConfiguration({})),
    ],
)
def test_synchronizers_reject_foreign_settings_before_backend_updates(
    synchronizer_type,
    configuration,
):
    """Backend synchronizers reject another backend's configuration before controller updates."""

    def unexpected_update(configuration):
        pytest.fail("Foreign settings must not reach the controller")

    controller = SimpleNamespace(modified_sweep_configuration=unexpected_update)
    synchronizer = synchronizer_type(backend_controller=cast(Any, controller))

    with (
        pytest.raises(TypeError, match="different backend"),
        synchronizer.modified_sweep_configuration(cast(Any, object()), configuration),
    ):
        pytest.fail("Foreign settings must not enter the scope")


def test_synchronizer_restores_generator_and_capture_models():
    """Both oscillator models and hardware return to entry state after acquisition errors."""
    from copy import deepcopy

    from qubex.backend.quel1.managers.frequency_configuration_manager import (
        Quel1FrequencyConfigurationManager,
    )
    from qubex.system.quel1.quel1_system_synchronizer import Quel1SystemSynchronizer
    from tests.backend.test_frequency_configuration_managers import Quel1Stub

    backend = Quel1Stub()
    configuration_manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    ports = {}
    for number, values in backend.cache["box"]["ports"].items():
        channels = values.get("channels", values.get("runits", {}))
        ports[number] = SimpleNamespace(
            number=number,
            box_id="box",
            lo_freq=values["lo_freq"],
            cnco_freq=values["cnco_freq"],
            channels=[SimpleNamespace(number=0, fnco_freq=channels[0]["fnco_freq"])],
        )
        ports[number].channels[0].port = ports[number]
    system = SimpleNamespace(
        control_system=SimpleNamespace(
            get_port=lambda box, number: ports[number],
            get_box=lambda box: SimpleNamespace(
                traits=SimpleNamespace(readout_cnco_center=1_500_000_000)
            ),
        ),
        get_target=lambda label: SimpleNamespace(
            channel=ports[1].channels[0], is_read=True, sideband="U"
        ),
        get_cap_target=lambda label: SimpleNamespace(channel=ports[2].channels[0]),
    )
    controller = SimpleNamespace(
        plan_frequency_sweep=configuration_manager.plan,
        modified_sweep_configuration=configuration_manager.modified,
    )
    synchronizer = Quel1SystemSynchronizer(backend_controller=cast(Any, controller))
    manager = SimpleNamespace(
        experiment_system=system, _resolve_system_synchronizer=lambda: synchronizer
    )
    configuration = SystemManager.plan_frequency_sweep(
        cast(Any, manager), "R00", [6.0]
    )[0].configuration
    assert isinstance(configuration, Quel1SweepConfiguration)
    original = deepcopy(backend.cache)
    with (  # noqa: PT012
        pytest.raises(RuntimeError, match="acquisition"),
        SystemManager.modified_sweep_configuration(cast(Any, manager), configuration),
    ):
        assert ports[1].lo_freq == configuration.settings["box"]["ports"][1]["lo_freq"]
        assert ports[2].channels[0].fnco_freq == 0
        raise RuntimeError("acquisition")
    assert ports[1].lo_freq == 10
    assert ports[2].channels[0].fnco_freq == 60
    assert backend.cache == backend.hardware == original


def test_quel3_configuration_flows_through_system_and_restores_after_failure():
    """Common system scopes retain QuEL-3 settings and restore instruments after acquisition errors."""
    from qubex.backend.quel3.managers.frequency_configuration_manager import (
        Quel3FrequencyConfigurationManager,
    )
    from qubex.backend.quel3.models import InstrumentSpec

    original = InstrumentSpec(
        port_id="unit:trx",
        alias="R00",
        role="TRANSCEIVER",
        frequency_range_min_hz=5.9e9,
        frequency_range_max_hz=6.1e9,
    )
    state = [original]
    deployments = []

    def deploy(*, instrument, append):
        assert append
        deployments.append(instrument)
        state[0] = instrument

    backend = SimpleNamespace(
        get_instrument_configuration=lambda: InstrumentConfiguration(
            instruments=tuple(state)
        ),
        deploy_instrument=deploy,
    )
    configuration_manager = Quel3FrequencyConfigurationManager(cast(Any, backend))
    synchronizer = Quel3SystemSynchronizer(
        backend_controller=cast(
            Any,
            SimpleNamespace(
                plan_frequency_sweep=configuration_manager.plan,
                modified_sweep_configuration=configuration_manager.modified,
            ),
        )
    )
    manager = SimpleNamespace(
        experiment_system=object(), _resolve_system_synchronizer=lambda: synchronizer
    )
    configuration = SystemManager.plan_frequency_sweep(
        cast(Any, manager), "R00", [6.5, 6.6]
    )[0].configuration
    assert isinstance(configuration, Quel3SweepConfiguration)
    assert not deployments

    with (  # noqa: PT012
        pytest.raises(RuntimeError, match="acquisition"),
        SystemManager.modified_sweep_configuration(cast(Any, manager), configuration),
    ):
        assert state == list(configuration.settings.instruments)
        raise RuntimeError("acquisition")

    assert state == [original]
    assert deployments == [configuration.settings.instruments[0], original]
