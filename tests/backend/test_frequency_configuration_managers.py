"""Backend-specific frequency configuration contracts without hardware."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel1.managers.frequency_configuration_manager import (
    Quel1FrequencyConfigurationManager,
)
from qubex.backend.quel1.sweep_configuration import (
    Quel1FrequencySettings,
    Quel1SweepConfiguration,
)
from qubex.backend.quel3.managers.frequency_configuration_manager import (
    Quel3FrequencyConfigurationManager,
)
from qubex.backend.quel3.models import InstrumentSpec
from qubex.backend.quel3.sweep_configuration import Quel3SweepConfiguration


class Quel1Stub:
    """Record oscillator state and simulate a single failed channel write."""

    def __init__(self):
        self.cache: Quel1FrequencySettings = {
            "box": {
                "ports": {
                    1: {
                        "lo_freq": 10,
                        "cnco_freq": 20,
                        "channels": {0: {"fnco_freq": 30}},
                    },
                    2: {
                        "lo_freq": 40,
                        "cnco_freq": 50,
                        "runits": {0: {"fnco_freq": 60}},
                    },
                }
            }
        }
        self.hardware = deepcopy(self.cache)
        self.resets = []
        self.fail = False

    def get_box_config_cache(self):
        """Return an isolated cache snapshot."""
        return deepcopy(self.cache)

    def update_box_config_cache(self, values):
        """Publish changed cache entries."""
        self.cache.update(deepcopy(values))

    def replace_box_config_cache(self, values):
        """Replace cache state."""
        self.cache = deepcopy(values)

    def config_port(self, box, *, port, lo_freq_hz, cnco_freq_hz):
        """Apply physical oscillator values."""
        self.hardware[box]["ports"][port].update(
            lo_freq=lo_freq_hz, cnco_freq=cnco_freq_hz
        )

    def config_channel(self, box, *, port, channel, fnco_freq_hz):
        """Fail once after a port write when requested."""
        if self.fail:
            self.fail = False
            raise RuntimeError("channel write")
        settings = self.hardware[box]["ports"][port]
        assert "channels" in settings
        settings["channels"][channel]["fnco_freq"] = fnco_freq_hz

    def config_runit(self, box, *, port, runit, fnco_freq_hz):
        """Apply a capture oscillator."""
        settings = self.hardware[box]["ports"][port]
        assert "runits" in settings
        settings["runits"][runit]["fnco_freq"] = fnco_freq_hz

    def initialize_awg_and_capunits(self, boxes):
        """Record preparation once per box."""
        self.resets.append(boxes)


def test_quel1_restores_partial_application_and_entry_values():
    """Native sweep settings restore distinct generator and capture entry values."""
    backend = Quel1Stub()
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    original = deepcopy(backend.cache)
    segments = manager.plan(
        [5.9, 6.1], configuration=original, sideband="U", cnco_center=1_500_000_000
    )
    assert isinstance(segments[0].configuration, Quel1SweepConfiguration)
    assert backend.hardware == original
    with manager.modified(segments[0].configuration):
        assert backend.hardware != original
    assert backend.hardware == backend.cache == original
    backend.fail = True
    with (
        pytest.raises(RuntimeError, match="channel write"),
        manager.modified(segments[0].configuration),
    ):
        pytest.fail("Application must fail")
    assert backend.hardware == backend.cache == original


def test_quel3_restores_native_instrument_configuration():
    """Instrument deployment occurs only at scope boundaries and restores entry state."""
    original = InstrumentSpec(
        port_id="unit:trx",
        alias="R00",
        role="TRANSCEIVER",
        frequency_range_min_hz=5.9e9,
        frequency_range_max_hz=6.1e9,
    )
    state = [original]
    calls = []

    def deploy(*, instrument, append):
        assert append
        calls.append(instrument)
        state[0] = instrument

    backend = SimpleNamespace(
        get_instrument_configuration=lambda: SimpleNamespace(instruments=tuple(state)),
        deploy_instrument=deploy,
    )
    manager = Quel3FrequencyConfigurationManager(cast(Any, backend))
    plan = manager.plan([6.5, 6.6], target="R00")
    assert isinstance(plan[0].configuration, Quel3SweepConfiguration)
    assert not calls
    with manager.modified(plan[0].configuration):
        assert len(calls) == 1
    assert state == [original]
    assert len(calls) == 2


@pytest.mark.parametrize("sideband", ["U", "L", None])
def test_quantized_settings_cover_requested_carriers(sideband):
    """Each mixing path realizes requested carriers inside its baseband interval."""
    backend = Quel1Stub()
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    plan = manager.plan(
        [5.9, 6.0, 6.1],
        configuration=backend.cache,
        sideband=sideband,
        cnco_center=1_500_000_000,
    )
    for segment in plan:
        port = segment.configuration.settings["box"]["ports"][1]
        assert port["cnco_freq"] is not None
        center = (
            (port["lo_freq"] or 0) + (-1 if sideband == "L" else 1) * port["cnco_freq"]
        ) * 1e-9
        assert all(abs(f - center) < 0.25 for f in segment.values)


def test_restoration_failure_invalidates_cache(monkeypatch):
    """A failed restore remains visible and invalidates affected cache entries."""
    backend = Quel1Stub()
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    configuration = manager.plan(
        [6.0], configuration=backend.cache, sideband="U", cnco_center=1_500_000_000
    )[0].configuration
    original = backend.config_port

    def fail_restore(box, *, port, lo_freq_hz, cnco_freq_hz):
        if lo_freq_hz == 10:
            raise RuntimeError("restore write")
        original(box, port=port, lo_freq_hz=lo_freq_hz, cnco_freq_hz=cnco_freq_hz)

    monkeypatch.setattr(backend, "config_port", fail_restore)
    with (
        pytest.raises(RuntimeError, match="Could not restore"),
        manager.modified(configuration),
    ):
        pass
    assert backend.cache == {}
    assert backend.hardware["box"]["ports"][2]["lo_freq"] == 40


def test_quel3_reuses_matching_instrument_without_redeployment():
    """Carriers inside the deployed range preserve the exact instrument definition."""
    original = InstrumentSpec(
        port_id="unit:trx",
        alias="R00",
        role="TRANSCEIVER",
        frequency_range_min_hz=5.93e9,
        frequency_range_max_hz=6.17e9,
    )
    backend = SimpleNamespace(
        get_instrument_configuration=lambda: SimpleNamespace(instruments=(original,))
    )
    manager = Quel3FrequencyConfigurationManager(cast(Any, backend))
    segment = manager.plan([6.0], target="R00")[0]
    assert not segment.requires_reconfiguration
    with manager.modified(segment.configuration):
        assert segment.configuration.settings.instruments == (original,)


def test_quel1_unchanged_settings_do_not_write_or_restore(monkeypatch):
    """Reusing entry settings performs no oscillator writes on entry or exit."""
    backend = Quel1Stub()
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))

    def unexpected(*args, **kwargs):
        pytest.fail("Unchanged oscillators must not be written")

    monkeypatch.setattr(backend, "config_port", unexpected)
    with manager.modified(Quel1SweepConfiguration(deepcopy(backend.cache))):
        pass


def test_quel1_explicit_center_controls_retuning():
    """An explicit carrier center determines frequency settings independently of scan extent."""
    from qubex.backend.quel1.mixing import MixingUtil

    backend = Quel1Stub()
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    segment = manager.plan(
        [6.0, 6.1],
        configuration=backend.cache,
        sideband="U",
        cnco_center=1_500_000_000,
        center_frequency=6.0,
    )[0]
    lo, cnco, _ = MixingUtil.calc_lo_cnco(6e9, 1_500_000_000, "U")
    assert segment.configuration.settings["box"]["ports"][1]["lo_freq"] == lo
    assert segment.configuration.settings["box"]["ports"][1]["cnco_freq"] == cnco


def test_quel3_overlap_retunes_when_reference_is_outside_deployed_range():
    """A deployed range is reused only when it also covers the boundary reference."""
    original = InstrumentSpec(
        port_id="unit:trx",
        alias="R00",
        role="TRANSCEIVER",
        frequency_range_min_hz=6.005e9,
        frequency_range_max_hz=6.205e9,
    )
    backend = SimpleNamespace(
        get_instrument_configuration=lambda: SimpleNamespace(instruments=(original,))
    )
    manager = Quel3FrequencyConfigurationManager(cast(Any, backend))
    plan = manager.plan(
        [5.98, 5.99, 6.0, 6.01, 6.02, 6.03],
        target="R00",
        subrange_width=0.03,
        overlap=True,
    )

    assert [segment.indices for segment in plan] == [(0, 1, 2), (3, 4, 5)]
    assert plan[1].reference_value == pytest.approx(6.0, rel=0, abs=1e-12)
    assert plan[1].requires_reconfiguration
    spec = plan[1].configuration.settings.instruments[0]
    assert (spec.frequency_range_min_hz + spec.frequency_range_max_hz) / 2 == (
        pytest.approx(6.02e9, rel=0, abs=1e-5)
    )
    assert spec.frequency_range_min_hz <= 6.0e9 <= spec.frequency_range_max_hz


@pytest.mark.parametrize(
    ("frequencies", "indices"),
    [
        ([6.0], [(0,)]),
        ([5.9, 6.0, 6.1], [(0,), (1,), (2,)]),
        ([6.0, 6.0, 6.1], [(0, 1), (2,)]),
        ([6.1, 6.0, 5.9], [(0,), (1,), (2,)]),
    ],
)
def test_quel3_zero_width_ranges_plan_and_restore(frequencies, indices):
    """Zero-margin instruments scan distinct carriers in separate restorable scopes."""
    original = InstrumentSpec(
        port_id="unit:trx",
        alias="Q00",
        role="TRANSMITTER",
        frequency_range_min_hz=6.0e9,
        frequency_range_max_hz=6.0e9,
    )
    state = [original]
    calls = []

    def deploy(*, instrument, append):
        assert append
        calls.append(instrument)
        state[0] = instrument

    backend = SimpleNamespace(
        get_instrument_configuration=lambda: SimpleNamespace(instruments=tuple(state)),
        deploy_instrument=deploy,
    )
    manager = Quel3FrequencyConfigurationManager(cast(Any, backend))
    plan = manager.plan(frequencies, target="Q00")

    assert not calls
    assert [segment.indices for segment in plan] == indices
    assert [value for segment in plan for value in segment.values] == frequencies
    for segment in plan:
        spec = segment.configuration.settings.instruments[0]
        assert spec.frequency_range_min_hz == spec.frequency_range_max_hz
        assert spec.frequency_range_min_hz == pytest.approx(
            segment.values[0] * 1e9, rel=0, abs=1e-5
        )
        with manager.modified(segment.configuration):
            assert state == [spec]
        assert state == [original]
    assert len(calls) == 2 * sum(value != 6.0 for value in frequencies)


def test_quel3_reuses_inclusive_range_endpoints():
    """Both deployed frequency-range endpoints remain valid without redeployment."""
    original = InstrumentSpec(
        port_id="unit:trx",
        alias="Q00",
        role="TRANSMITTER",
        frequency_range_min_hz=5.93e9,
        frequency_range_max_hz=6.17e9,
    )
    backend = SimpleNamespace(
        get_instrument_configuration=lambda: SimpleNamespace(instruments=(original,))
    )
    manager = Quel3FrequencyConfigurationManager(cast(Any, backend))
    plan = manager.plan([5.93, 6.17], target="Q00")

    assert len(plan) == 1
    assert plan[0].configuration.settings.instruments == (original,)
    assert not plan[0].requires_reconfiguration


def test_quel1_baseband_endpoints_require_separate_tuning():
    """QuEL-1 retains strict baseband limits when QuEL-3 endpoints are inclusive."""
    backend = Quel1Stub()
    for port in backend.cache["box"]["ports"].values():
        port.update(lo_freq=8_000_000_000, cnco_freq=2_000_000_000)
        for channel in port.get("channels", port.get("runits", {})).values():
            channel["fnco_freq"] = 0
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    plan = manager.plan(
        [5.75, 6.25],
        configuration=backend.cache,
        sideband="L",
        cnco_center=2_250_000_000,
        reuse_current_tuning=True,
    )

    assert [segment.indices for segment in plan] == [(0,), (1,)]
    assert all(segment.requires_reconfiguration for segment in plan)
    for segment in plan:
        port = segment.configuration.settings["box"]["ports"][1]
        assert port["lo_freq"] is not None
        assert port["cnco_freq"] is not None
        center = (port["lo_freq"] - port["cnco_freq"]) / 1e9
        assert all(abs(value - center) < 0.25 for value in segment.values)


def test_quel3_overlap_recenters_to_cover_reference_and_measurement():
    """Retuning may shift the center to cover a reference without widening the range."""
    original = InstrumentSpec(
        port_id="unit:trx",
        alias="R00",
        role="TRANSCEIVER",
        frequency_range_min_hz=6.0e9,
        frequency_range_max_hz=6.2e9,
    )
    backend = SimpleNamespace(
        get_instrument_configuration=lambda: SimpleNamespace(instruments=(original,))
    )
    manager = Quel3FrequencyConfigurationManager(cast(Any, backend))
    plan = manager.plan([5.9, 6.09], target="R00", subrange_width=0.1, overlap=True)

    assert [segment.indices for segment in plan] == [(0,), (1,)]
    assert plan[1].reference_value == pytest.approx(5.9, rel=0, abs=1e-12)
    spec = plan[1].configuration.settings.instruments[0]
    assert spec.frequency_range_min_hz <= 5.9e9
    assert spec.frequency_range_max_hz >= 6.09e9
    assert spec.frequency_range_max_hz - spec.frequency_range_min_hz == (
        pytest.approx(0.2e9, rel=0, abs=1e-5)
    )


def test_quel3_zero_width_cannot_bridge_distinct_reference():
    """Zero-margin windows reject distinct overlap references without deploying anything."""
    original = InstrumentSpec(
        port_id="unit:trx",
        alias="R00",
        role="TRANSCEIVER",
        frequency_range_min_hz=6.0e9,
        frequency_range_max_hz=6.0e9,
    )
    backend = SimpleNamespace(
        get_instrument_configuration=lambda: SimpleNamespace(instruments=(original,))
    )
    manager = Quel3FrequencyConfigurationManager(cast(Any, backend))

    with pytest.raises(ValueError, match="reference"):
        manager.plan([6.0, 6.1], target="R00", overlap=True)


def test_quel1_reuse_also_requires_reference_inside_baseband():
    """QuEL-1 retunes when measurement points fit but the preceding reference does not."""
    backend = Quel1Stub()
    for port in backend.cache["box"]["ports"].values():
        port.update(lo_freq=8_000_000_000, cnco_freq=1_900_000_000)
        for channel in port.get("channels", port.get("runits", {})).values():
            channel["fnco_freq"] = 0
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    plan = manager.plan(
        [5.8, 6.0],
        configuration=backend.cache,
        sideband="L",
        cnco_center=2_250_000_000,
        reuse_current_tuning=True,
        subrange_width=0.1,
        overlap=True,
    )

    assert [segment.indices for segment in plan] == [(0,), (1,)]
    assert plan[1].reference_value == pytest.approx(5.8, rel=0, abs=1e-12)
    assert plan[1].requires_reconfiguration
    port = plan[1].configuration.settings["box"]["ports"][1]
    assert port["lo_freq"] is not None
    assert port["cnco_freq"] is not None
    center = (port["lo_freq"] - port["cnco_freq"]) / 1e9
    assert abs(center - 5.8) < 0.25
    assert abs(center - 6.0) < 0.25
