"""Backend-specific frequency configuration contracts without hardware."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.backend.quel1.managers.frequency_configuration_manager import (
    Quel1FrequencyConfigurationManager,
)
from qubex.backend.quel1.sweep_configuration import (
    Quel1BoxFrequencySettings,
    Quel1SweepConfiguration,
)
from qubex.backend.quel3.managers.frequency_configuration_manager import (
    Quel3FrequencyConfigurationManager,
)
from qubex.backend.quel3.models import InstrumentConfiguration, InstrumentSpec
from qubex.backend.quel3.sweep_configuration import Quel3SweepConfiguration


class Quel1Stub:
    """Record oscillator state and simulate a single failed channel write."""

    def __init__(self):
        self.cache: dict[str, Quel1BoxFrequencySettings] = {
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
        port_id="unit:trx_p00p01",
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
    """A matching sweep window preserves the exact instrument definition."""
    original = InstrumentSpec(
        port_id="unit:trx_p00p01",
        alias="R00",
        role="TRANSCEIVER",
        frequency_range_min_hz=5.6e9,
        frequency_range_max_hz=6.4e9,
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


def test_quel1_interval_center_controls_retuning():
    """The interval midpoint determines tuning independently of the entry settings."""
    from qubex.backend.quel1.mixing import MixingUtil

    backend = Quel1Stub()
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    segment = manager.plan(
        [6.0, 6.1],
        configuration=backend.cache,
        sideband="U",
        cnco_center=1_500_000_000,
    )[0]
    lo, cnco, _ = MixingUtil.calc_lo_cnco(6.05e9, 1_500_000_000, "U")
    assert segment.configuration.settings["box"]["ports"][1]["lo_freq"] == lo
    assert segment.configuration.settings["box"]["ports"][1]["cnco_freq"] == cnco


def test_quel3_overlap_centers_window_on_reference_and_new_points():
    """Sweep windows center the boundary reference and new carriers together."""
    original = InstrumentSpec(
        port_id="unit:trx_p00p01",
        alias="R00",
        role="TRANSCEIVER",
        frequency_range_min_hz=6.005e9,
        frequency_range_max_hz=6.505e9,
    )
    backend = SimpleNamespace(
        get_instrument_configuration=lambda: SimpleNamespace(instruments=(original,))
    )
    manager = Quel3FrequencyConfigurationManager(cast(Any, backend))
    plan = manager.plan(
        [5.75, 5.875, 6.0, 6.125, 6.25],
        target="R00",
        subrange_width=0.25,
        overlap=True,
    )

    assert [segment.indices for segment in plan] == [(0, 1, 2), (3, 4)]
    assert plan[1].reference_value == pytest.approx(6.0, rel=0, abs=1e-12)
    assert plan[1].requires_reconfiguration
    spec = plan[1].configuration.settings.instruments[0]
    assert (spec.frequency_range_min_hz + spec.frequency_range_max_hz) / 2 == (
        pytest.approx(6.125e9, rel=0, abs=1e-5)
    )
    assert spec.frequency_range_min_hz <= 6.0e9 <= spec.frequency_range_max_hz


@pytest.mark.parametrize(
    ("frequencies", "indices"),
    [
        ([6.0], [(0,)]),
        ([3.0 + 56 * 0.005], [(0,)]),
        ([3.0 + 56 * 0.005] * 2, [(0, 1)]),
        ([3.0 + 56 * 0.005, 3.0 + 57 * 0.005], [(0, 1)]),
        ([3.0 + 56 * 0.005, 3.0 + 56 * 0.005 + 1e-10], [(0, 1)]),
        ([5.9, 6.0, 6.1], [(0, 1, 2)]),
        ([6.0, 6.0, 6.1], [(0, 1, 2)]),
        ([6.1, 6.0, 5.9], [(0, 1, 2)]),
        ([5.0, 6.8, 6.9], [(0,), (1, 2)]),
    ],
)
def test_quel3_zero_width_ranges_plan_and_restore(frequencies, indices):
    """Zero-margin instruments use fixed sweep widths and restore their entry ranges."""
    original = InstrumentSpec(
        port_id="unit:tx_p04",
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
        assert spec.frequency_range_max_hz - spec.frequency_range_min_hz == (
            pytest.approx(1.8e9, rel=0, abs=1e-5)
        )
        assert all(
            spec.frequency_range_min_hz <= f * 1e9 <= spec.frequency_range_max_hz
            for f in segment.values
        )
        assert (spec.frequency_range_min_hz + spec.frequency_range_max_hz) / 2 == (
            pytest.approx(
                (min(segment.values) + max(segment.values)) / 2 * 1e9,
                rel=0,
                abs=1e-5,
            )
        )
        with manager.modified(segment.configuration):
            assert state == [spec]
        assert state == [original]
    assert len(calls) == 2 * sum(segment.requires_reconfiguration for segment in plan)


@pytest.mark.parametrize(
    ("port_id", "width_ghz", "sweep_width_ghz"),
    [
        ("unit:tx_p04", 1.8, 1.6),
        ("unit:trx_p00p01", 0.8, 0.6),
    ],
)
@pytest.mark.parametrize(
    "role", ["TRANSMITTER", "TRANSCEIVER", "TRANSCEIVER_LOOPBACK", "RECEIVER"]
)
@pytest.mark.parametrize("entry_width_hz", [0, 0.24e9, 4e9])
@pytest.mark.parametrize("center_ghz", [6.0, 7.502, 7.602])
def test_quel3_sweep_width_is_independent_of_deployed_range(
    port_id, width_ghz, sweep_width_ghz, role, entry_width_hz, center_ghz
):
    """Port limits reserve the frequency margin independently of role and entry range."""
    original = InstrumentSpec(
        port_id=port_id,
        alias="Q00",
        role=role,
        frequency_range_min_hz=center_ghz * 1e9 - entry_width_hz / 2,
        frequency_range_max_hz=center_ghz * 1e9 + entry_width_hz / 2,
    )
    backend = SimpleNamespace(
        get_instrument_configuration=lambda: SimpleNamespace(instruments=(original,))
    )
    manager = Quel3FrequencyConfigurationManager(cast(Any, backend))
    lower, upper = center_ghz - sweep_width_ghz / 2, center_ghz + sweep_width_ghz / 2
    plan = manager.plan([lower, upper, upper + 1e-6], target="Q00")

    assert [segment.indices for segment in plan] == [(0, 1), (2,)]
    assert all(segment.requires_reconfiguration for segment in plan)
    for segment in plan:
        spec = segment.configuration.settings.instruments[0]
        assert spec.role == original.role
        assert spec.frequency_range_max_hz - spec.frequency_range_min_hz == (
            pytest.approx(width_ghz * 1e9, rel=0, abs=1e-5)
        )
        assert all(
            (spec.frequency_range_min_hz + 0.1e9) / 1e9
            <= frequency
            <= (spec.frequency_range_max_hz - 0.1e9) / 1e9
            for frequency in segment.values
        )
    spec = plan[0].configuration.settings.instruments[0]
    assert spec.frequency_range_min_hz == pytest.approx(
        (lower - 0.1) * 1e9, rel=0, abs=1e-5
    )
    assert spec.frequency_range_max_hz == pytest.approx(
        (upper + 0.1) * 1e9, rel=0, abs=1e-5
    )


@pytest.mark.parametrize("port_id", ["unit:mon", "unit:other"])
def test_quel3_sweep_rejects_unsupported_ports(port_id):
    """Unsupported ports fail planning without deploying an instrument."""
    original = InstrumentSpec(
        port_id=port_id,
        alias="R00",
        role="TRANSMITTER",
        frequency_range_min_hz=5.9e9,
        frequency_range_max_hz=6.1e9,
    )
    backend = SimpleNamespace(
        get_instrument_configuration=lambda: InstrumentConfiguration(
            instruments=(original,)
        )
    )
    with pytest.raises(ValueError, match="port"):
        Quel3FrequencyConfigurationManager(cast(Any, backend)).plan([6.0], target="R00")


def test_quel1_baseband_endpoints_are_inclusive():
    """The safe range includes both endpoints around a quantized carrier."""
    manager = Quel1FrequencyConfigurationManager(cast(Any, Quel1Stub()))
    plan = manager.plan(
        [5.8, 6.2],
        configuration=Quel1Stub().cache,
        sideband="U",
        cnco_center=1_500_000_000,
    )
    assert len(plan) == 1


def test_quel3_overlap_rejects_reference_exceeding_width():
    """References must obey the requested width even when hardware could cover them."""
    original = InstrumentSpec(
        port_id="unit:trx_p00p01",
        alias="R00",
        role="TRANSCEIVER",
        frequency_range_min_hz=6.0e9,
        frequency_range_max_hz=6.2e9,
    )
    backend = SimpleNamespace(
        get_instrument_configuration=lambda: SimpleNamespace(instruments=(original,))
    )
    manager = Quel3FrequencyConfigurationManager(cast(Any, backend))
    with pytest.raises(ValueError, match="reference"):
        manager.plan([5.9, 6.09], target="R00", subrange_width=0.1, overlap=True)


def test_quel3_overlap_rejects_reference_exceeding_sweep_width():
    """References outside the readout sweep width are rejected before deployment."""
    original = InstrumentSpec(
        port_id="unit:trx_p00p01",
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
        manager.plan([6.0, 6.600001], target="R00", overlap=True)


def test_quel1_overlap_rejects_reference_exceeding_width():
    """QuEL-1 rejects references outside the requested width before retuning."""
    backend = Quel1Stub()
    for port in backend.cache["box"]["ports"].values():
        port.update(lo_freq=8_000_000_000, cnco_freq=1_900_000_000)
        for channel in port.get("channels", port.get("runits", {})).values():
            channel["fnco_freq"] = 0
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    with pytest.raises(ValueError, match="reference"):
        manager.plan(
            [5.8, 6.0],
            configuration=backend.cache,
            sideband="L",
            cnco_center=2_250_000_000,
            subrange_width=0.1,
            overlap=True,
        )


def test_quel1_retunes_using_reference_and_new_measurements():
    """Retuning centers the entire candidate including its boundary reference."""
    backend = Quel1Stub()
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    plan = manager.plan(
        [6.0, 6.3, 6.6],
        configuration=backend.cache,
        sideband="U",
        cnco_center=2_250_000_000,
        subrange_width=0.75,
        overlap=True,
    )
    assert [segment.indices for segment in plan] == [(0, 1), (2,)]
    assert plan[1].reference_value == 6.3
    port = plan[1].configuration.settings["box"]["ports"][1]
    assert port["lo_freq"] is not None
    assert port["cnco_freq"] is not None
    center = (port["lo_freq"] + port["cnco_freq"]) / 1e9
    assert all(abs(frequency - center) <= 0.2 for frequency in (6.3, 6.6))


def test_quel1_splits_points_outside_safe_awg_range():
    """Points inside the hardware range but outside the safe range are split."""
    backend = Quel1Stub()
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    plan = manager.plan(
        [5.78, 6.22],
        configuration=backend.cache,
        sideband="U",
        cnco_center=1_500_000_000,
    )
    assert [segment.indices for segment in plan] == [(0,), (1,)]


@pytest.mark.parametrize("sideband", ["U", "L", None])
def test_quel1_tuning_does_not_depend_on_entry_oscillators(sideband):
    """Identical sweep points produce identical tuning from different entry settings."""
    backend = Quel1Stub()
    manager = Quel1FrequencyConfigurationManager(cast(Any, backend))
    first = manager.plan(
        [6.0, 6.1],
        configuration=backend.cache,
        sideband=sideband,
        cnco_center=1_500_000_000,
    )[0]
    unchanged = manager.plan(
        [6.0, 6.1],
        configuration=first.configuration.settings,
        sideband=sideband,
        cnco_center=1_500_000_000,
    )[0]
    assert unchanged.configuration == first.configuration
    assert not unchanged.requires_reconfiguration
    assert first.requires_reconfiguration


def test_quel3_centers_single_carrier_in_native_hertz():
    """A generated carrier centers a fixed sweep window regardless of its entry range."""
    frequency = 3.0 + 56 * 0.005
    original = InstrumentSpec(
        port_id="unit:tx_p04",
        alias="Q00",
        role="TRANSMITTER",
        frequency_range_min_hz=frequency * 1e9,
        frequency_range_max_hz=frequency * 1e9,
    )
    backend = SimpleNamespace(
        get_instrument_configuration=lambda: SimpleNamespace(instruments=(original,))
    )
    manager = Quel3FrequencyConfigurationManager(cast(Any, backend))
    segment = manager.plan([frequency], target="Q00")[0]
    assert segment.values == (frequency,)
    spec = segment.configuration.settings.instruments[0]
    assert spec.frequency_range_min_hz == pytest.approx(
        frequency * 1e9 - 0.9e9, rel=0, abs=1e-5
    )
    assert spec.frequency_range_max_hz == pytest.approx(
        frequency * 1e9 + 0.9e9, rel=0, abs=1e-5
    )
    assert segment.requires_reconfiguration
