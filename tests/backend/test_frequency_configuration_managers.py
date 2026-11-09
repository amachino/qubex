"""Backend-specific frequency configuration contracts without hardware."""

from copy import deepcopy
from typing import Any, cast

import pytest

from qubex.backend.quel1.managers.frequency_configuration_manager import (
    Quel1FrequencyConfigurationManager,
)
from qubex.backend.quel1.sweep_configuration import (
    Quel1BoxFrequencySettings,
    Quel1SweepConfiguration,
)


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
