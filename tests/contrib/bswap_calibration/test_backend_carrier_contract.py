"""Compare campaign carriers with real QuEL-1 adapter and DAC conversion."""

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from qxpulse import Arbitrary, Blank, PulseSchedule, Rect

from qubex.contrib.experiment.bswap_calibration.pulses import (
    compile_campaign,
    make_squad_pulse,
)
from qubex.measurement.adapters.quel1_backend_adapter import (
    Quel1MeasurementBackendAdapter,
)
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder
from qubex.measurement.models.capture_schedule import CaptureSchedule
from qubex.measurement.models.measurement_schedule import MeasurementSchedule

FIXED_FREQUENCY = 4.613
REFERENCES = {"A": 4.401, "P": 4.837}
GAIN = 0.636


def _recipes() -> dict[str, dict[str, Any]]:
    common = dict(
        amplitude=0.45,
        ramp_ns=16.0,
        cd_strength=0.5,
        design_delta_scale=1.0,
        window={"type": "hann"},
        gate_start_ns=24.0,
        cancel_amplitude_ratio=0.12,
        cancel_phase_rad=0.37,
    )
    return {
        "bswap": dict(
            common,
            gate_kind="bswap",
            frequency_ghz=4.612741332002298,
            duration_ns=264.0,
            phase_calibration=dict(
                pre_active_rad=0.0, post_active_rad=0.61, post_passive_rad=-0.82
            ),
        ),
        "sqrt_bswap": dict(
            common,
            gate_kind="sqrt_bswap",
            frequency_ghz=4.612693,
            duration_ns=144.0,
            phase_calibration=dict(
                pre_active_rad=0.73, post_active_rad=-0.31, post_passive_rad=0.24
            ),
        ),
    }


def _backend_samples(
    sequence: PulseSchedule,
    frequencies: Mapping[str, float],
    *,
    preamble_ns: float,
    skew_samples: int,
) -> dict[int, np.ndarray]:
    """Run the real conjugating adapter, multiplexing and integer DAC packing."""
    driver = pytest.importorskip("qxdriver_quel1.pulse")
    converter = pytest.importorskip("qxdriver_quel1.runtime.converter").Converter
    sysconf = pytest.importorskip("qxdriver_quel1.sysconf")
    aliases = pytest.importorskip("quel_ic_config").QUEL1_BOXTYPE_ALIAS
    profile = replace(
        MeasurementConstraintProfile.quel1(),
        require_workaround_capture=preamble_ns != 0.0,
    )
    assert preamble_ns == (
        profile.extra_capture_duration_ns if profile.require_workaround_capture else 0.0
    )
    builder = MeasurementScheduleBuilder(
        control_params=cast(Any, SimpleNamespace(readout_amplitude={})),
        pulse_factory=cast(Any, None),
        targets={
            label: cast(Any, SimpleNamespace(is_read=False))
            for label in sequence.labels
        },
        mux_dict={},
        constraint_profile=profile,
    )
    built = builder.build(
        schedule=sequence.copy(),
        final_measurement=False,
        capture_placement="entire_schedule",
        capture_targets=[next(iter(frequencies))],
    )
    sampled = built.pulse_schedule.get_sampled_sequences()
    # Control channels have no readout-phase compensation. Captures are omitted
    # here so the actual control adapter path is isolated without any hardware.
    with PulseSchedule(list(frequencies)) as controls:
        for label in frequencies:
            controls.add(label, Arbitrary(sampled[label]))
    system = SimpleNamespace(
        get_target=lambda _: SimpleNamespace(sideband=None),
        get_awg_frequency=lambda label: frequencies[label] - 4.6171875,
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, SimpleNamespace(driver=driver)),
        experiment_system=cast(Any, system),
    )
    generated, _ = adapter._create_sampled_sequences(  # noqa: SLF001
        schedule=MeasurementSchedule(
            pulse_schedule=controls, capture_schedule=CaptureSchedule(captures=[])
        )
    )
    box = sysconf.BoxSetting(
        box_name="B0", ipaddr_wss="127.0.0.1", boxtype=next(iter(aliases.values()))
    )
    resources: dict[str, Any] = {}
    configs: dict[str, Any] = {}
    for label, frequency in frequencies.items():
        port = 8 if label.startswith("D") else 9
        cnco, fnco = (
            (4_968_750_000.0, -351_562_500.0)
            if port == 8
            else (4_265_625_000.0, 351_562_500.0)
        )
        cast(Any, generated[label]).padding = skew_samples
        resources[label] = dict(
            box=box,
            port=sysconf.PortSetting(
                port_name=f"P{port}", box_name="B0", port=port, ndelay_or_nwait=(0, 0)
            ),
            channel_number=1,
            target={"frequency": frequency},
        )
        configs[label] = SimpleNamespace(
            lo_freq=None,
            cnco_freq=cnco,
            fnco_freq=fnco,
            sideband=None,
            dump_config={"direction": "out"},
            box_name="B0",
            port=port,
            channel=1,
        )
    converted = converter.convert_to_gen_device_specific_sequence(
        gen_sampled_sequence=generated,
        cap_sampled_sequence={},
        resource_map=resources,
        port_config=configs,
        repeats=1,
        interval=8192.0,
    )
    return {
        port: np.asarray(sequence.chunk(0).wave_data.samples)
        for (_, port, _), sequence in converted.items()
    }


def _manual_frequency_oracle(
    compiled: PulseSchedule, report: dict[str, Any], recipes: dict[str, dict[str, Any]]
) -> tuple[PulseSchedule, dict[str, float]]:
    """Use per-kind target aliases, with no encoded chirp, as the independent oracle."""
    frequencies = {
        f"{tone}_{kind}": recipe["frequency_ghz"]
        for kind, recipe in recipes.items()
        for tone in ("D", "C")
    }
    oracle = PulseSchedule(list(frequencies))
    with oracle:
        for event in report["events"]:
            kind = event["kind"]
            if kind == "local":
                continue
            recipe = recipes[kind]
            pulse = make_squad_pulse(
                recipe,
                rabi_ghz_per_amplitude=GAIN,
                transition_frequency_ghz=REFERENCES["A"],
            )
            local_start = event["start_ns"] - report["global_start_ns"]
            for tone, scale, phase in (
                ("D", 1.0, 0.0),
                ("C", recipe["cancel_amplitude_ratio"], recipe["cancel_phase_rad"]),
            ):
                label = f"{tone}_{kind}"
                oracle.add(
                    label, Blank(local_start - oracle.get_sequence(label).duration)
                )
                oracle.add(
                    label,
                    pulse.scaled(scale).shifted(
                        event["logical_drive_phase_rad"] + phase
                    ),
                )
        oracle.pad(total_duration=compiled.duration, pad_side="right")
    return oracle, frequencies


@pytest.mark.parametrize("gate", ["BSWAP", "RAW_SQRT_BSWAP", "mixed"])
@pytest.mark.parametrize("origin", [0.0, 126.0])
@pytest.mark.parametrize("skew_samples", [0, 3])
@pytest.mark.parametrize("preamble_ns", [0.0, 40.0])
def test_encoded_carriers_match_manual_target_dac_samples(
    gate: str, origin: float, skew_samples: int, preamble_ns: float
) -> None:
    """Fixed-target chirps match manual frequency lowering after all actual transforms."""
    recipes = _recipes()
    gates: list[Any] = [("VZ", 0.41, -0.29), ("XY", 0.3, -0.7)]
    gates += ["BSWAP", "ROOT_PAIR", "XX90"] if gate == "mixed" else [gate]
    compiled, report = compile_campaign(
        gates,
        recipes=recipes,
        qubits=("A", "P"),
        drive_label="D",
        cancel_label="C",
        target_frequencies_ghz={"D": FIXED_FREQUENCY, "C": FIXED_FREQUENCY},
        reference_frequencies_ghz=REFERENCES,
        rabi_ghz_per_amplitude=GAIN,
        x90={q: Rect(duration=16, amplitude=0.1) for q in REFERENCES},
        xpi={q: Rect(duration=24, amplitude=0.2) for q in REFERENCES},
        prepared=("+", "+i"),
        basis=None,
        delay_ns=18.0,
        initial_frame=(0.31, -0.73),
        global_start_ns=origin,
        backend_preamble_ns=preamble_ns,
    )
    manual, manual_frequencies = _manual_frequency_oracle(compiled, report, recipes)
    for schedule in (compiled, manual):
        schedule.pad(total_duration=schedule.duration + origin, pad_side="left")
    actual = _backend_samples(
        compiled,
        {"D": FIXED_FREQUENCY, "C": FIXED_FREQUENCY},
        preamble_ns=preamble_ns,
        skew_samples=skew_samples,
    )
    expected = _backend_samples(
        manual, manual_frequencies, preamble_ns=preamble_ns, skew_samples=skew_samples
    )
    assert report["backend_preamble_ns"] == preamble_ns
    for port in (8, 9):
        np.testing.assert_array_equal(actual[port], expected[port])


def test_omitted_measurement_preamble_is_a_phase_error_not_a_chirp_sign_error() -> None:
    """Omitting the real 40 ns prefix reproduces the known fixed phase mismatch."""
    assert MeasurementConstraintProfile.quel1().extra_capture_duration_ns == 40.0
    recipe = _recipes()["bswap"]
    pulse = make_squad_pulse(
        recipe, rabi_ghz_per_amplitude=GAIN, transition_frequency_ghz=REFERENCES["A"]
    )
    np.testing.assert_array_equal(pulse.times[:3], [0.0, 2.0, 4.0])
    delta = recipe["frequency_ghz"] - FIXED_FREQUENCY
    with PulseSchedule(["D"]) as manual:
        manual.add("D", Blank(24.0))
        manual.add("D", pulse)
    with PulseSchedule(["D"]) as omitted:
        omitted.add("D", Blank(24.0))
        omitted.add("D", pulse.detuned(delta).shifted(-2 * np.pi * delta * 24.0))
    expected = _backend_samples(
        manual, {"D": recipe["frequency_ghz"]}, preamble_ns=40.0, skew_samples=0
    )[8]
    actual = _backend_samples(
        omitted, {"D": FIXED_FREQUENCY}, preamble_ns=40.0, skew_samples=0
    )[8]
    expected_complex = expected[:, 0] + 1j * expected[:, 1]
    actual_complex = actual[:, 0] + 1j * actual[:, 1]
    phase = np.angle(np.vdot(expected_complex, actual_complex))
    assert phase == pytest.approx(-2 * np.pi * delta * 40.0, abs=2e-5)
    assert np.max(np.abs(actual - expected)) > 100
