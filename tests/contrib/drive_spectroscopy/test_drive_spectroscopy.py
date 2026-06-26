"""Tests for drive spectroscopy contrib helper."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

from numpy.testing import assert_allclose

from qubex.contrib.experiment.drive_spectroscopy import drive_spectroscopy
from qubex.pulse import FlatTop


class _Capture:
    def __init__(self, value: complex) -> None:
        self.kerneled = value


class _MeasurementResult:
    def __init__(self, data: dict[str, complex]) -> None:
        self.data = {target: _Capture(value) for target, value in data.items()}


class _PulseService:
    def __init__(self) -> None:
        self.readout_calls: list[dict[str, Any]] = []

    def readout(self, target: str, **kwargs: Any) -> FlatTop:
        self.readout_calls.append({"target": target, **kwargs})
        return FlatTop(duration=16, amplitude=1.0, tau=4)


class _Context:
    def __init__(self, read_labels: dict[str, str] | None = None) -> None:
        self.qubit_labels = ["Q00", "Q01"]
        self.read_labels = read_labels or {"Q00": "RQ00", "Q01": "RQ01"}
        self.params = SimpleNamespace(
            readout_amplitude={"Q00": 0.30, "Q01": 0.40},
        )
        self.targets = {
            "RQ00": SimpleNamespace(frequency=7.00),
            "RQ01": SimpleNamespace(frequency=7.10),
        }
        self.frequency_contexts: list[dict[str, float]] = []
        self.reset_calls: list[list[str]] = []
        self.backend_settings: list[dict[str, float | int | str | None]] = []
        self.port_configs: list[dict[str, Any]] = []
        self.channel_configs: list[dict[str, Any]] = []
        self.cache_updates: list[dict[str, Any]] = []
        self.replaced_caches: list[dict[str, Any]] = []
        self.initialized_boxes: list[list[str]] = []
        self.update_port_params_calls: list[dict[str, Any]] = []
        self._shared_port = SimpleNamespace(
            box_id="box0",
            number=0,
            sideband="L",
            lo_freq=10_000_000_000,
            cnco_freq=2_000_000_000,
        )
        self._ge_channel = SimpleNamespace(
            port=self._shared_port,
            number=0,
            fnco_freq=0,
        )
        self._drive_channel = SimpleNamespace(
            port=self._shared_port,
            number=1,
            fnco_freq=0,
        )
        self._box_cache = {
            "box0": {
                "ports": {
                    0: {
                        "lo_freq": self._shared_port.lo_freq,
                        "cnco_freq": self._shared_port.cnco_freq,
                        "channels": {
                            0: {"fnco_freq": self._ge_channel.fnco_freq},
                            1: {"fnco_freq": self._drive_channel.fnco_freq},
                        },
                    }
                }
            }
        }
        self._ge_target = SimpleNamespace(
            label="Q00",
            frequency=7.90,
            channel=self._ge_channel,
        )
        self.system_manager = SimpleNamespace(
            modified_backend_settings=self._modified_backend_settings,
            backend_controller=SimpleNamespace(
                config_port=self._config_port,
                config_channel=self._config_channel,
                initialize_awg_and_capunits=self._initialize_awg_and_capunits,
                update_box_config_cache=self._update_box_config_cache,
            ),
            _get_box_config_cache_snapshot=self._get_box_config_cache_snapshot,
            _replace_box_config_cache=self._replace_box_config_cache,
        )
        self._drive_target = SimpleNamespace(
            label="Q00-Q01-bSWAP",
            frequency=8.10,
            channel=self._drive_channel,
        )
        self.experiment_system = SimpleNamespace(
            get_target=self._get_target,
            get_control_box_for_qubit=lambda _qubit: SimpleNamespace(
                traits=SimpleNamespace(
                    default_control_frequency_range=(8.10, 8.12, 0.01)
                )
            ),
            resolve_ge_label=lambda label: label,
            update_port_params=self._update_port_params,
        )

    def resolve_qubit_label(self, target: str) -> str:
        if target == "Q00-Q01-bSWAP":
            return "Q00"
        return target

    def resolve_ge_label(self, target: str) -> str:
        return self.resolve_qubit_label(target)

    def resolve_read_label(self, target: str) -> str:
        return self.read_labels[target]

    def _get_target(self, target: str):
        if target == "Q00-Q01-bSWAP":
            return self._drive_target
        if target == "Q00":
            return self._ge_target
        raise KeyError(target)

    def reset_awg_and_capunits(self, *, qubits: list[str]) -> None:
        self.reset_calls.append(qubits)

    @contextmanager
    def modified_frequencies(self, frequencies: dict[str, float]):
        self.frequency_contexts.append(frequencies)
        yield

    @contextmanager
    def _modified_backend_settings(self, **settings: float | int | str | None):
        self.backend_settings.append(dict(settings))
        yield

    def _config_port(self, **kwargs: Any) -> None:
        self.port_configs.append(dict(kwargs))

    def _config_channel(self, **kwargs: Any) -> None:
        self.channel_configs.append(dict(kwargs))

    def _initialize_awg_and_capunits(self, box_ids: list[str]) -> None:
        self.initialized_boxes.append(list(box_ids))

    def _update_box_config_cache(self, cache: dict[str, Any]) -> None:
        self.cache_updates.append(deepcopy(cache))

    def _get_box_config_cache_snapshot(self) -> dict[str, Any]:
        return deepcopy(self._box_cache)

    def _replace_box_config_cache(self, cache: dict[str, Any]) -> None:
        self.replaced_caches.append(deepcopy(cache))
        self._box_cache = deepcopy(cache)

    def _update_port_params(
        self,
        label: str,
        *,
        lo_freq: int | None,
        cnco_freq: int,
        fnco_freq: int,
    ) -> None:
        target = self._get_target(label)
        target.channel.port.lo_freq = lo_freq
        target.channel.port.cnco_freq = cnco_freq
        target.channel.fnco_freq = fnco_freq
        self.update_port_params_calls.append(
            {
                "label": label,
                "lo_freq": lo_freq,
                "cnco_freq": cnco_freq,
                "fnco_freq": fnco_freq,
            }
        )


class _Experiment:
    def __init__(self, read_labels: dict[str, str] | None = None) -> None:
        self.ctx = _Context(read_labels=read_labels)
        self.pulse = _PulseService()
        self.measure_calls: list[dict[str, Any]] = []

    def measure(self, sequence: Any, **kwargs: Any) -> _MeasurementResult:
        self.measure_calls.append({"sequence": sequence, **kwargs})
        point_index = len(self.measure_calls)
        with self.ctx.modified_frequencies(kwargs["frequencies"]):
            return _MeasurementResult(
                {
                    qubit: complex(point_index, target_index + 1)
                    for target_index, qubit in enumerate(self.ctx.qubit_labels)
                }
            )


def test_drive_spectroscopy_drives_one_target_and_measures_all_context_qubits() -> None:
    """Given no measure_qubits, drive spectroscopy reads every context qubit."""
    exp = _Experiment()

    result = drive_spectroscopy(
        cast(Any, exp),
        target="Q00-Q01-bSWAP",
        drive_pulse=FlatTop(duration=16, amplitude=1.0, tau=4),
        frequency_range=[8.10, 8.11],
        amplitude_range=[0.10],
        n_shots=64,
        shot_interval=1234,
        plot=False,
        save_image=False,
    )

    assert len(exp.measure_calls) == 2
    assert exp.ctx.frequency_contexts == [
        {"Q00-Q01-bSWAP": 8.10, "RQ00": 7.00, "RQ01": 7.10},
        {"Q00-Q01-bSWAP": 8.11, "RQ00": 7.00, "RQ01": 7.10},
    ]
    assert exp.ctx.reset_calls == [["Q00", "Q01"]]
    assert exp.ctx.backend_settings == []
    assert len(exp.ctx.port_configs) == 2
    assert {call["channel"] for call in exp.ctx.channel_configs[:2]} == {0, 1}
    assert [call["label"] for call in exp.ctx.update_port_params_calls[:2]] == [
        "Q00-Q01-bSWAP",
        "Q00",
    ]
    assert set(result.data["frequency_bands"][0]["fnco_freqs"]) == {
        "Q00-Q01-bSWAP",
        "Q00",
    }
    assert (
        abs(cast(int, result.data["frequency_bands"][0]["fnco_freqs"]["Q00-Q01-bSWAP"]))
        <= 200_000_000
    )

    first_call = exp.measure_calls[0]
    assert first_call["sequence"].labels == ["Q00-Q01-bSWAP", "Q01"]
    assert list(first_call["sequence"].get_sampled_sequences()) == [
        "Q00-Q01-bSWAP",
        "Q01",
    ]
    assert first_call["n_shots"] == 64
    assert first_call["shot_interval"] == 1234
    assert first_call["reset_awg_and_capunits"] is False
    assert first_call["readout_amplitudes"] == {"Q00": 0.30, "Q01": 0.40}
    assert exp.pulse.readout_calls == []

    assert result.data["target"] == "Q00-Q01-bSWAP"
    assert result.data["target_qubit"] == "Q00"
    assert result.data["measure_qubits"] == ["Q00", "Q01"]
    assert result.data["data_kind"] == "magnitude"
    assert result.data["pca_data_kind"] == "per_band_iq_pca"
    assert_allclose(result.data["amplitude_range"], [0.10])
    assert set(result.data["measurements"]) == {"Q00", "Q01"}
    assert result.data["measurements"]["Q00"]["pca_projection"].shape == (1, 2)
    assert result.data["measurements"]["Q01"]["pca_projection"].shape == (1, 2)
    assert result.data["measurements"]["Q00"]["pca_projection_row_detrended"].shape == (
        1,
        2,
    )
    assert len(result.data["measurements"]["Q00"]["pca_projection_metadata"]) == 1
    assert result.figures is not None
    assert set(result.figures) == {"Q00", "Q01"}


def test_drive_spectroscopy_can_measure_selected_qubits() -> None:
    """Given measure_qubits, drive spectroscopy only reads those qubits."""
    exp = _Experiment()

    result = drive_spectroscopy(
        cast(Any, exp),
        target="Q00-Q01-bSWAP",
        drive_pulse=FlatTop(duration=16, amplitude=1.0, tau=4),
        measure_qubits=["Q01"],
        frequency_range=[8.10],
        amplitude_range=[0.20],
        readout_amplitude=0.50,
        readout_frequency={"Q01": 7.20},
        plot=False,
        save_image=False,
    )

    assert exp.ctx.frequency_contexts == [
        {"Q00-Q01-bSWAP": 8.10, "RQ01": 7.20},
    ]
    assert exp.ctx.reset_calls == [["Q00", "Q01"]]
    assert exp.measure_calls[0]["readout_amplitudes"] == {"Q01": 0.50}
    assert exp.measure_calls[0]["sequence"].labels == ["Q00-Q01-bSWAP", "Q01"]
    assert result.data["measure_qubits"] == ["Q01"]
    assert set(result.data["measurements"]) == {"Q01"}


def test_drive_spectroscopy_sweeps_all_amplitudes_inside_each_frequency_band() -> None:
    """Given multiple bands, all amplitudes are swept before moving to the next band."""
    exp = _Experiment()

    result = drive_spectroscopy(
        cast(Any, exp),
        target="Q00-Q01-bSWAP",
        drive_pulse=FlatTop(duration=16, amplitude=1.0, tau=4),
        frequency_range=[8.10, 8.11, 8.60],
        amplitude_range=[0.10, 0.20],
        plot=False,
        save_image=False,
    )

    assert exp.ctx.frequency_contexts == [
        {"Q00-Q01-bSWAP": 8.10, "RQ00": 7.00, "RQ01": 7.10},
        {"Q00-Q01-bSWAP": 8.11, "RQ00": 7.00, "RQ01": 7.10},
        {"Q00-Q01-bSWAP": 8.10, "RQ00": 7.00, "RQ01": 7.10},
        {"Q00-Q01-bSWAP": 8.11, "RQ00": 7.00, "RQ01": 7.10},
        {"Q00-Q01-bSWAP": 8.60, "RQ00": 7.00, "RQ01": 7.10},
        {"Q00-Q01-bSWAP": 8.60, "RQ00": 7.00, "RQ01": 7.10},
    ]
    assert len(exp.ctx.port_configs) == 4
    assert exp.ctx.reset_calls == [["Q00", "Q01"], ["Q00", "Q01"]]
    assert len(result.data["frequency_bands"]) == 2
    assert_allclose(
        result.data["measurements"]["Q00"]["signals"],
        [[1 + 1j, 2 + 1j, 5 + 1j], [3 + 1j, 4 + 1j, 6 + 1j]],
    )
    assert result.data["measurements"]["Q00"]["pca_projection"].shape == (2, 3)
    assert len(result.data["measurements"]["Q00"]["pca_projection_metadata"]) == 2
