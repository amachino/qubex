"""Tests for ChipInspector."""

from __future__ import annotations

from dataclasses import dataclass

from qubex.diagnostics import chip_inspector


@dataclass
class _FakeGraph:
    """Minimal graph stub."""

    qubit_nodes: dict[int, dict]
    qubit_edges: dict[int, dict]


class _FakeConfigLoader:
    """ConfigLoader stub recording selector inputs."""

    init_kwargs: dict[str, object] | None = None
    get_experiment_system_args: tuple[object, ...] | None = None

    def __init__(self, **kwargs: object) -> None:
        self.__class__.init_kwargs = kwargs

    def get_experiment_system(self, *args: object) -> object:
        """Return a fake experiment system."""
        self.__class__.get_experiment_system_args = args
        return _FakeExperimentSystem()

    def load_param_data(self, name: str) -> dict[str, float]:
        """Return empty parameter data."""
        del name
        return {}


class _FakeExperimentSystem:
    """Minimal experiment-system stub."""

    def __init__(self) -> None:
        self.quantum_system = _FakeQuantumSystem()


class _FakeQuantumSystem:
    """Minimal quantum-system stub."""

    def __init__(self) -> None:
        self.chip_graph = _FakeGraph(
            qubit_nodes={0: {"label": "Q000"}},
            qubit_edges={},
        )


def test_chip_inspector_uses_system_id_selector(monkeypatch) -> None:
    """Given system_id input, ChipInspector should initialize ConfigLoader with it."""
    monkeypatch.setattr(chip_inspector, "ConfigLoader", _FakeConfigLoader)

    chip_inspector.ChipInspector(
        system_id="144Q-LF-Q3",
        config_dir="config",
        params_dir="params",
    )

    assert _FakeConfigLoader.init_kwargs == {
        "chip_id": None,
        "system_id": "144Q-LF-Q3",
        "config_dir": "config",
        "params_dir": "params",
    }
    assert _FakeConfigLoader.get_experiment_system_args == ()


def test_chip_inspector_accepts_legacy_props_dir(monkeypatch) -> None:
    """Given props_dir input, ChipInspector should map it to params_dir."""
    monkeypatch.setattr(chip_inspector, "ConfigLoader", _FakeConfigLoader)

    chip_inspector.ChipInspector(
        chip_id="64Q-HF",
        config_dir="config",
        props_dir="props",
    )

    assert _FakeConfigLoader.init_kwargs == {
        "chip_id": "64Q-HF",
        "system_id": None,
        "config_dir": "config",
        "params_dir": "props",
    }
