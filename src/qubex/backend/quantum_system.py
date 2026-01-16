from __future__ import annotations

import math
from typing import Final

from pydantic.dataclasses import dataclass

from .lattice_graph import LatticeGraph
from .model import Model


@dataclass
class Chip(Model):
    id: str
    name: str
    qubits: tuple[Qubit, ...]
    resonators: tuple[Resonator, ...]
    muxes: tuple[Mux, ...]

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    @property
    def n_resonators(self) -> int:
        return len(self.resonators)

    @property
    def n_muxes(self) -> int:
        return len(self.muxes)

    @classmethod
    def new(
        cls,
        id: str,
        name: str,
        n_qubits: int,
    ) -> Chip:
        graph = LatticeGraph(n_qubits)

        qubits = tuple(
            Qubit(
                index=index,
                label=label,
                chip_id=id,
                resonator=graph.resonators[index],
            )
            for index, label in enumerate(graph.qubits)
        )
        resonators = tuple(
            Resonator(
                index=index,
                label=label,
                chip_id=id,
                qubit=graph.qubits[index],
            )
            for index, label in enumerate(graph.resonators)
        )
        muxes = tuple(
            Mux(
                index=index,
                label=label,
                chip_id=id,
                resonators=tuple(
                    resonators[index] for index in graph.get_indices_in_mux(label)
                ),
            )
            for index, label in enumerate(graph.muxes)
        )
        return cls(
            id=id,
            name=name,
            qubits=qubits,
            resonators=resonators,
            muxes=muxes,
        )


@dataclass
class Qubit(Model):
    index: int
    label: str
    chip_id: str
    resonator: str
    _bare_frequency: float | None = None
    _anharmonicity: float | None = None
    _control_frequency_ge: float | None = None
    _control_frequency_ef: float | None = None

    def __repr__(self) -> str:
        return f"Qubit('{self.label}', ω={self.frequency:.6f} GHz, α={self.anharmonicity:.6f} GHz, Ej/Ec={self.ej_over_ec:.2f})"

    @property
    def frequency(self) -> float:
        if self._control_frequency_ge is not None and not math.isnan(
            self._control_frequency_ge
        ):
            return self._control_frequency_ge
        elif self._bare_frequency is not None:
            return self._bare_frequency
        else:
            return math.nan

    @property
    def bare_frequency(self) -> float:
        if self._bare_frequency is not None:
            return self._bare_frequency
        else:
            return math.nan

    @property
    def anharmonicity(self) -> float:
        if self._anharmonicity is not None:
            return self._anharmonicity
        else:
            return -(1 / 19) * self.frequency  # E_J / E_C = 50

    @property
    def control_frequency_ge(self) -> float:
        if self._control_frequency_ge is not None:
            return self._control_frequency_ge
        else:
            return math.nan

    @property
    def control_frequency_ef(self) -> float:
        if self._control_frequency_ef is not None:
            return self._control_frequency_ef
        else:
            return self.frequency + self.anharmonicity

    @property
    def alpha(self) -> float:
        return 2 * math.pi * self.anharmonicity

    @property
    def charging_energy(self) -> float:
        return -self.anharmonicity

    @property
    def josephson_energy(self) -> float:
        if self.charging_energy == 0:
            return math.nan
        return (self.frequency + self.charging_energy) ** 2 / (8 * self.charging_energy)

    @property
    def ej_over_ec(self) -> float:
        if self.charging_energy == 0:
            return math.nan
        return self.josephson_energy / self.charging_energy

    @property
    def ec_over_ej(self) -> float:
        if self.josephson_energy == 0:
            return math.nan
        return self.charging_energy / self.josephson_energy

    @property
    def is_valid(self) -> bool:
        return not math.isnan(self.frequency) and not math.isnan(self.anharmonicity)


@dataclass
class Resonator(Model):
    index: int
    label: str
    chip_id: str
    qubit: str
    _frequency_g: float | None = None
    _frequency_e: float | None = None
    _readout_frequency: float | None = None

    def __repr__(self) -> str:
        repr = f"Resonator(label='{self.label}', frequency={self.frequency:.6f} GHz"
        if not math.isnan(self.dispersive_shift):
            return repr + f", dispersive_shift={self.dispersive_shift:.6f} GHz)"
        else:
            return repr + ")"

    @property
    def frequency(self) -> float:
        if self._readout_frequency is not None and not math.isnan(
            self._readout_frequency
        ):
            return self._readout_frequency
        elif self._frequency_g is not None:
            return self._frequency_g
        else:
            return math.nan

    @property
    def frequency_g(self) -> float:
        if self._frequency_g is not None:
            return self._frequency_g
        else:
            return math.nan

    @property
    def frequency_e(self) -> float:
        if self._frequency_e is not None:
            return self._frequency_e
        else:
            return math.nan

    @property
    def readout_frequency(self) -> float:
        if self._readout_frequency is not None:
            return self._readout_frequency
        else:
            return math.nan

    @property
    def dispersive_shift(self) -> float:
        return (self.frequency_e - self.frequency_g) / 2

    @property
    def chi(self) -> float:
        return 2 * math.pi * self.dispersive_shift

    @property
    def is_valid(self) -> bool:
        return not math.isnan(self.frequency)


@dataclass
class Mux(Model):
    index: int
    label: str
    chip_id: str
    resonators: tuple[Resonator, ...]

    @property
    def is_valid(self) -> bool:
        return all(resonator.is_valid for resonator in self.resonators)

    @property
    def is_not_available(self) -> bool:
        return all(not resonator.is_valid for resonator in self.resonators)


class QuantumSystem:
    def __init__(
        self,
        chip: Chip,
    ):
        self._graph: Final = LatticeGraph(chip.n_qubits)
        self._chip: Final = chip
        self._qubit_dict: Final = {q.label: q for q in chip.qubits}
        self._resonator_dict: Final = {r.label: r for r in chip.resonators}
        self._mux_dict: Final = {m.label: m for m in chip.muxes}

    @property
    def chip(self) -> Chip:
        return self._chip

    @property
    def chip_graph(self) -> LatticeGraph:
        return self._graph

    @property
    def hash(self) -> int:
        return self.chip.hash

    @property
    def n_qubits(self) -> int:
        return self.chip.n_qubits

    @property
    def qubits(self) -> list[Qubit]:
        return list(self.chip.qubits)

    @property
    def resonators(self) -> list[Resonator]:
        return list(self.chip.resonators)

    @property
    def n_muxes(self) -> int:
        return self.chip.n_muxes

    @property
    def muxes(self) -> list[Mux]:
        return list(self.chip.muxes)

    def get_qubit(
        self,
        label: int | str,
    ) -> Qubit:
        try:
            if isinstance(label, int):
                return self.qubits[label]
            else:
                return self._qubit_dict[label]
        except KeyError:
            raise KeyError(f"Qubit `{label}` not found.") from None

    def get_resonator(
        self,
        label: int | str,
    ) -> Resonator:
        try:
            if isinstance(label, int):
                return self.resonators[label]
            else:
                return self._resonator_dict[label]
        except KeyError:
            raise KeyError(f"Resonator `{label}` not found.") from None

    def get_mux(
        self,
        label: int | str,
    ) -> Mux:
        try:
            if isinstance(label, int):
                return self.muxes[label]
            else:
                return self._mux_dict[label]
        except KeyError:
            raise KeyError(f"Mux `{label}` not found.") from None

    def get_qubits_in_mux(
        self,
        mux: int | str,
    ) -> list[Qubit]:
        labels = self._graph.get_qubits_in_mux(mux)
        return [self.get_qubit(label) for label in labels]

    def get_spectator_qubits(
        self,
        qubit: int | str,
        *,
        in_same_mux: bool = False,
    ) -> list[Qubit]:
        labels = self._graph.get_spectator_qubits(qubit, in_same_mux=in_same_mux)
        return [self.get_qubit(label) for label in labels]

    def set_qubit_params(
        self,
        qubit: int | str,
        *,
        frequency: float | None = None,
        anharmonicity: float | None = None,
    ) -> None:
        obj = self.get_qubit(qubit)
        if frequency is not None:
            obj._bare_frequency = frequency
        if anharmonicity is not None:
            obj._anharmonicity = anharmonicity

    def set_resonator_params(
        self,
        resonator: int | str,
        *,
        frequency: float | None = None,
    ) -> None:
        obj = self.get_resonator(resonator)
        if frequency is not None:
            obj._frequency_g = frequency
