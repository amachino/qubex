"""Quantum system data models and access helpers."""

from __future__ import annotations

import math
from typing import Final

from pydantic import Field

from qubex.core import MutableModel

from .lattice_graph import LatticeGraph


class Chip(MutableModel):
    """Chip metadata and collections of qubits and resonators."""

    id: str
    name: str
    qubits: tuple[Qubit, ...]
    resonators: tuple[Resonator, ...]
    muxes: tuple[Mux, ...]

    @property
    def n_qubits(self) -> int:
        """Return the number of qubits."""
        return len(self.qubits)

    @property
    def n_resonators(self) -> int:
        """Return the number of resonators."""
        return len(self.resonators)

    @property
    def n_muxes(self) -> int:
        """Return the number of muxes."""
        return len(self.muxes)

    @classmethod
    def new(
        cls,
        id: str,
        name: str,
        n_qubits: int,
    ) -> Chip:
        """Create a new chip model with a lattice layout."""
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


class Qubit(MutableModel):
    """Qubit metadata and derived frequency helpers."""

    index: int
    label: str
    chip_id: str
    resonator: str
    bare_frequency_value: float | None = Field(default=None, alias="_bare_frequency")
    anharmonicity_value: float | None = Field(default=None, alias="_anharmonicity")
    control_frequency_ge_value: float | None = Field(
        default=None,
        alias="_control_frequency_ge",
    )
    control_frequency_ef_value: float | None = Field(
        default=None,
        alias="_control_frequency_ef",
    )

    @property
    def _bare_frequency(self) -> float | None:
        """Backward-compatible alias for bare frequency storage."""
        return self.bare_frequency_value

    @_bare_frequency.setter
    def _bare_frequency(self, value: float | None) -> None:
        """Set bare frequency via legacy attribute name."""
        self.bare_frequency_value = value

    @property
    def _anharmonicity(self) -> float | None:
        """Backward-compatible alias for anharmonicity storage."""
        return self.anharmonicity_value

    @_anharmonicity.setter
    def _anharmonicity(self, value: float | None) -> None:
        """Set anharmonicity via legacy attribute name."""
        self.anharmonicity_value = value

    @property
    def _control_frequency_ge(self) -> float | None:
        """Backward-compatible alias for GE frequency storage."""
        return self.control_frequency_ge_value

    @_control_frequency_ge.setter
    def _control_frequency_ge(self, value: float | None) -> None:
        """Set GE control frequency via legacy attribute name."""
        self.control_frequency_ge_value = value

    @property
    def _control_frequency_ef(self) -> float | None:
        """Backward-compatible alias for EF frequency storage."""
        return self.control_frequency_ef_value

    @_control_frequency_ef.setter
    def _control_frequency_ef(self, value: float | None) -> None:
        """Set EF control frequency via legacy attribute name."""
        self.control_frequency_ef_value = value

    def __repr__(self) -> str:
        """Return the debug representation of the qubit."""
        return f"Qubit('{self.label}', ω={self.frequency:.6f} GHz, α={self.anharmonicity:.6f} GHz, Ej/Ec={self.ej_over_ec:.2f})"

    @property
    def frequency(self) -> float:
        """Return the control GE frequency in GHz."""
        if self.control_frequency_ge_value is not None and not math.isnan(
            self.control_frequency_ge_value
        ):
            return self.control_frequency_ge_value
        elif self.bare_frequency_value is not None:
            return self.bare_frequency_value
        else:
            return math.nan

    @property
    def bare_frequency(self) -> float:
        """Return the bare frequency in GHz."""
        if self.bare_frequency_value is not None:
            return self.bare_frequency_value
        else:
            return math.nan

    @property
    def anharmonicity(self) -> float:
        """Return the anharmonicity in GHz."""
        if self.anharmonicity_value is not None:
            return self.anharmonicity_value
        else:
            return -(1 / 19) * self.frequency  # E_J / E_C = 50

    @property
    def control_frequency_ge(self) -> float:
        """Return the configured GE control frequency in GHz."""
        if self.control_frequency_ge_value is not None:
            return self.control_frequency_ge_value
        else:
            return math.nan

    @property
    def control_frequency_ef(self) -> float:
        """Return the configured EF control frequency in GHz."""
        if self.control_frequency_ef_value is not None:
            return self.control_frequency_ef_value
        else:
            return self.frequency + self.anharmonicity

    @property
    def alpha(self) -> float:
        """Return the angular anharmonicity in rad/ns."""
        return 2 * math.pi * self.anharmonicity

    @property
    def charging_energy(self) -> float:
        """Return the charging energy in GHz."""
        return -self.anharmonicity

    @property
    def josephson_energy(self) -> float:
        """Return the Josephson energy in GHz."""
        if self.charging_energy == 0:
            return math.nan
        return (self.frequency + self.charging_energy) ** 2 / (8 * self.charging_energy)

    @property
    def ej_over_ec(self) -> float:
        """Return the ratio $E_J/E_C$."""
        if self.charging_energy == 0:
            return math.nan
        return self.josephson_energy / self.charging_energy

    @property
    def ec_over_ej(self) -> float:
        """Return the ratio $E_C/E_J$."""
        if self.josephson_energy == 0:
            return math.nan
        return self.charging_energy / self.josephson_energy

    @property
    def is_valid(self) -> bool:
        """Return whether qubit parameters are valid."""
        return not math.isnan(self.frequency) and not math.isnan(self.anharmonicity)


class Resonator(MutableModel):
    """Resonator metadata and frequency helpers."""

    index: int
    label: str
    chip_id: str
    qubit: str
    frequency_g_value: float | None = Field(default=None, alias="_frequency_g")
    frequency_e_value: float | None = Field(default=None, alias="_frequency_e")
    readout_frequency_value: float | None = Field(
        default=None, alias="_readout_frequency"
    )

    @property
    def _frequency_g(self) -> float | None:
        """Backward-compatible alias for ground-state frequency storage."""
        return self.frequency_g_value

    @_frequency_g.setter
    def _frequency_g(self, value: float | None) -> None:
        """Set ground-state frequency via legacy attribute name."""
        self.frequency_g_value = value

    @property
    def _frequency_e(self) -> float | None:
        """Backward-compatible alias for excited-state frequency storage."""
        return self.frequency_e_value

    @_frequency_e.setter
    def _frequency_e(self, value: float | None) -> None:
        """Set excited-state frequency via legacy attribute name."""
        self.frequency_e_value = value

    @property
    def _readout_frequency(self) -> float | None:
        """Backward-compatible alias for readout frequency storage."""
        return self.readout_frequency_value

    @_readout_frequency.setter
    def _readout_frequency(self, value: float | None) -> None:
        """Set readout frequency via legacy attribute name."""
        self.readout_frequency_value = value

    def __repr__(self) -> str:
        """Return the debug representation of the resonator."""
        repr = f"Resonator(label='{self.label}', frequency={self.frequency:.6f} GHz"
        if not math.isnan(self.dispersive_shift):
            return repr + f", dispersive_shift={self.dispersive_shift:.6f} GHz)"
        else:
            return repr + ")"

    @property
    def frequency(self) -> float:
        """Return the readout frequency in GHz."""
        if self.readout_frequency_value is not None and not math.isnan(
            self.readout_frequency_value
        ):
            return self.readout_frequency_value
        elif self.frequency_g_value is not None:
            return self.frequency_g_value
        else:
            return math.nan

    @property
    def frequency_g(self) -> float:
        """Return the ground-state resonator frequency in GHz."""
        if self.frequency_g_value is not None:
            return self.frequency_g_value
        else:
            return math.nan

    @property
    def frequency_e(self) -> float:
        """Return the excited-state resonator frequency in GHz."""
        if self.frequency_e_value is not None:
            return self.frequency_e_value
        else:
            return math.nan

    @property
    def readout_frequency(self) -> float:
        """Return the readout frequency in GHz."""
        if self.readout_frequency_value is not None:
            return self.readout_frequency_value
        else:
            return math.nan

    @property
    def dispersive_shift(self) -> float:
        """Return the dispersive shift in GHz."""
        return (self.frequency_e - self.frequency_g) / 2

    @property
    def chi(self) -> float:
        """Return the dispersive shift in rad/ns."""
        return 2 * math.pi * self.dispersive_shift

    @property
    def is_valid(self) -> bool:
        """Return whether resonator parameters are valid."""
        return not math.isnan(self.frequency)


class Mux(MutableModel):
    """Mux metadata and resonator grouping."""

    index: int
    label: str
    chip_id: str
    resonators: tuple[Resonator, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether all resonators are valid."""
        return all(resonator.is_valid for resonator in self.resonators)

    @property
    def is_not_available(self) -> bool:
        """Return whether all resonators are invalid."""
        return all(not resonator.is_valid for resonator in self.resonators)


class QuantumSystem:
    """Container for chip-level models and lookup helpers."""

    def __init__(
        self,
        chip: Chip,
    ):
        """Initialize the QuantumSystem with a chip model."""
        self._graph: Final = LatticeGraph(chip.n_qubits)
        self._chip: Final = chip
        self._qubit_dict: Final = {q.label: q for q in chip.qubits}
        self._resonator_dict: Final = {r.label: r for r in chip.resonators}
        self._mux_dict: Final = {m.label: m for m in chip.muxes}

    @property
    def chip(self) -> Chip:
        """Return the chip model."""
        return self._chip

    @property
    def chip_graph(self) -> LatticeGraph:
        """Return the lattice graph for the chip."""
        return self._graph

    @property
    def hash(self) -> str:
        """Return a hash of the chip model."""
        return self.chip.hash

    @property
    def n_qubits(self) -> int:
        """Return the number of qubits."""
        return self.chip.n_qubits

    @property
    def qubits(self) -> list[Qubit]:
        """Return a list of qubit objects."""
        return list(self.chip.qubits)

    @property
    def resonators(self) -> list[Resonator]:
        """Return a list of resonator objects."""
        return list(self.chip.resonators)

    @property
    def n_muxes(self) -> int:
        """Return the number of muxes."""
        return self.chip.n_muxes

    @property
    def muxes(self) -> list[Mux]:
        """Return a list of mux objects."""
        return list(self.chip.muxes)

    def get_qubit(
        self,
        label: int | str,
    ) -> Qubit:
        """Return a qubit by index or label."""
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
        """Return a resonator by index or label."""
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
        """Return a mux by index or label."""
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
        """Return qubits that belong to a mux."""
        labels = self._graph.get_qubits_in_mux(mux)
        return [self.get_qubit(label) for label in labels]

    def get_spectator_qubits(
        self,
        qubit: int | str,
        *,
        in_same_mux: bool = False,
    ) -> list[Qubit]:
        """Return spectator qubits for the given qubit."""
        labels = self._graph.get_spectator_qubits(qubit, in_same_mux=in_same_mux)
        return [self.get_qubit(label) for label in labels]

    def set_qubit_params(
        self,
        qubit: int | str,
        *,
        frequency: float | None = None,
        anharmonicity: float | None = None,
    ) -> None:
        """Set qubit parameters for the specified qubit."""
        obj = self.get_qubit(qubit)
        # TODO: Fix SLF001
        if frequency is not None:
            obj._bare_frequency = frequency  # noqa: SLF001
        if anharmonicity is not None:
            obj._anharmonicity = anharmonicity  # noqa: SLF001

    def set_resonator_params(
        self,
        resonator: int | str,
        *,
        frequency: float | None = None,
    ) -> None:
        """Set resonator parameters for the specified resonator."""
        obj = self.get_resonator(resonator)
        # TODO: Fix SLF001
        if frequency is not None:
            obj._frequency_g = frequency  # noqa: SLF001
