from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..backend.config_loader import ConfigLoader

CONDITIONS = {
    "max_frequency": 9.5,
    "min_frequency": 6.5,
    "max_detuning": 1.3,
    "min_t1": 3e3,
    "min_t2": 3e3,
    "b1": 0.2,
    "b2": 0.75,
}

DEFAULTS = {
    "t1": 3e3,
    "t2_echo": 3e3,
    "coupling": 0.008,
}


CollisionType = Literal[
    "Type0A",
    "Type0B",
    "Type1A",
    "Type1B",
    "Type1C",
    "Type2A",
    "Type2B",
    "Type3A",
    "Type3B",
    "Type7",
    "Type8",
    "Type9",
]


@dataclass
class Collision:
    name: str
    description: str
    n_nodes: int
    directed: bool
    node_data: dict
    edge_data: dict

    @property
    def collision_nodes(self) -> dict[str, list[str]]:
        return {k: v["log"] for k, v in self.node_data.items() if v["collision"]}

    @property
    def collision_edges(self) -> dict[str, list[str]]:
        return {k: v["log"] for k, v in self.edge_data.items() if v["collision"]}


class CollisionChecker:
    def __init__(
        self,
        chip_id: str,
        params_dir: str | None = None,
        conditions: dict | None = None,
    ):
        if params_dir is None:
            config_loader = ConfigLoader()
        else:
            config_loader = ConfigLoader(params_dir=params_dir)

        experiment_system = config_loader.get_experiment_system(chip_id)
        self.graph = experiment_system.quantum_system._graph
        props = config_loader._props_dict[chip_id]

        for node in self.graph.qubit_nodes.values():
            label = node["label"]
            node["properties"] = {
                "frequency": props["qubit_frequency"].get(label),
                "anharmonicity": props["anharmonicity"].get(label),
                "t1": props["t1"].get(label),
                "t2_echo": props["t2_echo"].get(label),
            }

        for edge in self.graph.qubit_edges.values():
            label = edge["label"]
            edge["properties"] = {
                "coupling": props["qubit_qubit_coupling_strength"].get(label),
            }

        self.conditions = CONDITIONS
        if conditions is not None:
            self.conditions.update(conditions)

        self.collisions: dict[CollisionType, Collision] = {}

    def get_value(
        self,
        target: int | tuple[int, int],
        property: str,
        default: float = np.nan,
    ) -> float:
        if isinstance(target, int):
            value = self.graph.qubit_nodes[target]["properties"][property]
            if value is None or np.isnan(value):
                return default
            else:
                return value
        elif isinstance(target, tuple):
            value = self.graph.qubit_edges[target]["properties"][property]
            if value is None or np.isnan(value):
                return default
            else:
                return value
        else:
            raise ValueError("Invalid target type.")

    def check(
        self,
        *collision: CollisionType,
    ) -> dict[CollisionType, Collision]:
        result = {}
        for type in collision:
            result[type] = getattr(self, f"check_{type.lower()}")()

        self.collisions = result
        return result

    def check_type0a(
        self,
        min_frequency: float | None = None,
        max_frequency: float | None = None,
        min_t1: float | None = None,
        min_t2: float | None = None,
    ) -> Collision:
        """
        conditions:
            (1) qubit unmeasured
            (2) qubit frequency out of range
            (3) qubit t1, t2 is too short
        """
        result = {}
        for i, data in self.graph.qubit_nodes.items():
            flag = False
            label = data["label"]
            log = []
            f = self.get_value(i, "frequency", np.nan)
            f_min = min_frequency or self.conditions["min_frequency"]
            f_max = max_frequency or self.conditions["max_frequency"]
            t1 = self.get_value(i, "t1", DEFAULTS["t1"])
            t1_min = min_t1 or self.conditions["min_t1"]
            t2 = self.get_value(i, "t2_echo", DEFAULTS["t2_echo"])
            t2_min = min_t2 or self.conditions["min_t2"]
            if np.isnan(f):
                flag = True
                log.append(f"Frequency of {label} is not defined.")
            if f < f_min:
                flag = True
                log.append(
                    f"Frequency of {label} ({f:.3f} GHz) is lower than {f_min:.3f} GHz."
                )
            if f > f_max:
                flag = True
                log.append(
                    f"Frequency of {label} ({f:.3f} GHz) is higher than {f_max:.3f} GHz."
                )
            if t1 < t1_min:
                flag = True
                log.append(
                    f"T1 of {label} ({t1 * 1e-3:.1f} μs) is lower than {t1_min * 1e-3:.1f} μs."
                )
            if t2 < t2_min:
                flag = True
                log.append(
                    f"T2 of {label} ({t2 * 1e-3:.1f} μs) is lower than {t2_min * 1e-3:.1f} μs."
                )

            result[label] = {
                "collision": flag,
                "log": log,
            }

        result = dict(sorted(result.items()))
        return Collision(
            name="Type0A",
            description="Qubit unmeasured, frequency out of range, T1, T2 too short.",
            n_nodes=1,
            directed=False,
            node_data=result,
            edge_data={},
        )

    def check_type0b(
        self,
        max_detuning: float | None = None,
    ):
        """
        conditions:
            (1) ge(i)-ge(j) is too far to implement the fast CR(i>j) or CR(j>i)
        """
        result = {}
        for (i, j), data in self.graph.qubit_links.items():
            flag = False
            label = data["label"]
            log = []
            f_i = self.get_value(i, "frequency")
            f_j = self.get_value(j, "frequency")
            f_diff = abs(f_i - f_j)
            f_diff_max = max_detuning or self.conditions["max_detuning"]
            if f_diff > f_diff_max:
                flag = True
                log.append(
                    f"Detuning of {label} ({f_diff:.3f} GHz) is higher than {f_diff_max:.3f} GHz."
                )

            result[label] = {
                "collision": flag,
                "log": log,
            }

        result = dict(sorted(result.items()))
        return Collision(
            name="Type0B",
            description="Detuning too far to implement the fast CR.",
            n_nodes=2,
            directed=False,
            node_data={},
            edge_data=result,
        )

    def check_type1a(self): ...

    def check_type1b(self): ...

    def check_type1c(self): ...

    def check_type2a(self): ...

    def check_type2b(self): ...

    def check_type3a(self): ...

    def check_type3b(self): ...

    def check_type7(self): ...

    def check_type8(self): ...

    def check_type9(self): ...
