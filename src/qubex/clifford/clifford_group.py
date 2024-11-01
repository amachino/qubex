from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

FILE_1Q = "clifford_group_1q.json"
FILE_1Qx1Q = "clifford_group_1q1q.json"
FILE_2Q = "clifford_group_2q.json"

PAULI_1Q = [
    "I",
    "X",
    "Y",
    "Z",
]

PAULI_2Q = [
    "II",
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
]


class Pauli:
    """
    Represents a Pauli operator for single or two qubits.

    Attributes
    ----------
    coefficient : complex
        The coefficient of the Pauli operator. Must be one of ±1, ±i.
    operator : str
        The type of the Pauli operator. Must be one of the single- or two-qubit combinations from 'I', 'X', 'Y', 'Z'.
        Valid operators include 'I', 'X', 'Y', 'Z' for single qubit and combinations like 'IX', 'XY', 'ZZ' for two qubits.
    """

    def __init__(
        self,
        coefficient: complex,
        operator: str,
    ):
        if coefficient not in {1, -1, 1j, -1j}:
            raise ValueError("Invalid coefficient. Must be one of ±1, ±i.")
        # Allow single and two-qubit Pauli operators
        valid_operators = PAULI_1Q + PAULI_2Q
        if operator not in valid_operators:
            raise ValueError(
                f"Invalid Pauli operator. Must be one of {valid_operators}."
            )
        self.coefficient = coefficient
        self.operator = operator

    def to_string(self) -> str:
        """Returns a string representation of the Pauli operator with its coefficient."""
        sign = {1: "", -1: "-", 1j: "i", -1j: "-i"}[self.coefficient]
        return f"{sign}{self.operator}"

    def print(self):
        """Prints the string representation of the Pauli operator."""
        print(self.to_string())

    def __repr__(self) -> str:
        coefficient = {1: "1", -1: "-1", 1j: "1j", -1j: "-1j"}[self.coefficient]
        return f"Pauli({coefficient}, '{self.operator}')"

    def __hash__(self):
        return hash((self.coefficient, self.operator))

    def __eq__(self, other) -> bool:
        return self.coefficient == other.coefficient and self.operator == other.operator


class Clifford:
    """
    Represents a Clifford operator.

    Attributes
    ----------
    name : str
        The name of the Clifford operator.
    map : dict[str, Pauli]
        The Pauli transformation map of the Clifford operator,
        supporting single and two qubit Pauli operators.
    """

    def __init__(
        self,
        name: str,
        map: dict[str, Pauli],
    ):
        self.name = name
        self.map = map

    @classmethod
    def I(cls) -> Clifford:  # noqa: E743
        """
        Create the single identity Clifford transformation.

        Returns
        -------
        Clifford
            An identity Clifford with no operations and a map that does not change Pauli operators.
        """
        return Clifford(
            name="I",
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(1, "X"),
                "Y": Pauli(1, "Y"),
                "Z": Pauli(1, "Z"),
            },
        )

    @classmethod
    def X90(cls) -> Clifford:
        """
        Create the Clifford transformation for a 90-degree rotation around the X-axis.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the X-axis.
        """
        return Clifford(
            name="X90",
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(1, "X"),
                "Y": Pauli(1, "Z"),
                "Z": Pauli(-1, "Y"),
            },
        )

    @classmethod
    def Z90(cls) -> Clifford:
        """
        Create the Clifford transformation for a 90-degree rotation around the Z-axis.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the Z-axis.
        """
        return Clifford(
            name="Z90",
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(1, "Y"),
                "Y": Pauli(-1, "X"),
                "Z": Pauli(1, "Z"),
            },
        )

    @classmethod
    def II(cls) -> Clifford:
        """
        Create the two-qubit identity Clifford transformation.

        Returns
        -------
        Clifford
            An identity Clifford with no operations and a map that does not change Pauli operators.
        """
        return Clifford(
            name="II",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "IX"),
                "IY": Pauli(1, "IY"),
                "IZ": Pauli(1, "IZ"),
                "XI": Pauli(1, "XI"),
                "XX": Pauli(1, "XX"),
                "XY": Pauli(1, "XY"),
                "XZ": Pauli(1, "XZ"),
                "YI": Pauli(1, "YI"),
                "YX": Pauli(1, "YX"),
                "YY": Pauli(1, "YY"),
                "YZ": Pauli(1, "YZ"),
                "ZI": Pauli(1, "ZI"),
                "ZX": Pauli(1, "ZX"),
                "ZY": Pauli(1, "ZY"),
                "ZZ": Pauli(1, "ZZ"),
            },
        )

    @classmethod
    def XI90(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation for a 90-degree rotation around the X-axis of the first qubit.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the X-axis of the first qubit.
        """
        return Clifford(
            name="XI90",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "IX"),
                "IY": Pauli(1, "IY"),
                "IZ": Pauli(1, "IZ"),
                "XI": Pauli(1, "XI"),
                "XX": Pauli(1, "XX"),
                "XY": Pauli(1, "XY"),
                "XZ": Pauli(1, "XZ"),
                "YI": Pauli(1, "ZI"),
                "YX": Pauli(1, "ZX"),
                "YY": Pauli(1, "ZY"),
                "YZ": Pauli(1, "ZZ"),
                "ZI": Pauli(-1, "YI"),
                "ZX": Pauli(-1, "YX"),
                "ZY": Pauli(-1, "YY"),
                "ZZ": Pauli(-1, "YZ"),
            },
        )

    @classmethod
    def IX90(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation for a 90-degree rotation around the X-axis of the second qubit.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the X-axis of the second qubit.
        """
        return Clifford(
            name="IX90",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "IX"),
                "IY": Pauli(1, "IZ"),
                "IZ": Pauli(-1, "IY"),
                "XI": Pauli(1, "XI"),
                "XX": Pauli(1, "XX"),
                "XY": Pauli(1, "XZ"),
                "XZ": Pauli(-1, "XY"),
                "YI": Pauli(1, "YI"),
                "YX": Pauli(1, "YX"),
                "YY": Pauli(1, "YZ"),
                "YZ": Pauli(-1, "YY"),
                "ZI": Pauli(1, "ZI"),
                "ZX": Pauli(1, "ZX"),
                "ZY": Pauli(1, "ZZ"),
                "ZZ": Pauli(-1, "ZY"),
            },
        )

    @classmethod
    def ZI90(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation for a 90-degree rotation around the Z-axis of the first qubit.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the Z-axis of the first qubit.
        """
        return Clifford(
            name="ZI90",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "IX"),
                "IY": Pauli(1, "IY"),
                "IZ": Pauli(1, "IZ"),
                "XI": Pauli(1, "YI"),
                "XX": Pauli(1, "YX"),
                "XY": Pauli(1, "YY"),
                "XZ": Pauli(1, "YZ"),
                "YI": Pauli(-1, "XI"),
                "YX": Pauli(-1, "XX"),
                "YY": Pauli(-1, "XY"),
                "YZ": Pauli(-1, "XZ"),
                "ZI": Pauli(1, "ZI"),
                "ZX": Pauli(1, "ZX"),
                "ZY": Pauli(1, "ZY"),
                "ZZ": Pauli(1, "ZZ"),
            },
        )

    @classmethod
    def IZ90(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation for a 90-degree rotation around the Z-axis of the second qubit.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the Z-axis of the second qubit.
        """
        return Clifford(
            name="IZ90",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "IY"),
                "IY": Pauli(-1, "IX"),
                "IZ": Pauli(1, "IZ"),
                "XI": Pauli(1, "XI"),
                "XX": Pauli(1, "XY"),
                "XY": Pauli(-1, "XX"),
                "XZ": Pauli(1, "XZ"),
                "YI": Pauli(1, "YI"),
                "YX": Pauli(1, "YY"),
                "YY": Pauli(-1, "YX"),
                "YZ": Pauli(1, "YZ"),
                "ZI": Pauli(1, "ZI"),
                "ZX": Pauli(1, "ZY"),
                "ZY": Pauli(-1, "ZX"),
                "ZZ": Pauli(1, "ZZ"),
            },
        )

    @classmethod
    def ZX90(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation by ZX90.

        Returns
        -------
        Clifford
            A Clifford transformation by ZX90.
        """
        return Clifford(
            name="ZX90",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "IX"),
                "IY": Pauli(1, "ZZ"),
                "IZ": Pauli(-1, "ZY"),
                "XI": Pauli(1, "YX"),
                "XX": Pauli(1, "YI"),
                "XY": Pauli(1, "XY"),
                "XZ": Pauli(1, "XZ"),
                "YI": Pauli(-1, "XX"),
                "YX": Pauli(-1, "XI"),
                "YY": Pauli(1, "YY"),
                "YZ": Pauli(1, "YZ"),
                "ZI": Pauli(1, "ZI"),
                "ZX": Pauli(1, "ZX"),
                "ZY": Pauli(1, "IZ"),
                "ZZ": Pauli(-1, "IY"),
            },
        )

    @classmethod
    def ZZ90(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation by ZZ90.

        Returns
        -------
        Clifford
            A Clifford transformation by ZZ90.
        """
        return Clifford(
            name="ZZ90",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "ZY"),
                "IY": Pauli(-1, "ZX"),
                "IZ": Pauli(1, "IZ"),
                "XI": Pauli(1, "YZ"),
                "XX": Pauli(1, "XX"),
                "XY": Pauli(1, "XY"),
                "XZ": Pauli(1, "YI"),
                "YI": Pauli(-1, "XZ"),
                "YX": Pauli(1, "YX"),
                "YY": Pauli(1, "YY"),
                "YZ": Pauli(-1, "XI"),
                "ZI": Pauli(1, "ZI"),
                "ZX": Pauli(1, "IY"),
                "ZY": Pauli(-1, "IX"),
                "ZZ": Pauli(1, "ZZ"),
            },
        )

    @classmethod
    def CNOT(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation by CNOT.

        Returns
        -------
        Clifford
            A Clifford transformation by CNOT.
        """
        return Clifford(
            name="CNOT",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "IX"),
                "IY": Pauli(1, "ZY"),
                "IZ": Pauli(1, "ZZ"),
                "XI": Pauli(1, "XX"),
                "XX": Pauli(1, "XI"),
                "XY": Pauli(1, "YZ"),
                "XZ": Pauli(-1, "YY"),
                "YI": Pauli(1, "YX"),
                "YX": Pauli(1, "YI"),
                "YY": Pauli(-1, "XZ"),
                "YZ": Pauli(1, "XY"),
                "ZI": Pauli(1, "ZI"),
                "ZX": Pauli(1, "ZX"),
                "ZY": Pauli(1, "IY"),
                "ZZ": Pauli(1, "IZ"),
            },
        )

    @classmethod
    def CZ(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation by CZ.

        Returns
        -------
        Clifford
            A Clifford transformation by CZ.
        """
        return Clifford(
            name="CZ",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "ZX"),
                "IY": Pauli(1, "ZY"),
                "IZ": Pauli(1, "IZ"),
                "XI": Pauli(1, "XZ"),
                "XX": Pauli(1, "YY"),
                "XY": Pauli(-1, "YX"),
                "XZ": Pauli(1, "XI"),
                "YI": Pauli(1, "YZ"),
                "YX": Pauli(-1, "XY"),
                "YY": Pauli(1, "XX"),
                "YZ": Pauli(1, "YI"),
                "ZI": Pauli(1, "ZI"),
                "ZX": Pauli(1, "IX"),
                "ZY": Pauli(1, "IY"),
                "ZZ": Pauli(1, "ZZ"),
            },
        )

    @classmethod
    def SWAP(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation by SWAP.

        Returns
        -------
        Clifford
            A Clifford transformation by SWAP.
        """
        return Clifford(
            name="SWAP",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "XI"),
                "IY": Pauli(1, "YI"),
                "IZ": Pauli(1, "ZI"),
                "XI": Pauli(1, "IX"),
                "XX": Pauli(1, "XX"),
                "XY": Pauli(1, "YX"),
                "XZ": Pauli(1, "ZX"),
                "YI": Pauli(1, "IY"),
                "YX": Pauli(1, "XY"),
                "YY": Pauli(1, "YY"),
                "YZ": Pauli(1, "ZY"),
                "ZI": Pauli(1, "IZ"),
                "ZX": Pauli(1, "XZ"),
                "ZY": Pauli(1, "YZ"),
                "ZZ": Pauli(1, "ZZ"),
            },
        )

    @classmethod
    def ISWAP(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation by ISWAP.

        Returns
        -------
        Clifford
            A Clifford transformation by ISWAP.
        """
        return Clifford(
            name="ISWAP",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "YZ"),
                "IY": Pauli(-1, "XZ"),
                "IZ": Pauli(1, "ZI"),
                "XI": Pauli(1, "ZY"),
                "XX": Pauli(1, "XX"),
                "XY": Pauli(1, "YX"),
                "XZ": Pauli(1, "IY"),
                "YI": Pauli(-1, "ZX"),
                "YX": Pauli(1, "XY"),
                "YY": Pauli(1, "YY"),
                "YZ": Pauli(-1, "IX"),
                "ZI": Pauli(1, "IZ"),
                "ZX": Pauli(1, "YI"),
                "ZY": Pauli(-1, "XI"),
                "ZZ": Pauli(1, "ZZ"),
            },
        )

    @classmethod
    def BSWAP(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation by BSWAP.

        Returns
        -------
        Clifford
            A Clifford transformation by BSWAP.
        """
        return Clifford(
            name="BSWAP",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(-1, "YZ"),
                "IY": Pauli(-1, "XZ"),
                "IZ": Pauli(-1, "ZI"),
                "XI": Pauli(-1, "ZY"),
                "XX": Pauli(1, "XX"),
                "XY": Pauli(-1, "YX"),
                "XZ": Pauli(1, "IY"),
                "YI": Pauli(-1, "ZX"),
                "YX": Pauli(-1, "XY"),
                "YY": Pauli(1, "YY"),
                "YZ": Pauli(1, "IX"),
                "ZI": Pauli(-1, "IZ"),
                "ZX": Pauli(1, "YI"),
                "ZY": Pauli(1, "XI"),
                "ZZ": Pauli(1, "ZZ"),
            },
        )

    @property
    def inverse(self) -> Clifford:
        """
        Compute the inverse of the Clifford transformation.

        Returns
        -------
        Clifford
            The inverse of the current Clifford transformation.
        """
        inverse_map = {}
        for operator, pauli in self.map.items():
            coefficient = 1 / pauli.coefficient
            inverse_map[pauli.operator] = Pauli(coefficient, operator)
        map = {operator: inverse_map[operator] for operator in self.map}
        return Clifford(
            name=f"({self.name})^-1",
            map=map,
        )

    def is_identity(self) -> bool:
        """Check if the Clifford transformation is the identity."""
        return all(pauli.coefficient == 1 for pauli in self.map.values())

    def compose(
        self,
        other: Clifford,
    ) -> Clifford:
        """
        Compose two Clifford transformations.

        Parameters
        ----------
        other : Clifford
            The other Clifford transformation to compose with.

        Returns
        -------
        Clifford
            The resulting Clifford transformation after composing the two input transformations.
        """
        composed_map = {}
        for operator, pauli in self.map.items():
            composed_map[operator] = other.apply_to(pauli)
        return Clifford(
            name=f"{self.name}->{other.name}",
            map=composed_map,
        )

    def apply_to(
        self,
        pauli: Pauli,
    ) -> Pauli:
        """
        Apply the Clifford transformation to a given Pauli operator.

        Parameters
        ----------
        pauli : Pauli
            The Pauli operator to transform.

        Returns
        -------
        Pauli
            The resulting Pauli operator after the transformation.
        """
        mapped_pauli = self.map[pauli.operator]
        new_coefficient = pauli.coefficient * mapped_pauli.coefficient
        return Pauli(new_coefficient, mapped_pauli.operator)

    def to_string(self) -> str:
        map = ", ".join(
            f"{operator}->{pauli.to_string()}" for operator, pauli in self.map.items()
        )
        return f"{{{map}}}"

    def to_dict(self) -> dict:
        return {
            operator: [
                pauli.coefficient,
                pauli.operator,
            ]
            for operator, pauli in self.map.items()
        }

    def print(self):
        print(self.to_string())

    def __repr__(self) -> str:
        return f"Clifford({self.to_string()})"

    def __hash__(self):
        return hash(tuple(self.map.items()))

    def __eq__(self, other) -> bool:
        return self.map == other.map


class CliffordSequence:
    """
    Represents a sequence of Clifford operators.

    Attributes
    ----------
    sequence : list[Clifford]
        The sequence of Clifford operators.
    clifford : Clifford
        The cumulative Clifford operator of the sequence.
    """

    def __init__(
        self,
        sequence: list[Clifford],
        clifford: Clifford,
    ):
        self.sequence = sequence
        self.clifford = clifford

    @classmethod
    def I(cls) -> CliffordSequence:  # noqa: E743
        """
        Create an identity Clifford sequence.

        Returns
        -------
        CliffordSequence
            An identity Clifford sequence with no operations and an identity Clifford.
        """
        return cls(sequence=[], clifford=Clifford.I())

    @classmethod
    def II(cls) -> CliffordSequence:  # noqa: E743
        """
        Create an two-qubit identity Clifford sequence.

        Returns
        -------
        CliffordSequence
            An identity Clifford sequence with no operations and an identity Clifford.
        """
        return cls(sequence=[], clifford=Clifford.II())

    @property
    def gate_sequence(self) -> list[str]:
        """Returns the sequence of gate names."""
        return [clifford.name for clifford in self.sequence]

    @property
    def length(self) -> int:
        """Returns the length of the sequence."""
        return len(self.sequence)

    def count(self, clifford: Clifford) -> int:
        """
        Count the number of occurrences of a Clifford operator in the sequence.

        Parameters
        ----------
        clifford : Clifford
            The Clifford operator to count.

        Returns
        -------
        int
            The number of occurrences of the input Clifford operator in the sequence.
        """
        return self.sequence.count(clifford)

    def compose(
        self,
        other: Clifford | CliffordSequence,
    ) -> CliffordSequence:
        """
        Compose a Clifford transformation with the current sequence.

        Parameters
        ----------
        other : Clifford | CliffordSequence
            The Clifford transformation to compose with the current sequence.

        Returns
        -------
        CliffordSequence
            The resulting Clifford sequence after composing the input transformation.
        """
        if isinstance(other, Clifford):
            other = CliffordSequence(sequence=[other], clifford=other)
        composed_sequence = self.sequence + other.sequence
        composed_clifford = self.clifford.compose(other.clifford)
        return CliffordSequence(sequence=composed_sequence, clifford=composed_clifford)

    def __hash__(self) -> int:
        return hash(tuple(self.sequence))

    def __repr__(self) -> str:
        return f"CliffordSequence({self.gate_sequence})"


class CliffordGroup:
    generators = {
        "I": Clifford.I(),
        "X90": Clifford.X90(),
        "Z90": Clifford.Z90(),
        "II": Clifford.II(),
        "XI90": Clifford.XI90(),
        "IX90": Clifford.IX90(),
        "ZI90": Clifford.ZI90(),
        "IZ90": Clifford.IZ90(),
        "ZX90": Clifford.ZX90(),
    }

    def __init__(self):
        self._clifford_1q = dict[Clifford, CliffordSequence]()
        self._clifford_1q1q = dict[Clifford, CliffordSequence]()
        self._clifford_2q = dict[Clifford, CliffordSequence]()
        self.load()

    def get_clifford_dict(
        self,
        type: Literal["1Q", "1Qx1Q", "2Q"] = "1Q",
    ) -> dict[Clifford, CliffordSequence]:
        if type == "1Q":
            return self._clifford_1q
        elif type == "1Qx1Q":
            return self._clifford_1q1q
        else:
            return self._clifford_2q

    def get_clifford_sequences(
        self,
        type: Literal["1Q", "1Qx1Q", "2Q"] = "1Q",
    ) -> list[CliffordSequence]:
        """

        Parameters
        ----------
        type : Literal["1Q", "1Qx1Q", "2Q"], optional
        """
        if type == "1Q":
            return list(self._clifford_1q.values())
        elif type == "1Qx1Q":
            return list(self._clifford_1q1q.values())
        else:
            return list(self._clifford_2q.values())

    def get_clifford_list(
        self,
        type: Literal["1Q", "1Qx1Q", "2Q"] = "1Q",
    ) -> list[dict]:
        clifford_sequences = self.get_clifford_sequences(type)
        return [
            {
                "index": index,
                "sequence": clifford_sequence.gate_sequence,
                "map": clifford_sequence.clifford.to_dict(),
            }
            for index, clifford_sequence in enumerate(clifford_sequences)
        ]

    def get_clifford(
        self,
        index: int,
        type: Literal["1Q", "1Qx1Q", "2Q"] = "1Q",
    ) -> dict:
        """ """
        return self.get_clifford_list(type)[index]

    def get_random_clifford_sequences(
        self,
        n: int,
        type: Literal["1Q", "1Qx1Q", "2Q"] = "1Q",
        seed: int | None = None,
    ) -> list[CliffordSequence]:
        """Returns a list of n random Clifford operators."""
        random.seed(seed)
        clifford_sequences = self.get_clifford_sequences(type)
        return random.choices(clifford_sequences, k=n)

    def get_inverse(
        self,
        clifford_sequence: CliffordSequence,
        type: Literal["1Q", "1Qx1Q", "2Q"] = "1Q",
    ) -> CliffordSequence:
        """Returns the inverse of a given Clifford operator."""

        clifford = clifford_sequence.clifford
        clifford_dict = self.get_clifford_dict(type)
        inverse = clifford_dict.get(clifford.inverse, None)
        if inverse is None:
            raise ValueError("Clifford operator not found in the group.")

        return inverse

    def create_rb_sequences(
        self,
        n: int,
        type: Literal["1Q", "1Qx1Q", "2Q"] = "1Q",
        seed: int | None = None,
    ) -> tuple[list[list[str]], list[str]]:
        """
        Create a set of random Clifford operators for randomized benchmarking.

        Parameters
        ----------
        n : int
            The number of random Clifford operators to return.
        seed : int, optional
            The seed for the random number generator.

        Returns
        -------
        tuple[list[list[str]], list[str]]
            A tuple containing the list of random Clifford operators and their total inverse.
        """
        # Get the random Clifford operators
        clifford_sequences = self.get_random_clifford_sequences(n, type, seed)

        # Compose the Clifford operators
        composed = CliffordSequence.I()
        for clifford_sequence in clifford_sequences:
            # Apply the random Clifford operator
            composed = composed.compose(clifford_sequence.clifford)
        # Compute the total inverse of the composed Clifford operators
        composed_inverse = self.get_inverse(composed)

        # Return the Clifford operators and their total inverse
        return [
            clifford_sequence.gate_sequence for clifford_sequence in clifford_sequences
        ], composed_inverse.gate_sequence

    def create_irb_sequences(
        self,
        n: int,
        interleave: dict[str, tuple[complex, str]],
        type: Literal["1Q", "1Qx1Q", "2Q"] = "1Q",
        seed: int | None = None,
    ) -> tuple[list[list[str]], list[str]]:
        """
        Create a set of random Clifford operators for interleaved randomized benchmarking.

        Parameters
        ----------
        n : int
            The number of random Clifford operators to return.
        interlieve : dict[str, tuple[complex, str]]
            The interleaved gate to apply after each Clifford operator.
        seed : int, optional
            The seed for the random number generator.

        Returns
        -------
        tuple[list[list[str]], list[str]]
            A tuple containing the list of random Clifford operators and their total inverse.
        """
        # Get the random Clifford operators
        clifford_sequences = self.get_random_clifford_sequences(n, type, seed)

        # Create the interleaved Clifford operator
        interleave_cliford = Clifford(
            name="U",
            map={operator: Pauli(*interleave[operator]) for operator in interleave},
        )
        # Compose the Clifford operators with the interleaved operator
        composed = CliffordSequence.I()
        for clifford_sequence in clifford_sequences:
            # Apply the random Clifford operator
            composed = composed.compose(clifford_sequence.clifford)
            # Apply the interleaved Clifford operator
            composed = composed.compose(interleave_cliford)
        # Compute the total inverse of the composed Clifford operators
        composed_inverse = self.get_inverse(composed, type)

        # Return the Clifford operators and their total inverse
        return [
            clifford_sequence.gate_sequence for clifford_sequence in clifford_sequences
        ], composed_inverse.gate_sequence

    def generate_1q_cliffords(
        self,
        max_gates: int = 7,
    ) -> dict[Clifford, CliffordSequence]:
        """
        Generate unique Clifford sequences up to a specified gate count.

        Parameters
        ----------
        max_gates : int
            The maximum number of gates in the Clifford sequences.

        Returns
        -------
        dict[Clifford, CliffordSequence]
            A dictionary of unique Clifford operators and their sequences.
        """
        # Start with the identity Clifford
        identity = CliffordSequence.I()

        # Dictionary to store found Clifford operators
        found_cliffords: dict[Clifford, CliffordSequence] = {
            identity.clifford: identity
        }

        # Clifford generators
        x90 = Clifford.X90()
        z90 = Clifford.Z90()

        # Recursive function to generate Clifford sequences
        def generate_clifford_sequences(
            sequence: CliffordSequence,
            gate_count: int,
        ):
            if gate_count == 1:
                # Base case: reached the maximum number of gates
                return
            for clifford in (x90, z90):
                # Compose the current sequence with the Clifford generator
                new_sequence = sequence.compose(clifford)
                # Add the new sequence to the dictionary if it is unique
                if new_sequence.clifford not in found_cliffords:
                    found_cliffords[new_sequence.clifford] = new_sequence
                # Update the existing sequence if the new sequence is shorter
                else:
                    existing_sequence = found_cliffords[new_sequence.clifford]
                    if new_sequence.count(x90) < existing_sequence.count(x90):
                        found_cliffords[new_sequence.clifford] = new_sequence
                    elif new_sequence.count(x90) == existing_sequence.count(x90):
                        if new_sequence.count(z90) < existing_sequence.count(z90):
                            found_cliffords[new_sequence.clifford] = new_sequence
                # Recurse with the new sequence
                generate_clifford_sequences(new_sequence, gate_count - 1)

        # Generate Clifford sequences starting from the identity
        generate_clifford_sequences(identity, max_gates + 1)  # +1 for the identity

        # Store the Clifford sequences in the group
        self._clifford_1q = found_cliffords
        print(f"Generated {len(found_cliffords)} unique Clifford sequences.")
        print(
            f"Maximum gate count: {max(sequence.length for sequence in self._clifford_1q.values())}"
        )
        return self._clifford_1q

    def generate_1q1q_cliffords(
        self,
    ) -> dict[Clifford, CliffordSequence]:
        """
        Generate unique 1Qx1Q Clifford sequences.

        Returns
        -------
        dict[Clifford, CliffordSequence]
            A dictionary of unique 1Qx1Q Clifford operators and their sequences.
        """
        clifford_sequences_1q = self.get_clifford_sequences("1Q")
        for seq_1q_1 in clifford_sequences_1q:
            seq_2q_1 = CliffordSequence.II()
            for seq in seq_1q_1.sequence:
                if seq.name == "X90":
                    seq_2q_1 = seq_2q_1.compose(Clifford.XI90())
                elif seq.name == "Z90":
                    seq_2q_1 = seq_2q_1.compose(Clifford.ZI90())
                else:
                    raise ValueError("Invalid 1Q Clifford sequence.")
            for seq_1q_2 in clifford_sequences_1q:
                seq_2q_2 = CliffordSequence.II()
                for seq in seq_1q_2.sequence:
                    if seq.name == "X90":
                        seq_2q_2 = seq_2q_2.compose(Clifford.IX90())
                    elif seq.name == "Z90":
                        seq_2q_2 = seq_2q_2.compose(Clifford.IZ90())
                    else:
                        raise ValueError("Invalid 1Q Clifford sequence.")
                seq_2q = seq_2q_1.compose(seq_2q_2)
                self._clifford_1q1q[seq_2q.clifford] = seq_2q

        print(f"Generated {len(self._clifford_1q1q)} unique 1Qx1Q Clifford sequences.")
        print(
            f"Maximum gate count: {max(sequence.length for sequence in self._clifford_1q1q.values())}"
        )
        return self._clifford_1q1q

    def generate_2q_cliffords(
        self,
        two_qubit_gate: Clifford = Clifford.ZX90(),
    ) -> dict[Clifford, CliffordSequence]:
        """
        Generate unique 2Q Clifford sequences.

        Returns
        -------
        dict[Clifford, CliffordSequence]
            A dictionary of unique 2Q Clifford operators and their sequences.
        """
        clifford_sequences_1q1q = self.get_clifford_sequences("1Qx1Q")
        found_cliffords_1: dict[Clifford, CliffordSequence] = {
            seq.clifford: seq for seq in clifford_sequences_1q1q
        }
        print(f"1. 1Qx1Q : {len(found_cliffords_1)}")

        found_cliffords_2: dict[Clifford, CliffordSequence] = {} | found_cliffords_1
        for found_sequence_0 in found_cliffords_1.values():
            composed = found_sequence_0.compose(two_qubit_gate)
            if composed.clifford not in found_cliffords_2:
                found_cliffords_2[composed.clifford] = composed
        print(f"2. 1Qx1Q - 2Q : {len(found_cliffords_2)}")

        found_cliffords_3: dict[Clifford, CliffordSequence] = {} | found_cliffords_2
        for found_sequence_1 in found_cliffords_2.values():
            for seq in clifford_sequences_1q1q:
                composed = found_sequence_1.compose(seq)
                if composed.clifford not in found_cliffords_3:
                    found_cliffords_3[composed.clifford] = composed
        print(f"3. 1Qx1Q - 2Q - 1Qx1Q : {len(found_cliffords_3)}")

        found_cliffords_4: dict[Clifford, CliffordSequence] = {} | found_cliffords_3
        for found_sequence_2 in found_cliffords_3.values():
            composed = found_sequence_2.compose(two_qubit_gate)
            if composed.clifford not in found_cliffords_4:
                found_cliffords_4[composed.clifford] = composed
        print(f"4. 1Qx1Q - 2Q - 1Qx1Q - 2Q : {len(found_cliffords_4)}")

        found_cliffords_5: dict[Clifford, CliffordSequence] = {} | found_cliffords_4
        for found_sequence_3 in found_cliffords_4.values():
            for seq in clifford_sequences_1q1q:
                composed = found_sequence_3.compose(seq)
                if composed.clifford not in found_cliffords_5:
                    found_cliffords_5[composed.clifford] = composed
        print(f"5. 1Qx1Q - 2Q - 1Qx1Q - 2Q - 1Qx1Q : {len(found_cliffords_5)}")

        found_cliffords_6: dict[Clifford, CliffordSequence] = {} | found_cliffords_5
        for found_sequence_4 in found_cliffords_5.values():
            composed = found_sequence_4.compose(two_qubit_gate)
            if composed.clifford not in found_cliffords_6:
                found_cliffords_6[composed.clifford] = composed
        print(f"6. 1Qx1Q - 2Q - 1Qx1Q - 2Q - 1Qx1Q - 2Q : {len(found_cliffords_6)}")

        self._clifford_2q = found_cliffords_6
        print(f"Generated {len(self._clifford_2q)} unique 2Q Clifford sequences.")

        print(
            f"Maximum gate count: {max(sequence.length for sequence in self._clifford_2q.values())}"
        )
        return self._clifford_2q

    def get_file_path(
        self,
        type: Literal["1Q", "1Qx1Q", "2Q"] = "1Q",
    ):
        current_dir = Path(__file__).parent
        if type == "1Q":
            return current_dir / FILE_1Q
        elif type == "1Qx1Q":
            return current_dir / FILE_1Qx1Q
        else:
            return current_dir / FILE_2Q

    def save(
        self,
        type: Literal["1Q", "1Qx1Q", "2Q"] = "1Q",
    ):
        """Save the Clifford group to a JSON file."""
        file_path = self.get_file_path(type)
        clifford_list = self.get_clifford_list(type)
        with open(file_path, "w") as file:
            json.dump(clifford_list, file, indent=4)

    def load(
        self,
        type: Literal["1Q", "1Qx1Q", "2Q"] = "1Q",
    ):
        """Load the Clifford group from a JSON file."""
        file_path = self.get_file_path(type)
        with open(file_path, "r") as file:
            data = json.load(file)

        for item in data:
            sequence = []
            for gate in item["sequence"]:
                if gate not in self.generators:
                    raise ValueError("Invalid gate name.")
                sequence.append(self.generators[gate])

            map = {operator: Pauli(*item["map"][operator]) for operator in item["map"]}

            clifford_sequence = CliffordSequence(
                sequence=sequence,
                clifford=Clifford(
                    name=f"#{item['index']}",
                    map=map,
                ),
            )

            if type == "1Q":
                self._clifford_1q[clifford_sequence.clifford] = clifford_sequence
            elif type == "1Qx1Q":
                self._clifford_1q1q[clifford_sequence.clifford] = clifford_sequence
            else:
                self._clifford_2q[clifford_sequence.clifford] = clifford_sequence

        print(f"Loaded {len(data)} Clifford sequences.")
