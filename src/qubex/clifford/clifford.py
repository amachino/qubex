from __future__ import annotations

from .pauli import Pauli


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
    def X180(cls) -> Clifford:
        """
        Create the Clifford transformation for a 180-degree rotation around the X-axis.

        Returns
        -------
        Clifford
            A Clifford representing a 180-degree rotation around the X-axis.
        """
        return Clifford(
            name="X180",
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(1, "X"),
                "Y": Pauli(-1, "Y"),
                "Z": Pauli(-1, "Z"),
            },
        )

    @classmethod
    def Y90(cls) -> Clifford:
        """
        Create the Clifford transformation for a 90-degree rotation around the Y-axis.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the Y-axis.
        """
        return Clifford(
            name="Y90",
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(-1, "Z"),
                "Y": Pauli(1, "Y"),
                "Z": Pauli(1, "X"),
            },
        )

    @classmethod
    def Y180(cls) -> Clifford:
        """
        Create the Clifford transformation for a 180-degree rotation around the Y-axis.

        Returns
        -------
        Clifford
            A Clifford representing a 180-degree rotation around the Y-axis.
        """
        return Clifford(
            name="Y180",
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(-1, "X"),
                "Y": Pauli(1, "Y"),
                "Z": Pauli(-1, "Z"),
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
    def Z180(cls) -> Clifford:
        """
        Create the Clifford transformation for a 180-degree rotation around the Z-axis.

        Returns
        -------
        Clifford
            A Clifford representing a 180-degree rotation around the Z-axis.
        """
        return Clifford(
            name="Z180",
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(-1, "X"),
                "Y": Pauli(-1, "Y"),
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
    def IY90(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation for a 90-degree rotation around the Y-axis of the second qubit.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the Y-axis of the second qubit.
        """
        return Clifford(
            name="IY90",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(-1, "IZ"),
                "IY": Pauli(1, "IY"),
                "IZ": Pauli(1, "IX"),
                "XI": Pauli(1, "XI"),
                "XX": Pauli(-1, "XZ"),
                "XY": Pauli(1, "XY"),
                "XZ": Pauli(1, "XX"),
                "YI": Pauli(1, "YI"),
                "YX": Pauli(-1, "YZ"),
                "YY": Pauli(1, "YY"),
                "YZ": Pauli(1, "YX"),
                "ZI": Pauli(1, "ZI"),
                "ZX": Pauli(-1, "ZZ"),
                "ZY": Pauli(1, "ZY"),
                "ZZ": Pauli(1, "ZX"),
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
    def YI90(cls) -> Clifford:
        """
        Create the two-qubit Clifford transformation for a 90-degree rotation around the Y-axis of the first qubit.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the Y-axis of the first qubit.
        """
        return Clifford(
            name="YI90",
            map={
                "II": Pauli(1, "II"),
                "IX": Pauli(1, "IX"),
                "IY": Pauli(1, "IY"),
                "IZ": Pauli(1, "IZ"),
                "XI": Pauli(1, "ZI"),
                "XX": Pauli(1, "ZX"),
                "XY": Pauli(1, "ZY"),
                "XZ": Pauli(1, "ZZ"),
                "YI": Pauli(1, "YI"),
                "YX": Pauli(1, "YX"),
                "YY": Pauli(1, "YY"),
                "YZ": Pauli(1, "YZ"),
                "ZI": Pauli(-1, "XI"),
                "ZX": Pauli(-1, "XX"),
                "ZY": Pauli(-1, "XY"),
                "ZZ": Pauli(-1, "XZ"),
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
