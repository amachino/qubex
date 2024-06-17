from __future__ import annotations


class Pauli:
    """
    Represents a Pauli operator.

    Attributes
    ----------
    coefficient : complex
        The coefficient of the Pauli operator. Must be one of ±1, ±i.
    operator : str
        The type of the Pauli operator. Must be one of 'I', 'X', 'Y', 'Z'.
    """

    def __init__(
        self,
        coefficient: complex,
        operator: str,
    ):
        if coefficient not in {1, -1, 1j, -1j}:
            raise ValueError("Invalid coefficient. Must be one of ±1, ±i.")
        if operator not in {"I", "X", "Y", "Z"}:
            raise ValueError(
                "Invalid Pauli operator. Must be one of 'I', 'X', 'Y', 'Z'."
            )
        self.coefficient = coefficient
        self.operator = operator

    def __repr__(self):
        return f"Pauli({self.coefficient}, {self.operator})"

    def __hash__(self):
        return hash((self.coefficient, self.operator))

    def __eq__(self, other) -> bool:
        return self.coefficient == other.coefficient and self.operator == other.operator


class Clifford:
    """
    Represents a Clifford operator.

    Attributes
    ----------
    sequence : list[str]
        The decomposition of the Clifford operator into a sequence of generators.
    map : dict[str, Pauli]
        The Clifford transformation map that maps Pauli operators to new Pauli operators.
    """

    # Generator names
    X90 = "X90"
    Z90 = "Z90"

    def __init__(
        self,
        *,
        sequence: list[str],
        map: dict[str, Pauli],
    ):
        self.sequence = sequence
        self.map = map

    @classmethod
    def identity(cls) -> Clifford:
        """
        Create the identity Clifford transformation.

        Returns
        -------
        Clifford
            An identity Clifford with no operations and a map that does not change Pauli operators.
        """
        return Clifford(
            sequence=[],
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(1, "X"),
                "Y": Pauli(1, "Y"),
                "Z": Pauli(1, "Z"),
            },
        )

    @classmethod
    def x90(cls) -> Clifford:
        """
        Create the Clifford transformation for a 90-degree rotation around the X-axis.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the X-axis.
        """
        return Clifford(
            sequence=[cls.X90],
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(1, "X"),
                "Y": Pauli(1, "Z"),
                "Z": Pauli(-1, "Y"),
            },
        )

    @classmethod
    def z90(cls) -> Clifford:
        """
        Create the Clifford transformation for a 90-degree rotation around the Z-axis.

        Returns
        -------
        Clifford
            A Clifford representing a 90-degree rotation around the Z-axis.
        """
        return Clifford(
            sequence=[cls.Z90],
            map={
                "I": Pauli(1, "I"),
                "X": Pauli(1, "Y"),
                "Y": Pauli(-1, "X"),
                "Z": Pauli(1, "Z"),
            },
        )

    @classmethod
    def compose(
        cls,
        c1: Clifford,
        c2: Clifford,
    ) -> Clifford:
        """
        Compose two Clifford transformations and return the resulting Clifford transformation.

        Parameters
        ----------
        c1 : Clifford
            The first Clifford transformation.
        c2 : Clifford
            The second Clifford transformation.

        Returns
        -------
        Clifford
            The resulting Clifford transformation from composing c1 and c2.
        """
        identity_map = cls.identity().map
        composed_map = {
            key: c2.apply_to(c1.apply_to(value)) for key, value in identity_map.items()
        }
        return Clifford(
            sequence=c1.sequence + c2.sequence,
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

    def __repr__(self):
        return f"Clifford({self.sequence}, {self.map})"


class CliffordGenerator:
    """
    Generates Clifford transformations from two generators X90 and Z90.

    Methods
    -------
    generate_clifford(bit_sequence: int) -> Clifford:
        Generate a Clifford transformation from X90 and Z90 generators.
    find_clifford_group(max_gate_count: int = 10) -> list[Clifford]:
        Find the group of unique Clifford transformations up to a specified gate count.
    """

    @classmethod
    def generate_clifford(
        cls,
        bit_representation: int,
    ) -> Clifford:
        """
        Generate a Clifford transformation from X90 and Z90 generators.

        Parameters
        ----------
        bit_representation : int
            The bit representation of the gate sequence,
            e.g. 0 -> X90, 1 -> Z90, 1011 -> Z90-X90-Z90-Z90

        Returns
        -------
        Clifford
            The generated Clifford transformation.
        """
        gate_sequence: list[str] = []
        identity_clifford = Clifford.identity()
        current_clifford = identity_clifford

        # Decompose the bit representation into X90 and Z90 generators
        while bit_representation > 1:
            # Apply the X90 or Z90 generator based on the least significant bit
            if bit_representation % 2 == 0:
                # Apply the X90 generator
                gate_sequence.append(Clifford.X90)
                current_clifford = Clifford.compose(current_clifford, Clifford.x90())
            else:
                # Apply the Z90 generator
                gate_sequence.append(Clifford.Z90)
                current_clifford = Clifford.compose(current_clifford, Clifford.z90())
            bit_representation //= 2
        return Clifford(sequence=gate_sequence, map=current_clifford.map)

    @classmethod
    def find_clifford_group(cls, max_gate_count: int = 10) -> list[Clifford]:
        """
        Find the group of unique Clifford transformations up to a specified gate count.

        Parameters
        ----------
        max_gate_count : int, optional
            The maximum number of gate sequences to consider (default is 10).

        Returns
        -------
        list of Clifford
            A list of unique Clifford transformations.
        """
        # Generate all possible Clifford transformations up to the maximum gate count
        clifford_group: list[Clifford] = []
        # The number of combinations of X90 and Z90 generators is 2^(max_gate_count)
        for bit_representation in range(2**max_gate_count):
            # Generate a Clifford transformation from the bit representation
            clifford = cls.generate_clifford(bit_representation)
            found_same_clifford = False
            for existing in clifford_group:
                # Check if the Clifford transformation is the same as an existing one
                if existing.map == clifford.map:
                    found_same_clifford = True
                    # If the new Clifford transformation has fewer X90 gates, replace the existing one
                    existing_X90_count = existing.sequence.count(Clifford.X90)
                    new_X90_count = clifford.sequence.count(Clifford.X90)
                    if new_X90_count < existing_X90_count:
                        clifford_group.remove(existing)
                        clifford_group.append(clifford)
                    break
            # If the Clifford transformation is unique, add it to the group
            if not found_same_clifford:
                clifford_group.append(clifford)
            # If the group contains all 24 unique Clifford transformations, stop searching
            if len(clifford_group) == 24:
                break
        return clifford_group
