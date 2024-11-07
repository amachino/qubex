from __future__ import annotations

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
