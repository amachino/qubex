"""Clifford sequence composition utilities."""

from __future__ import annotations

from .clifford import Clifford


class CliffordSequence:
    """
    Represents a sequence of Clifford operators.

    Attributes
    ----------
    sequence : list[Clifford]
        Sequence of Clifford operators.

    clifford : Clifford
        Cumulative Clifford operator of the sequence.

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
    def II(cls) -> CliffordSequence:
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
            Clifford operator to count.


        Returns
        -------
        int
            Number of occurrences of the input Clifford operator in the sequence.

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
            Clifford transformation to compose with the current sequence.


        Returns
        -------
        CliffordSequence
            Resulting Clifford sequence after composing the input transformation.

        """
        if isinstance(other, CliffordSequence):
            composed_sequence = self.sequence + other.sequence
            composed_clifford = self.clifford.compose(other.clifford)
        else:
            composed_sequence = [*self.sequence, other]
            composed_clifford = self.clifford.compose(other)
        return CliffordSequence(sequence=composed_sequence, clifford=composed_clifford)

    def __hash__(self) -> int:
        """Return a hash based on the sequence content."""
        return hash(tuple(self.sequence))

    def __repr__(self) -> str:
        """Return the debug representation of the sequence."""
        return f"CliffordSequence({self.gate_sequence})"
