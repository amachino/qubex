from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from .clifford import Clifford
from .clifford_sequence import CliffordSequence
from .pauli import Pauli

CLIFFORD_LIST_DIR = "clifford_list"
CLIFFORD_LIST_1Q = "clifford_list_1q"
CLIFFORD_LIST_1Q1Q = "clifford_list_1q1q"
CLIFFORD_LIST_2Q = "clifford_list_2q"


class CliffordGenerator:
    generators = {
        "I": Clifford.I(),
        "X90": Clifford.X90(),
        "Y90": Clifford.Y90(),
        "Z90": Clifford.Z90(),
        "II": Clifford.II(),
        "IX90": Clifford.IX90(),
        "IY90": Clifford.IY90(),
        "IZ90": Clifford.IZ90(),
        "XI90": Clifford.XI90(),
        "YI90": Clifford.YI90(),
        "ZI90": Clifford.ZI90(),
        "ZX90": Clifford.ZX90(),
        "ZZ90": Clifford.ZZ90(),
        "CNOT": Clifford.CNOT(),
        "CZ": Clifford.CZ(),
        "SWAP": Clifford.SWAP(),
        "ISWAP": Clifford.ISWAP(),
        "BSWAP": Clifford.BSWAP(),
    }

    def __init__(self):
        self._cliffords_1q = dict[Clifford, CliffordSequence]()
        self._cliffords_1q1q = dict[Clifford, CliffordSequence]()
        self._cliffords_2q = dict[Clifford, CliffordSequence]()
        self.load("1Q")
        self.load("1Q1Q")
        self.load("2Q")

    def get_cliffords(
        self,
        type: Literal["1Q", "1Q1Q", "2Q"] = "1Q",
    ) -> dict[Clifford, CliffordSequence]:
        """
        Returns a dictionary of Clifford operators.

        Parameters
        ----------
        type : Literal["1Q", "1Q1Q", "2Q"], optional
            Clifford operator type.

        Returns
        -------
        dict[Clifford, CliffordSequence]
            Dictionary of Clifford operators.
        """
        if type == "1Q":
            return self._cliffords_1q
        elif type == "1Q1Q":
            return self._cliffords_1q1q
        else:
            return self._cliffords_2q

    def get_clifford_sequences(
        self,
        type: Literal["1Q", "1Q1Q", "2Q"] = "1Q",
    ) -> list[CliffordSequence]:
        """
        Returns a list of Clifford operators.

        Parameters
        ----------
        type : Literal["1Q", "1Q1Q", "2Q"], optional
            Clifford operator type.

        Returns
        -------
        list[CliffordSequence]
            List of Clifford operators.
        """
        if type == "1Q":
            return list(self._cliffords_1q.values())
        elif type == "1Q1Q":
            return list(self._cliffords_1q1q.values())
        else:
            return list(self._cliffords_2q.values())

    def get_clifford_list(
        self,
        type: Literal["1Q", "1Q1Q", "2Q"] = "1Q",
    ) -> list[dict]:
        """
        Returns a list of Clifford operators.

        Parameters
        ----------
        type : Literal["1Q", "1Q1Q", "2Q"], optional
            Clifford operator type.

        Returns
        -------
        list[dict]
            List of Clifford operators.
        """
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
        type: Literal["1Q", "1Q1Q", "2Q"] = "1Q",
    ) -> dict:
        """
        Returns a Clifford operator by index.

        Parameters
        ----------
        index : int
            The index of the Clifford operator.
        type : Literal["1Q", "1Q1Q", "2Q"], optional
            Clifford operator type.

        Returns
        -------
        dict
            Clifford operator.
        """
        return self.get_clifford_list(type)[index]

    def get_random_clifford_sequences(
        self,
        n: int,
        type: Literal["1Q", "1Q1Q", "2Q"] = "1Q",
        seed: int | None = None,
    ) -> list[CliffordSequence]:
        """
        Returns a list of n random Clifford operators.

        Parameters
        ----------
        n : int
            The number of random Clifford operators to return.
        type : Literal["1Q", "1Q1Q", "2Q"], optional
            Clifford operator type.
        seed : int, optional
            The seed for the random number generator.

        Returns
        -------
        list[CliffordSequence]
            List of random Clifford operators.
        """
        random.seed(seed)
        clifford_sequences = self.get_clifford_sequences(type)
        return random.choices(clifford_sequences, k=n)

    def get_inverse(
        self,
        clifford_sequence: CliffordSequence,
        type: Literal["1Q", "1Q1Q", "2Q"] = "1Q",
    ) -> CliffordSequence:
        """
        Returns the inverse of a given Clifford operator.

        Parameters
        ----------
        clifford_sequence : CliffordSequence
            The Clifford operator.
        type : Literal["1Q", "1Q1Q", "2Q"], optional
            Clifford operator type.

        Returns
        -------
        CliffordSequence
            The inverse of the Clifford operator.
        """
        clifford = clifford_sequence.clifford
        clifford_dict = self.get_cliffords(type)
        inverse = clifford_dict.get(clifford.inverse, None)
        if inverse is None:
            raise ValueError("Clifford operator not found in the group.")
        return inverse

    def create_rb_sequences(
        self,
        n: int,
        type: Literal["1Q", "1Q1Q", "2Q"] = "1Q",
        seed: int | None = None,
    ) -> tuple[list[list[str]], list[str]]:
        """
        Create a set of random Clifford operators for randomized benchmarking.

        Parameters
        ----------
        n : int
            The number of random Clifford operators to return.
        type : Literal["1Q", "1Q1Q", "2Q"] = "1Q"
            The type of Clifford operators to generate.
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
        composed = CliffordSequence.I() if type == "1Q" else CliffordSequence.II()
        for clifford_sequence in clifford_sequences:
            # Apply the random Clifford operator
            composed = composed.compose(clifford_sequence.clifford)
        # Compute the total inverse of the composed Clifford operators
        composed_inverse = self.get_inverse(composed, type)

        # Return the Clifford operators and their total inverse
        return [
            clifford_sequence.gate_sequence for clifford_sequence in clifford_sequences
        ], composed_inverse.gate_sequence

    def create_irb_sequences(
        self,
        n: int,
        interleave: Clifford | dict[str, tuple[complex, str]],
        type: Literal["1Q", "1Q1Q", "2Q"] = "1Q",
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
        type : Literal["1Q", "1Q1Q", "2Q"] = "1Q"
            The type of Clifford operators to generate.
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
        if isinstance(interleave, Clifford):
            interleave_cliford = interleave
        else:
            interleave_cliford = Clifford(
                name="U",
                map={operator: Pauli(*interleave[operator]) for operator in interleave},
            )
        # Compose the Clifford operators with the interleaved operator
        composed = CliffordSequence.I() if type == "1Q" else CliffordSequence.II()
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

        found_clifford_list = list(found_cliffords.values())
        list_count = len(found_clifford_list)
        max_gate_count = max(sequence.length for sequence in found_clifford_list)
        max_x90_count = max(sequence.count(x90) for sequence in found_clifford_list)
        sum_x90_count = sum(sequence.count(x90) for sequence in found_clifford_list)
        avg_x90_count = sum_x90_count / list_count

        print(f"Generated {list_count} unique 1Q Clifford sequences.")
        print()
        print(f"  Maximum gate count per Clifford: {max_gate_count}")
        print(f"  Maximum X90 count per Clifford: {max_x90_count}")
        print(f"  Total X90 count: {sum_x90_count}")
        print(f"  Average X90 count: {avg_x90_count}")

        self._cliffords_1q = found_cliffords
        return self._cliffords_1q

    def generate_1q1q_cliffords(
        self,
    ) -> dict[Clifford, CliffordSequence]:
        """
        Generate unique 1Q1Q Clifford sequences.

        Returns
        -------
        dict[Clifford, CliffordSequence]
            A dictionary of unique 1Q1Q Clifford operators and their sequences.
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
                self._cliffords_1q1q[seq_2q.clifford] = seq_2q

        found_clifford_list = list(self._cliffords_1q1q.values())
        list_count = len(found_clifford_list)
        max_gate_count = max(sequence.length for sequence in found_clifford_list)

        print(f"Generated {list_count} unique 1Q1Q Clifford sequences.")
        print()
        print(f"  Maximum gate count per Clifford: {max_gate_count}")

        return self._cliffords_1q1q

    def generate_2q_cliffords(
        self,
        two_qubit_gate: Clifford = Clifford.ZX90(),
    ) -> dict[Clifford, CliffordSequence]:
        """
        Generate unique 2Q Clifford sequences.

        Parameters
        ----------
        two_qubit_gate : Clifford, optional
            The two-qubit gate to use in the Clifford sequences.

        Returns
        -------
        dict[Clifford, CliffordSequence]
            A dictionary of unique 2Q Clifford operators and their sequences.
        """

        IX90 = Clifford.IX90()
        XI90 = Clifford.XI90()
        GATE_2Q = two_qubit_gate

        clifford_sequences_1q1q = self.get_clifford_sequences("1Q1Q")

        def count_1q_gate(seq: CliffordSequence) -> int:
            return seq.count(IX90) + seq.count(XI90)

        def count_2q_gate(seq: CliffordSequence) -> int:
            return seq.count(GATE_2Q)

        def choose_sequence(
            seq1: CliffordSequence,
            seq2: CliffordSequence,
        ) -> CliffordSequence:
            seq1_2q_count = count_2q_gate(seq1)
            seq2_2q_count = count_2q_gate(seq2)
            if seq2_2q_count < seq1_2q_count:
                return seq2
            elif seq2_2q_count == seq1_2q_count:
                seq1_1q_count = count_1q_gate(seq1)
                seq2_1q_count = count_1q_gate(seq2)
                if seq2_1q_count < seq1_1q_count:
                    return seq2
            return seq1

        def apply_1q1q_clifford(
            cliffords: dict[Clifford, CliffordSequence],
        ) -> dict[Clifford, CliffordSequence]:
            new_cliffords: dict[Clifford, CliffordSequence] = {} | cliffords
            for seq in cliffords.values():
                for c_1q1q in clifford_sequences_1q1q:
                    composed = seq.compose(c_1q1q)
                    clifford = composed.clifford
                    if clifford not in new_cliffords:
                        new_cliffords[clifford] = composed
                    else:
                        existing = new_cliffords[clifford]
                        new_cliffords[clifford] = choose_sequence(existing, composed)
            return new_cliffords

        def apply_2q_clifford(
            cliffords: dict[Clifford, CliffordSequence],
        ) -> dict[Clifford, CliffordSequence]:
            new_cliffords: dict[Clifford, CliffordSequence] = {} | cliffords
            for seq in cliffords.values():
                composed = seq.compose(GATE_2Q)
                clifford = composed.clifford
                if clifford not in new_cliffords:
                    new_cliffords[clifford] = composed
                else:
                    existing = new_cliffords[clifford]
                    new_cliffords[clifford] = choose_sequence(existing, composed)
            return new_cliffords

        found_cliffords_1 = self.get_cliffords("1Q1Q")
        print(f"1. 1Q1Q : {len(found_cliffords_1)}")

        found_cliffords_2 = apply_2q_clifford(found_cliffords_1)
        print(f"2. 1Q1Q - 2Q : {len(found_cliffords_2)}")

        found_cliffords_3 = apply_1q1q_clifford(found_cliffords_2)
        print(f"3. 1Q1Q - 2Q - 1Q1Q : {len(found_cliffords_3)}")

        found_cliffords_4 = apply_2q_clifford(found_cliffords_3)
        print(f"4. 1Q1Q - 2Q - 1Q1Q - 2Q : {len(found_cliffords_4)}")

        found_cliffords_5 = apply_1q1q_clifford(found_cliffords_4)
        print(f"5. 1Q1Q - 2Q - 1Q1Q - 2Q - 1Q1Q : {len(found_cliffords_5)}")

        found_cliffords_6 = apply_2q_clifford(found_cliffords_5)
        print(f"6. 1Q1Q - 2Q - 1Q1Q - 2Q - 1Q1Q - 2Q : {len(found_cliffords_6)}")

        self._cliffords_2q = found_cliffords_6

        found_clifford_list = list(self._cliffords_2q.values())
        list_count = len(found_clifford_list)
        max_gate_count = max(sequence.length for sequence in found_clifford_list)
        max_1q_count = max(count_1q_gate(sequence) for sequence in found_clifford_list)
        sum_1q_count = sum(count_1q_gate(sequence) for sequence in found_clifford_list)
        avg_1q_count = sum_1q_count / list_count
        max_2q_count = max(count_2q_gate(sequence) for sequence in found_clifford_list)
        sum_2q_count = sum(count_2q_gate(sequence) for sequence in found_clifford_list)
        max_2q_count = max(count_2q_gate(sequence) for sequence in found_clifford_list)
        avg_2q_count = sum_2q_count / list_count

        print(f"Generated {list_count} unique 2Q Clifford sequences.")
        print()
        print(f"  Maximum gate count per Clifford: {max_gate_count}")
        print(f"  Maximum 1Q gate count per Clifford: {max_1q_count}")
        print(f"  Total 1Q gate count: {sum_1q_count}")
        print(f"  Average 1Q gate count: {avg_1q_count}")
        print(f"  Maximum 2Q gate count per Clifford: {max_2q_count}")
        print(f"  Total 2Q gate count: {sum_2q_count}")
        print(f"  Average 2Q gate count: {avg_2q_count}")

        return self._cliffords_2q

    def get_file_path(
        self,
        type: Literal["1Q", "1Q1Q", "2Q"] = "1Q",
    ):
        dir = Path(__file__).parent / CLIFFORD_LIST_DIR
        if type == "1Q":
            return dir / f"{CLIFFORD_LIST_1Q}.json"
        elif type == "1Q1Q":
            return dir / f"{CLIFFORD_LIST_1Q1Q}.json"
        else:
            return dir / f"{CLIFFORD_LIST_2Q}.json"

    def save(
        self,
        type: Literal["1Q", "1Q1Q", "2Q"] = "1Q",
    ):
        """Save the Clifford group to a JSON file."""
        file_path = self.get_file_path(type)
        clifford_list = self.get_clifford_list(type)
        with open(file_path, "w") as file:
            json.dump(clifford_list, file, indent=4)

    def load(
        self,
        type: Literal["1Q", "1Q1Q", "2Q"],
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
                self._cliffords_1q[clifford_sequence.clifford] = clifford_sequence
            elif type == "1Q1Q":
                self._cliffords_1q1q[clifford_sequence.clifford] = clifford_sequence
            else:
                self._cliffords_2q[clifford_sequence.clifford] = clifford_sequence

        print(f"Loaded {len(data)} Clifford sequences.")
