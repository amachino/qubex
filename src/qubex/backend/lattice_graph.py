from __future__ import annotations

import math
from typing import Final

MUX_SIZE = 4

PREFIX_QUBIT = "Q"
PREFIX_RESONATOR = "RQ"
PREFIX_MUX = "MUX"


class LatticeGraph:
    """
    ex1) n_qubits = 16
    |00|01|04|05|
    |02|03|06|07|
    |08|09|12|13|
    |10|11|14|15|

    ex2) n_qubits = 64
    |00|01|04|05|08|09|12|13|
    |02|03|06|07|10|11|14|15|
    |16|17|20|21|24|25|28|29|
    |18|19|22|23|26|27|30|31|
    |32|33|36|37|40|41|44|45|
    |34|35|38|39|42|43|46|47|
    |48|49|52|53|56|57|60|61|
    |50|51|54|55|58|59|62|63|

    ex3) n_qubits = 144
    |000|001|004|005|008|009|012|013|016|017|020|021|
    |002|003|006|007|010|011|014|015|018|019|022|023|
    |024|025|028|029|032|033|036|037|040|041|044|045|
    |026|027|030|031|034|035|038|039|042|043|046|047|
    |048|049|052|053|056|057|060|061|064|065|068|069|
    |050|051|054|055|058|059|062|063|066|067|070|071|
    |072|073|076|077|080|081|084|085|088|089|092|093|
    |074|075|078|079|082|083|086|087|090|091|094|095|
    |096|097|100|101|104|105|108|109|112|113|116|117|
    |098|099|102|103|106|107|110|111|114|115|118|119|
    |120|121|124|125|128|129|132|133|136|137|140|141|
    |122|123|126|127|130|131|134|135|138|139|142|143|
    """

    def __init__(
        self,
        n_qubits: int,
    ):
        if n_qubits % MUX_SIZE != 0:
            raise ValueError("n_qubits must be a multiple of MUX_SIZE.")
        n_muxes = n_qubits // MUX_SIZE
        mux_side_length = math.isqrt(n_muxes)
        if mux_side_length**2 != n_muxes:
            raise ValueError("n_qubits must be a square number.")
        self.n_qubits: Final = n_qubits
        self.max_digit: Final = len(str(self.n_qubits - 1))
        self.edges: Final = self._create_edges(mux_side_length, mux_side_length)

    @property
    def n_muxes(
        self,
    ) -> int:
        """
        Get number of MUXes.

        Returns
        -------
        int
            Number of MUXes.
        """
        return self.n_qubits // MUX_SIZE

    @property
    def indices(
        self,
    ) -> list[int]:
        """
        Get qubit indices.

        Returns
        -------
        list[int]
            List of qubit indices.
        """
        return list(range(self.n_qubits))

    @property
    def qubits(
        self,
        prefix: str = PREFIX_QUBIT,
    ) -> list[str]:
        """
        Get qubit labels.

        Parameters
        ----------
        prefix : str, optional
            Prefix of qubit labels, by default "Q".

        Returns
        -------
        list[str]
            List of qubit labels.
        """
        return [f"{prefix}{str(i).zfill(self.max_digit)}" for i in self.indices]

    @property
    def resonators(
        self,
        prefix: str = PREFIX_RESONATOR,
    ) -> list[str]:
        """
        Get resonator labels.

        Parameters
        ----------
        prefix : str, optional
            Prefix of resonator labels, by default "RQ".

        Returns
        -------
        list[str]
            List of resonator labels.
        """
        return [f"{prefix}{str(i).zfill(self.max_digit)}" for i in self.indices]

    @property
    def muxes(
        self,
        prefix: str = PREFIX_MUX,
    ) -> list[str]:
        """
        Get MUX labels.

        Parameters
        ----------
        prefix : str, optional
            Prefix of MUX labels, by default "MUX".

        Returns
        -------
        list[str]
            List of MUX labels.
        """
        return [f"{prefix}{i}" for i in range(self.n_muxes)]

    @property
    def qubit_edges(
        self,
    ) -> list[tuple[str, str]]:
        """
        Get qubit edges.

        Returns
        -------
        list[tuple[str, str]]
            List of qubit edges.
        """
        return [(self.qubits[edge[0]], self.qubits[edge[1]]) for edge in self.edges]

    def get_indices_in_mux(
        self,
        mux: int | str,
    ) -> list[int]:
        """
        Get qubit indices in the input MUX.

        Parameters
        ----------
        mux : int, str
            MUX number or label.

        Returns
        -------
        list[int]
            List of qubit indices.
        """
        if isinstance(mux, str):
            mux = self.muxes.index(mux)
        base_qubit = mux * MUX_SIZE
        return [base_qubit + i for i in range(MUX_SIZE)]

    def get_qubits_in_mux(
        self,
        mux: int | str,
    ) -> list[str]:
        """
        Get qubit labels in the input MUX.

        Parameters
        ----------
        mux : int, str
            MUX number or label.

        Returns
        -------
        list[str]
            List of qubit labels.
        """
        return [self.qubits[i] for i in self.get_indices_in_mux(mux)]

    def get_resonators_in_mux(
        self,
        mux: int | str,
    ) -> list[str]:
        """
        Get resonator labels in the input MUX.

        Parameters
        ----------
        mux : int, str
            MUX number or label.

        Returns
        -------
        list[str]
            List of resonator labels.
        """
        return [self.resonators[i] for i in self.get_indices_in_mux(mux)]

    def get_mux_of_qubit(
        self,
        qubit: str | int,
    ) -> str:
        """
        Get MUX label of the input qubit.

        Parameters
        ----------
        qubit : str, int
            Qubit label or index.

        Returns
        -------
        str
            MUX label.
        """
        if isinstance(qubit, int):
            qubit = self.qubits[qubit]
        mux = self.qubits.index(qubit) // MUX_SIZE
        return self.muxes[mux]

    def get_mux_of_resonator(
        self,
        resonator: str | int,
    ) -> str:
        """
        Get MUX label of the input resonator.

        Parameters
        ----------
        resonator : str, int
            Resonator label or index.

        Returns
        -------
        str
            MUX label.
        """
        if isinstance(resonator, int):
            resonator = self.resonators[resonator]
        mux = self.resonators.index(resonator) // MUX_SIZE
        return self.muxes[mux]

    def get_spectator_indices(
        self,
        qubit: int | str,
        *,
        in_same_mux: bool = False,
    ) -> list[int]:
        """
        Get spectator indices of the input qubit.

        Parameters
        ----------
        qubit : int, str
            Qubit index or label.
        in_same_mux : bool, optional
            Whether to get only spectators in the same MUX, by default False.

        Returns
        -------
        list[int]
            List of spectator indices.
        """
        if isinstance(qubit, str):
            qubit = self.qubits.index(qubit)

        if in_same_mux:
            mux = self.get_mux_of_qubit(qubit)

        spectators = []
        for edge in self.edges:
            if edge[0] == qubit:
                spectator = edge[1]
            elif edge[1] == qubit:
                spectator = edge[0]
            else:
                continue
            if in_same_mux:
                if self.get_mux_of_qubit(spectator) != mux:
                    continue
            spectators.append(spectator)
        return spectators

    def get_spectator_qubits(
        self,
        qubit: int | str,
        *,
        in_same_mux: bool = False,
    ) -> list[str]:
        """
        Get spectator labels of the input qubit.

        Parameters
        ----------
        qubit : int, str
            Qubit index or label.
        in_same_mux : bool, optional
            Whether to get only spectators in the same MUX, by default False.

        Returns
        -------
        list[str]
            List of spectator labels.
        """
        return [
            self.qubits[spectator]
            for spectator in self.get_spectator_indices(qubit, in_same_mux=in_same_mux)
        ]

    def _create_edges(
        self,
        n_row: int,
        n_col: int,
    ) -> list[tuple[int, int]]:
        """
        Create edges of the lattice chip.

        Parameters
        ----------
        n_row : int
            Number of rows of MUXes.
        n_col : int
            Number of columns of MUXes.

        Returns
        -------
        list[tuple[int, int]]
            List of edges.
        """
        edge_set: set[tuple[int, int]] = set()

        for row in range(n_row):
            for col in range(n_col):
                # MUX number
                mux_number = row * n_col + col

                # Base qubit of the MUX
                base_qubit = mux_number * MUX_SIZE

                # Qubits in the MUX
                qubits = [base_qubit + i for i in range(MUX_SIZE)]

                # Internal MUX connections (within the same MUX)
                edge_set.add((qubits[0], qubits[1]))
                edge_set.add((qubits[0], qubits[2]))
                edge_set.add((qubits[1], qubits[3]))
                edge_set.add((qubits[2], qubits[3]))

                # Connections to adjacent MUXes
                # Right neighbor
                if col < n_col - 1:
                    right_base_qubit = base_qubit + MUX_SIZE
                    right_qubits = [right_base_qubit + i for i in range(MUX_SIZE)]
                    edge_set.add((qubits[1], right_qubits[0]))
                    edge_set.add((qubits[3], right_qubits[2]))

                # Down neighbor
                if row < n_row - 1:
                    down_base_qubit = base_qubit + n_col * MUX_SIZE
                    down_qubits = [down_base_qubit + i for i in range(MUX_SIZE)]
                    edge_set.add((qubits[2], down_qubits[0]))
                    edge_set.add((qubits[3], down_qubits[1]))

        edge_list = list(edge_set)
        edge_list.sort()
        return edge_list
