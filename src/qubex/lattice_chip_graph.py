from typing import Final

MUX_SIZE = 4


class LatticeChipGraph:
    """
    Lattice chip graph.

    ex1) n_row = 2, n_col = 2
    |00|01|04|05|
    |02|03|06|07|
    |08|09|12|13|
    |10|11|14|15|

    ex2) n_row = 4, n_col = 4
    |00|01|04|05|08|09|12|13|
    |02|03|06|07|10|11|14|15|
    |16|17|20|21|24|25|28|29|
    |18|19|22|23|26|27|30|31|
    |32|33|36|37|40|41|44|45|
    |34|35|38|39|42|43|46|47|
    |48|49|52|53|56|57|60|61|
    |50|51|54|55|58|59|62|63|

    ex3) n_row = 4, n_col = 8
    |000|001|004|005|008|009|012|013|016|017|020|021|024|025|028|029|
    |002|003|006|007|010|011|014|015|018|019|022|023|026|027|030|031|
    |032|033|036|037|040|041|044|045|048|049|052|053|056|057|060|061|
    |034|035|038|039|042|043|046|047|050|051|054|055|058|059|062|063|
    |064|065|068|069|072|073|076|077|080|081|084|085|088|089|092|093|
    |066|067|070|071|074|075|078|079|082|083|086|087|090|091|094|095|
    |096|097|100|101|104|105|108|109|112|113|116|117|120|121|124|125|
    |098|099|102|103|106|107|110|111|114|115|118|119|122|123|126|127|
    """

    def __init__(
        self,
        n_row: int,
        n_col: int,
    ):
        """
        Initialize the lattice chip graph.

        Parameters
        ----------
        n_row : int
            Number of rows of MUXes.
        n_col : int
            Number of columns of MUXes.
        """
        self.n_row: Final = n_row
        self.n_col: Final = n_col
        self.n_qubits: Final = self.n_row * self.n_col * MUX_SIZE
        self.max_digit: Final = len(str(self.n_qubits - 1))
        self.edges: Final = self.create_edges(n_row, n_col)

    @property
    def qubits(
        self,
        prefix: str = "Q",
    ) -> list[str]:
        """
        Get qubit names.

        Parameters
        ----------
        prefix : str, optional
            Prefix of qubit names, by default "Q".

        Returns
        -------
        list[str]
            List of qubit names.
        """
        return [f"{prefix}{str(i).zfill(self.max_digit)}" for i in range(self.n_qubits)]

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

    def get_spectators(
        self,
        qubit: str,
    ) -> list[str]:
        """
        Get spectators of the input qubit.

        Parameters
        ----------
        qubit : str
            Qubit label.

        Returns
        -------
        list[str]
            List of spectator qubit labels.
        """
        spectators = []
        for edge in self.qubit_edges:
            if edge[0] == qubit:
                spectator = edge[1]
            elif edge[1] == qubit:
                spectator = edge[0]
            else:
                continue
            spectators.append(spectator)
        return spectators

    def create_edges(
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
                    right_base_qubit = base_qubit + 4
                    right_qubits = [right_base_qubit + i for i in range(MUX_SIZE)]
                    edge_set.add((qubits[1], right_qubits[0]))
                    edge_set.add((qubits[3], right_qubits[2]))

                # Down neighbor
                if row < n_row - 1:
                    down_base_qubit = base_qubit + n_col * 4
                    down_qubits = [down_base_qubit + i for i in range(MUX_SIZE)]
                    edge_set.add((qubits[2], down_qubits[0]))
                    edge_set.add((qubits[3], down_qubits[1]))

        edge_list = list(edge_set)
        edge_list.sort()
        return edge_list
