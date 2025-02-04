from __future__ import annotations

import math
from typing import Collection, Final

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from ..analysis.visualization import save_figure_image

MUX_SIZE = 4
NODE_SIZE = 36


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
            raise ValueError(
                f"n_qubits ({n_qubits}) must be a multiple of MUX_SIZE ({MUX_SIZE})."
            )
        qubit_side_length = math.isqrt(n_qubits)
        if qubit_side_length**2 != n_qubits:
            raise ValueError(f"n_qubits ({n_qubits}) must be a perfect square.")

        n_muxes = n_qubits // MUX_SIZE
        mux_side_length = math.isqrt(n_muxes)
        if mux_side_length**2 != n_muxes:
            raise ValueError(f"n_muxes ({n_muxes}) must be a perfect square.")

        self.n_qubits: Final = n_qubits
        self.n_qubit_rows: Final = qubit_side_length
        self.n_qubit_cols: Final = qubit_side_length
        self.qubit_max_digit: Final = len(str(self.n_qubits - 1))
        qubit_graph = nx.grid_2d_graph(qubit_side_length, qubit_side_length)
        self.qubit_graph: Final = qubit_graph.to_directed()

        self.n_muxes: Final = n_muxes
        self.n_mux_rows: Final = mux_side_length
        self.n_mux_cols: Final = mux_side_length
        self.mux_max_digit: Final = len(str(self.n_muxes - 1))
        self.mux_graph: Final = nx.grid_2d_graph(mux_side_length, mux_side_length)

        self._init_qubit_nodes()
        self._init_qubit_edges()
        self._init_mux_nodes()

    def get_qubit_node(
        self,
        node: tuple[int, int],
    ) -> dict:
        return self.qubit_graph.nodes[node]

    def set_qubit_node(
        self,
        node: tuple[int, int],
        data: dict,
    ):
        self.qubit_graph.nodes[node].update(data)

    def get_qubit_edge(
        self,
        edge: tuple[tuple[int, int], tuple[int, int]],
    ) -> dict:
        return self.qubit_graph.edges[edge]

    def set_qubit_edge(
        self,
        edge: tuple[tuple[int, int], tuple[int, int]],
        data: dict,
    ):
        self.qubit_graph.edges[edge].update(data)

    def get_mux_node(
        self,
        node: tuple[int, int],
    ) -> dict:
        return self.mux_graph.nodes[node]

    def set_mux_node(
        self,
        node: tuple[int, int],
        data: dict,
    ):
        self.mux_graph.nodes[node].update(data)

    def _init_qubit_nodes(self):
        for x, y in self.qubit_graph.nodes():
            len_q = self.n_qubit_cols
            len_m = self.n_mux_cols
            len_qm = len_q // len_m
            col_m = x // len_qm
            row_m = y // len_qm
            idx_m = row_m * len_m + col_m
            idx_qm = (x % len_qm) + (y % len_qm) * len_qm
            idx_q = MUX_SIZE * idx_m + idx_qm
            self.set_qubit_node(
                (x, y),
                {
                    "index": idx_q,
                    "label": f"Q{idx_q:0{self.qubit_max_digit}d}",
                    "position": (x, y),
                    "mux": f"MUX{idx_m:0{self.mux_max_digit}d}",
                },
            )

    def _init_qubit_edges(self):
        for node0, node1 in self.qubit_graph.edges():
            qubit0 = self.get_qubit_node(node0)
            qubit1 = self.get_qubit_node(node1)
            self.set_qubit_edge(
                (node0, node1),
                {
                    "index": (qubit0["index"], qubit1["index"]),
                    "label": f"{qubit0['label']}-{qubit1['label']}",
                    "weight": np.random.rand(),
                },
            )

    def _init_mux_nodes(self):
        for x, y in self.mux_graph.nodes():
            idx = y * self.n_mux_cols + x
            self.set_mux_node(
                (x, y),
                {
                    "index": idx,
                    "label": f"MUX{idx:0{self.mux_max_digit}d}",
                    "position": (x * 2 + 0.5, y * 2 + 0.5),
                },
            )

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
        return [f"{prefix}{str(i).zfill(self.qubit_max_digit)}" for i in self.indices]

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
        return [f"{prefix}{str(i).zfill(self.qubit_max_digit)}" for i in self.indices]

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
        return [
            f"{prefix}{str(i).zfill(self.mux_max_digit)}" for i in range(self.n_muxes)
        ]

    @property
    def edges(
        self,
    ) -> list[tuple[int, int]]:
        """
        Get qubit edges.

        Returns
        -------
        list[tuple[int, int]]
            List of qubit edges.
        """
        return [data["index"] for _, _, data in self.qubit_graph.edges(data=True)]

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
        return [data["label"] for _, _, data in self.qubit_graph.edges(data=True)]

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

    def create_qubit_edge_trace(
        self,
        values: dict | None = None,
        texts: dict | None = None,
        hovertexts: dict | None = None,
    ) -> list[go.Scatter]:
        edges = self.qubit_graph.edges(data=True)

        weights = [data["weight"] for _, _, data in edges]
        w_min, w_max = min(weights), max(weights)

        trace = []
        for n0, n1, data in edges:
            qubit0 = self.get_qubit_node(n0)
            qubit1 = self.get_qubit_node(n1)
            x0, y0 = qubit0["position"]
            x1, y1 = qubit1["position"]
            weight = data["weight"]
            label = data["label"]

            if weight < 0.3:
                continue

            offset = 0.058
            margin = 0.32
            if x0 == x1:
                if y0 < y1:
                    x = [x0 + offset, x1 + offset]
                    y = [y0 + margin, y1 - margin]
                else:
                    x = [x0 - offset, x1 - offset]
                    y = [y0 - margin, y1 + margin]
            elif y0 == y1:
                if x0 < x1:
                    x = [x0 + margin, x1 - margin]
                    y = [y0 - offset, y1 - offset]
                else:
                    x = [x0 - margin, x1 + margin]
                    y = [y0 + offset, y1 + offset]

            norm_weight = (weight - w_min) / (w_max - w_min)
            color = sample_colorscale("Blues", norm_weight)[0]
            trace.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers+lines",
                    marker=dict(
                        symbol="arrow",
                        size=11,
                        color=color,
                        angleref="previous",
                        standoff=0,
                    ),
                    line=dict(
                        width=3,
                        color=color,
                    ),
                    text=f"{label}: {weight:.2f}",
                    hoverinfo="text",
                )
            )
        return trace

    def create_qubit_trace(self) -> go.Scatter:
        x = []
        y = []
        text = []
        for _, data in self.qubit_graph.nodes(data=True):
            pos = data["position"]
            x.append(pos[0])
            y.append(pos[1])
            text.append(data["label"])

        return go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker=dict(
                color="white",
                size=NODE_SIZE,
                line_width=2,
                line_color="black",
                showscale=False,
            ),
            text=text,
            textposition="middle center",
            textfont=dict(
                family="sans-serif",
                color="black",
                size=NODE_SIZE // 3,
            ),
            hoverinfo="text",
        )

    def create_mux_trace(self) -> go.Scatter:
        x = []
        y = []
        text = []
        for _, data in self.mux_graph.nodes(data=True):
            pos = data["position"]
            x.append(pos[0])
            y.append(pos[1])
            text.append(data["label"])

        return go.Scatter(
            x=x,
            y=y,
            mode="text",
            text=text,
            textposition="middle center",
            textfont=dict(
                family="sans-serif",
                color="lightgrey",
                weight="bold",
                size=NODE_SIZE // 3,
            ),
        )

    def plot_graph_data(
        self,
        *,
        title: str = "Graph Data",
        edge_values: dict | None = None,
        edge_texts: dict | None = None,
        edge_hovertexts: dict | None = None,
        image_name: str = "graph_data",
        images_dir: str = "./images",
        save_image: bool = False,
    ):
        layout = go.Layout(
            title=title,
            width=NODE_SIZE * 2 * self.n_qubit_cols,
            height=NODE_SIZE * 2 * self.n_qubit_rows,
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(
                ticks="",
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                ticks="",
                autorange="reversed",
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            plot_bgcolor="white",
            hovermode="closest",
            showlegend=False,
        )

        qubit_trace = self.create_qubit_trace()
        mux_trace = self.create_mux_trace()
        qubit_edge_trace = self.create_qubit_edge_trace(
            values=edge_values,
            texts=edge_texts,
            hovertexts=edge_hovertexts,
        )

        fig = go.Figure(
            data=[qubit_trace, mux_trace] + qubit_edge_trace,
            layout=layout,
        )
        fig.show()

        if save_image:
            save_figure_image(
                fig,
                name=image_name,
                images_dir=images_dir,
                format="png",
                width=NODE_SIZE * 2 * self.n_qubit_cols,
                height=NODE_SIZE * 2 * self.n_qubit_rows,
                scale=3,
            )

    def plot_lattice_data(
        self,
        *,
        title: str = "Latice Data",
        values: list | None = None,
        texts: list[str] | None = None,
        hovertexts: list[str] | None = None,
        image_name: str = "lattice_data",
        images_dir: str = "./images",
        save_image: bool = False,
    ):
        value_matrix = self.create_data_matrix(values) if values else None
        text_matrix = self.create_data_matrix(texts) if texts else None
        hovertext_matrix = self.create_data_matrix(hovertexts) if hovertexts else None

        fig = go.Figure(
            go.Heatmap(
                z=value_matrix,
                text=text_matrix,
                colorscale="Viridis",
                hoverinfo="text",
                hovertext=hovertext_matrix or text_matrix,
                texttemplate="%{text}",
                showscale=False,
                textfont=dict(
                    family="sans-serif",
                    size=12,
                    weight="bold",
                ),
            )
        )

        fig.update_layout(
            title=title,
            showlegend=False,
            margin=dict(b=30, l=30, r=30, t=60),
            xaxis=dict(
                ticks="",
                # showline=False,
                linewidth=1,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                ticks="",
                autorange="reversed",
                # showline=False,
                linewidth=1,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            height=144 * self.n_mux_rows,
            width=144 * self.n_mux_cols,
        )

        fig.show()

        if save_image:
            save_figure_image(
                fig,
                name=image_name,
                images_dir=images_dir,
                format="png",
                width=144 * self.n_mux_cols,
                height=144 * self.n_mux_rows,
                scale=3,
            )

    def create_data_matrix(
        self,
        data: Collection,
    ) -> list[list]:
        data = list(data)

        if len(data) != self.n_qubits:
            raise ValueError(
                f"Length of data ({len(data)}) must be equal to the number of qubits ({self.n_qubits})."
            )

        data_matrix = []

        for qubit_index, data in enumerate(data):
            mux_index = qubit_index // MUX_SIZE
            qubit_index_in_mux = qubit_index % MUX_SIZE
            mux_col = mux_index // self.n_mux_cols
            row_in_mux = qubit_index_in_mux // (MUX_SIZE // 2)

            if mux_col == 0 and row_in_mux == 0:
                data_matrix.append([])

            row = mux_col * 2 + row_in_mux
            data_matrix[row].append(data)

        return data_matrix
