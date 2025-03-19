from __future__ import annotations

import numpy as np
import plotly.graph_objs as go
from IPython.display import display
from ipywidgets import Output

from ..style import get_colors, get_config
from ..typing import IQArray, TargetMap


class IQPlotter:
    def __init__(
        self,
        state_centers: TargetMap[dict[int, complex]] | None = None,
    ):
        self._state_centers = state_centers or {}
        self._colors = [f"rgba{color}" for color in get_colors(alpha=0.8)]
        self._num_scatters = -1
        self._output = Output()
        self._widget = go.FigureWidget()
        self._widget.update_layout(
            title="I/Q plane",
            xaxis_title="In-phase (arb. units)",
            yaxis_title="Quadrature (arb. units)",
            width=500,
            height=400,
            margin=dict(l=120, r=120),
            xaxis=dict(
                zeroline=True,
                zerolinecolor="black",
                tickformat=".2g",
                showticklabels=True,
                showgrid=True,
            ),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                tickformat=".2g",
                showticklabels=True,
                zeroline=True,
                zerolinecolor="black",
                showgrid=True,
            ),
            showlegend=True,
        )

        if state_centers is not None:
            colors = get_colors(alpha=0.1)
            for idx, (label, centers) in enumerate(state_centers.items()):
                color = colors[idx % len(colors)]
                center_values = list(centers.values())
                x = np.real(center_values)
                y = np.imag(center_values)
                self._widget.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="text+markers",
                        text=[f"{state}" for state in centers],
                        name=f"{label}",
                        hoverinfo="name",
                        showlegend=True,
                        marker=dict(
                            symbol="circle",
                            size=24,
                            color=f"rgba{color}",
                        ),
                        textfont=dict(
                            size=12,
                            color="rgba(0, 0, 0, 0.8)",
                            family="sans-serif",
                            weight="bold",
                        ),
                        legendgroup="state",
                    )
                )

    def update(self, data: TargetMap[IQArray]):
        if self._num_scatters == -1:
            display(self._output)
            with self._output:
                display(self._widget)
            for idx, qubit in enumerate(data):
                if qubit in self._state_centers:
                    idx = list(self._state_centers.keys()).index(qubit)
                color = self._colors[idx % len(self._colors)]
                self._widget.add_scatter(
                    name=qubit,
                    meta=qubit,
                    mode="markers",
                    marker=dict(size=4, color=color),
                    legendrank=idx,
                )
            self._num_scatters = len(data)
        if len(data) != self._num_scatters:
            raise ValueError("Number of scatters does not match")

        max_val = np.max([np.max(np.abs(data[qubit])) for qubit in data])
        axis_range = [-max_val * 1.1, max_val * 1.1]
        dtick = max_val / 2

        self._widget.update_layout(
            xaxis=dict(
                range=axis_range,
                dtick=dtick,
            ),
            yaxis=dict(
                range=axis_range,
                dtick=dtick,
            ),
        )

        for qubit in data:
            for trace in self._widget.data:
                scatter: go.Scatter = trace  # type: ignore
                if scatter.meta == qubit:
                    scatter.x = np.real(data[qubit])
                    scatter.y = np.imag(data[qubit])

    def clear(self):
        self._output.clear_output()
        self._output.close()

    def show(self):
        self.clear()
        self._widget.show(config=get_config())


class IQPlotterPolar:
    def __init__(self, normalize: bool = True):
        self._normalize = normalize
        self._num_scatters: int | None = None
        self._widget = go.FigureWidget()
        self._widget.update_layout(
            title="I/Q plane",
            width=500,
            height=400,
            margin=dict(l=120, r=120),
            showlegend=True,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    showline=True,
                    showticklabels=True,
                    side="counterclockwise",
                    ticks="inside",
                    dtick=0.5,
                    gridcolor="lightgrey",
                    gridwidth=1,
                ),
                angularaxis=dict(
                    visible=True,
                    showline=True,
                    showticklabels=False,
                    ticks="inside",
                    dtick=30,
                    gridcolor="lightgrey",
                    gridwidth=1,
                ),
            ),
        )

    def update(
        self,
        data: TargetMap[IQArray],
    ):
        if self._num_scatters is None:
            display(self._widget)
            for qubit in data:
                self._widget.add_scatterpolar(name=qubit, mode="markers")
            self._num_scatters = len(data)
        if len(data) != self._num_scatters:
            raise ValueError("Number of scatters does not match")

        if self._normalize:
            self._widget.update_layout(polar_radialaxis_range=[0, 1.1])

        signals = {}
        for qubit, iq_list in data.items():
            iq_array = np.array(iq_list)
            if self._normalize:
                r = np.abs(iq_array[0])
                theta = np.angle(iq_array[0])
                signals[qubit] = (iq_array * np.exp(-1j * theta)) / r
            else:
                signals[qubit] = iq_array

        for idx, qubit in enumerate(data):
            scatterpolar: go.Scatterpolar = self._widget.data[idx]  # type: ignore
            scatterpolar.r = np.abs(signals[qubit])
            scatterpolar.theta = np.angle(signals[qubit], deg=True)
