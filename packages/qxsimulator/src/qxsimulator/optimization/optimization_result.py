"""Define results returned by pulse optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
import qutip as qt
from numpy.typing import NDArray
from qxvisualizer.figure import show_figure


@dataclass
class OptimizationResult:
    """
    Store optimized controls and their simulated outcome.

    Attributes
    ----------
    params : Any
        Optimizer parameter pytree. Results from `PulseOptimizer` contain one
        `(n_segments, 2)` JAX array per target, with I and Q components in
        rad/ns.
    infidelity : float
        Final unitary trace-overlap infidelity. This value is dimensionless.
    unitary : qt.Qobj
        Final propagator in the full system Hilbert space.
    state : qt.Qobj
        Initial state propagated by `unitary`.
    times : NDArray[np.float64]
        Segment-boundary times in ns with shape `(n_segments + 1,)`.
    waveforms : dict[str, NDArray[np.complex128]]
        Complex control amplitudes `I + 1j * Q` in rad/ns. Each target maps to
        an array with shape `(n_segments,)`.
    history : NDArray[np.float64]
        Dimensionless unitary-infidelity values recorded after optimizer
        updates, with shape `(n_iterations,)`.
    """

    params: Any
    infidelity: float
    unitary: qt.Qobj
    state: qt.Qobj
    times: NDArray[np.float64]
    waveforms: dict[str, NDArray[np.complex128]]
    history: NDArray[np.float64]

    def plot_waveforms(self) -> None:
        """
        Display the optimized I and Q waveforms for each control target.

        Notes
        -----
        Amplitudes are converted from rad/ns to MHz for display. This method
        invokes the configured Plotly renderer once per target.
        """
        for target, waveform in self.waveforms.items():
            dt = self.times[1] - self.times[0]
            times = np.append(self.times, self.times[-1] + dt)
            waveform = waveform / (2 * np.pi) * 1e3
            real = np.append(waveform.real, waveform.real[-1])
            imag = np.append(waveform.imag, waveform.imag[-1])

            waveform = waveform / (2 * np.pi) * 1e3
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=real,
                    mode="lines",
                    name="I",
                    line_shape="hv",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=imag,
                    mode="lines",
                    name="Q",
                    line_shape="hv",
                )
            )
            fig.update_layout(
                title=f"Waveform : {target}",
                xaxis_title="Time (ns)",
                yaxis_title="Amplitude (MHz)",
                template="qubex",
            )
            show_figure(fig, filename=f"waveform_{target}")

    def plot_history(self) -> None:
        """
        Display the unitary-infidelity history on a logarithmic scale.

        Notes
        -----
        This method invokes the configured Plotly renderer.
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=self.history,
                mode="lines",
            )
        )
        fig.update_layout(
            title="Optimization history",
            xaxis_title="Number of iterations",
            yaxis_title="Loss function",
            yaxis=dict(
                type="log",
            ),
            template="qubex",
        )
        show_figure(fig, filename="optimization_history")
