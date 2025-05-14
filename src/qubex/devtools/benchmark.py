from __future__ import annotations

import io
import sys
from typing import Any, Callable, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from numpy.typing import NDArray


def benchmark(
    func: Callable[[Any], Any],
    params: Sequence | NDArray,
    *,
    n_trials: int = 1,
    title: str = "Benchmark Results",
    xlabel: str = "Parameters",
    ylabel: str = "Time (seconds)",
    plot: bool = True,
) -> dict:
    """
    Benchmark a function with different parameters and plot the results.

    Parameters
    ----------

    func : Callable[[Any], Any]
        Function to benchmark.
    params : Sequence | NDArray
        Parameters to test the function with.
    n_trials : int, optional
        Number of trials to run for each parameter. Default is 1.

    Returns
    -------
    dict
        Dictionary containing the parameters, times, and figure.
        - 'params' : NDArray
            Parameters used for benchmarking.
        - 'times' : NDArray
            Times taken for each parameter.
        - 'fig' : plotly.graph_objects.Figure
            Figure object containing the benchmark results.
    """
    import time

    times = []
    times_stdev = []
    params = params

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_renderer = pio.renderers.default
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        pio.renderers.default = None
        for param in params:
            time_trials = []
            for _ in range(n_trials):
                start_time = time.time()
                func(param)
                time_trials.append(time.time() - start_time)
            times.append(np.mean(time_trials))
            times_stdev.append(np.std(time_trials))
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        pio.renderers.default = original_renderer

    params = np.asarray(params)
    times = np.asarray(times)
    times_stdev = np.asarray(times_stdev)

    fig = go.Figure(
        data=go.Scatter(
            x=params,
            y=times,
            error_y=dict(
                type="data",
                array=times_stdev,
                visible=True,
            ),
            mode="lines+markers",
        ),
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
    )
    if plot:
        fig.show()

    return {
        "params": params,
        "times": times,
        "fig": fig,
    }
