"""Schedule and sequencer visualization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
from qxvisualizer.figure import DEFAULT_TEMPLATE, make_figure, show_figure

from .style import COLORS

if TYPE_CHECKING:
    from qubex.measurement.models.measurement_schedule import MeasurementSchedule

MEASUREMENT_SCHEDULE_DEFAULT_WIDTH = 900
MEASUREMENT_SCHEDULE_ROW_HEIGHT = 90
MEASUREMENT_SCHEDULE_BASE_HEIGHT = 170

SEQUENCER_TIMELINE_DEFAULT_WIDTH = 900
SEQUENCER_TIMELINE_MIN_HEIGHT = 320
SEQUENCER_TIMELINE_BASE_HEIGHT = 120
SEQUENCER_TIMELINE_LANE_HEIGHT = 42

_MEASUREMENT_BLANK_FILL_COLOR = "#EEF3F8"
_MEASUREMENT_CAPTURE_FILL_COLOR = "#F6C58F"
_MEASUREMENT_CAPTURE_LINE_COLOR = "#D48A3D"
_MEASUREMENT_CAPTURE_LABEL_COLOR = "#8C4F1F"
_MEASUREMENT_CAPTURE_BAND_Y0 = 0.0
_MEASUREMENT_CAPTURE_BAND_Y1 = 0.2
_MEASUREMENT_CAPTURE_LABEL_Y = 0.1
_SEQUENCER_WAVEFORM_FILL_COLOR = "#6A88A8"
_SEQUENCER_WAVEFORM_LINE_COLOR = "#48657F"


def _x_axis_reference(index: int) -> str:
    """Return Plotly x-axis reference string for one subplot row."""
    return "x" if index == 1 else f"x{index}"


def _primary_y_axis_reference(index: int) -> str:
    """
    Return primary Plotly y-axis reference for one row with secondary y-axes.

    With `secondary_y=True`, subplot rows use y-axis numbering:
    row1 -> y, row2 -> y3, row3 -> y5, ...
    """
    axis_number = 1 + 2 * (index - 1)
    return "y" if axis_number == 1 else f"y{axis_number}"


def make_measurement_schedule_figure(
    schedule: MeasurementSchedule,
    *,
    show_physical_pulse: bool = False,
    hide_workaround_capture: bool = True,
    title: str = "Measurement Schedule",
    width: int = MEASUREMENT_SCHEDULE_DEFAULT_WIDTH,
    n_samples: int | None = None,
    divide_by_two_pi: bool = False,
    line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """Build a measurement-schedule figure with pulse and capture overlays."""
    pulse_schedule = schedule.pulse_schedule
    sequences = pulse_schedule.get_sequences(copy=False)
    blank_ranges = pulse_schedule.get_blank_ranges()
    n_channels = len(sequences)

    if n_channels == 0:
        raise ValueError("MeasurementSchedule must include at least one pulse channel.")

    figure = make_figure(template=template, width=width)
    figure.set_subplots(
        rows=n_channels,
        cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}] for _ in range(n_channels)],
    )

    blank_legend_added = False
    capture_legend_added = False
    for row_index, (label, sequence) in enumerate(sequences.items(), start=1):
        if sequence.length == 0:
            times = np.array([0.0], dtype=np.float64)
            real = np.array([0.0], dtype=np.float64)
            imag = np.array([0.0], dtype=np.float64)
            phase = np.array([0.0], dtype=np.float64)
        else:
            times = np.append(
                sequence.times, sequence.times[-1] + sequence.SAMPLING_PERIOD
            )
            values = sequence.get_values(apply_frame_shifts=show_physical_pulse)
            real = np.append(np.real(values), np.real(values)[-1])
            imag = np.append(np.imag(values), np.imag(values)[-1])
            phase = -np.append(sequence.frame_shifts, sequence.final_frame_shift)
            phase = (phase + np.pi) % (2 * np.pi) - np.pi

        if n_samples is not None and len(times) > n_samples:
            indices = np.linspace(0, len(times) - 1, n_samples).astype(int)
            times = times[indices]
            real = real[indices]
            imag = imag[indices]
            phase = phase[indices]

        if divide_by_two_pi:
            real = real / (2 * np.pi * 1e-3)
            imag = imag / (2 * np.pi * 1e-3)

        figure.add_trace(
            go.Scatter(
                x=times,
                y=real,
                mode="lines",
                line_shape=line_shape,
                line={"color": COLORS[0]},
                name="I" if show_physical_pulse else "X",
                showlegend=(row_index == 1),
            ),
            row=row_index,
            col=1,
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=imag,
                mode="lines",
                line_shape=line_shape,
                line={"color": COLORS[1]},
                name="Q" if show_physical_pulse else "Y",
                showlegend=(row_index == 1),
            ),
            row=row_index,
            col=1,
            secondary_y=False,
        )

        if not show_physical_pulse:
            figure.add_trace(
                go.Scatter(
                    x=times,
                    y=phase,
                    mode="lines",
                    line_shape=line_shape,
                    line={"color": COLORS[2], "dash": "dot"},
                    name="φ",
                    showlegend=(row_index == 1),
                ),
                row=row_index,
                col=1,
                secondary_y=True,
            )

        channel_blank_ranges = blank_ranges.get(label, [])
        for blank_range in channel_blank_ranges:
            x_ref = _x_axis_reference(row_index)
            y_ref = f"{_primary_y_axis_reference(row_index)} domain"
            blank_start = float(blank_range.start * sequence.SAMPLING_PERIOD)
            blank_end = float(blank_range.stop * sequence.SAMPLING_PERIOD)
            figure.add_shape(
                type="rect",
                xref=x_ref,
                yref=y_ref,
                x0=blank_start,
                x1=blank_end,
                y0=0.0,
                y1=1.0,
                fillcolor=_MEASUREMENT_BLANK_FILL_COLOR,
                opacity=0.4,
                line_width=0,
                layer="below",
            )
        if channel_blank_ranges and not blank_legend_added:
            figure.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": _MEASUREMENT_BLANK_FILL_COLOR,
                        "symbol": "square",
                    },
                    name="Blank",
                    showlegend=True,
                ),
                row=row_index,
                col=1,
                secondary_y=False,
            )
            blank_legend_added = True

        captures = sorted(
            schedule.capture_schedule.channels.get(label, []),
            key=lambda capture: capture.start_time,
        )
        if hide_workaround_capture:
            captures = [
                capture
                for capture in captures
                if not getattr(capture, "is_workaround", False)
            ]
        for capture_index, capture in enumerate(captures):
            x_ref = _x_axis_reference(row_index)
            y_ref = f"{_primary_y_axis_reference(row_index)} domain"
            capture_start = float(capture.start_time)
            capture_end = float(capture.start_time + capture.duration)
            figure.add_shape(
                type="rect",
                xref=x_ref,
                yref=y_ref,
                x0=capture_start,
                x1=capture_end,
                y0=_MEASUREMENT_CAPTURE_BAND_Y0,
                y1=_MEASUREMENT_CAPTURE_BAND_Y1,
                fillcolor=_MEASUREMENT_CAPTURE_FILL_COLOR,
                opacity=0.55,
                line={"color": _MEASUREMENT_CAPTURE_LINE_COLOR, "width": 1},
                layer="below",
            )
            figure.add_annotation(
                x=(capture_start + capture_end) / 2,
                y=_MEASUREMENT_CAPTURE_LABEL_Y,
                xref=x_ref,
                yref=y_ref,
                text=f"<b>capture_{capture_index}</b>",
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font={
                    "size": 10,
                    "color": _MEASUREMENT_CAPTURE_LABEL_COLOR,
                    "family": "Avenir Next, Avenir, Helvetica Neue, Arial, sans-serif",
                },
            )
            if not capture_legend_added:
                figure.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker={
                            "size": 10,
                            "color": _MEASUREMENT_CAPTURE_FILL_COLOR,
                            "line": {
                                "color": _MEASUREMENT_CAPTURE_LINE_COLOR,
                                "width": 1,
                            },
                            "symbol": "square",
                        },
                        name="Capture",
                        showlegend=True,
                    ),
                    row=row_index,
                    col=1,
                    secondary_y=False,
                )
                capture_legend_added = True

        y_max = (
            float(np.max(np.abs(np.concatenate([real, imag]))))
            if len(real) > 0
            else 1.0
        )
        if y_max == 0:
            y_max = 1.0

        figure.update_yaxes(
            row=row_index,
            col=1,
            title_text=label,
            range=[-1.2 * y_max, 1.2 * y_max],
            secondary_y=False,
        )
        if not show_physical_pulse:
            figure.update_yaxes(
                row=row_index,
                col=1,
                range=[-np.pi * 1.2, np.pi * 1.2],
                tickvals=[-np.pi, 0, np.pi],
                ticktext=["-π", "0", "π"],
                secondary_y=True,
            )

    figure.update_layout(
        title=title,
        height=MEASUREMENT_SCHEDULE_ROW_HEIGHT * n_channels
        + MEASUREMENT_SCHEDULE_BASE_HEIGHT,
    )
    figure.update_xaxes(row=n_channels, col=1, title_text="Time (ns)")
    return figure


def plot_measurement_schedule(
    schedule: MeasurementSchedule,
    *,
    show_physical_pulse: bool = False,
    hide_workaround_capture: bool = True,
    title: str = "Measurement Schedule",
    width: int = MEASUREMENT_SCHEDULE_DEFAULT_WIDTH,
    n_samples: int | None = None,
    divide_by_two_pi: bool = False,
    line_shape: Literal["hv", "vh", "hvh", "vhv", "spline", "linear"] = "hv",
    template: str = DEFAULT_TEMPLATE,
) -> None:
    """Plot a measurement schedule and show the figure."""
    figure = make_measurement_schedule_figure(
        schedule,
        show_physical_pulse=show_physical_pulse,
        hide_workaround_capture=hide_workaround_capture,
        title=title,
        width=width,
        n_samples=n_samples,
        divide_by_two_pi=divide_by_two_pi,
        line_shape=line_shape,
        template=template,
    )
    show_figure(
        figure,
        filename="measurement_schedule",
        width=width,
    )


def make_sequencer_timeline_figure(
    sequencer: Any,
    *,
    title: str = "Sequencer Timeline",
    width: int = SEQUENCER_TIMELINE_DEFAULT_WIDTH,
    template: str = DEFAULT_TEMPLATE,
) -> go.Figure:
    """Build a sequencer timeline figure with event and capture lanes."""
    state = vars(sequencer)
    waveform_library = state["_waveform_library"]
    alias_to_events = state["_alias_to_events"]
    alias_to_capwin = state["_alias_to_capwin"]

    lanes = [f"w:{alias}" for alias in alias_to_events]
    for alias in alias_to_capwin:
        lane = f"c:{alias}"
        if lane not in lanes:
            lanes.append(lane)
    if len(lanes) == 0:
        raise ValueError("Sequencer timeline is empty.")
    left_margin = max(120, 20 + max((len(lane) for lane in lanes), default=0) * 7)

    figure = make_figure(template=template, width=width)
    pulse_legend_added = False
    capture_legend_added = False

    for alias, events in alias_to_events.items():
        lane = f"w:{alias}"
        for event in events:
            waveform = waveform_library[event.waveform_name]
            duration_ns = len(waveform.iq_array) * float(waveform.sampling_period_ns)
            start_ns = float(event.start_offset_ns)
            figure.add_trace(
                go.Bar(
                    x=[duration_ns],
                    y=[lane],
                    base=[start_ns],
                    orientation="h",
                    marker={
                        "color": _SEQUENCER_WAVEFORM_FILL_COLOR,
                        "line": {
                            "color": _SEQUENCER_WAVEFORM_LINE_COLOR,
                            "width": 1,
                        },
                    },
                    opacity=0.9,
                    name="Waveform Event",
                    showlegend=not pulse_legend_added,
                    hovertemplate=(
                        f"lane={lane}<br>"
                        f"waveform={event.waveform_name}<br>"
                        f"start={start_ns:.3f} ns<br>"
                        f"duration={duration_ns:.3f} ns<br>"
                        f"gain={float(event.gain):.4g}<br>"
                        f"phase={float(event.phase_offset_deg):.2f} deg"
                        "<extra></extra>"
                    ),
                )
            )
            pulse_legend_added = True

    for alias, windows in alias_to_capwin.items():
        lane = f"c:{alias}"
        for window in windows:
            duration_ns = float(window.length_ns)
            start_ns = float(window.start_offset_ns)
            figure.add_trace(
                go.Bar(
                    x=[duration_ns],
                    y=[lane],
                    base=[start_ns],
                    orientation="h",
                    marker={
                        "color": _MEASUREMENT_CAPTURE_FILL_COLOR,
                        "line": {
                            "color": _MEASUREMENT_CAPTURE_LINE_COLOR,
                            "width": 1,
                        },
                    },
                    opacity=0.75,
                    name="Capture Window",
                    showlegend=not capture_legend_added,
                    hovertemplate=(
                        f"lane={lane}<br>"
                        f"window={window.name}<br>"
                        f"start={start_ns:.3f} ns<br>"
                        f"duration={duration_ns:.3f} ns"
                        "<extra></extra>"
                    ),
                )
            )
            capture_legend_added = True

    figure.update_layout(
        title=title,
        barmode="overlay",
        height=max(
            SEQUENCER_TIMELINE_MIN_HEIGHT,
            SEQUENCER_TIMELINE_BASE_HEIGHT
            + SEQUENCER_TIMELINE_LANE_HEIGHT * len(lanes),
        ),
        xaxis_title="Time (ns)",
        yaxis_title="Lane",
        margin={"l": left_margin},
    )
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=lanes[::-1],
        automargin=True,
    )
    return figure


def plot_sequencer_timeline(
    sequencer: Any,
    *,
    title: str = "Sequencer Timeline",
    width: int = SEQUENCER_TIMELINE_DEFAULT_WIDTH,
    template: str = DEFAULT_TEMPLATE,
) -> None:
    """Plot a sequencer timeline and show the figure."""
    figure = make_sequencer_timeline_figure(
        sequencer,
        title=title,
        width=width,
        template=template,
    )
    show_figure(
        figure,
        filename="sequencer_timeline",
        width=width,
    )
