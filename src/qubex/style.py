import plotly.colors as colors
import plotly.graph_objects as go
import plotly.io as pio

COLORS = colors.DEFAULT_PLOTLY_COLORS
FONT_FAMILY = "Times New Roman"
WIDTH = 600
HEIGHT = 300
MARGIN_L = 70
MARGIN_R = 70
MARGIN_B = 70
MARGIN_T = 70
TITLE_FONT_SIZE = 18
AXIS_TITLEFONT_SIZE = 16
AXIS_TICKFONT_SIZE = 14
LEGEND_FONT_SIZE = 14

pio.templates["qubex"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT_FAMILY),
        title=dict(font=dict(size=TITLE_FONT_SIZE), x=0.5),
        xaxis=dict(
            mirror=True,
            showline=True,
            zeroline=False,
            showgrid=False,
            linecolor="black",
            linewidth=1,
            ticks="inside",
            tickfont=dict(size=AXIS_TICKFONT_SIZE),
            titlefont=dict(size=AXIS_TITLEFONT_SIZE),
            domain=[0.0, 1.0],
        ),
        yaxis=dict(
            mirror=True,
            showline=True,
            zeroline=False,
            showgrid=False,
            linecolor="black",
            linewidth=1,
            ticks="inside",
            tickfont=dict(size=AXIS_TICKFONT_SIZE),
            titlefont=dict(size=AXIS_TITLEFONT_SIZE),
            domain=[0.0, 1.0],
        ),
        legend=dict(font=dict(size=LEGEND_FONT_SIZE)),
        autosize=False,
        width=WIDTH,
        height=HEIGHT,
        margin=dict(l=MARGIN_L, r=MARGIN_R, b=MARGIN_B, t=MARGIN_T),
        colorway=[
            "#0C5DA5",
            "#00B945",
            "#FF9500",
            "#FF2C00",
            "#845B97",
            "#474747",
            "#9e9e9e",
        ],
        plot_bgcolor="white",
        paper_bgcolor="white",
    ),
    data=dict(
        scatter=[
            go.Scatter(
                mode="markers",
                marker=dict(size=6),
            )
        ]
    ),
)


def apply_template(template: str):
    pio.templates.default = template
