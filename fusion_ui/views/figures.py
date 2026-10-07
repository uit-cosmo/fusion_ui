"""Plotly plumbing the Fields builders share: a grid of equal-aspect cells, a message, colours.

``make_subplots`` plus ``add_trace(row=, col=)`` costs milliseconds per trace, and the pixel level
draws a few dozen. Every builder here therefore lays its own axes out as a grid of cells and hands
Plotly its traces in one batch, each already naming its axes.
"""

import numpy as np
import plotly.graph_objects as go

#: Marker colours. The three tracks and the two TDE lines keep the colours the group's own figures
#: give them (``plotting_scripts.twodca_plots.TRACK_STYLES``, ``figures.TDE_COLORS``), so a figure
#: here reads like the one in the deck.
TRACK_COLOURS = {"max": "#1f77b4", "com": "#7fb8e0", "2dcc": "#ff7f0e"}
TRACK_SYMBOLS = {"max": "triangle-up", "com": "circle", "2dcc": "square"}
TDE_COLOURS = {"3": "#2ca02c", "2": "#9467bd"}

#: How a pixel that is not drawn as a number is marked, and why. A dead pixel was never computed; a
#: failed fit is a live pixel whose estimator returned nothing; a cut pixel has a number the view
#: cuts leave out. They look different because they mean different things.
DEAD_STYLE = dict(
    symbol="square", size=8, color="#c7c7c7", line=dict(width=1, color="#7f7f7f")
)
FAILED_STYLE = dict(
    symbol="x", size=9, color="#d62728", line=dict(width=1.5, color="#d62728")
)
CUT_STYLE = dict(
    symbol="circle-open", size=9, color="#7f7f7f", line=dict(width=1.5, color="#7f7f7f")
)
SELECTED_STYLE = dict(
    symbol="circle-open",
    size=17,
    color="rgba(0,0,0,0)",
    line=dict(width=2.5, color="#d62728"),
)


def message_figure(text, height=170):
    """An empty figure that says ``text``: what a builder returns when there is nothing to draw.

    A pixel with no events, a dead pixel and a missing product are not errors. The page puts this
    where the figure would be, so the reason is on the screen and not in a traceback.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=text,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#555"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def axis_name(k, kind="x"):
    """Plotly's name for the k-th axis (1-based): ``x``, ``x2``, ``x3``, …."""
    return kind if k == 1 else f"{kind}{k}"


def layout_key(k, kind="x"):
    """The layout property of that axis: ``xaxis``, ``xaxis2``, …."""
    return f"{kind}axis" if k == 1 else f"{kind}axis{k}"


def grid_cells(
    rows,
    cols,
    left=0.045,
    right=0.9,
    bottom=0.07,
    top=0.93,
    gap_x=0.012,
    gap_y=0.1,
    title=0.06,
):
    """``[(x_domain, y_domain), …]``, row by row from the top, for a ``rows x cols`` grid.

    ``title`` is the fraction of the figure's height kept free above every row for its titles.
    """
    width = (right - left) / cols
    height = (top - bottom) / rows
    cells = []
    for row in range(rows):
        y1 = top - row * height - title
        y0 = top - (row + 1) * height + gap_y * height
        for col in range(cols):
            x0 = left + col * width + gap_x * width
            x1 = left + (col + 1) * width - gap_x * width
            cells.append(([x0, x1], [y0, y1]))
    return cells


def cell_axes(
    layout,
    k,
    domains,
    x_range,
    y_range,
    *,
    x_labels=True,
    y_labels=True,
    x_title=None,
    y_title=None,
):
    """Put the k-th cell's two axes into ``layout`` (a dict): equal aspect, fixed ranges.

    Both ranges are fixed so that every cell of a figure shows the same stretch of the array at the
    same scale: arrows drawn in two cells then have comparable lengths on the screen, and a figure's
    cells are not each zoomed to their own contents. ``constrain="domain"`` keeps the aspect by
    shrinking the cell, never by moving the range.
    """
    x_domain, y_domain = domains
    common = dict(
        showgrid=False,
        zeroline=False,
        mirror=True,
        showline=True,
        linecolor="#9e9e9e",
        ticks="outside",
    )
    layout[layout_key(k, "x")] = dict(
        domain=x_domain,
        anchor=axis_name(k, "y"),
        range=list(x_range),
        constrain="domain",
        showticklabels=x_labels,
        title=dict(text=x_title if x_labels else None, standoff=2),
        nticks=4,
        tickformat=".3f",
        tickangle=0,
        **common,
    )
    layout[layout_key(k, "y")] = dict(
        domain=y_domain,
        anchor=axis_name(k, "x"),
        range=list(y_range),
        scaleanchor=axis_name(k, "x"),
        scaleratio=1,
        constrain="domain",
        showticklabels=y_labels,
        title=dict(text=y_title if y_labels else None, standoff=2),
        nticks=4,
        tickformat=".3f",
        **common,
    )


def cell_title(domains, text, size=13):
    """An annotation centred above a cell."""
    x_domain, y_domain = domains
    return dict(
        text=text,
        x=0.5 * (x_domain[0] + x_domain[1]),
        y=y_domain[1],
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="bottom",
        yshift=4,
        showarrow=False,
        font=dict(size=size),
    )


def padded_range(values, pad):
    """``(lo - pad, hi + pad)`` of the finite ``values``; ``(0, 1)`` when there are none."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    return float(finite.min() - pad), float(finite.max() + pad)
