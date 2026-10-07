"""The shot level: every method's velocity field side by side, on one scale.

Seven panels, one per method (2DCA max, 2DCA centroid, 2DCC, 3TDE and 2TDE off the record, 3TDE and
2TDE on the conditional average), drawn either as **arrows** or as **maps** of v_R or v_Z. What
makes the methods comparable by eye is that nothing is rescaled between panels:

- One arrow scale for the whole figure, taken from the speeds of all panels together, and one key
  that states it. A per-panel scale would make every field look equally fast.
- One diverging colour scale for the maps, symmetric about zero.
- Every panel shows the same stretch of the array at the same size, at equal aspect, or the arrow
  directions lie.

A pixel that is not drawn as a number says why: the mask marks it dead (never computed), the
method found no fit at a live pixel, or the view cuts leave it out. The three marks differ, and
hovering any pixel of any panel shows its numbers. Every pixel carries ``[x, y]`` as customdata, so
a click on it in any panel names the pixel the pixel level opens on.

Layout: two rows of four cells. The 2DCA methods fill the first three cells of the top row and the
arrow key the fourth; the four TDE panels fill the bottom row.
"""

import numpy as np
import plotly.graph_objects as go

from fusion_ui.views import arrows
from fusion_ui.views.figures import (
    CUT_STYLE,
    DEAD_STYLE,
    FAILED_STYLE,
    SELECTED_STYLE,
    axis_name,
    cell_axes,
    cell_title,
    grid_cells,
    layout_key,
    message_figure,
    padded_range,
)
from fusion_ui.views.geometry import grid_axes, pitch
from fusion_ui.views.methods import METHODS, Status, panel
from fusion_ui.views.overlays import Legend

ROWS, COLUMNS = 2, 4
#: The grid cell (1-based, row by row) of each of the seven methods, and of the arrow key.
CELL_OF = (1, 2, 3, 5, 6, 7, 8)
KEY_CELL = 4
MODES = {"arrows": "Arrows", "vr": "v_R map", "vz": "v_Z map"}
#: How much of the map colour range the largest speeds may take: the 95th percentile of |v|, since
#: a two-point TDE blows up where a component is small and would otherwise set the range alone.
MAP_PERCENTILE = 95

_MARKS = (
    (Status.FAILED, "no fit", FAILED_STYLE),
    (Status.CUT, "cut by the view", CUT_STYLE),
    (Status.DEAD, "dead pixel (mask)", DEAD_STYLE),
)


def _axes(cell):
    return axis_name(cell, "x"), axis_name(cell, "y")


def _customdata(p, mask):
    """``[x, y, v_R, v_Z, |v|, lags, events, CC]`` of the pixels in ``mask``, as plain nested lists.

    Lists, never a numpy array: Plotly 6 and later serialise an array as a typed array, and the
    browser hands one back in a selection event as an object keyed "0", "1", … where the page expects
    ``[x, y]``. NaN where a panel has no such number.
    """
    ys, xs = np.nonzero(mask)
    nan = np.full(len(xs), np.nan)
    return (
        np.column_stack(
            [
                xs,
                ys,
                p.vr[ys, xs],
                p.vz[ys, xs],
                p.speed[ys, xs],
                p.nlags[ys, xs] if p.nlags is not None else nan,
                p.events[ys, xs],
                p.cc[ys, xs] if p.cc is not None else nan,
            ]
        )
        .astype(float)
        .tolist()
    )


def _hover(p):
    lines = [
        "pixel (x=%{customdata[0]:.0f}, y=%{customdata[1]:.0f})",
        "v_R = %{customdata[2]:.0f} m/s, v_Z = %{customdata[3]:.0f} m/s",
        "|v| = %{customdata[4]:.0f} m/s",
    ]
    if p.nlags is not None:
        lines.append("lags %{customdata[5]:.0f}")
    lines.append(
        "CC %{customdata[7]:.2f}" if p.cc is not None else "events %{customdata[6]:.0f}"
    )
    return "<br>".join(lines) + f"<extra>{p.method.label}</extra>"


def _dots(p, R, Z, axes):
    """A small dot on every drawn pixel: what hover and click land on."""
    ys, xs = np.nonzero(p.ok)
    return go.Scatter(
        x=R[ys, xs],
        y=Z[ys, xs],
        mode="markers",
        marker=dict(size=4, color="#333333"),
        customdata=_customdata(p, p.ok),
        hovertemplate=_hover(p),
        showlegend=False,
        meta=dict(kind="pixels", panel=p.method.key),
        xaxis=axes[0],
        yaxis=axes[1],
    )


def _mark_traces(p, R, Z, axes, legend):
    """The pixels that are not numbers: no fit, cut by the view, dead -- each its own mark."""
    traces = []
    for status, name, style in _MARKS:
        ys, xs = np.nonzero(p.status == status)
        if not len(xs):
            continue
        traces.append(
            go.Scatter(
                x=R[ys, xs],
                y=Z[ys, xs],
                mode="markers",
                marker=dict(**style),
                name=name,
                legendgroup=name,
                showlegend=legend.first(name),
                customdata=np.column_stack([xs, ys]).tolist(),
                text=p.reason[ys, xs],
                hovertemplate="pixel (x=%{customdata[0]}, y=%{customdata[1]})<br>%{text}"
                f"<extra>{p.method.label}</extra>",
                meta=dict(kind=status.name.lower(), panel=p.method.key),
                xaxis=axes[0],
                yaxis=axes[1],
            )
        )
    return traces


def _selected_trace(bundle, R, Z, axes):
    """A ring round the pixel the pixel level is showing."""
    if bundle.pixel is None:
        return []
    x, y = bundle.pixel
    if not (0 <= y < R.shape[0] and 0 <= x < R.shape[1]):
        return []
    return [
        go.Scatter(
            x=[R[y, x]],
            y=[Z[y, x]],
            mode="markers",
            marker=dict(**SELECTED_STYLE),
            hoverinfo="skip",
            showlegend=False,
            meta=dict(kind="selected"),
            xaxis=axes[0],
            yaxis=axes[1],
        )
    ]


def _arrow_traces(p, R, Z, axes, scale):
    ys, xs = np.nonzero(p.ok)
    if not len(xs) or scale is None:
        return []
    q = arrows.quiver(R[ys, xs], Z[ys, xs], p.vr[ys, xs], p.vz[ys, xs], scale)
    return [
        go.Scatter(
            x=q["shaft_x"],
            y=q["shaft_y"],
            mode="lines",
            line=dict(color="#333333", width=1.3),
            hoverinfo="skip",
            showlegend=False,
            meta=dict(kind="shafts", panel=p.method.key),
            xaxis=axes[0],
            yaxis=axes[1],
        ),
        go.Scatter(
            x=q["tip_x"],
            y=q["tip_y"],
            mode="markers",
            marker=dict(
                symbol="arrow",
                size=9,
                angle=q["angle"],
                angleref="up",
                color=p.colour[ys, xs],
                coloraxis="coloraxis2" if p.cc is not None else "coloraxis",
                line=dict(width=0),
            ),
            hoverinfo="skip",
            showlegend=False,
            meta=dict(kind="heads", panel=p.method.key),
            xaxis=axes[0],
            yaxis=axes[1],
        ),
    ]


def _key(traces, x_range, y_range, spacing, scale, key_speed):
    """The arrow key, in the key cell: one arrow of ``key_speed`` at the figure's scale.

    The key cell has the same axis ranges as every panel, so a metre is as long in it as in any of
    them and the key can be held up against the arrows. Returns its annotations.
    """
    if scale is None or key_speed is None:
        return []
    axes = _axes(KEY_CELL)
    x0 = x_range[0] + 0.5 * spacing
    y0 = 0.5 * (y_range[0] + y_range[1])
    q = arrows.quiver([x0], [y0], [key_speed], [0.0], scale)
    traces.append(
        go.Scatter(
            x=q["shaft_x"],
            y=q["shaft_y"],
            mode="lines",
            line=dict(color="#333333", width=1.8),
            hoverinfo="skip",
            showlegend=False,
            meta=dict(kind="key-shaft"),
            xaxis=axes[0],
            yaxis=axes[1],
        )
    )
    traces.append(
        go.Scatter(
            x=q["tip_x"],
            y=q["tip_y"],
            mode="markers",
            marker=dict(
                symbol="arrow",
                size=11,
                angle=q["angle"],
                angleref="up",
                color="#333333",
            ),
            hoverinfo="skip",
            showlegend=False,
            meta=dict(kind="key-head"),
            xaxis=axes[0],
            yaxis=axes[1],
        )
    )
    return [
        dict(
            text=f"<b>{key_speed:g} m/s</b>",
            x=x0,
            y=y0,
            xref=axes[0],
            yref=axes[1],
            xanchor="left",
            yanchor="bottom",
            yshift=6,
            showarrow=False,
        ),
        dict(
            text="the same scale in every panel",
            x=x0,
            y=y0,
            xref=axes[0],
            yref=axes[1],
            xanchor="left",
            yanchor="top",
            yshift=-8,
            showarrow=False,
            font=dict(size=11, color="#666"),
        ),
    ]


def velocity_panels(bundle, mode="arrows", arrow_gain=1.0):
    """The seven panels of ``bundle.fields`` as one figure; ``mode`` is ``"arrows"``, ``"vr"`` or ``"vz"``.

    ``arrow_gain`` makes every arrow that many times longer: the scale is still one for all the
    panels, only moved, for a field whose arrows come out short (a TDE sets the scale and the 2DCA
    arrows are stubs) or tangled. ``fig.layout.meta`` records what the figure was drawn at:
    ``arrow_scale`` (velocity per metre) and ``key_speed`` for arrows, ``colour_limit`` for maps.
    """
    if bundle.fields is None:
        return message_figure("The velocity fields of this shot are not computed.")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}, not {mode!r}")
    if not arrow_gain > 0:
        raise ValueError(f"arrow_gain must be positive, not {arrow_gain!r}")
    R, Z = bundle.grid
    panels = [panel(bundle, method) for method in METHODS]
    x_label, y_label = bundle.labels
    spacing = pitch(R, Z)
    x_range, y_range = padded_range(R, spacing), padded_range(Z, spacing)
    cells = grid_cells(ROWS, COLUMNS, title=0.07, gap_x=0.07)

    layout = {}
    for cell in range(1, ROWS * COLUMNS + 1):
        row, column = divmod(cell - 1, COLUMNS)
        cell_axes(
            layout,
            cell,
            cells[cell - 1],
            x_range,
            y_range,
            x_labels=row == ROWS - 1,
            y_labels=column == 0,
            x_title=x_label,
            y_title=y_label,
        )

    traces, legend = [], Legend()
    meta = dict(mode=mode)
    annotations = [
        cell_title(cells[CELL_OF[i] - 1], p.method.label) for i, p in enumerate(panels)
    ]

    if mode == "arrows":
        speeds = np.concatenate([p.speed[p.ok] for p in panels])
        scale = arrows.arrow_scale(speeds, float(np.nanmax(R) - np.nanmin(R)))
        if scale is not None:
            scale /= arrow_gain
        key_speed = arrows.nice_speed(speeds)
        meta.update(arrow_scale=scale, key_speed=key_speed)
        events = np.concatenate([p.colour[p.ok] for p in panels if p.cc is None])
        events = events[np.isfinite(events)]
        low, high = (
            (float(events.min()), float(events.max())) if events.size else (0.0, 1.0)
        )
        layout["coloraxis"] = dict(
            colorscale="Viridis",
            cmin=low,
            cmax=high if high > low else low + 1.0,
            colorbar=dict(
                title=dict(text="events"),
                x=0.915,
                xanchor="left",
                y=0.76,
                len=0.4,
                thickness=12,
            ),
        )
        layout["coloraxis2"] = dict(
            colorscale="Viridis",
            cmin=bundle.min_cc,
            cmax=1.0,
            colorbar=dict(
                title=dict(text="CC"),
                x=0.915,
                xanchor="left",
                y=0.27,
                len=0.4,
                thickness=12,
            ),
        )
        for i, p in enumerate(panels):
            axes = _axes(CELL_OF[i])
            traces += _arrow_traces(p, R, Z, axes, scale)
            if p.ok.any():
                traces.append(_dots(p, R, Z, axes))
            traces += _mark_traces(p, R, Z, axes, legend)
            traces += _selected_trace(bundle, R, Z, axes)
        annotations += _key(traces, x_range, y_range, spacing, scale, key_speed)
    else:
        component = "vr" if mode == "vr" else "vz"
        label = "v_R" if mode == "vr" else "v_Z"
        shown = [np.where(p.ok, getattr(p, component), np.nan) for p in panels]
        pooled = np.abs(np.concatenate([s[np.isfinite(s)] for s in shown]))
        limit = float(np.percentile(pooled, MAP_PERCENTILE)) if pooled.size else 1.0
        limit = limit if limit > 0 else 1.0
        meta.update(colour_limit=limit)
        layout["coloraxis"] = dict(
            colorscale="RdBu",
            reversescale=True,
            cmin=-limit,
            cmax=limit,
            cmid=0,
            colorbar=dict(
                title=dict(text=f"{label} [m/s]"),
                x=0.915,
                xanchor="left",
                y=0.5,
                len=0.8,
                thickness=14,
            ),
        )
        layout["plot_bgcolor"] = "#f1f1f1"
        x_axis, y_axis = grid_axes(R, Z)
        for i, (p, values) in enumerate(zip(panels, shown)):
            axes = _axes(CELL_OF[i])
            traces.append(
                go.Heatmap(
                    z=values,
                    x=x_axis,
                    y=y_axis,
                    coloraxis="coloraxis",
                    zsmooth=False,
                    xgap=1,
                    ygap=1,
                    hoverinfo="skip",
                    meta=dict(kind="map", panel=p.method.key),
                    xaxis=axes[0],
                    yaxis=axes[1],
                )
            )
            if p.ok.any():
                traces.append(_dots(p, R, Z, axes))
            traces += _mark_traces(p, R, Z, axes, legend)
            traces += _selected_trace(bundle, R, Z, axes)

    # The key cell holds the arrow key or nothing: its axes carry no ticks either way.
    layout[layout_key(KEY_CELL, "x")]["visible"] = False
    layout[layout_key(KEY_CELL, "y")]["visible"] = False
    layout.update(
        annotations=annotations,
        height=780,
        margin=dict(l=55, r=10, t=30, b=60),
        legend=dict(orientation="h", x=0.0, y=-0.01, yanchor="top"),
        meta=meta,
        hovermode="closest",
    )
    return go.Figure(data=traces, layout=go.Layout(**layout))
