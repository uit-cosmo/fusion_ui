"""One large frame of the average, at a lag the page's slider picks, and the reference pixel's trace.

As ``plots/two_dca.py`` draws a conditional average: a frame of ``cond_av``, ``cond_repr`` or
``cross_corr`` with the reference marked, on a colour scale fixed across the sweep (a per-frame
rescale makes a decaying average look as if it never decays). It also draws what the strip draws
over a frame: the contour at the level the average implies and each track's position at this lag.

The lag is view state. It is a time in microseconds, snapped to the bank's lag axis.
"""

import numpy as np
import plotly.graph_objects as go

from fusion_ui.views import pixel
from fusion_ui.views.figures import cell_axes, message_figure, padded_range
from fusion_ui.views.geometry import grid_axes, pitch
from fusion_ui.views.overlays import Legend, contours, frame_trace, overlay_traces
from fusion_ui.views.pixel import FIELDS, FIELD_TRACKS


def lag_index(bundle, lag):
    """Index into the bank's lag axis of ``lag`` (microseconds), the nearest one."""
    return int(np.argmin(np.abs(pixel.lags(bundle) - lag * 1e-6)))


def frame_figure(bundle, name="cond_av", lag=0.0):
    """The frame of field ``name`` at ``lag`` microseconds, with contour, reference and tracks."""
    why = pixel.problem(bundle)
    if why:
        return message_figure(why)
    if name not in FIELDS:
        raise ValueError(f"field must be one of {sorted(FIELDS)}, not {name!r}")
    time = pixel.lags(bundle)
    index = lag_index(bundle, lag)
    x, y = bundle.pixel
    R, Z = bundle.grid
    x_axis, y_axis = grid_axes(R, Z)
    spacing = pitch(R, Z)
    field = pixel.reference_field(bundle, name)
    low, high = pixel.colour_range(field)
    level = pixel.contour_level(bundle, name, field)
    outline = contours(field, [index], level).get(index)
    axes = ("x", "y")
    layout = {}
    cell_axes(
        layout,
        1,
        ([0.08, 0.86], [0.1, 0.9]),
        padded_range(R, 0.6 * spacing),
        padded_range(Z, 0.6 * spacing),
        x_title=bundle.labels[0],
        y_title=bundle.labels[1],
    )
    layout["coloraxis"] = dict(
        colorscale="Viridis",
        cmin=low,
        cmax=high,
        colorbar=dict(
            title=dict(text=name), x=0.88, xanchor="left", y=0.5, len=0.8, thickness=14
        ),
    )
    values = np.asarray(field.values, dtype=float)
    traces = [frame_trace(values[:, :, index], x_axis, y_axis, axes, "coloraxis", name)]
    traces += overlay_traces(
        axes,
        Legend(),
        reference=(R[y, x], Z[y, x]),
        contour=outline,
        tracks=pixel.track_positions(bundle, FIELD_TRACKS[name], index),
        contour_name=(
            f"contour ({level:.2f} × max)" if np.isfinite(level) else "contour"
        ),
    )
    layout.update(
        height=520,
        margin=dict(l=60, r=10, t=50, b=50),
        title=dict(
            text=f"{FIELDS[name]} at τ = {time[index] * 1e6:+.1f} µs, reference pixel (x={x}, y={y})",
            font=dict(size=14),
            x=0.08,
            xanchor="left",
        ),
        legend=dict(orientation="h", x=0.0, y=-0.12, yanchor="top"),
        meta=dict(index=index, lag=float(time[index]), level=level),
    )
    return go.Figure(data=traces, layout=go.Layout(**layout))


def trace_figure(bundle, name="cond_av", lag=0.0):
    """The field at the reference pixel against the lag, with the frame's lag marked."""
    why = pixel.problem(bundle)
    if why:
        return message_figure(why)
    time = pixel.lags(bundle)
    x, y = bundle.pixel
    at_reference = np.asarray(pixel.reference_field(bundle, name).values, dtype=float)[
        y, x
    ]
    figure = go.Figure(
        go.Scatter(
            x=time * 1e6,
            y=at_reference,
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            showlegend=False,
        )
    )
    figure.add_vline(
        x=time[lag_index(bundle, lag)] * 1e6, line_dash="dot", line_color="grey"
    )
    figure.update_layout(
        xaxis_title="lag [µs]",
        yaxis_title=f"{name} at the reference pixel",
        height=300,
        margin=dict(l=60, r=10, t=10, b=50),
    )
    return figure
