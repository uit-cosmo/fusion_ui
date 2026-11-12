"""The lag strip: the conditional average and the cross-correlation at a few lags, with contour and tracks.

One row per field, one column per lag; the reference pixel is marked, the contour is drawn at the
level the average implies, and each track's position is marked at each lag. Each row keeps one
colour scale across its lags: a per-frame rescale makes a decaying average look as if it never
decays.

The lags are view state: a span (± the largest lag) and a number of panels, or lags typed in. The
default span is the lags the centroid track's slope was fitted on, widened by a quarter on each
side, and the deck's lags when the track did not fit. They stop at the bank's window.

The panel nearest τ = 0 also carries each track's whole trajectory in R and Z
(``overlays.path_traces``), on its own field's row. They are read off ``method_fields``, and a
strip without it is drawn without them.
"""

import re

import numpy as np
import plotly.graph_objects as go

from fusion_ui.views import pixel
from fusion_ui.views.figures import (
    axis_name,
    cell_axes,
    cell_title,
    grid_cells,
    message_figure,
    padded_range,
)
from fusion_ui.views.geometry import grid_axes, pitch
from fusion_ui.views.overlays import Legend, contours, frame_trace, overlay_traces

#: The lags of the group's deck, in frames: the span when the centroid track did not fit, so the
#: pixel has no span of its own.
DECK_FRAMES = 20
#: The fitted lags are widened by this fraction of their width on each side.
FIT_MARGIN = 0.25
DEFAULT_PANELS = 5
MAX_PANELS = 9

#: The rows: the fields that have tracks, and the tracks read off each.
ROWS = tuple((name, keys) for name, keys in pixel.FIELD_TRACKS.items() if keys)


def parse_lags(text):
    """Lags in microseconds out of free text: ``"-6, -3  0; 3 6"`` -> ``[-6.0, -3.0, 0.0, 3.0, 6.0]``.

    Raises ``ValueError`` on a part it cannot read.
    """
    cleaned = text.replace("−", "-").replace("µs", " ").replace("us", " ")
    return [float(token) for token in re.split(r"[,;\s]+", cleaned.strip()) if token]


def _step(time):
    return float(np.median(np.diff(time))) if time.size > 1 else 1.0


def default_span_us(bundle):
    """``(span, source)``: +- the largest lag of the strip when the span is not set, in microseconds.

    The lags the centroid track's slope was fitted on (``fit_com``), widened by a quarter of their
    width on each side (source ``"fit"``); the deck's ``DECK_FRAMES`` lags when that track has no
    velocity or fewer than two fitted lags (``"deck"``). Never beyond the bank's window, never
    narrower than two lags, rounded up to a whole lag.
    """
    time = pixel.lags(bundle)
    step = _step(time)
    window = float(np.max(np.abs(time)))
    x, y = bundle.pixel
    span, source = DECK_FRAMES * step, "deck"
    fields = bundle.fields
    if fields is not None and "fit_com" in fields:
        fitted = np.asarray(fields["fit_com"].values)[y, x].astype(bool)
        velocity = float(fields["vr_com"].values[y, x]) if "vr_com" in fields else 0.0
        if fitted.sum() >= 2 and np.isfinite(velocity):
            low, high = float(time[fitted].min()), float(time[fitted].max())
            margin = FIT_MARGIN * (high - low)
            span = max(abs(low - margin), abs(high + margin), 2 * step)
            span, source = float(np.ceil(span / step - 1e-9) * step), "fit"
    return round(min(span, window) * 1e6, 6), source


def _spread(time, span_us, n):
    """``n`` lags from ``-span`` to ``+span`` microseconds, snapped to the lag axis, without repeats."""
    targets = np.linspace(-span_us, span_us, int(n)) * 1e-6
    return sorted({int(np.argmin(np.abs(time - t))) for t in targets})


def choose_lags(bundle, n=DEFAULT_PANELS, span_us=None, typed=""):
    """``(indices, source)``: the lags the strip shows, ascending indices into the bank's lag axis.

    Typed lags win when any can be read and is inside the bank's window (``"typed"``); otherwise
    ``span_us`` (``"span"``), and otherwise the pixel's own default span. All in microseconds; lags
    snap to the bank's lag axis.
    """
    time = pixel.lags(bundle)
    window = float(np.max(np.abs(time)))
    try:
        asked = parse_lags(typed or "")
    except ValueError:
        asked = []
    inside = [t for t in asked if abs(t) * 1e-6 <= window + _step(time) / 2]
    if inside:
        indices = sorted({int(np.argmin(np.abs(time - t * 1e-6))) for t in inside})
        return indices, "typed"
    if span_us is None:
        span_us, source = default_span_us(bundle)
    else:
        span_us, source = min(float(span_us), window * 1e6), "span"
    return _spread(time, span_us, n), source


def zero_lag_column(time):
    """Index of the lag nearest zero among ``time`` (ascending); the earlier of two equally near.

    "Equally near" is to a part in 10^9, so that rounding in a symmetric lag axis does not decide
    which panel carries the trajectories.
    """
    distance = np.abs(np.asarray(time, dtype=float))
    return int(
        np.flatnonzero(np.isclose(distance, distance.min(), rtol=1e-9, atol=0))[0]
    )


def _height(columns):
    """Figure height in pixels for ``columns`` panels per row: few lags make big panels and a tall figure."""
    panel = 0.85 * 1000 / columns
    return int(min(900, max(430, 2 * (1.1 * panel + 75) + 100)))


def lag_strip(bundle, n=DEFAULT_PANELS, span_us=None, lags=""):
    """The strip at the pixel in view: ``n`` lags over +-``span_us``, or the lags typed in ``lags``.

    ``span_us`` and ``lags`` are in microseconds; ``lags`` is free text (see :func:`parse_lags`).
    """
    why = pixel.problem(bundle)
    if why:
        return message_figure(why)
    time = pixel.lags(bundle)
    indices, source = choose_lags(bundle, n, span_us, lags)
    x, y = bundle.pixel
    R, Z = bundle.grid
    x_axis, y_axis = grid_axes(R, Z)
    spacing = pitch(R, Z)
    x_range, y_range = padded_range(R, 0.6 * spacing), padded_range(Z, 0.6 * spacing)
    x_label, y_label = bundle.labels
    columns = len(indices)
    # The column the trajectories go on: the lag nearest zero.
    at_zero = zero_lag_column(time[indices])
    cells = grid_cells(
        len(ROWS), columns, right=0.9, gap_x=0.04, title=0.09, bottom=0.08, gap_y=0.16
    )

    layout = {}
    traces, annotations, legend = [], [], Legend()
    levels = {}
    for row, (name, track_keys) in enumerate(ROWS):
        field = pixel.reference_field(bundle, name)
        low, high = pixel.colour_range(field)
        level = levels[name] = pixel.contour_level(bundle, name, field)
        outline = contours(field, indices, level)
        coloraxis = "coloraxis" if row == 0 else "coloraxis2"
        layout[coloraxis] = dict(
            colorscale="Viridis",
            cmin=low,
            cmax=high,
            colorbar=dict(
                title=dict(text=name),
                x=0.925,
                xanchor="left",
                y=0.76 - 0.5 * row,
                len=0.4,
                thickness=12,
            ),
        )
        contour_name = (
            f"{name} contour ({level:.2f} × max)" if np.isfinite(level) else "contour"
        )
        values = np.asarray(field.values, dtype=float)
        paths = pixel.track_paths(bundle, track_keys)
        for column, index in enumerate(indices):
            cell = row * columns + column + 1
            axes = (axis_name(cell, "x"), axis_name(cell, "y"))
            cell_axes(
                layout,
                cell,
                cells[cell - 1],
                x_range,
                y_range,
                x_labels=row == len(ROWS) - 1,
                y_labels=column == 0,
                x_title=x_label,
                y_title=f"{name}: {y_label}",
            )
            traces.append(
                frame_trace(values[:, :, index], x_axis, y_axis, axes, coloraxis, name)
            )
            traces += overlay_traces(
                axes,
                legend,
                reference=(R[y, x], Z[y, x]),
                contour=outline.get(index),
                tracks=pixel.track_positions(bundle, track_keys, index),
                paths=paths if column == at_zero else (),
                contour_name=contour_name,
            )
            if row == 0:
                annotations.append(
                    cell_title(cells[cell - 1], f"τ = {time[index] * 1e6:+.1f} µs")
                )
    layout.update(
        annotations=annotations,
        height=_height(columns),
        margin=dict(l=70, r=10, t=30, b=70),
        legend=dict(orientation="h", x=0.0, y=-0.01, yanchor="top"),
        meta=dict(lags=[float(time[i]) for i in indices], source=source, levels=levels),
    )
    return go.Figure(data=traces, layout=go.Layout(**layout))
