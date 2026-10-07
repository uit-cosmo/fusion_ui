"""The lag strip: the conditional average and the cross-correlation at a few lags, with contour and tracks.

One row per field, one column per lag; the reference pixel is marked, the contour is drawn at the
level the average implies, and each track's position is marked at each lag. The reference drawing
is ``figures.fig_lags`` through ``plotting_scripts.plot_frames_with_contour``.

**The lags are view state, and flexible.** Some shots have much faster dynamics, and +-10 us then
shows a structure long gone. They are a span (+- the largest lag) and a number of panels, or lags
typed in. The default span comes from the pixel's own data: the lags the centroid track's slope was
fitted on, widened by a quarter on each side; the deck's lags when the track did not fit. The lags
offered stop at the bank's window: a shot that needs a longer one has its own parameter set.

Each row keeps **one colour scale across its lags**, because a per-frame rescale makes a decaying
average look as if it never decays.
"""

import re
from dataclasses import dataclass
from typing import Optional

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

#: The deck's lags (``twodca_manuscript.datasets.cmod.SPEC.lags``), in frames: the fallback when the
#: centroid track did not fit, so there is nothing of the pixel's own to take a span from.
DECK_FRAMES = 20
#: The fitted lags are widened by this fraction of their width on each side.
FIT_MARGIN = 0.25
DEFAULT_PANELS = 5
MAX_PANELS = 9

#: The rows: a field, and the tracks that are read off it.
ROWS = (("cond_av", ("max", "com")), ("cross_corr", ("2dcc",)))


@dataclass(frozen=True)
class LagChoice:
    """Which lags the strip shows, and where they came from."""

    indices: tuple  # into the bank's lag axis, ascending
    source: str  # "fit", "deck", "span" or "typed"
    span_us: Optional[float] = (
        None  # +- the largest lag, in microseconds; None for typed lags
    )
    dropped: tuple = ()  # typed lags (us) outside the bank's window
    error: Optional[str] = None  # why typed lags could not be used


def parse_lags(text):
    """Lags in microseconds out of free text: ``"-6, -3  0; 3 6"`` -> ``[-6.0, -3.0, 0.0, 3.0, 6.0]``.

    Raises ``ValueError`` naming the part it could not read.
    """
    cleaned = text.replace("−", "-").replace("µs", " ").replace("us", " ")
    values = []
    for token in re.split(r"[,;\s]+", cleaned.strip()):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            raise ValueError(
                f"cannot read {token!r} as a lag in microseconds"
            ) from None
    return values


def _step(time):
    return float(np.median(np.diff(time))) if time.size > 1 else 1.0


def default_span_us(bundle):
    """``(span, source)``: +- the largest lag of the strip when the span is not set, in microseconds.

    The lags the centroid track's slope was fitted on (``fit_com``), widened by a quarter of their
    width on each side; the deck's +-20 frames when that track did not fit. A track fitted when it
    has a velocity and at least two lags behind it: one lag is no slope, and its width is nothing
    to take a span from. Never beyond the bank's window, never narrower than two lags, rounded up
    to a whole lag.
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
    return tuple(sorted({int(np.argmin(np.abs(time - t))) for t in targets}))


def choose_lags(bundle, n=DEFAULT_PANELS, span_us=None, typed=""):
    """The :class:`LagChoice` for these view settings.

    Typed lags win when there are any inside the bank's window; otherwise ``span_us``, and
    otherwise the pixel's own default span. All in microseconds; lags snap to the bank's lag axis.
    """
    time = pixel.lags(bundle)
    window = float(np.max(np.abs(time)))
    error, dropped = None, ()
    if typed and typed.strip():
        try:
            asked = parse_lags(typed)
        except ValueError as problem:
            asked, error = [], str(problem)
        inside = [t for t in asked if abs(t) * 1e-6 <= window + _step(time) / 2]
        dropped = tuple(t for t in asked if t not in inside)
        if inside:
            indices = sorted({int(np.argmin(np.abs(time - t * 1e-6))) for t in inside})
            return LagChoice(tuple(indices), "typed", None, dropped, error)
        error = error or "none of the typed lags is inside the bank's window"
    if span_us is None:
        span_us, source = default_span_us(bundle)
    else:
        span_us, source = min(float(span_us), window * 1e6), "span"
    return LagChoice(_spread(time, span_us, n), source, span_us, dropped, error)


def _height(columns):
    """Figure height in pixels for ``columns`` panels per row, so that they fill a page-wide figure.

    A panel is as wide as the figure allows and a little taller than wide (the array is), so the
    height follows the number of columns: few lags make big panels and a tall figure.
    """
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
    choice = choose_lags(bundle, n, span_us, lags)
    indices = list(choice.indices)
    x, y = bundle.pixel
    R, Z = bundle.grid
    x_axis, y_axis = grid_axes(R, Z)
    spacing = pitch(R, Z)
    x_range, y_range = padded_range(R, 0.6 * spacing), padded_range(Z, 0.6 * spacing)
    x_label, y_label = bundle.labels
    columns = len(indices)
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
        meta=dict(
            lags=[float(time[i]) for i in indices], source=choice.source, levels=levels
        ),
    )
    return go.Figure(data=traces, layout=go.Layout(**layout))
