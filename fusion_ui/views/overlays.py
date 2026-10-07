"""What is drawn over a frame of a conditional average: the contour, the reference, the tracks.

The contour of ``plotting_scripts.plot_frames_with_contour``, with its semantics kept:

- **One absolute level for every lag**: ``level`` times the maximum of the field over *all* lags and
  pixels, as ``imaging_methods.get_contour_evolution`` thresholds. On a conditional average that
  maximum is the zero-lag peak, so a lag whose peak has decayed below the level has no contour,
  and a contour that shrinks and vanishes is the average decaying, not a drawing fault.
- **The level is read off the average**, not set: ``neighbour_level`` is the mean of the zero-lag
  peak's four neighbours as a fraction of the peak (``twodca_manuscript.contour_level``). It is a
  lookup, not a search. ``method_fields`` stores the level the centroid track used; the other
  fields get the same rule applied to their own zero-lag frame.

The outline is found as ``imaging_methods.get_contour_evolution`` finds it, so what is drawn is
what the stored centroid track was computed from: marching squares on the frame
(``skimage.measure.find_contours``), the contour of highest mass when there are several, and its
points mapped to R and Z by interpolating the grid (``imaging_methods.indexes_to_coordinates``). The
rest of what that function computes (the ellipse fit, the convex hull, the polygon's centroid)
costs about 20 ms a frame and some BLAS threads, and a figure that draws a few outlines has no use
for it. ``tests/test_views_pixel.py`` holds the two to the same points.
"""

import numpy as np
import plotly.graph_objects as go

from fusion_ui.views.figures import TRACK_COLOURS, TRACK_SYMBOLS


def zero_lag_index(time):
    """Index of the lag closest to zero."""
    return int(np.argmin(np.abs(np.asarray(time, dtype=float))))


def neighbour_level(field, time, step=1):
    """The contour level a conditional average implies: its peak's neighbours over its peak, at zero lag.

    ``field`` is one reference's ``(y, x, time)`` array. The four neighbours are ``step`` pixels
    from the peak, fewer when the peak is on the border; their ratios to the peak are averaged.
    ``NaN`` when there is no positive peak.
    """
    field = np.asarray(field, dtype=float)
    frame = field[:, :, zero_lag_index(time)]
    if not np.isfinite(frame).any():
        return float("nan")
    y0, x0 = np.unravel_index(np.nanargmax(frame), frame.shape)
    peak = frame[y0, x0]
    if not peak > 0:
        return float("nan")
    ny, nx = frame.shape
    ratios = [
        frame[y0 + dy, x0 + dx] / peak
        for dy, dx in ((-step, 0), (step, 0), (0, -step), (0, step))
        if 0 <= y0 + dy < ny and 0 <= x0 + dx < nx
    ]
    return float(np.mean(ratios)) if ratios else float("nan")


def outline(frame, R, Z, threshold):
    """``(r, z)`` of the contour of ``frame`` at ``threshold`` as the library picks it, or ``None``.

    ``frame`` is ``(y, x)`` and ``R`` and ``Z`` are its 2-D coordinates. Several contours: the one
    that encloses the most (``compute_contour_mass``), as ``get_contour_evolution`` chooses.
    """
    import imaging_methods as im
    from skimage import measure

    found = measure.find_contours(frame, threshold)
    if not found:
        return None
    contour = (
        found[0]
        if len(found) == 1
        else max(found, key=lambda c: im.compute_contour_mass(c, frame, R, Z))
    )
    r, z = im.indexes_to_coordinates(R, Z, contour)
    keep = np.isfinite(r) & np.isfinite(z)
    return (r[keep], z[keep]) if keep.any() else None


def contours(field, indices, level):
    """``{lag index: (r, z)}``: the contour of each requested lag at ``level`` times the field's maximum.

    ``field`` is one reference's ``(y, x, time)`` DataArray with 2-D ``R`` and ``Z`` coordinates.
    A lag with no contour at that level is left out: one that has decayed below it is not even
    looked at, and one that cannot be contoured (a ``ValueError`` from the marching squares) leaves
    its own panel without an outline and no other.
    """
    peak = float(field.max())
    if not np.isfinite(level) or not np.isfinite(peak) or peak <= 0:
        return {}
    threshold = level * peak
    R, Z = np.asarray(field["R"].values), np.asarray(field["Z"].values)
    values = np.asarray(field.values, dtype=float)
    out = {}
    for index in (int(i) for i in indices):
        frame = values[:, :, index]
        if not np.nanmax(frame) > threshold:
            continue  # decayed below the level: nothing to outline
        try:
            points = outline(frame, R, Z, threshold)
        except (RuntimeError, ValueError):
            continue
        if points is not None:
            out[index] = points
    return out


class Legend:
    """Shows a trace's legend entry the first time its name turns up in a figure, and no more."""

    def __init__(self):
        self._seen = set()

    def first(self, name):
        if name in self._seen:
            return False
        self._seen.add(name)
        return True


def frame_trace(z, x_axis, y_axis, axes, coloraxis, name):
    """The heatmap of one frame on the cell ``axes`` = ``(xaxis, yaxis)``, on a shared colour axis."""
    return go.Heatmap(
        z=z,
        x=x_axis,
        y=y_axis,
        xaxis=axes[0],
        yaxis=axes[1],
        coloraxis=coloraxis,
        zsmooth=False,
        xgap=1,
        ygap=1,
        hovertemplate=f"R %{{x:.4f}} m<br>Z %{{y:.4f}} m<br>{name} %{{z:.3g}}<extra></extra>",
    )


def overlay_traces(
    axes, legend, reference=None, contour=None, tracks=(), contour_name="contour"
):
    """The contour, the reference pixel and the tracks' positions, over one frame.

    ``reference`` is ``(r, z)``; ``contour`` is ``(r, z)`` arrays or ``None``; ``tracks`` is
    ``[(Track, (r, z)), …]`` with NaN where a track is untracked at this lag, which draws nothing.
    ``contour_name`` is what the contour is called in the legend, so a figure with two rows
    contoured at different levels can say so.
    """
    traces = []
    if contour is not None:
        traces.append(
            go.Scatter(
                x=contour[0],
                y=contour[1],
                mode="lines",
                line=dict(color="black", width=1.5, dash="dash"),
                name=contour_name,
                legendgroup=contour_name,
                showlegend=legend.first(contour_name),
                hoverinfo="skip",
                xaxis=axes[0],
                yaxis=axes[1],
            )
        )
    for track, (r, z) in tracks:
        if not (np.isfinite(r) and np.isfinite(z)):
            continue
        traces.append(
            go.Scatter(
                x=[r],
                y=[z],
                mode="markers",
                marker=dict(
                    symbol=TRACK_SYMBOLS[track.key],
                    size=10,
                    color=TRACK_COLOURS[track.key],
                    line=dict(width=1.5, color="white"),
                ),
                name=track.label,
                legendgroup=track.key,
                showlegend=legend.first(track.key),
                hovertemplate=f"{track.label}<br>R %{{x:.4f}} m, Z %{{y:.4f}} m<extra></extra>",
                xaxis=axes[0],
                yaxis=axes[1],
            )
        )
    if reference is not None:
        traces.append(
            go.Scatter(
                x=[reference[0]],
                y=[reference[1]],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=9,
                    color="#d62728",
                    line=dict(width=2, color="white"),
                ),
                name="reference pixel",
                legendgroup="reference",
                showlegend=legend.first("reference"),
                hoverinfo="skip",
                xaxis=axes[0],
                yaxis=axes[1],
            )
        )
    return traces
