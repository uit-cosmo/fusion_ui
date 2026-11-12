"""What is drawn over a frame of a conditional average: the contour, the reference, the tracks.

The contour:

- **One absolute level for every lag**: ``level`` times the maximum of the field over all lags and
  pixels, as ``imaging_methods.get_contour_evolution`` thresholds. A lag whose peak has decayed
  below the level has no contour: a contour that shrinks and vanishes is the average decaying.
- **The level is read off the average**: ``neighbour_level`` is the mean of the zero-lag peak's
  four neighbours as a fraction of the peak. ``method_fields`` stores the level the centroid track
  used; the other fields get the same rule applied to their own zero-lag frame.

The outline is found as ``get_contour_evolution`` finds it (marching squares, the contour of
highest mass, its points mapped to R and Z), so what is drawn is what the stored centroid track was
computed from, without the ellipse fit and convex hull that function also pays for.
``tests/test_views_pixel.py`` holds the two to the same points.
"""

import numpy as np
import plotly.graph_objects as go

from fusion_ui.views.figures import TRACK_COLOURS, TRACK_SYMBOLS


def neighbour_level(field, time, step=1):
    """The contour level a conditional average implies: its peak's neighbours over its peak, at zero lag.

    ``field`` is one reference's ``(y, x, time)`` array. The four neighbours are ``step`` pixels
    from the peak, fewer when the peak is on the border; their ratios to the peak are averaged.
    ``NaN`` when there is no positive peak.
    """
    field = np.asarray(field, dtype=float)
    frame = field[:, :, int(np.argmin(np.abs(np.asarray(time, dtype=float))))]
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
    A lag with no contour at that level is left out: one that has decayed below it, or one the
    marching squares cannot contour.
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


def path_traces(axes, path):
    """One track's trajectory over a frame: every lag, the lags its slope rests on, the stored velocity.

    ``path`` is a :class:`fusion_ui.views.pixel.TrackPath`. Three layers, in the track's colour,
    symbol and legend group (its legend entry hides them too):

    - every lag of the bank, a faint line with small markers; a lag with no position leaves a gap;
    - the lags the slope rests on that have a position, thicker;
    - the stored velocity, dashed: a straight path along ``(v_R, v_Z)`` through the mean of the
      fitted points, over the fitted lags. Not drawn when the fit failed.
    """
    track = path.track
    colour, symbol = TRACK_COLOURS[track.key], TRACK_SYMBOLS[track.key]
    common = dict(
        legendgroup=track.key,
        showlegend=False,
        connectgaps=False,
        xaxis=axes[0],
        yaxis=axes[1],
    )
    lag_us = path.lag * 1e6
    where = "τ %{customdata:+.1f} µs<br>R %{x:.4f} m, Z %{y:.4f} m<extra></extra>"
    fitted = path.fitted
    traces = [
        go.Scatter(
            x=path.r,
            y=path.z,
            mode="lines+markers",
            line=dict(color=colour, width=1),
            marker=dict(color=colour, symbol=symbol, size=3),
            opacity=0.55,
            customdata=lag_us,
            name=f"{track.label} trajectory",
            hovertemplate=f"{track.label}<br>{where}",
            **common,
        ),
        go.Scatter(
            x=np.where(fitted, path.r, np.nan),
            y=np.where(fitted, path.z, np.nan),
            mode="lines+markers",
            line=dict(color=colour, width=2),
            marker=dict(color=colour, symbol=symbol, size=5),
            customdata=lag_us,
            name=f"{track.label} trajectory: lags of the fit",
            hovertemplate=f"{track.label}, in the fit<br>{where}",
            **common,
        ),
    ]
    straight = path.straight()
    if straight is not None:
        lag, r, z = straight
        traces.append(
            go.Scatter(
                x=r,
                y=z,
                mode="lines",
                line=dict(color=colour, width=1.5, dash="dash"),
                customdata=lag * 1e6,
                name=f"{track.label} trajectory: stored velocity",
                hovertemplate=(
                    f"{track.label}: stored velocity<br>"
                    f"v_R = {path.vr:.0f} m/s, v_Z = {path.vz:.0f} m/s<br>{where}"
                ),
                **common,
            )
        )
    return traces


def overlay_traces(
    axes,
    legend,
    reference=None,
    contour=None,
    tracks=(),
    contour_name="contour",
    paths=(),
):
    """The contour, the reference pixel and the tracks' positions, over one frame.

    ``reference`` is ``(r, z)``; ``contour`` is ``(r, z)`` arrays or ``None``; ``tracks`` is
    ``[(Track, (r, z)), …]`` with NaN where a track is untracked at this lag, which draws nothing.
    ``paths`` is ``[TrackPath, …]``, the trajectories to draw over this frame, over the contour and
    under the markers.
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
    for path in paths:
        traces += path_traces(axes, path)
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
