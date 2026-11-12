"""The blob parameters at every pixel: size, area and duration of the structure behind each average.

``compute`` is ``imaging_methods.decorrelation.blobs``, returned unchanged (R and Z in metres). It reads
the stored conditional average and the pixel's own series, and takes about a minute a shot once the bank
exists. The blob holds every variable the library computes; the app shows the six of
``fusion_ui.views.numbers.BLOB_PARAMETERS``:

- ``nevents``: the events the conditional average rests on;
- ``area`` [m^2]: the area of the contour at zero lag;
- ``lr`` and ``lz`` [m]: the FWHM along the reference's row and column;
- ``taud`` [s] and ``lam``: the duration time and the asymmetry of the two-sided exponential pulse (in
  [0, 1]), fitted to the PSD of the pixel's own record.

**Settings.** ``averages`` selects the bank. ``neighbour_step`` sets the contour level the area is read at;
it is the same knob as ``method_fields``'s ``tracking.neighbour_step`` but a separate parameter, so a
change to one leaves the other's key alone. ``blobs`` holds the settings of the library's ellipse fit and
of the duration time.
"""

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go

from fusion_ui.core import registry
from fusion_ui.plots import pixel_averages
from fusion_ui.plots._pipeline import (
    Averages,
    Blobs,
    Tracking,
    average_at,
    bank_mask,
    live_scalars,
    pipeline,
    record,
)
from fusion_ui.views.figures import (
    DEAD_STYLE,
    FAILED_STYLE,
    axis_name,
    cell_axes,
    cell_title,
    grid_cells,
    message_figure,
    padded_range,
)
from fusion_ui.views.geometry import grid_axes, pitch
from fusion_ui.views.numbers import BLOB_PARAMETERS as TABLE
from fusion_ui.views.overlays import Legend

#: The scalars this writes per live pixel: each name, as the store records it, next to the variable of the
#: blob it is read from. ``taud`` and ``lam`` take the names the ``taud_psd`` spec writes for the same fit,
#: so that the two line up on one multi-shot axis. ``nevents`` is ``method_fields``'s ``number_events``.
SCALARS = {
    "area": "area",
    "lr": "lr",
    "lz": "lz",
    "taud_psd": "taud",
    "lambda_psd": "lam",
}


@dataclass
class BlobParametersParams:
    """
    averages: The 2DCA the blob parameters are read off, as ``pixel_averages`` was computed with. It
        selects the bank this is built on.
    neighbour_step: Pixels between the peak and the neighbours the contour level is read off. The one
        tracking setting the blob parameters read.
    blobs: The settings of the ellipse fit and of the duration time fit.
    """

    averages: Averages = field(default_factory=Averages)
    neighbour_step: int = Tracking().neighbour_step
    blobs: Blobs = field(default_factory=Blobs)


def compute(ds, params, upstream):
    """The blob parameters at every live pixel, from the stored bank and the record.

    ``upstream`` is the ``pixel_averages`` result: the mask is the bank's own, and every live
    reference's average is read back out of it.
    """
    return pipeline.blobs(
        record(ds),
        average_at(upstream),
        params.averages,
        params.neighbour_step,
        params.blobs,
        bank_mask(upstream),
    )


def scalars(result):
    """The :data:`SCALARS` at every live pixel."""
    return live_scalars(result, SCALARS)


#: Maps per row of the view: the six parameters shown make two full rows.
COLUMNS = 3


def render(result, params, target):
    """One map of the array per parameter, each on its own colour scale.

    A dead pixel is a grey square, since it was never computed; a live pixel whose estimate failed is
    a red cross (a contour that never closed, a fit that did not converge). The parameters differ in
    kind and in unit, so the maps share no scale.
    """
    shown = [row for row in TABLE if row[0] in result]
    if not shown:
        return message_figure("This blob has none of the blob parameters.")
    R = np.asarray(result["R"].values, dtype=float)
    Z = np.asarray(result["Z"].values, dtype=float)
    dead = np.asarray(result["dead"].values, dtype=bool)
    x_axis, y_axis = grid_axes(R, Z)
    spacing = pitch(R, Z)
    x_range, y_range = padded_range(R, 0.6 * spacing), padded_range(Z, 0.6 * spacing)
    rows = -(-len(shown) // COLUMNS)
    cells = grid_cells(
        rows, COLUMNS, left=0.045, right=0.99, gap_x=0.3, gap_y=0.25, title=0.05
    )

    layout, traces, annotations, legend = {}, [], [], Legend()
    for k, (name, meaning, unit) in enumerate(shown, start=1):
        cell = cells[k - 1]
        cell_axes(
            layout,
            k,
            cell,
            x_range,
            y_range,
            x_labels=k + COLUMNS > len(shown),
            y_labels=(k - 1) % COLUMNS == 0,
            x_title="R [m]",
            y_title="Z [m]",
        )
        axes = (axis_name(k, "x"), axis_name(k, "y"))
        values = np.asarray(result[name].values, dtype=float)
        x_domain, y_domain = cell
        traces.append(
            go.Heatmap(
                z=np.where(dead, np.nan, values),
                x=x_axis,
                y=y_axis,
                colorscale="Viridis",
                zsmooth=False,
                xgap=1,
                ygap=1,
                colorbar=dict(
                    x=x_domain[1] + 0.006,
                    xanchor="left",
                    y=0.5 * (y_domain[0] + y_domain[1]),
                    len=0.8 * (y_domain[1] - y_domain[0]),
                    thickness=8,
                    tickfont=dict(size=9),
                ),
                hovertemplate="R=%{x:.4f} m, Z=%{y:.4f} m<br>%{z:.4g}<extra>"
                + name
                + "</extra>",
                meta=dict(kind="map", parameter=name),
                xaxis=axes[0],
                yaxis=axes[1],
            )
        )
        for mask, label, style in (
            (dead, "dead pixel (mask)", DEAD_STYLE),
            (~dead & ~np.isfinite(values), "no estimate", FAILED_STYLE),
        ):
            ys, xs = np.nonzero(mask)
            if len(xs):
                traces.append(
                    go.Scatter(
                        x=R[ys, xs],
                        y=Z[ys, xs],
                        mode="markers",
                        marker=dict(**style),
                        name=label,
                        legendgroup=label,
                        showlegend=legend.first(label),
                        hoverinfo="skip",
                        meta=dict(kind=label, parameter=name),
                        xaxis=axes[0],
                        yaxis=axes[1],
                    )
                )
        annotations.append(
            cell_title(cell, f"{name} [{unit}]" if unit else name, size=12)
        )
    layout.update(
        annotations=annotations,
        height=max(420, 230 * rows + 120),
        margin=dict(l=60, r=10, t=30, b=60),
        legend=dict(orientation="h", x=0.0, y=-0.01, yanchor="top"),
        hovermode="closest",
    )
    return go.Figure(data=traces, layout=go.Layout(**layout))


SPEC = registry.register(
    registry.PlotSpec(
        key="blob_parameters",
        label="Blob parameters at every pixel",
        diagnostics=("apd",),
        params=BlobParametersParams,
        render=render,
        compute=compute,
        scalars=scalars,
        requires="pixel_averages",
        upstream_params=pixel_averages.lifted_from,
        preprocessed=True,
        description=(
            "Size, area and duration time of the structure at every live pixel: the FWHM along the"
            " row and column, the area of the zero-lag contour, the duration time from the PSD."
            " Read off pixel_averages; emits five names per live pixel."
        ),
    )
)
