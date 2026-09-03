"""Blob velocity from the contour of the conditional average.

Ported from ``density_scan/utils.py:get_contour_parameters``, which is the
estimator behind the ``vx_c`` / ``vy_c`` / ``area_c`` columns in the group's
results: take the conditional average, trace a contour at a fixed fraction of
its peak amplitude in every frame, smooth the track of contour centroids, keep
the stretch of it where the structure is still both near the reference pixel
and bright, and reduce that to one velocity.

The first spec with an upstream. It declares ``requires="two_dca"`` and
``upstream_params``, so the store hands ``compute`` the cached conditional
average instead of it re-running a half-minute analysis that three other plots
in this phase will also want. Note that ``compute`` never looks at ``ds``: the
input to this analysis really is the other spec's output.
"""

from dataclasses import dataclass, field

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from imaging_methods.method_parameters import (
    ContouringParams,
    PositionFilterParams,
    TwoDcaParams,
    VelocityParams,
)

from fusion_ui.core import loader, registry
from fusion_ui.plots import two_dca

#: R and Z are stored in centimetres, so a velocity off this grid is cm/s and
#: an area cm^2. The group reports m/s and m^2, and the seeded density_scan
#: rows are in those units -- convert once, here, rather than in each caller.
CM = 100.0

#: The field the contour is traced on. "cross_corr" is the other tracked field
#: upstream supports; it is a separate estimator, not a knob on this one.
VARIABLE = "cond_av"


def _apd_defaults(name):
    return lambda: getattr(im.get_default_apd_method_params(), name)


@dataclass
class ContourVelocityParams:
    #: Identifies the upstream conditional average -- see ``upstream_params``.
    two_dca: TwoDcaParams = field(default_factory=_apd_defaults("two_dca"))
    contouring: ContouringParams = field(default_factory=_apd_defaults("contouring"))
    position_filter: PositionFilterParams = field(
        default_factory=_apd_defaults("position_filter")
    )
    velocity: VelocityParams = field(default_factory=_apd_defaults("velocity"))


def upstream_params(params):
    """The 2DCA parameters, lifted out of this spec's own.

    Reading them out rather than defaulting them is what keeps the two cache
    keys in step: change the threshold here and both the average and this
    velocity get a new entry.
    """
    return two_dca.TwoDcaSpecParams(two_dca=params.two_dca)


def compute(ds, params, upstream):
    """Track the contour centroid through the conditional average."""
    average = upstream
    contours = im.get_contour_evolution(
        average[VARIABLE],
        params.contouring.threshold_factor,
        max_displacement_threshold=None,
    )

    within = None
    if params.position_filter.require_within_boundaries:
        within = contours["within_boundaries"].values

    track, start, end = im.smooth_da(
        contours.centroid, params.position_filter, return_start_end=True
    )
    if within is not None:
        # smooth_da clips both ends of the track; realign the per-time flags
        # with what it actually returned.
        within = within[start:end]

    mask = im.get_combined_mask(
        average.isel(time=slice(start, end)),
        VARIABLE,
        track,
        params.position_filter,
        extra=within,
    )
    vx, vy = im.get_averaged_velocity_from_position(
        position_da=track, mask=mask, estimator=params.velocity.estimator
    )

    at_peak = dict(time=0, method="nearest")
    size = contours["size"].sel(**at_peak).values
    return xr.Dataset(
        {
            "track": (("track_time", "coord"), np.asarray(track.values)),
            # int8, not bool: netCDF has no boolean type and xarray reads one
            # back as int8 anyway, so storing what comes back avoids a render
            # that works before a round trip and not after.
            "tracked": ("track_time", np.asarray(mask, dtype="int8")),
            "contour": (
                ("point_idx", "coord"),
                np.asarray(contours["contours"].sel(**at_peak).values),
            ),
            "peak_frame": (("y", "x"), np.asarray(average[VARIABLE].sel(**at_peak))),
            "vx": float(vx) / CM,
            "vy": float(vy) / CM,
            "area": float(contours.area.sel(**at_peak)) / CM**2,
            "lx": float(size[0]) / CM,
            "ly": float(size[1]) / CM,
            "theta": float(contours.theta.sel(**at_peak)),
            "refx": int(average["refx"]),
            "refy": int(average["refy"]),
            "number_events": int(average["number_events"]),
        },
        coords={
            "track_time": np.asarray(track.time.values),
            "coord": ["r", "z"],
            "R": (("y", "x"), np.asarray(average["R"].values)),
            "Z": (("y", "x"), np.asarray(average["Z"].values)),
        },
    )


# ---------------------------------------------------------------------------


def _axes(result):
    r_grid, z_grid = loader.pixel_grid(result)
    if r_grid is None:
        shape = result["peak_frame"].shape
        return np.arange(shape[1]), np.arange(shape[0]), "x", "y"
    return r_grid[0, :], z_grid[:, 0], "R [cm]", "Z [cm]"


def render(result, params, target):
    """The track over the frame it was traced on -- the only honest check.

    A velocity is one number and looks equally plausible whatever it is. Drawn
    on top of the contour it came from, an estimate that latched onto the array
    edge or wandered off the structure is obvious at a glance.
    """
    refx, refy = int(result["refx"]), int(result["refy"])
    x_axis, y_axis, x_label, y_label = _axes(result)
    track = result["track"].values
    used = result["tracked"].values.astype(bool)
    vx, vy = float(result["vx"]), float(result["vy"])

    figure = go.Figure(
        go.Heatmap(
            z=result["peak_frame"].values,
            x=x_axis,
            y=y_axis,
            colorscale="Plasma",
            colorbar=dict(title="cond. av."),
        )
    )

    contour = result["contour"].values
    finite = np.isfinite(contour).all(axis=1)
    if finite.any():
        closed = contour[finite]
        closed = np.vstack([closed, closed[:1]])
        figure.add_trace(
            go.Scatter(
                x=closed[:, 0],
                y=closed[:, 1],
                mode="lines",
                line=dict(color="white", width=2),
                name=f"contour at {params.contouring.threshold_factor:g}·max",
            )
        )

    figure.add_trace(
        go.Scatter(
            x=track[:, 0],
            y=track[:, 1],
            mode="lines+markers",
            line=dict(color="lightgrey", width=1),
            marker=dict(size=4, color="lightgrey"),
            name="centroid track",
        )
    )
    if used.any():
        figure.add_trace(
            go.Scatter(
                x=track[used, 0],
                y=track[used, 1],
                mode="markers",
                marker=dict(size=9, color="cyan", line=dict(width=1, color="black")),
                name=f"averaged over ({int(used.sum())} lags)",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[x_axis[refx]],
            y=[y_axis[refy]],
            mode="markers",
            marker=dict(color="white", size=12, symbol="x", line=dict(width=2)),
            name="reference pixel",
        )
    )

    figure.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.15),
        title=(
            f"v = ({vx:.0f}, {vy:.0f}) m/s"
            f" · area {float(result['area']):.3g} m²"
            f" · {int(result['number_events'])} events"
        ),
    )
    return figure


def scalars(result):
    """The three names density_scan reports for this estimator.

    ``lx`` / ``ly`` / ``theta`` are computed and drawn but deliberately not
    written: the seeded ``lx_f`` / ``ly_f`` / ``theta_f`` are the Gaussian fit,
    a different estimator, and putting a contour size under those names would
    silently mix two quantities on one axis in the multi-shot view.
    """
    x, y = int(result["refx"]), int(result["refy"])
    return {
        (x, y, "vx_c"): float(result["vx"]),
        (x, y, "vy_c"): float(result["vy"]),
        (x, y, "area_c"): float(result["area"]),
    }


SPEC = registry.register(
    registry.PlotSpec(
        key="velocity_contour",
        label="Blob velocity (contour tracking)",
        diagnostics=("apd", "phantom"),
        params=ContourVelocityParams,
        render=render,
        compute=compute,
        scalars=scalars,
        choices=two_dca.choices,
        requires="two_dca",
        upstream_params=upstream_params,
        description=(
            "Tracks the contour of the conditional average and reduces it to "
            "one velocity. Emits vx_c, vy_c and area_c."
        ),
    )
)
