"""The tracked structure's path through the conditional average, two ways.

Ported from ``twodca_plots.py:plot_trajectories`` and its helpers
(``get_positions_and_mask``, ``_lsq_fit``, ``_track_pairs``, ``_track_style``,
``_lag_units``). A velocity is one number and looks equally plausible whatever
it is, so this puts two independent trackers on the same axes: the
conditional average's contour **centroid** (``cond_av``, tracked by
``"contouring"``) and the cross-correlation's sub-pixel **maximum**
(``cross_corr``, tracked by ``"max"``, via ``im.compute_maximum_trajectory_da``).
Drawn together, over the lags actually averaged, a track that has latched onto
the array edge or never straightens out is obvious at a glance -- which the
velocity number alone cannot show.

Chained like ``velocity_contour``: ``requires="two_dca"``, ``compute`` never
looks at ``ds``, only at the cached conditional average.

Upstream's ``params_for(variable)`` exists because the mask threshold
(``position_filter.mask_signal_factor``) is a fraction of *each field's own*
maximum over lags, and the cross-correlation sits on a pedestal well above
zero while the conditional average decays towards it -- one number cannot
serve both. ``TrajectoryParams.cross_corr`` is that override, read out in
:func:`_params_for` exactly as upstream's ``params_for`` does; every other
field of ``params`` (including ``contouring``, which the cross-correlation
track does not need since it is tracked by "max") is shared.
"""

import collections
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from imaging_methods.method_parameters import (
    ContouringParams,
    PositionFilterParams,
    TwoDcaParams,
)
from plotly.subplots import make_subplots

from fusion_ui.core import registry
from fusion_ui.plots import two_dca

#: R and Z are stored in centimetres; a slope through them (a velocity) is
#: cm/s, and the group reports m/s -- convert once, here.
CM = 100.0

#: Seconds -> microseconds for the lag axes. Fixed, unlike upstream's
#: ``_lag_units``: this app only ever sees physical (APD/phantom) time, never
#: the dimensionless synthetic ``tau_d`` units that function also handles.
LAG_SCALE = 1e6

#: Each field's own tracker -- not a knob, a fact about the field: the
#: centroid needs a closed contour, which the cross-correlation need not draw
#: at all, and the maximum is the same sub-pixel tracker every field supports.
TRACKS = (("cond_av", "contouring"), ("cross_corr", "max"))

#: One colour per field, so "blue is the conditional average, orange the
#: cross-correlation" holds in every caption -- ported from upstream's
#: TRACK_STYLES (matplotlib hex colours carry over to Plotly unchanged).
TRACK_STYLES = {
    "cond_av": dict(color="#1f77b4", marker="circle"),
    "cross_corr": dict(color="#ff7f0e", marker="square"),
}

METHOD_NAMES = {"contouring": "centroid", "max": "max"}


def _apd_defaults(name):
    return lambda: getattr(im.get_default_apd_method_params(), name)


@dataclass
class CrossCorrParams:
    #: Overrides ``position_filter.mask_signal_factor`` for the cross_corr
    #: track only. The 2D cross-correlation sits on a pedestal well above zero
    #: rather than decaying to it like cond_av does, so the fraction-of-max
    #: mask needs its own floor to bind at all -- upstream's own datasets keep
    #: it at the default 0.75 while lowering cond_av's; see
    #: ``fusion_scripts/twodca_manuscript/datasets/{w7x,cmod}.py``.
    mask_signal_factor: float = 0.75


@dataclass
class TrajectoryParams:
    #: Identifies the upstream conditional average -- see ``upstream_params``.
    two_dca: TwoDcaParams = field(default_factory=_apd_defaults("two_dca"))
    contouring: ContouringParams = field(default_factory=_apd_defaults("contouring"))
    position_filter: PositionFilterParams = field(
        default_factory=_apd_defaults("position_filter")
    )
    #: Deliberately no ``velocity`` block. ``VelocityParams.estimator`` picks
    #: how ``im.get_averaged_velocity_from_position`` reduces a track, and this
    #: spec does not use it -- it always reports the least-squares slope, which
    #: is the line it draws. Carrying the field anyway would put a working-
    #: looking selectbox in the form that changes nothing, and, worse, hash
    #: into ``param_sets``: flipping it would mint a second cache key for a
    #: byte-identical result. Use ``velocity_contour`` for the estimator knob.
    cross_corr: CrossCorrParams = field(default_factory=CrossCorrParams)


def upstream_params(params):
    """The 2DCA parameters, lifted out of this spec's own.

    Reading them out rather than defaulting them is what keeps the two cache
    keys in step: change the threshold here and both the average and these
    tracks get a new entry.
    """
    return two_dca.TwoDcaSpecParams(two_dca=params.two_dca)


def _params_for(params, variable):
    """Upstream's ``params_for``: every field but the cross-correlation reads
    ``params`` unchanged; the cross-correlation gets its own
    ``mask_signal_factor``, everything else shared."""
    if variable != "cross_corr":
        return params
    return SimpleNamespace(
        contouring=params.contouring,
        position_filter=replace(
            params.position_filter,
            mask_signal_factor=params.cross_corr.mask_signal_factor,
        ),
    )


def _positions_and_mask(average, variable, var_params, position_method):
    """Same steps ``im.get_averaged_velocity`` runs internally, kept apart so
    the raw positions can also be stored and drawn. ``valid`` flags lags where
    the raw position exists -- ``smooth_da`` interpolates/extrapolates the
    rest, which can place a track point far outside the field of view."""
    within = None
    if position_method == "contouring":
        contours = im.get_contour_evolution(
            average[variable],
            var_params.contouring.threshold_factor,
            max_displacement_threshold=None,
        )
        raw = contours.centroid
        if var_params.position_filter.require_within_boundaries:
            within = contours.within_boundaries.values
    elif position_method == "max":
        raw = im.compute_maximum_trajectory_da(average, variable)
    else:
        raise NotImplementedError(f"unknown position method {position_method!r}")

    pos, start, end = im.smooth_da(
        raw, var_params.position_filter, return_start_end=True
    )
    if within is not None:
        # smooth_da clips both ends; realign the per-time flags with what it
        # actually returned.
        within = within[start:end]
    mask = im.get_combined_mask(
        average.isel(time=slice(start, end)),
        variable,
        pos,
        var_params.position_filter,
        extra=within,
    )
    return pos, np.asarray(mask, dtype=bool), np.asarray(pos.valid.values, dtype=bool)


def _lsq_fit(times, values):
    """Least-squares slope and intercept of ``values`` against ``times``, plus
    the slope's standard error.

    This is upstream's ``_lsq_fit`` -- the same fit
    ``im.get_averaged_velocity_from_position``'s "lsq" estimator runs -- with
    one addition: upstream only ever draws the fitted line, so it has no need
    of an uncertainty; this spec reports the velocity as a number, so the
    standard error of the slope (from ``np.polyfit``'s covariance) is computed
    here. Non-finite values are dropped first, exactly as upstream's version
    drops them. Two finite points determine a line with no residual to judge
    it by, so the standard error is NaN below three; fewer than two is no fit
    at all.
    """
    finite = np.isfinite(values)
    times, values = times[finite], values[finite]
    if times.size < 2:
        return np.nan, np.nan, np.nan
    if times.size < 3:
        slope, intercept = np.polyfit(times, values, 1)
        return float(slope), float(intercept), np.nan
    (slope, intercept), cov = np.polyfit(times, values, 1, cov=True)
    return float(slope), float(intercept), float(np.sqrt(cov[0, 0]))


def _track(average, variable, method, params):
    """One track's positions, masks, peak amplitude and lsq fit -- everything
    the figure draws for it.

    Returns all-NaN arrays, sized to the full (unclipped) lag axis, rather
    than raising: a track that fails entirely (no contour anywhere, a
    maximum-tracker that finds nothing usable) must not take the other track
    down with it. The peak-amplitude trace is computed before the tracker
    runs, since the coherence panel wants it whether or not a track exists.
    """
    n_full = average.sizes["time"]
    peak = np.asarray(average[variable].max(dim=["x", "y"]).values)
    try:
        var_params = _params_for(params, variable)
        pos, mask, valid = _positions_and_mask(average, variable, var_params, method)
    except Exception:
        lag = np.asarray(average["time"].values)
        return dict(
            lag=lag,
            pos=np.full((n_full, 2), np.nan),
            valid=np.zeros(n_full, dtype=bool),
            mask=np.zeros(n_full, dtype=bool),
            peak=peak,
            slope_r=np.nan,
            intercept_r=np.nan,
            se_r=np.nan,
            slope_z=np.nan,
            intercept_z=np.nan,
            se_z=np.nan,
        )

    lag = np.asarray(pos.time.values)
    r, z = pos.values[:, 0], pos.values[:, 1]
    # Note the mask, not mask & valid: the library fits every masked lag,
    # including positions smooth_da extrapolated (see _lsq_fit's docstring
    # upstream).
    t_fit = lag[mask]
    slope_r, intercept_r, se_r = _lsq_fit(t_fit, r[mask])
    slope_z, intercept_z, se_z = _lsq_fit(t_fit, z[mask])
    return dict(
        lag=lag,
        pos=np.asarray(pos.values),
        valid=valid,
        mask=mask,
        peak=peak,
        slope_r=slope_r,
        intercept_r=intercept_r,
        se_r=se_r,
        slope_z=slope_z,
        intercept_z=intercept_z,
        se_z=se_z,
    )


def compute(ds, params, upstream):
    """Track the contour centroid and the cross-correlation maximum through
    the same conditional average."""
    average = upstream
    refx, refy = int(average["refx"]), int(average["refy"])
    ref_r = float(average.R.isel(x=refx, y=refy))
    ref_z = float(average.Z.isel(x=refx, y=refy))

    tracks = {
        variable: _track(average, variable, method, params)
        for variable, method in TRACKS
    }

    data_vars = {}
    coords = {
        "coord": ["r", "z"],
        "time_full": np.asarray(average["time"].values),
        "R": (("y", "x"), np.asarray(average["R"].values)),
        "Z": (("y", "x"), np.asarray(average["Z"].values)),
    }
    for variable, _method in TRACKS:
        t = tracks[variable]
        lag_dim = f"lag_{variable}"
        # Each track keeps its own lag dimension rather than sharing one: the
        # two fields are smoothed and clipped independently, and a bool array
        # has no netCDF representation, so flags round-trip as int8 (see
        # velocity_contour.py's "tracked").
        coords[lag_dim] = t["lag"]
        data_vars[f"pos_{variable}"] = ((lag_dim, "coord"), t["pos"])
        data_vars[f"valid_{variable}"] = (lag_dim, t["valid"].astype("int8"))
        data_vars[f"mask_{variable}"] = (lag_dim, t["mask"].astype("int8"))
        data_vars[f"peak_{variable}"] = ("time_full", t["peak"])
        data_vars[f"slope_r_{variable}"] = float(t["slope_r"]) / CM
        data_vars[f"se_r_{variable}"] = float(t["se_r"]) / CM
        data_vars[f"intercept_r_{variable}"] = float(t["intercept_r"])
        data_vars[f"slope_z_{variable}"] = float(t["slope_z"]) / CM
        data_vars[f"se_z_{variable}"] = float(t["se_z"]) / CM
        data_vars[f"intercept_z_{variable}"] = float(t["intercept_z"])

    data_vars["refx"] = refx
    data_vars["refy"] = refy
    data_vars["ref_r"] = ref_r / CM
    data_vars["ref_z"] = ref_z / CM
    data_vars["number_events"] = int(average["number_events"])

    return xr.Dataset(data_vars, coords=coords)


# ---------------------------------------------------------------------------


#: Vertical step between two annotations that chose the same corner, in axis
#: domain units. Upstream's ``_text_corner`` staggers by 0.12 per entry; the
#: same value keeps two lines of 10pt text clear of each other here.
CORNER_STEP = 0.12


def _corner(slope):
    """Top corner on the side this track descends towards -- upstream's
    ``_text_corner``, per track rather than averaged over several: close to a
    straight line, a positive slope leaves the top-left free and a negative
    one the top-right."""
    return (0.02, "left") if not np.isfinite(slope) or slope >= 0 else (0.98, "right")


def render(result, params, target):
    """Two tracks, over the frame sequence and the lags they were fitted on --
    the only honest check on a velocity, which otherwise looks equally
    plausible whatever it is.

    Ported from ``plot_trajectories``'s ``2x2`` layout: R-Z track (top left),
    R and Z against lag (top right, bottom right), peak-amplitude coherence
    against lag (bottom left). The three lag panels share one x-axis via
    Plotly's ``matches`` (matplotlib's ``sharex`` has no direct analogue in a
    ``make_subplots`` grid), so zooming one moves the other two with it.
    """
    refx, refy = int(result["refx"]), int(result["refy"])
    ref_r, ref_z = float(result["ref_r"]), float(result["ref_z"])
    events = int(result["number_events"])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "R–Z track",
            "R vs lag",
            "peak amplitude (coherence)",
            "Z vs lag",
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
    )
    fig.update_xaxes(matches="x2", row=2, col=1)
    fig.update_xaxes(matches="x2", row=2, col=2)

    r_grid = np.asarray(result["R"].values).ravel()
    z_grid = np.asarray(result["Z"].values).ravel()
    fig.add_trace(
        go.Scatter(
            x=r_grid,
            y=z_grid,
            mode="markers",
            marker=dict(color="lightgrey", size=4),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[ref_r],
            y=[ref_z],
            mode="markers",
            marker=dict(color="black", size=11, symbol="x"),
            name="ref. pixel",
        ),
        row=1,
        col=1,
    )

    for row, ref, name in ((1, ref_r, "R*"), (2, ref_z, "Z*")):
        fig.add_hline(
            y=ref, line=dict(color="grey", dash="dash", width=1), row=row, col=2
        )
        fig.add_annotation(
            text=name,
            x=0.02,
            y=ref,
            xref=f"x{2 if row == 1 else 4} domain",
            yref=f"y{2 if row == 1 else 4}",
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=9, color="dimgrey"),
        )
    for row, col in ((1, 2), (2, 1), (2, 2)):
        fig.add_vline(x=0, line=dict(color="grey", dash="dot", width=1), row=row, col=col)

    placed = collections.Counter()
    for variable, method in TRACKS:
        style = TRACK_STYLES[variable]
        color, marker = style["color"], style["marker"]
        label = f"{variable} ({METHOD_NAMES[method]})"

        lag = np.asarray(result[f"lag_{variable}"].values) * LAG_SCALE
        pos = np.asarray(result[f"pos_{variable}"].values)
        valid = result[f"valid_{variable}"].values.astype(bool)
        mask = result[f"mask_{variable}"].values.astype(bool)
        seg = mask & valid
        r, z = pos[:, 0], pos[:, 1]

        fig.add_trace(
            go.Scatter(
                x=r[valid],
                y=z[valid],
                mode="markers",
                marker=dict(color=color, symbol=marker, size=6, opacity=0.6),
                name=label,
                legendgroup=variable,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=r[seg],
                y=z[seg],
                mode="lines",
                line=dict(color=color, width=3),
                name=label,
                legendgroup=variable,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        if valid.any():
            i0 = int(np.abs(lag).argmin())
            if valid[i0]:
                fig.add_trace(
                    go.Scatter(
                        x=[r[i0]],
                        y=[z[i0]],
                        mode="markers",
                        marker=dict(
                            color=color,
                            size=16,
                            symbol="star",
                            line=dict(color="white", width=1),
                        ),
                        showlegend=False,
                        legendgroup=variable,
                    ),
                    row=1,
                    col=1,
                )

        slope_r = float(result[f"slope_r_{variable}"])
        se_r = float(result[f"se_r_{variable}"])
        intercept_r = float(result[f"intercept_r_{variable}"])
        slope_z = float(result[f"slope_z_{variable}"])
        se_z = float(result[f"se_z_{variable}"])
        intercept_z = float(result[f"intercept_z_{variable}"])

        panels = (
            (1, 2, r, slope_r, se_r, intercept_r, "v_R"),
            (2, 2, z, slope_z, se_z, intercept_z, "v_Z"),
        )
        for row, col, comp, slope, se, intercept, name in panels:
            fig.add_trace(
                go.Scatter(
                    x=lag[valid],
                    y=comp[valid],
                    mode="markers",
                    marker=dict(color=color, symbol=marker, size=6, opacity=0.6),
                    legendgroup=variable,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=lag[seg],
                    y=comp[seg],
                    mode="lines",
                    line=dict(color=color, width=3),
                    legendgroup=variable,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            if np.isfinite(slope):
                t_fit = lag[seg]
                fitted = slope * CM * (t_fit / LAG_SCALE) + intercept
                fig.add_trace(
                    go.Scatter(
                        x=t_fit,
                        y=fitted,
                        mode="lines",
                        line=dict(color=color, width=1.4, dash="dash"),
                        legendgroup=variable,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
                text = (
                    f"{name} = {slope:.0f} ± {se:.0f} m/s"
                    if np.isfinite(se)
                    else f"{name} = {slope:.0f} m/s"
                )
                x, ha = _corner(slope)
                # Both tracks' slopes usually share a sign, so both pick the
                # same corner; without this counter their labels land on the
                # identical point and neither is readable. Upstream staggers
                # the same way.
                bucket = (row, col, ha)
                depth = placed[bucket]
                placed[bucket] += 1
                fig.add_annotation(
                    text=text,
                    xref=f"x{2 if col == 2 and row == 1 else 4} domain",
                    yref=f"y{2 if col == 2 and row == 1 else 4} domain",
                    x=x,
                    y=0.96 - CORNER_STEP * depth,
                    xanchor=ha,
                    yanchor="top",
                    showarrow=False,
                    font=dict(size=10, color=color),
                    bgcolor="rgba(255,255,255,0.7)",
                )

        peak = np.asarray(result[f"peak_{variable}"].values)
        peak_max = float(np.nanmax(peak)) if np.isfinite(peak).any() else np.nan
        normalised = peak / peak_max if peak_max else peak
        fig.add_trace(
            go.Scatter(
                x=np.asarray(result["time_full"].values) * LAG_SCALE,
                y=normalised,
                mode="lines",
                line=dict(color=color, width=2),
                name=label,
                legendgroup=variable,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # The event count, in the corner the velocity text does not use in either
    # lag panel: the same fit off 30 events and off 300 is not the same
    # measurement, and unlike everything else in the figure it cannot be read
    # off an axis.
    fig.add_annotation(
        text=f"{events} events",
        xref="x2 domain",
        yref="y2 domain",
        x=0.98,
        y=0.04,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=9, color="dimgrey"),
    )

    fig.update_xaxes(title_text="R [cm]", row=1, col=1)
    fig.update_yaxes(title_text="Z [cm]", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="lag [µs]", row=2, col=1)
    fig.update_yaxes(title_text="peak amplitude (norm.)", row=2, col=1)
    fig.update_xaxes(title_text="lag [µs]", row=2, col=2)
    fig.update_yaxes(title_text="R [cm]", row=1, col=2)
    fig.update_yaxes(title_text="Z [cm]", row=2, col=2)

    fig.update_layout(
        height=760,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.08),
        title=(
            f"reference pixel (x={refx}, y={refy}) · {events} events"
        ),
    )
    return fig


def scalars(result):
    """The least-squares slope velocity of each track, in m/s, at the
    reference pixel.

    ``vx_2dca_lsq`` / ``vy_2dca_lsq`` / ``vx_ccf_lsq`` / ``vy_ccf_lsq`` are new
    names -- not in the seeded ``density_scan`` rows, which never ran this
    estimator on the cross-correlation track. Worth confirming before phase 04
    puts them on an axis next to ``vx_c`` / ``vy_c``.
    """
    x, y = int(result["refx"]), int(result["refy"])
    return {
        (x, y, "vx_2dca_lsq"): float(result["slope_r_cond_av"]),
        (x, y, "vy_2dca_lsq"): float(result["slope_z_cond_av"]),
        (x, y, "vx_ccf_lsq"): float(result["slope_r_cross_corr"]),
        (x, y, "vy_ccf_lsq"): float(result["slope_z_cross_corr"]),
    }


SPEC = registry.register(
    registry.PlotSpec(
        key="trajectories",
        label="Tracked trajectories (2DCA)",
        diagnostics=("apd", "phantom"),
        params=TrajectoryParams,
        render=render,
        compute=compute,
        scalars=scalars,
        choices=two_dca.choices,
        requires="two_dca",
        upstream_params=upstream_params,
        description=(
            "Two independent trackers -- the conditional average's contour "
            "centroid and the cross-correlation's maximum -- through the same "
            "average, with their least-squares velocities. Emits "
            "vx_2dca_lsq, vy_2dca_lsq, vx_ccf_lsq and vy_ccf_lsq."
        ),
    )
)
