"""The 2DCA velocity at every pixel, not just at one reference.

Ported from ``twodca_manuscript/velocity_field.py``'s ``compute`` /
``compute_many`` loop and ``summary``, and the drawing in
``plotting_scripts/twodca_plots.py``'s ``draw_velocity_field_quiver``,
``auto_quiver_scale`` and ``nice_speed``.

``velocity_contour`` quotes one velocity per record, from one reference pixel.
This spec reruns the whole pipeline -- event detection, conditional average,
contour track, mask, slope fit -- with **every** pixel in turn as the
reference, and the field of results is the answer to how representative that
one number is.

Not chained. ``velocity_contour`` and ``fwhm_sizes`` declare
``requires="two_dca"`` because the 2DCA at one reference pixel is expensive and
every derived quantity at *that* pixel wants the same average; here there is
no single reference to share -- the whole point is to run the 2DCA once per
pixel -- so ``compute(ds, params)`` takes the raw dataset directly, the same
two-argument shape as ``spectra.py``.

Cost. ``find_events_and_2dca`` computes the cross-correlation over the whole
grid on every call, so one reference costs O(pixels) and this spec's full
double loop costs O(pixels^2): about 21 s per reference on a real preprocessed
APD shot, times ~90 pixels, is roughly half an hour for the first click. That
is acceptable *because* this is a cached spec behind a Compute button whose
cache never evicts -- but it is exactly why ``compute`` must stay pure (no
progress bar, no Streamlit call of any kind) and why the button's description
says the cost plainly before anyone presses it.

Per-pixel settings are used unchanged, as tuned at the reference pixel -- that
is the point of the figure, so a pixel where the tracking fails comes back NaN
rather than being retuned with settings of its own.
"""

from dataclasses import dataclass, field

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from imaging_methods.method_parameters import (
    ContouringParams,
    PositionFilterParams,
    TwoDcaParams,
    VelocityParams,
)
from scipy.spatial import QhullError

#: Where the "Minimum lags" slider starts. Upstream's MIN_LAGS.
DEFAULT_MIN_LAGS = 5

from fusion_ui.core import registry

#: R and Z are stored in centimetres; the tracked position (and so the raw
#: velocity coming out of get_averaged_velocity_from_position) is in the same
#: units, cm/s. The group reports m/s -- convert once, here.
CM = 100.0

_APD_TWO_DCA = im.get_default_apd_method_params().two_dca


def _apd_defaults(name):
    return lambda: getattr(im.get_default_apd_method_params(), name)


@dataclass
class VelocityFieldParams:
    """
    threshold: Threshold value for event detection, in standard deviations.
    window: Size of the window extracted around each event peak, in samples.
    check_max: Radius over which the reference pixel is checked to be the
        spatial maximum at peak time. 0 skips the check.
    single_counting: If True, keeps events at least one window apart.

    Deliberately no ``min_lags``: it cuts what the figure draws, never what is
    computed, so it is view state and lives in ``st.session_state`` keyed off
    ``Target.key`` (see ``render``). Hashing it would mint a second cache key
    -- and a second half-hour of compute -- for a byte-identical field.
    """

    # Only the 2DCA settings this analysis actually honours. TwoDcaParams
    # itself also carries refx/refy, and those are overwritten at every pixel
    # -- if they were part of this dataclass, two runs differing only in a
    # refx/refy nobody reads would hash to two different cache entries for a
    # byte-identical result. TwoDcaParams is built fresh inside compute().
    threshold: float = _APD_TWO_DCA.threshold
    window: int = _APD_TWO_DCA.window
    check_max: int = _APD_TWO_DCA.check_max
    single_counting: bool = _APD_TWO_DCA.single_counting
    contouring: ContouringParams = field(default_factory=_apd_defaults("contouring"))
    position_filter: PositionFilterParams = field(
        default_factory=_apd_defaults("position_filter")
    )
    velocity: VelocityParams = field(default_factory=_apd_defaults("velocity"))


def compute(ds, params):
    """Velocity field over the whole view, as a Dataset with dims (y, x).

    A plain double loop over the pixels, each iteration repeating exactly the
    contour-tracking reduction ``velocity_contour.compute`` performs at its one
    reference: ``find_events_and_2dca``, then ``get_contour_evolution``,
    ``smooth_da``, ``get_combined_mask`` and
    ``get_averaged_velocity_from_position``. Nothing here invents a second
    version of that reduction -- the field is only informative if it is the
    same estimator run 81 (or however many) times, not a lookalike.

    Every failure mode lands as NaN for that pixel and lets the loop continue,
    with the reasoning kept next to each case below:

    - no events survive the threshold/check_max/window cuts at this
      reference -- ``find_events_and_2dca`` returns no windows, which is what
      the single-pixel ``two_dca`` spec turns into a raised error; here it is
      one NaN pixel among many, not a failed run;
    - the tracking fails to produce anything to fit -- a contour that never
      closes, or closes but never spends a frame near the reference pixel
      while bright, leaves ``get_combined_mask``'s mask empty, and
      ``get_averaged_velocity_from_position`` already returns (nan, nan) for
      an empty mask with no exception at all;
    - a genuine exception out of the tracking: a contour whose points are
      (near-)collinear makes ``scipy.spatial.ConvexHull`` raise
      ``QhullError`` inside ``get_contour_evolution``'s convexity
      calculation -- a ``RuntimeError`` subclass that function's own
      ``except (ValueError, ZeroDivisionError)`` does not catch. Caught
      narrowly to that one exception type here, so a real bug elsewhere in
      the reduction still surfaces as a failed run rather than a silent NaN.

    ``nlags`` is stored for every pixel and nothing here cuts on it: the
    cut belongs to whoever reads the field, and is applied in ``render``.
    Summary numbers are not stored either, for the same reason -- they depend
    on that cut, and are a few numpy calls over arrays already in hand.
    """
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    vr_cms = np.full((ny, nx), np.nan)
    vz_cms = np.full((ny, nx), np.nan)
    nevents = np.zeros((ny, nx), dtype=int)
    nlags = np.zeros((ny, nx), dtype=int)

    for y in range(ny):
        for x in range(nx):
            two_dca_params = TwoDcaParams(
                refx=x,
                refy=y,
                threshold=params.threshold,
                window=params.window,
                check_max=params.check_max,
                single_counting=params.single_counting,
            )
            events, average = im.find_events_and_2dca(
                ds, two_dca_params, verbose=False
            )
            if not events or not average.data_vars:
                continue
            nevents[y, x] = int(average["number_events"])

            try:
                contours = im.get_contour_evolution(
                    average["cond_av"],
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
                    within = within[start:end]
                mask = im.get_combined_mask(
                    average.isel(time=slice(start, end)),
                    "cond_av",
                    track,
                    params.position_filter,
                    extra=within,
                )
                nlags[y, x] = int(np.sum(mask))
                with np.errstate(invalid="ignore"):
                    v, w = im.get_averaged_velocity_from_position(
                        position_da=track,
                        mask=mask,
                        estimator=params.velocity.estimator,
                    )
                vr_cms[y, x], vz_cms[y, x] = v, w
            except QhullError:
                continue

    vr = vr_cms / CM
    vz = vz_cms / CM

    return xr.Dataset(
        {
            "vr": (["y", "x"], vr),
            "vz": (["y", "x"], vz),
            "nevents": (["y", "x"], nevents),
            "nlags": (["y", "x"], nlags),
            "npixels": int(vr.size),
        },
        coords={
            "R": (("y", "x"), np.asarray(ds["R"].values)),
            "Z": (("y", "x"), np.asarray(ds["Z"].values)),
        },
    )


# ---------------------------------------------------------------------------
# The quiver. plotly has no quiver primitive of its own -- figure_factory's
# create_quiver cannot colour arrows individually, which here is the thing
# that matters (an arrow's event count is how much to trust it) -- so the
# arrows are drawn as plain line segments, one go.Scatter with a None after
# every pair of points, plus a marker layer at the tips carrying the colour.
# ---------------------------------------------------------------------------

#: Length of a representative arrow, as a fraction of the R extent. Ported
#: from twodca_plots.ARROW_FRACTION.
ARROW_FRACTION = 0.18


def auto_quiver_scale(speeds, R, fraction=ARROW_FRACTION):
    """Arrow scale (velocity per unit length) sizing a field's arrows sensibly.

    Ported from ``twodca_plots.auto_quiver_scale``. A quiver arrow is drawn
    ``v / scale`` long, so a field of a few hundred m/s and one of a few km/s
    need scales an order of magnitude apart to be equally readable -- and so
    do two records of the same shot, whose flows can differ by a factor of
    several. Taking the scale from the field itself, rather than fixing one,
    is what keeps a slow record from being drawn as a field of stubs. The key
    drawn alongside it is what makes the arrows quantitative, so a per-figure
    scale costs nothing as long as every figure carries one.

    ``speeds`` and ``R`` must be in the same length unit (both cm here) --
    mixing units would size the arrows against the wrong fraction of the view.
    """
    speeds = np.asarray(speeds, dtype=float)
    speeds = speeds[np.isfinite(speeds) & (speeds > 0)]
    width = float(np.nanmax(R) - np.nanmin(R))
    if speeds.size == 0 or not np.isfinite(width) or width <= 0:
        return None  # nothing to scale off
    return float(np.percentile(speeds, 90)) / (fraction * width)


def nice_speed(speeds):
    """A round number near the arrows' upper end, for the key: 1, 2 or 5 x 10^k.

    Ported from ``twodca_plots.nice_speed``.
    """
    speeds = np.asarray(speeds, dtype=float)
    speeds = speeds[np.isfinite(speeds) & (speeds > 0)]
    if speeds.size == 0:
        return None
    v = float(np.percentile(speeds, 90))
    exponent = np.floor(np.log10(v))
    step = min((1, 2, 5, 10), key=lambda s: abs(s - v / 10**exponent))
    return int(step * 10**exponent)


def render(result, params, target):
    """Read the lag cut off a slider, then draw the field at it.

    Returns ``None``: a slider means this has to draw itself. The figure is
    built by :func:`figure`, which stays a pure function of the result and the
    cut so it can be tested without a Streamlit runtime.
    """
    nlags = np.asarray(result["nlags"].values)
    lag_key = f"velocity_field.min_lags.{target.key}"
    highest = int(np.nanmax(nlags)) if np.isfinite(nlags).any() else 2
    st.plotly_chart(
        figure(
            result,
            st.slider(
                "Minimum lags",
                1,
                max(2, highest),
                key=lag_key,
                value=st.session_state.get(lag_key, DEFAULT_MIN_LAGS),
                help=(
                    "Fewest lags a pixel's slope may rest on to be drawn. A "
                    "slope through two or three lags is a secant, not a fit, "
                    "and those pixels carry the field's most extreme "
                    "velocities. The cut is on the fit, not on the answer -- "
                    "every pixel stays in the stored result, so moving this "
                    "never recomputes anything."
                ),
            ),
        ),
        use_container_width=True,
    )
    return None


def figure(result, min_lags=DEFAULT_MIN_LAGS):
    """The field as a quiver: one arrow per pixel that fitted, coloured by
    how many events it rests on.

    Ported from ``twodca_plots.draw_velocity_field_quiver``. Unlike every
    other plot in this app, **no reference pixel is marked**: the field is
    the whole pipeline rerun with each pixel in turn as the reference, so
    every arrow *is* a reference pixel, and singling one out with the usual
    white x would claim a distinction the figure does not have.

    Pixels with no usable fit -- no events, no contour, or too few lags --
    are still marked, with a grey x rather than left blank: a gap in a
    quiver reads as a zero velocity, and that is not what a NaN means here.

    ``min_lags`` is a slider here rather than a parameter. It changes only
    which pixels are drawn, so making it a parameter would put half an hour
    of recompute behind a control that cannot change a single stored number.
    Draws into Streamlit and returns ``None`` for that reason, as
    ``two_dca.render`` does for its lag slider.

    **A pixel passing the cut is not therefore trustworthy.** On the synthetic
    fixture the two columns either side of the array edge return velocities of
    the wrong sign resting on 40 lags -- well past any cut worth setting -- and
    they draw as ordinary, well-supported arrows. The cut removes fits that are
    not fits; it cannot remove a confident wrong answer, which is exactly what
    a whole field is drawn to expose.
    """
    R = np.asarray(result["R"].values)
    Z = np.asarray(result["Z"].values)
    vr = np.asarray(result["vr"].values)  # m/s
    vz = np.asarray(result["vz"].values)
    nevents = np.asarray(result["nevents"].values, dtype=float)
    nlags = np.asarray(result["nlags"].values)

    # The cut is on the fit, not on the answer:
    # a slope through two or three lags is not a fit, so compute() keeps every
    # pixel and this is where the cut is actually made.
    ok = np.isfinite(vr) & np.isfinite(vz) & (nlags >= min_lags)

    fig = go.Figure()

    if (~ok).any():
        fig.add_trace(
            go.Scatter(
                x=R[~ok],
                y=Z[~ok],
                mode="markers",
                marker=dict(symbol="x", size=7, color="grey", line=dict(width=1)),
                name="no fit",
                hoverinfo="skip",
            )
        )

    if ok.any():
        # Both derived from this field's own arrows: a fixed scale would make
        # a slow pixel a field of stubs and a fast one an unreadable tangle.
        speeds_cm_s = np.hypot(vr[ok], vz[ok]) * CM  # cm/s, matching R/Z's units
        scale = auto_quiver_scale(speeds_cm_s, R)
        key_speed = nice_speed(np.hypot(vr[ok], vz[ok]))  # m/s, what the key shows

        if scale is not None:
            r0, z0 = R[ok], Z[ok]
            r1 = r0 + vr[ok] * CM / scale
            z1 = z0 + vz[ok] * CM / scale
            xs, ys = [], []
            for a, b, c, d in zip(r0, z0, r1, z1):
                xs += [a, c, None]
                ys += [b, d, None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="black", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=r1,
                    y=z1,
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=nevents[ok],
                        colorscale="Viridis",
                        colorbar=dict(title="events"),
                    ),
                    name="arrow tip",
                )
            )
            if key_speed is not None:
                # Arrow key: a short reference segment drawn in the same data
                # units as the field, so it is directly comparable to the
                # arrows rather than a caption asking to be taken on trust.
                height = float(np.nanmax(Z) - np.nanmin(Z))
                key_x0 = float(np.nanmin(R))
                key_y0 = float(np.nanmax(Z)) + 0.08 * height
                key_x1 = key_x0 + key_speed * CM / scale
                fig.add_trace(
                    go.Scatter(
                        x=[key_x0, key_x1],
                        y=[key_y0, key_y0],
                        mode="lines",
                        line=dict(color="black", width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_annotation(
                    x=(key_x0 + key_x1) / 2,
                    y=key_y0,
                    yshift=14,
                    text=f"{key_speed:g} m/s",
                    showarrow=False,
                    font=dict(size=11),
                )

    npixels = int(result["npixels"])
    nfitted = int(ok.sum())
    median_vr = float(np.median(vr[ok])) if nfitted else float("nan")

    fig.update_layout(
        xaxis_title="R [cm]",
        yaxis_title="Z [cm]",
        height=560,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.12),
        title=(
            f"{nfitted}/{npixels} pixels fitted (≥{min_lags} lags) "
            f"· median v_R = {median_vr:.0f} m/s"
        ),
    )
    # Equal aspect, or the arrow directions would lie.
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def scalars(result):
    """Every pixel, not just one -- the first spec to write a full field.

    NaN pixels are written through rather than skipped: ``store.write_scalars``
    already turns a NaN into a SQL NULL rather than dropping the row (see its
    own comment -- "keeps 'the fit did not converge' readable in the table"),
    which is exactly the distinction that matters here between "this pixel was
    tried and failed" and "this pixel was never computed". Skipping it would
    throw that distinction away for no benefit, since the store handles it.
    """
    vr = np.asarray(result["vr"].values)
    vz = np.asarray(result["vz"].values)
    nevents = np.asarray(result["nevents"].values)
    nlags = np.asarray(result["nlags"].values)

    out = {}
    for y in range(vr.shape[0]):
        for x in range(vr.shape[1]):
            out[(x, y, "vx_field")] = float(vr[y, x])
            out[(x, y, "vy_field")] = float(vz[y, x])
            out[(x, y, "number_events_field")] = float(nevents[y, x])
            out[(x, y, "nlags_field")] = float(nlags[y, x])
    return out


SPEC = registry.register(
    registry.PlotSpec(
        key="velocity_field",
        label="Blob velocity field (2DCA at every pixel)",
        diagnostics=("apd", "phantom"),
        params=VelocityFieldParams,
        render=render,
        compute=compute,
        scalars=scalars,
        description=(
            "Reruns the whole 2DCA + contour-tracking chain with every pixel "
            "in turn as the reference. Expensive: about 21 s per pixel on a "
            "real preprocessed shot, so roughly half an hour for ~90 pixels "
            "on the first click -- cached after that. Emits vx_field, "
            "vy_field, number_events_field and nlags_field at every pixel."
        ),
    )
)
