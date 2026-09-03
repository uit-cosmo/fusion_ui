"""Blob velocity from time-delay estimation on the raw record.

Ported from ``density_scan/utils.py:get_tde_velocities``: at the reference
pixel, find the nearest usable neighbour in each of the four directions
(radial-in, radial-out, poloidal-up, poloidal-down), estimate the time delay
to each with a cross-conditional-average, and solve the resulting system for
a radial and poloidal velocity. "Usable" means the neighbour is not dead and
its cross-correlation with the reference peaks at a lag of at least
``ccf_min_lag`` samples -- a neighbour with no measurable delay is skipped
rather than silently reported as zero.

Unlike every other phase-03 spec, this one is **not chained**: it conditions
on nothing and is estimated once over the whole record rather than from a
2DCA average, so ``compute(ds, params)`` takes two arguments, there is no
``requires``, and it is a sibling of ``spectra.py`` rather than of
``velocity_contour.py``.

``velocity_estimation.EstimationOptions``, ``CAOptions`` and
``NeighbourOptions`` all carry ``@dataclass`` but declare their own
``__init__``, so ``dataclasses.fields()`` on any of them is empty -- exactly
the trap ``fusion_ui.core.params_ui`` guards against (verified directly:
``fields(EstimationOptions()) == ()``, and likewise for the other two). None
of them may ever be a params field; ``TdeVelocityParams`` below carries only
the handful of settings ``density_scan`` actually varies, and the upstream
objects are built fresh inside ``compute``.
"""

from dataclasses import dataclass

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import velocity_estimation as ve
import xarray as xr
from velocity_estimation.two_dim_velocity_estimates import _find_neighbors

from fusion_ui.core import registry

#: R and Z are stored in centimetres, so a velocity off this grid is cm/s. The
#: group reports m/s -- convert once, here, exactly as velocity_contour.py and
#: fwhm_sizes.py do. (This is also, concretely, upstream's own ``/ 100``.)
CM = 100.0

#: The C-Mod APD camera's dead-pixel mask: 9 pixels wide (x) by 10 tall (y).
#: Copied from ``fusion_scripts/density_scan/dead_pixel_mask.py`` rather than
#: imported -- importing ``density_scan`` pulls in its package ``__init__``,
#: which reads ``config`` (and hence the environment) at import time. Rows
#: are listed top-to-bottom in Z as upstream wrote them; upstream reverses the
#: list when building its DataArray, so :func:`_dead_pixel_mask` does the same.
_DEAD_PIXEL_ROWS = [
    [True, True, False, True, False, False, True, True, True],
    [False, False, False, False, False, False, False, True, True],
    [True, False, False, False, True, False, False, False, True],
    [False, False, False, True, False, False, False, False, False],
    [True, False, False, False, False, False, False, False, True],
    [False, False, False, False, False, False, False, False, True],
    [False, False, False, False, False, False, False, True, False],
    [False, True, False, True, False, False, False, False, True],
    [True, True, False, False, False, False, False, False, False],
    [False, False, True, False, False, False, False, False, False],
]


def _dead_pixel_mask():
    """The mask as a ``(y, x)`` boolean array, oriented as upstream builds it."""
    return np.asarray(_DEAD_PIXEL_ROWS[::-1], dtype=bool)


@dataclass
class TdeVelocityParams:
    """
    refx: X index of the reference pixel.
    refy: Y index of the reference pixel.
    ccf_min_lag: Minimum lag, in samples, that the cross-correlation between
        the reference and a candidate neighbour must peak at for that
        neighbour to be used. 0 only requires the neighbour not be dead --
        useful when the expected delay in one direction is close to zero,
        but then not distinguishable from a bad estimate.
    min_separation, max_separation: Allowed pixel separation, in pixel
        widths, when searching outward for a usable neighbour in each of the
        four directions.
    use_3point_method: If True, solve for vx and vy together from one radial
        and one poloidal neighbour (the 2D estimator). If False, estimate
        each independently from its own axis (the naive 1D estimator).
    min_threshold, max_threshold: Conditional-average event thresholds, in
        standard deviations of the signal. max_threshold=None means no upper
        cut -- upstream's ``np.inf``, which does not round-trip through the
        canonical JSON this app hashes, so None is converted inside compute.
    length_of_return: Length of the averaged window around each event, in
        seconds.
    distance: Minimum distance between two peaks used to define separate
        events, in samples.
    interpolate: If True, the time of the conditional-average maximum is
        found by spline interpolation rather than snapped to the nearest
        sample.
    mask_dead_pixels: Apply the hardcoded APD dead-pixel mask -- only takes
        effect when the opened array is actually the 9x10 grid it was built
        for; see the module docstring on shape mismatch.
    """

    refx: int = 6
    refy: int = 6
    ccf_min_lag: int = 1
    min_separation: int = 1
    max_separation: int = 1
    use_3point_method: bool = True
    min_threshold: float = 2.5
    max_threshold: float = None
    length_of_return: float = 1e-4
    distance: int = 0
    interpolate: bool = True
    mask_dead_pixels: bool = True


def choices(ds, path, chosen):
    """Pixel indices, bounded by the array that was actually opened."""
    if path == "refx":
        return tuple(range(ds.sizes["x"]))
    if path == "refy":
        return tuple(range(ds.sizes["y"]))
    return None


def _image_variable(ds):
    """``"frames"`` for imaging data, or the only 3D variable if it is named
    something else -- phantom files have been seen both ways."""
    if "frames" in ds:
        return "frames"
    for name, variable in ds.data_vars.items():
        if variable.ndim == 3:
            return name
    raise KeyError("no 3D image variable in this dataset")


def _neighbours_used(refx, refy, interface, neighbour_options):
    """Which of the four candidate neighbours the estimator actually kept.

    ``estimate_velocities_for_pixel`` runs this same search internally but
    does not hand back which pixels it settled on. Re-deriving it once more,
    cheaply, is what lets the render show honestly what the velocity is (or
    is not) built from, rather than just trusting the number.
    """
    horizontal, vertical = _find_neighbors(refx, refy, interface, neighbour_options)
    return list(horizontal), list(vertical)


def compute(ds, params):
    """The time-delay velocity at one pixel, averaged over the whole record.

    Not chained: unlike every other phase-03 spec, the input here really is
    the raw (or preprocessed) frames, not a conditional average -- this
    estimator conditions on nothing itself and reads the whole record.
    """
    _image_variable(ds)  # raises KeyError if this genuinely is not imaging data
    ny, nx = ds.sizes["y"], ds.sizes["x"]

    mask_grid = _dead_pixel_mask()
    mask_matches_shape = mask_grid.shape == (ny, nx)
    apply_mask = bool(params.mask_dead_pixels) and mask_matches_shape
    mask = None
    if apply_mask:
        mask = xr.DataArray(
            mask_grid,
            dims=["y", "x"],
            coords={"y": np.arange(ny), "x": np.arange(nx)},
        )

    interface = im.MaskedImagingDataInterface(ds, mask)

    eo = ve.EstimationOptions()
    eo.method = ve.TDEMethod.CA
    eo.cache = False
    eo.use_3point_method = bool(params.use_3point_method)
    eo.neighbour_options = ve.NeighbourOptions(
        ccf_min_lag=int(params.ccf_min_lag),
        max_separation=int(params.max_separation),
        min_separation=int(params.min_separation),
    )
    eo.ca_options.length_of_return = float(params.length_of_return)
    eo.ca_options.distance = int(params.distance)
    eo.ca_options.min_threshold = float(params.min_threshold)
    eo.ca_options.max_threshold = (
        np.inf if params.max_threshold is None else float(params.max_threshold)
    )
    eo.ca_options.interpolate = bool(params.interpolate)

    refx, refy = int(params.refx), int(params.refy)
    estimate_failed = False
    try:
        pixel = ve.estimate_velocities_for_pixel(refx, refy, interface, eo)
        vx, vy = pixel.vx / CM, pixel.vy / CM
        confidence, events, is_dead = pixel.confidence, pixel.events, pixel.is_dead
        r_ref, z_ref = float(pixel.r_pos), float(pixel.z_pos)
    except ValueError:
        # Upstream's own "nothing to estimate here" signal, caught exactly as
        # density_scan's get_tde_velocities catches it, and returned as NaN.
        #
        # Narrower than it looks: a pixel with no usable neighbour does NOT
        # raise -- estimate_velocities_for_pixel returns PixelData(vx=nan)
        # for that. The only ValueError in this chain comes from
        # get_2d_velocities_from_time_delays, when the two-equation system is
        # fully degenerate (its common denominator vanishes). Deliberately the
        # only exception caught: a KeyError from a malformed dataset, or
        # anything else, must still fail loudly.
        estimate_failed = True
        vx = vy = float("nan")
        confidence, events, is_dead = float("nan"), 0, False
        r_ref = float(ds["R"].isel(x=refx, y=refy))
        z_ref = float(ds["Z"].isel(x=refx, y=refy))

    horizontal, vertical = _neighbours_used(refx, refy, interface, eo.neighbour_options)
    used_grid = np.zeros((ny, nx), dtype="int8")
    for used_x, used_y in horizontal + vertical:
        if 0 <= used_y < ny and 0 <= used_x < nx:
            used_grid[used_y, used_x] = 1

    # int8, not bool: netCDF has no boolean type, and storing what actually
    # comes back off a round trip is what makes the render trustworthy after
    # a cache hit, not just on the first, in-memory pass.
    dead_grid = (
        np.asarray(mask.values, dtype="int8")
        if mask is not None
        else np.zeros((ny, nx), dtype="int8")
    )

    return xr.Dataset(
        {
            "vx": float(vx),
            "vy": float(vy),
            "confidence": float(confidence) if confidence is not None else float("nan"),
            "events": float(events) if events is not None else 0.0,
            "is_dead": int(bool(is_dead)),
            "estimate_failed": int(estimate_failed),
            "mask_applied": int(mask is not None),
            "mask_shape_mismatch": int(bool(params.mask_dead_pixels) and not mask_matches_shape),
            "neighbours_used": (("y", "x"), used_grid),
            # Kept per axis, not just as a total: both the 3-point and the
            # naive path pair one horizontal neighbour with one vertical one,
            # so an axis with none is the whole reason there is no velocity --
            # and a total that is merely non-zero cannot say which axis it is.
            "n_horizontal": len(horizontal),
            "n_vertical": len(vertical),
            "dead_pixels": (("y", "x"), dead_grid),
            "refx": refx,
            "refy": refy,
            "ref_r": r_ref,
            "ref_z": z_ref,
        },
        coords={
            "R": (("y", "x"), np.asarray(ds["R"].values)),
            "Z": (("y", "x"), np.asarray(ds["Z"].values)),
        },
    )


# ---------------------------------------------------------------------------


def _reason(result):
    """Why there is no arrow, or ``None`` when there is one.

    A velocity is one number and looks equally plausible whatever it is --
    the whole point of drawing on the pixel grid is to make a bad estimate
    visibly bad, and a missing one visibly explained rather than a blank plot.
    """
    if np.isfinite(float(result["vx"])):
        return None
    if bool(result["estimate_failed"]):
        return "no conditional-average event crossed the threshold here or at every neighbour"
    if bool(result["is_dead"]):
        return "the reference pixel is marked dead"
    horizontal, vertical = int(result["n_horizontal"]), int(result["n_vertical"])
    if not horizontal or not vertical:
        # Both the 3-point and the naive path pair one neighbour from each
        # axis, so one empty list means no estimate at all -- not a partial
        # one. Purely radial motion hits this at the default ccf_min_lag=1:
        # the poloidal neighbour's cross-correlation peaks at exactly lag 0
        # and is rejected as unmeasurable.
        missing = "poloidal (Z)" if not vertical else "radial (R)"
        return (
            f"no {missing} neighbour satisfied ccf_min_lag -- every estimate "
            "pairs one neighbour from each axis, so one empty side leaves no "
            "velocity at all. Motion purely along the other axis does exactly "
            "this: the cross-correlation peaks at zero lag and is rejected as "
            "unmeasurable. Lowering ccf_min_lag to 0 accepts it."
        )
    return "no combination of neighbours produced an estimate"


def render(result, params, target):
    """The pixel grid, the neighbours and dead pixels the estimate saw, and
    the velocity as an arrow from the reference pixel."""
    refx, refy = int(result["refx"]), int(result["refy"])
    r_grid = result["R"].values
    z_grid = result["Z"].values
    used = result["neighbours_used"].values.astype(bool)
    dead = result["dead_pixels"].values.astype(bool)
    vx, vy = float(result["vx"]), float(result["vy"])
    speed = float(np.hypot(vx, vy))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=r_grid.ravel(),
            y=z_grid.ravel(),
            mode="markers",
            marker=dict(size=6, color="lightgrey", line=dict(width=1, color="grey")),
            name="pixels",
            hoverinfo="skip",
        )
    )
    if dead.any():
        figure.add_trace(
            go.Scatter(
                x=r_grid[dead],
                y=z_grid[dead],
                mode="markers",
                marker=dict(size=9, color="black", symbol="x"),
                name="dead (masked)",
            )
        )
    if used.any():
        figure.add_trace(
            go.Scatter(
                x=r_grid[used],
                y=z_grid[used],
                mode="markers",
                marker=dict(
                    size=11, color="orange", symbol="diamond",
                    line=dict(width=1, color="black"),
                ),
                name="neighbours used",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[r_grid[refy, refx]],
            y=[z_grid[refy, refx]],
            mode="markers",
            marker=dict(color="white", size=13, symbol="x", line=dict(width=2, color="black")),
            name="reference pixel",
        )
    )

    reason = _reason(result)
    # speed == 0 is reachable (the naive 1-point path with both delays at 0):
    # the estimate exists, but has no direction to draw, and normalising by it
    # would place the arrow head at NaN.
    if reason is None and speed > 0:
        # The arrow's length is fixed (a third of the grid's larger extent)
        # and only its direction carries the estimate -- the title carries
        # the actual numbers. A literal cm/s-scaled displacement would either
        # vanish or run off the plot depending on the shot.
        extent = max(
            float(r_grid.max() - r_grid.min()), float(z_grid.max() - z_grid.min())
        )
        arrow_length = 0.3 * extent
        dx, dy = (vx / speed) * arrow_length, (vy / speed) * arrow_length
        figure.add_annotation(
            x=r_grid[refy, refx] + dx,
            y=z_grid[refy, refx] + dy,
            ax=r_grid[refy, refx],
            ay=z_grid[refy, refx],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowwidth=2,
            arrowcolor="cyan",
            text="",
        )
    else:
        figure.add_annotation(
            text=f"no estimate: {reason}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.08,
            showarrow=False,
            font=dict(color="crimson"),
        )

    if bool(result["mask_shape_mismatch"]):
        figure.add_annotation(
            text=(
                f"dead-pixel mask not applied: mask is "
                f"{_dead_pixel_mask().shape[1]}x{_dead_pixel_mask().shape[0]} "
                f"(x, y), this array is {r_grid.shape[1]}x{r_grid.shape[0]}"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.14,
            showarrow=False,
            font=dict(color="darkorange"),
        )

    title = (
        f"v = ({vx:.0f}, {vy:.0f}) m/s · speed {speed:.0f} m/s"
        if reason is None
        else "no velocity estimate"
    )
    figure.update_layout(
        xaxis_title="R [cm]",
        yaxis_title="Z [cm]",
        height=520,
        margin=dict(l=10, r=10, t=60, b=60),
        legend=dict(orientation="h", y=-0.22),
        title=title,
    )
    return figure


def scalars(result):
    """``vx_tde`` / ``vy_tde``, the two names density_scan reports for this
    estimator, written at the reference pixel."""
    x, y = int(result["refx"]), int(result["refy"])
    return {
        (x, y, "vx_tde"): float(result["vx"]),
        (x, y, "vy_tde"): float(result["vy"]),
    }


SPEC = registry.register(
    registry.PlotSpec(
        key="velocity_tde",
        label="Blob velocity (TDE, whole record)",
        diagnostics=("apd", "phantom"),
        params=TdeVelocityParams,
        render=render,
        compute=compute,
        scalars=scalars,
        choices=choices,
        description=(
            "Time-delay velocity at one pixel from the raw record, not from "
            "a conditional average: nearest usable neighbour in each "
            "direction, averaged over every event in the whole record. "
            "Emits vx_tde and vy_tde."
        ),
    )
)
