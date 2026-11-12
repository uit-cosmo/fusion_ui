"""Blob velocity from time-delay estimation on the record, the estimator of
the group's PPCF 2026 paper.

At the reference pixel, find the nearest usable neighbour in each of the four
directions (radial in and out, poloidal up and down), estimate the time delay
to each with a cross-conditional average, and solve for a radial and a
poloidal velocity. A neighbour is usable when it is not dead and its
cross-correlation with the reference peaks at a lag of at least
``ccf_min_lag`` samples: one with no measurable delay is skipped, not reported
as zero.

``velocity_estimation``'s ``EstimationOptions``, ``CAOptions`` and
``NeighbourOptions`` carry ``@dataclass`` but declare their own ``__init__``,
so ``dataclasses.fields()`` on them is empty and they would hash to nothing.
None of them may be a params field: ``TdeVelocityParams`` carries the settings
and ``compute`` builds the upstream objects from them.
"""

from dataclasses import dataclass

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import velocity_estimation as ve
import xarray as xr
from velocity_estimation.two_dim_velocity_estimates import _find_neighbors

from fusion_ui.core import loader, registry

#: R and Z are stored in centimetres, so a velocity off this grid is cm/s. The
#: group reports m/s.
CM = 100.0


def _dead_pixel_mask():
    """The hand-made 1160616 mask of ``experimental_database`` as a ``(y, x)``
    boolean array, 10 tall by 9 wide.

    It is applied to any shot whose array has that shape, whatever the run day.
    """
    from experimental_database import hand_made_dead_pixel_mask

    return np.asarray(hand_made_dead_pixel_mask(1160616).values, dtype=bool)


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
        for.
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


def compute(ds, params):
    """The time-delay velocity at one pixel, over the whole of ``ds``."""
    loader.image_variable(ds)  # raises KeyError if this is not imaging data
    ny, nx = ds.sizes["y"], ds.sizes["x"]

    mask_grid = _dead_pixel_mask()
    mask_matches_shape = mask_grid.shape == (ny, nx)
    mask = None
    if params.mask_dead_pixels and mask_matches_shape:
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
        # Upstream raises ValueError only when the two-equation system is
        # degenerate (get_2d_velocities_from_time_delays); a pixel with no
        # usable neighbour comes back as NaN without raising. Nothing else is
        # caught: a malformed dataset must still fail loudly.
        estimate_failed = True
        vx = vy = float("nan")
        confidence, events, is_dead = float("nan"), 0, False
        r_ref = float(ds["R"].isel(x=refx, y=refy))
        z_ref = float(ds["Z"].isel(x=refx, y=refy))

    # The estimator does not say which neighbours it settled on: run its
    # search once more, to draw them.
    horizontal, vertical = _find_neighbors(refx, refy, interface, eo.neighbour_options)
    used_grid = np.zeros((ny, nx), dtype="int8")
    for used_x, used_y in list(horizontal) + list(vertical):
        if 0 <= used_y < ny and 0 <= used_x < nx:
            used_grid[used_y, used_x] = 1

    # int8, not bool: netCDF has no boolean type.
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
            # Per axis: every estimate pairs one horizontal neighbour with one
            # vertical one, so an axis with none is why there is no velocity.
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


def _reason(result):
    """Why there is no estimate, or ``None`` when there is one."""
    if np.isfinite(float(result["vx"])):
        return None
    if bool(result["estimate_failed"]):
        return "no conditional-average event crossed the threshold here or at every neighbour"
    if bool(result["is_dead"]):
        return "the reference pixel is marked dead"
    horizontal, vertical = int(result["n_horizontal"]), int(result["n_vertical"])
    if not horizontal or not vertical:
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
    # speed == 0 is reachable (the naive path with both delays at 0): an
    # estimate with no direction to draw.
    if reason is None and speed > 0:
        # The arrow has a fixed length and carries only the direction; the
        # numbers are in the title.
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
        mask_ny, mask_nx = _dead_pixel_mask().shape
        figure.add_annotation(
            text=(
                f"dead-pixel mask not applied: mask is {mask_nx}x{mask_ny} "
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
    """``vx_tde`` and ``vy_tde`` in m/s, written at the reference pixel."""
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
