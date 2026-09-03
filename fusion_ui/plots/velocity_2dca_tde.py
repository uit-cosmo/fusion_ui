"""Blob velocity from time delays across the conditional average (3-point TDE).

Ported from ``density_scan/utils.py``'s ``get_2dca_tde_velocities`` (which
calls ``get_3tde_velocities`` -> ``get_delays`` -> ``get_maximum_time`` ->
``find_maximum_interpolate`` / ``gaussian_convolve``): read the lag at which
each of the reference pixel's four neighbours peaks in the conditional
average, and solve the horizontal and vertical pairs together with the
standard three-point time-delay estimator. It is the same estimator the group
runs on the raw record, applied instead to ``cond_av`` -- the average is much
less noisy than a single event, at the cost of only seeing the "typical" blob.

``density_scan``'s package ``__init__`` reads environment at import time, so
those five helpers are reimplemented below as private pure functions rather
than imported. ``velocity_estimation`` carries no such cost and is imported
directly, as upstream does inside ``get_3tde_velocities``.

Chained the same way as ``velocity_contour`` and ``fwhm_sizes``:
``requires="two_dca"``, and ``compute`` never looks at ``ds``.
"""

from dataclasses import dataclass, field

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import velocity_estimation as ve
import xarray as xr
from imaging_methods.method_parameters import TwoDcaParams
from scipy.interpolate import InterpolatedUnivariateSpline

from fusion_ui.core import registry
from fusion_ui.plots import two_dca

#: R and Z are stored in centimetres; the group reports velocities in m/s.
CM = 100.0

#: (x, y) pixel offset of each neighbour from the reference pixel, in the
#: order ``get_delays`` reads them.
_OFFSETS = {"right": (1, 0), "left": (-1, 0), "up": (0, 1), "down": (0, -1)}


def _apd_two_dca_defaults():
    return im.get_default_apd_method_params().two_dca


@dataclass
class TdeParams:
    """Knobs ``get_maximum_time`` takes that upstream never wrapped in a class."""

    #: Smooth the reference-pixel and neighbour waveforms with a Gaussian
    #: before locating their maxima. Upstream's own default is off.
    gauss_convolve: bool = False
    #: Standard deviation of that Gaussian, in samples. Only read when
    #: gauss_convolve is set; upstream hardcodes this at 3 in the one place it
    #: calls ``gaussian_convolve``.
    sigma: float = 3.0


@dataclass
class Velocity2dcaTdeParams:
    #: Identifies the upstream conditional average -- see ``upstream_params``.
    two_dca: TwoDcaParams = field(default_factory=_apd_two_dca_defaults)
    tde: TdeParams = field(default_factory=TdeParams)


def upstream_params(params):
    """The 2DCA parameters, lifted out of this spec's own.

    Reading them out rather than defaulting them is what keeps the two cache
    keys in step: change the threshold here and both the average and this
    velocity get a new entry.
    """
    return two_dca.TwoDcaSpecParams(two_dca=params.two_dca)


# ---------------------------------------------------------------------------
# Reimplemented from density_scan/utils.py -- see the module docstring for why
# these are not imported from there.
# ---------------------------------------------------------------------------


def _gaussian_convolve(x, times, sigma):
    """Ported from ``density_scan.utils.gaussian_convolve``."""
    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
    center = kernel_size // 2
    kernel = np.exp(-((np.arange(-center, center + 1) / sigma) ** 2))
    kernel = kernel / kernel.sum()
    return times[center:-center], np.convolve(x, kernel, mode="valid")


def _find_maximum_interpolate(x, y):
    """Ported from ``density_scan.utils.find_maximum_interpolate``.

    Degree 4 specifically: the maximum is located from the roots of the
    spline's derivative, which only works from degree 4 up. Also checks the
    interval endpoints, since the derivative can be nonzero everywhere and the
    true maximum still sit on the boundary.

    Returns ``(tau, value, edge)`` -- upstream only ``warnings.warn``s when
    ``edge`` is true, which nobody would see in a browser; the caller stores
    it instead.
    """
    spline = InterpolatedUnivariateSpline(x, y, k=4)
    possible_maxima = spline.derivative().roots()
    possible_maxima = np.append(possible_maxima, (x[0], x[-1]))
    values = spline(possible_maxima)
    max_index = np.argmax(values)
    max_time = possible_maxima[max_index]
    edge = bool(max_time == x[0] or max_time == x[-1])
    return float(max_time), float(values[max_index]), edge


def _get_maximum_time(average, x, y, gauss_convolve, sigma):
    """Ported from ``density_scan.utils.get_maximum_time``.

    Returns ``None`` when ``(x, y)`` is off the conditional average's grid --
    the caller drops those rather than raising, so a reference pixel on the
    array edge still yields a velocity from whichever neighbour survives.
    """
    if not (0 <= x < average.sizes["x"] and 0 <= y < average.sizes["y"]):
        return None
    times = average["time"].values
    trace = average["cond_av"].isel(x=x, y=y).values
    if gauss_convolve:
        times, trace = _gaussian_convolve(trace, times, sigma)
    tau, value, edge = _find_maximum_interpolate(times, trace)
    return tau, value, edge, times, trace


def _pair_delay(tau_ref, tau_pos, tau_neg):
    """Ported from ``density_scan.utils.get_delays``, one axis at a time.

    Averages whichever side of the pair is finite: an edge pixel that has
    lost one neighbour still yields a delay from the other, rather than a
    NaN velocity.
    """
    deltas = []
    if np.isfinite(tau_pos):
        deltas.append(tau_pos - tau_ref)
    if np.isfinite(tau_neg):
        deltas.append(tau_ref - tau_neg)
    return float(np.mean(deltas)) if deltas else np.nan


def compute(ds, params, upstream):
    """The three-point time-delay velocity, read off the conditional average."""
    average = upstream
    refx, refy = int(average["refx"]), int(average["refy"])
    gauss_convolve = bool(params.tde.gauss_convolve)
    sigma = float(params.tde.sigma)

    ref = _get_maximum_time(average, refx, refy, gauss_convolve, sigma)
    if ref is None:
        # Cannot happen in practice -- the reference pixel is where the 2DCA
        # events were found in the first place -- but guard rather than let a
        # stray IndexError surface a hundred lines downstream.
        raise ValueError(
            f"reference pixel (x={refx}, y={refy}) is off the conditional "
            "average's own grid"
        )
    tau_ref, val_ref, edge_ref, lag, trace_ref = ref

    waveforms = {}
    for name, (dx, dy) in _OFFSETS.items():
        got = _get_maximum_time(average, refx + dx, refy + dy, gauss_convolve, sigma)
        if got is None:
            # Off the array: NaN, at the same length as every other waveform
            # -- not a silently shorter array that render or a downstream
            # reader would have to special-case.
            waveforms[name] = dict(
                tau=np.nan,
                val=np.nan,
                edge=-1,  # sentinel: no maximum was located at all
                trace=np.full_like(lag, np.nan, dtype=float),
            )
        else:
            tau, val, edge, _, trace = got
            waveforms[name] = dict(tau=tau, val=val, edge=int(edge), trace=trace)

    taux = _pair_delay(tau_ref, waveforms["right"]["tau"], waveforms["left"]["tau"])
    tauy = _pair_delay(tau_ref, waveforms["up"]["tau"], waveforms["down"]["tau"])

    # Pixel spacing, read off the data. Fixed at pixels (0,0)/(1,0)/(0,1), not
    # at the reference pixel, exactly as upstream does -- so this still works
    # with the reference on the edge. R varies along x and Z along y in this
    # dataset's (y, x) layout, but isel is by dimension name, so which axis
    # comes first in the underlying array does not matter here.
    deltax = average.R.isel(x=1, y=0).item() - average.R.isel(x=0, y=0).item()
    deltay = average.Z.isel(x=0, y=1).item() - average.Z.isel(x=0, y=0).item()

    vx, vy = ve.get_2d_velocities_from_time_delays(taux, tauy, deltax, 0, 0, deltay)

    return xr.Dataset(
        {
            "lag": ("lag", np.asarray(lag)),
            "trace_ref": ("lag", np.asarray(trace_ref)),
            "trace_right": ("lag", np.asarray(waveforms["right"]["trace"])),
            "trace_left": ("lag", np.asarray(waveforms["left"]["trace"])),
            "trace_up": ("lag", np.asarray(waveforms["up"]["trace"])),
            "trace_down": ("lag", np.asarray(waveforms["down"]["trace"])),
            "tau_ref": float(tau_ref),
            "val_ref": float(val_ref),
            # int8, not bool: netCDF has no boolean type, and -1 marks a
            # neighbour that was off the array rather than merely not on the
            # window edge.
            "edge_ref": np.int8(int(edge_ref)),
            "tau_right": float(waveforms["right"]["tau"]),
            "val_right": float(waveforms["right"]["val"]),
            "edge_right": np.int8(waveforms["right"]["edge"]),
            "tau_left": float(waveforms["left"]["tau"]),
            "val_left": float(waveforms["left"]["val"]),
            "edge_left": np.int8(waveforms["left"]["edge"]),
            "tau_up": float(waveforms["up"]["tau"]),
            "val_up": float(waveforms["up"]["val"]),
            "edge_up": np.int8(waveforms["up"]["edge"]),
            "tau_down": float(waveforms["down"]["tau"]),
            "val_down": float(waveforms["down"]["val"]),
            "edge_down": np.int8(waveforms["down"]["edge"]),
            "taux": float(taux),
            "tauy": float(tauy),
            "vx": float(vx) / CM,
            "vy": float(vy) / CM,
            "refx": refx,
            "refy": refy,
            "number_events": int(average["number_events"]),
        }
    )


# ---------------------------------------------------------------------------


_WAVEFORMS = (
    ("trace_ref", "tau_ref", "val_ref", "edge_ref", "reference", "#444444"),
    ("trace_right", "tau_right", "val_right", "edge_right", "+x neighbour", "crimson"),
    ("trace_left", "tau_left", "val_left", "edge_left", "-x neighbour", "royalblue"),
    ("trace_up", "tau_up", "val_up", "edge_up", "+y neighbour", "seagreen"),
    ("trace_down", "tau_down", "val_down", "edge_down", "-y neighbour", "darkorange"),
)


def render(result, params, target):
    """The five waveforms against lag, each marked at its located maximum.

    Five staggered peaks *are* the measurement: a velocity computed from
    waveforms whose peaks all sit on top of each other, or whose maximum
    landed on the window edge, is visibly untrustworthy in a way the two
    velocity numbers alone are not.
    """
    lag_us = result["lag"].values * 1e6
    vx, vy = float(result["vx"]), float(result["vy"])
    n_events = int(result["number_events"])

    figure = go.Figure()
    any_edge = False
    for trace_key, tau_key, val_key, edge_key, label, color in _WAVEFORMS:
        trace = result[trace_key].values
        if not np.isfinite(trace).any():
            continue  # off the array -- see compute's comment on NaN filling
        figure.add_trace(
            go.Scatter(
                x=lag_us,
                y=trace,
                mode="lines",
                name=label,
                line=dict(color=color),
            )
        )
        edge = int(result[edge_key])
        on_edge = edge == 1
        any_edge = any_edge or on_edge
        figure.add_trace(
            go.Scatter(
                x=[float(result[tau_key]) * 1e6],
                y=[float(result[val_key])],
                mode="markers",
                marker=dict(
                    color=color,
                    size=16 if on_edge else 9,
                    symbol="x" if on_edge else "circle",
                    line=dict(width=2, color="black"),
                ),
                showlegend=False,
                hovertext=f"{label} maximum" + (" -- AT WINDOW EDGE" if on_edge else ""),
                hoverinfo="text",
            )
        )

    title = f"v = ({vx:.0f}, {vy:.0f}) m/s · {n_events} events"
    if any_edge:
        title += " -- maximum at window edge (✕): widen the 2DCA window"

    figure.update_layout(
        xaxis_title="lag [µs]",
        yaxis_title="conditional average at pixel",
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.15),
        title=title,
    )
    return figure


def scalars(result):
    """``vx_2dca_tde`` / ``vy_2dca_tde``, the two names density_scan reports.

    Written even when a delay could not be estimated on one axis (which comes
    back as NaN out of ``compute``, not as a missing key) -- the multi-shot
    view is what surfaces "this shot has no answer here", and an omitted key
    would instead just look like the plot was never run.
    """
    x, y = int(result["refx"]), int(result["refy"])
    return {
        (x, y, "vx_2dca_tde"): float(result["vx"]),
        (x, y, "vy_2dca_tde"): float(result["vy"]),
    }


SPEC = registry.register(
    registry.PlotSpec(
        key="velocity_2dca_tde",
        label="Blob velocity (2DCA time delay)",
        diagnostics=("apd", "phantom"),
        params=Velocity2dcaTdeParams,
        render=render,
        compute=compute,
        scalars=scalars,
        choices=two_dca.choices,
        requires="two_dca",
        upstream_params=upstream_params,
        description=(
            "Three-point time-delay velocity from the lag at which the "
            "reference pixel's four neighbours peak in the conditional "
            "average. Emits vx_2dca_tde and vy_2dca_tde."
        ),
    )
)
