"""Blob size from a Gaussian ellipse fitted to the conditional average.

Ported from ``density_scan/utils.py:get_gaussian_fit_sizes``, which is the
estimator behind the ``lx_f`` / ``ly_f`` / ``theta_f`` columns in the group's
results: fit a tilted, penalised 2D Gaussian to the zero-lag frame of the
conditional average, centred on the reference pixel, and read off its two
semi-axes and its tilt.

``imaging_methods.fit_ellipse_to_event`` takes the conditional average and the
reference pixel plus the three penalty factors from ``GaussFitParams``; upstream's
own ``get_gaussian_fit_sizes`` does not pass ``size_max`` even though
``GaussFitParams`` carries it, but it is a real knob the form should show, so
this spec passes it through. Chained the same way as ``velocity_contour`` and
``fwhm_sizes``: ``requires="two_dca"``, and ``compute`` never looks at ``ds``.
"""

from dataclasses import dataclass, field

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from imaging_methods.method_parameters import GaussFitParams, TwoDcaParams

from fusion_ui.core import loader, registry
from fusion_ui.plots import two_dca

#: R and Z are stored in centimetres; the group reports sizes in metres, and
#: the seeded density_scan rows are in those units -- convert once, here.
CM = 100.0


def _apd_defaults(name):
    return lambda: getattr(im.get_default_apd_method_params(), name)


@dataclass
class GaussianSizeParams:
    #: Identifies the upstream conditional average -- see ``upstream_params``.
    two_dca: TwoDcaParams = field(default_factory=_apd_defaults("two_dca"))
    gauss_fit: GaussFitParams = field(default_factory=_apd_defaults("gauss_fit"))


def upstream_params(params):
    """The 2DCA parameters, lifted out of this spec's own.

    Reading them out rather than defaulting them is what keeps the two cache
    keys in step: change the threshold here and both the average and this
    size get a new entry.
    """
    return two_dca.TwoDcaSpecParams(two_dca=params.two_dca)


def compute(ds, params, upstream):
    """Fit a tilted Gaussian ellipse to the zero-lag conditional average."""
    average = upstream
    fit = params.gauss_fit
    lx, ly, theta = im.fit_ellipse_to_event(
        average.cond_av,
        int(average["refx"]),
        int(average["refy"]),
        size_penalty_factor=fit.size_penalty,
        aspect_ratio_penalty_factor=fit.aspect_penalty,
        theta_penalty_factor=fit.tilt_penalty,
        size_max=fit.size_max,
    )

    refx, refy = int(average["refx"]), int(average["refy"])
    peak_frame = average.cond_av.sel(time=0, method="nearest")

    return xr.Dataset(
        {
            "peak_frame": (("y", "x"), np.asarray(peak_frame.values)),
            "lx": float(lx) / CM,
            "ly": float(ly) / CM,
            "theta": float(theta),
            "refx": refx,
            "refy": refy,
            "number_events": int(average["number_events"]),
        },
        coords={
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
    """The zero-lag average with the fitted ellipse drawn over it.

    A fit that ran away to the array edge or collapsed to a point has to be
    visible as a wrong-looking ellipse -- printing lx/ly/theta as numbers
    would look equally plausible whatever the fit actually did.
    """
    refx, refy = int(result["refx"]), int(result["refy"])
    x_axis, y_axis, x_label, y_label = _axes(result)
    lx, ly, theta = float(result["lx"]), float(result["ly"]), float(result["theta"])

    figure = go.Figure(
        go.Heatmap(
            z=result["peak_frame"].values,
            x=x_axis,
            y=y_axis,
            colorscale="Plasma",
            colorbar=dict(title="cond. av."),
        )
    )

    # Drawn in the grid's own centimetres, so convert the stored metres back.
    r_ref, z_ref = x_axis[refx], y_axis[refy]
    alpha = np.linspace(0, 2 * np.pi, 200)
    lx_cm, ly_cm = lx * CM, ly * CM
    ellipse_r = lx_cm * np.cos(alpha) * np.cos(theta) - ly_cm * np.sin(alpha) * np.sin(
        theta
    ) + r_ref
    ellipse_z = lx_cm * np.cos(alpha) * np.sin(theta) + ly_cm * np.sin(alpha) * np.cos(
        theta
    ) + z_ref
    figure.add_trace(
        go.Scatter(
            x=ellipse_r,
            y=ellipse_z,
            mode="lines",
            line=dict(color="white", width=2),
            name="fitted ellipse",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[r_ref],
            y=[z_ref],
            mode="markers",
            marker=dict(color="white", size=12, symbol="x", line=dict(width=2)),
            name="reference pixel",
        )
    )

    figure.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.15),
        title=(
            f"lx = {lx:.3g} m · ly = {ly:.3g} m · θ = {theta:.2f} rad"
            f" · {int(result['number_events'])} events"
        ),
    )
    return figure


def scalars(result):
    """``lx_f`` / ``ly_f`` / ``theta_f``, the three names density_scan reports."""
    x, y = int(result["refx"]), int(result["refy"])
    return {
        (x, y, "lx_f"): float(result["lx"]),
        (x, y, "ly_f"): float(result["ly"]),
        (x, y, "theta_f"): float(result["theta"]),
    }


SPEC = registry.register(
    registry.PlotSpec(
        key="gaussian_sizes",
        label="Blob size (Gaussian fit)",
        diagnostics=("apd", "phantom"),
        params=GaussianSizeParams,
        render=render,
        compute=compute,
        scalars=scalars,
        choices=two_dca.choices,
        requires="two_dca",
        upstream_params=upstream_params,
        description=(
            "Fits a tilted, penalised Gaussian ellipse to the zero-lag "
            "conditional average. Emits lx_f, ly_f and theta_f."
        ),
    )
)
