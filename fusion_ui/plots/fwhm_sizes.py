"""Blob size from the full-width-half-maximum of the conditional average.

Ported from ``density_scan/utils.py:plot_and_estimate_fwhm_sizes``, which is
the estimator behind the ``lr`` / ``lz`` columns in the group's results: take
the peak-lag frame of the conditional average, walk out from the reference
pixel along the radial and poloidal cuts through it, and interpolate where
each cut first crosses half its value at the reference pixel.

``imaging_methods.estimate_fwhm_sizes`` takes only the conditional average --
there is no dedicated parameter class upstream, so this spec's own params
dataclass carries nothing but the ``two_dca`` field every chained spec needs.
Chained the same way as ``velocity_contour``: ``requires="two_dca"``, and
``compute`` never looks at ``ds``.
"""

from dataclasses import dataclass, field

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from imaging_methods.method_parameters import TwoDcaParams

from fusion_ui.core import registry
from fusion_ui.plots import two_dca

#: R and Z are stored in centimetres; the group reports sizes in metres, and
#: the seeded density_scan rows are in those units -- convert once, here.
CM = 100.0


def _apd_two_dca_defaults():
    return im.get_default_apd_method_params().two_dca


@dataclass
class FwhmSizeParams:
    #: Identifies the upstream conditional average -- see ``upstream_params``.
    two_dca: TwoDcaParams = field(default_factory=_apd_two_dca_defaults)


def upstream_params(params):
    """The 2DCA parameters, lifted out of this spec's own.

    Reading them out rather than defaulting them is what keeps the two cache
    keys in step: change the threshold here and both the average and this
    size get a new entry.
    """
    return two_dca.TwoDcaSpecParams(two_dca=params.two_dca)


def compute(ds, params, upstream):
    """The radial and poloidal FWHM cuts through the peak-lag average."""
    average = upstream
    rp_fwhm, rn_fwhm, zp_fwhm, zn_fwhm = im.estimate_fwhm_sizes(average)

    refx, refy = int(average["refx"]), int(average["refy"])
    peak = average.cond_av.sel(time=0, method="nearest")
    r_ref = average.R.isel(x=refx, y=refy).item()
    z_ref = average.Z.isel(x=refx, y=refy).item()

    poloidal_var = peak.isel(x=refx).values
    poloidal_pos = average.Z.isel(x=refx).values - z_ref
    radial_var = peak.isel(y=refy).values
    radial_pos = average.R.isel(y=refy).values - r_ref

    return xr.Dataset(
        {
            "poloidal_pos": ("y", np.asarray(poloidal_pos)),
            "poloidal_var": ("y", np.asarray(poloidal_var)),
            "radial_pos": ("x", np.asarray(radial_pos)),
            "radial_var": ("x", np.asarray(radial_var)),
            "rp_fwhm": float(rp_fwhm),
            "rn_fwhm": float(rn_fwhm),
            "zp_fwhm": float(zp_fwhm),
            "zn_fwhm": float(zn_fwhm),
            "lr": float((rp_fwhm - rn_fwhm) / CM),
            "lz": float((zp_fwhm - zn_fwhm) / CM),
            "refx": refx,
            "refy": refy,
            "number_events": int(average["number_events"]),
        }
    )


# ---------------------------------------------------------------------------


def render(result, params, target):
    """The radial and poloidal cuts, with the FWHM crossings marked.

    Ported from the matplotlib version's two profile lines and vlines -- the
    only honest check for an estimator that is one interpolation each side of
    a peak: a cut that never crosses half its reference value should come back
    as a visibly missing dashed line, not a silently NaN size.
    """
    refy = int(result["refy"])
    poloidal_pos = result["poloidal_pos"].values
    poloidal_var = result["poloidal_var"].values
    radial_pos = result["radial_pos"].values
    radial_var = result["radial_var"].values
    ref_val = float(poloidal_var[refy])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=poloidal_pos,
            y=poloidal_var,
            mode="lines",
            name="Φ(Z−Z*)",
            line=dict(color="royalblue"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=radial_pos,
            y=radial_var,
            mode="lines",
            name="Φ(R−R*)",
            line=dict(color="seagreen"),
        )
    )
    for value, color in (
        (float(result["zp_fwhm"]), "royalblue"),
        (float(result["zn_fwhm"]), "royalblue"),
        (float(result["rp_fwhm"]), "seagreen"),
        (float(result["rn_fwhm"]), "seagreen"),
    ):
        if np.isfinite(value):
            figure.add_shape(
                type="line",
                x0=value,
                x1=value,
                y0=0,
                y1=ref_val,
                line=dict(color=color, dash="dash", width=1),
            )

    figure.update_layout(
        xaxis_title="R−R*, Z−Z* [cm]",
        yaxis_title="conditional average at zero lag",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.15),
        title=(
            f"lr = {float(result['lr']):.3g} m · lz = {float(result['lz']):.3g} m"
            f" · {int(result['number_events'])} events"
        ),
    )
    return figure


def scalars(result):
    """``lr`` / ``lz``, the two names density_scan reports for this estimator."""
    x, y = int(result["refx"]), int(result["refy"])
    return {
        (x, y, "lr"): float(result["lr"]),
        (x, y, "lz"): float(result["lz"]),
    }


SPEC = registry.register(
    registry.PlotSpec(
        key="fwhm_sizes",
        label="Blob size (FWHM)",
        diagnostics=("apd", "phantom"),
        params=FwhmSizeParams,
        render=render,
        compute=compute,
        scalars=scalars,
        choices=two_dca.choices,
        requires="two_dca",
        upstream_params=upstream_params,
        description=(
            "Radial and poloidal full-width-half-maximum of the conditional "
            "average at zero lag. Emits lr and lz."
        ),
    )
)
