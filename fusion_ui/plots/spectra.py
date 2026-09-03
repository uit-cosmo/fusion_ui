"""Duration time from a power spectral density fit.

The first *cached* spec, and the pattern every later analysis copies: a
``compute`` that returns a small derived ``xr.Dataset``, a ``render`` that only
reads it, and a ``scalars`` that says what belongs in the multi-shot store.
None of the three touches Streamlit, the database or the filesystem -- the
store does all of that.

Ported from ``density_scan/utils.py:get_taud_from_psd``, so a value computed
here is directly comparable with the ``taud_psd`` / ``lambda_psd`` rows the
``density_scan_import`` seed writes for the same shot and pixel.
"""

from dataclasses import dataclass, field

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from imaging_methods.method_parameters import TaudEstimationParams
from imaging_methods.utils import power_spectral_density

from fusion_ui.core import registry


def _apd_taud_defaults():
    # nperseg = 2e3 rather than TaudEstimationParams' own 1e3: this is what
    # density_scan ran, and matching it is what makes a fresh number and a
    # seeded one comparable.
    return im.get_default_apd_method_params().taud_estimation


@dataclass
class TaudPsdParams:
    """
    refx: X index of the pixel whose spectrum is fitted.
    refy: Y index of the pixel whose spectrum is fitted.
    """

    refx: int = 6
    refy: int = 6
    taud_estimation: TaudEstimationParams = field(default_factory=_apd_taud_defaults)


def choices(ds, path, chosen):
    """Pixel indices, bounded by the array that was actually opened.

    A free number input would let someone ask for pixel 40 of a 9x10 array and
    get an ``IndexError`` recorded as a failed run.
    """
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


def compute(ds, params):
    """Welch PSD of one pixel, plus the two-sided-exponential FPP fit to it."""
    series = ds[_image_variable(ds)].isel(x=int(params.refx), y=int(params.refy)).values
    dt = im.get_dt(ds)
    settings = params.taud_estimation

    estimator = im.DurationTimeEstimator(
        im.SecondOrderStatistic.PSD, im.Analytics.TwoSided
    )
    taud, lam = estimator.estimate_duration_time(
        series, dt, cutoff=settings.cutoff, nperseg=settings.nperseg
    )
    # The plotted spectrum has to be the one that was fitted, so it comes from
    # the estimator rather than from a second Welch call here that could drift
    # out of step with its normalisation. Private for now; worth promoting
    # upstream, not worth blocking on.
    omega, psd = estimator._get_second_order_statistic(
        series, dt, settings.cutoff, nperseg=settings.nperseg
    )

    return xr.Dataset(
        {
            "psd": ("omega", np.asarray(psd)),
            "psd_fit": ("omega", np.asarray(power_spectral_density(omega, taud, lam))),
            "taud": float(taud),
            "lam": float(lam),
            "refx": int(params.refx),
            "refy": int(params.refy),
            "dt": float(dt),
            "samples": int(series.size),
        },
        coords={"omega": np.asarray(omega)},
    )


def render(result, params, target):
    taud = float(result["taud"])
    lam = float(result["lam"])
    omega = result["omega"].values

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=omega, y=result["psd"].values, mode="lines", name="Welch PSD")
    )
    figure.add_trace(
        go.Scatter(
            x=omega,
            y=result["psd_fit"].values,
            mode="lines",
            line=dict(dash="dash"),
            name=f"fit: τ_d = {taud:.3g} s, λ = {lam:.3g}",
        )
    )
    figure.update_layout(
        xaxis_title="angular frequency [rad/s]",
        yaxis_title="power spectral density",
        xaxis_type="log",
        yaxis_type="log",
        height=440,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=-0.2),
        title=f"pixel (x={int(result['refx'])}, y={int(result['refy'])})",
    )
    return figure


def scalars(result):
    """Written at the pixel, not at the shot: which pixel was fitted is the
    whole content of the number, and it is how the seeded density_scan rows are
    laid out too."""
    x, y = int(result["refx"]), int(result["refy"])
    return {
        (x, y, "taud_psd"): float(result["taud"]),
        (x, y, "lambda_psd"): float(result["lam"]),
    }


SPEC = registry.register(
    registry.PlotSpec(
        key="taud_psd",
        label="Duration time (PSD fit)",
        diagnostics=("apd", "phantom"),
        params=TaudPsdParams,
        render=render,
        compute=compute,
        scalars=scalars,
        choices=choices,
        description=(
            "Welch spectrum of one pixel fitted with the two-sided-exponential "
            "FPP model. Emits taud_psd and lambda_psd."
        ),
    )
)
