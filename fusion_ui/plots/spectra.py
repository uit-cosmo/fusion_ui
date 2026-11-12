"""Duration time from a power spectral density fit: the Welch spectrum of one
pixel, fitted with the two-sided-exponential FPP model.

A cached spec computed straight off the frames, and the one to copy for
another: ``compute`` returns a small derived ``xr.Dataset``, ``render`` only
reads it, and ``scalars`` says what goes to the multi-shot store.
"""

from dataclasses import dataclass, field

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from imaging_methods.method_parameters import TaudEstimationParams
from imaging_methods.utils import power_spectral_density

from fusion_ui.core import loader, registry


def _apd_taud_defaults():
    # The APD defaults, with nperseg = 2e3 where TaudEstimationParams' own
    # default is 1e3.
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
    """Pixel indices, bounded by the array that was opened."""
    if path == "refx":
        return tuple(range(ds.sizes["x"]))
    if path == "refy":
        return tuple(range(ds.sizes["y"]))
    return None


def compute(ds, params):
    """Welch PSD of one pixel, plus the two-sided-exponential FPP fit to it."""
    series = ds[loader.image_variable(ds)].isel(x=int(params.refx), y=int(params.refy)).values
    dt = im.get_dt(ds)
    settings = params.taud_estimation

    estimator = im.DurationTimeEstimator(
        im.SecondOrderStatistic.PSD, im.Analytics.TwoSided
    )
    taud, lam = estimator.estimate_duration_time(
        series, dt, cutoff=settings.cutoff, nperseg=settings.nperseg
    )
    # The plotted spectrum has to be the one that was fitted, so it comes from
    # the estimator's own (private) method, not from a second Welch call here.
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


def overlay(items, params, target):
    """Every selected pixel's PSD and its fit on one log-log axis.

    One colour per pixel, the fit dashed, both under one ``legendgroup`` so
    that a click on the legend hides the pair.
    """
    from plotly.colors import qualitative

    cycle = qualitative.Plotly
    figure = go.Figure()
    for index, ((x, y), result) in enumerate(items):
        colour = cycle[index % len(cycle)]
        group = f"pixel-{x}-{y}"
        taud = float(result["taud"])
        omega = result["omega"].values
        figure.add_trace(
            go.Scatter(
                x=omega,
                y=result["psd"].values,
                mode="lines",
                line=dict(color=colour),
                legendgroup=group,
                name=f"(x={x}, y={y})  τ_d={taud:.3g} s",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=omega,
                y=result["psd_fit"].values,
                mode="lines",
                line=dict(color=colour, dash="dash"),
                legendgroup=group,
                showlegend=False,
            )
        )
    figure.update_layout(
        xaxis_title="angular frequency [rad/s]",
        yaxis_title="power spectral density",
        xaxis_type="log",
        yaxis_type="log",
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
        title=f"Duration time at {len(items)} pixels",
    )
    return figure


def scalars(result):
    """Written at the pixel that was fitted."""
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
        overlay=overlay,
        choices=choices,
        description=(
            "Welch spectrum of one pixel fitted with the two-sided-exponential "
            "FPP model. Emits taud_psd and lambda_psd."
        ),
    )
)
