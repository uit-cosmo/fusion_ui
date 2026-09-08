"""Autocorrelation of one trace, with an optional duration-time fit."""

from dataclasses import dataclass

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from imaging_methods.utils import autocorrelation

from fusion_ui.core import statistics


@dataclass
class AcfParams:
    """
    max_lag: Plotted lag range, in seconds (lags beyond it are dropped).
    biased: Use the biased (1/N) correlation estimator rather than the unbiased (1/(N-|k|)) one.
    fit: Fit the two-sided-exponential FPP model and draw it dashed.
    """

    max_lag: float = 1e-4
    biased: bool = False
    fit: bool = False


def compute(trace, params):
    """``fppanalysis.corr_fun`` of one trace against itself, truncated."""
    from fppanalysis import corr_fun

    values = np.asarray(trace.value, dtype=float)
    dt = float(trace.dt)
    lags, curve = corr_fun(values, values, dt, norm=True, biased=bool(params.biased))
    lags = np.asarray(lags, dtype=float)
    curve = np.asarray(curve, dtype=float)
    mask = np.abs(lags) <= float(params.max_lag)
    lags, curve = lags[mask], curve[mask]

    data = {"acf": ("lag", curve)}
    if params.fit:
        estimator = im.DurationTimeEstimator(
            im.SecondOrderStatistic.ACF, im.Analytics.TwoSided
        )
        taud, lam = estimator.estimate_duration_time(values, dt)
        data["acf_fit"] = ("lag", np.asarray(autocorrelation(lags, taud, lam)))
        data["taud"] = float(taud)
        data["lam"] = float(lam)
    return xr.Dataset(data, coords={"lag": lags})


def render(items, params):
    """One autocorrelation per trace; each fit dashed in its curve's colour."""
    from plotly.colors import qualitative

    cycle = qualitative.Plotly
    figure = go.Figure()
    for index, (trace, result) in enumerate(items):
        colour = cycle[index % len(cycle)]
        group = f"trace-{index}"
        lags = result["lag"].values
        name = trace.label or trace.ref.key
        if "taud" in result:
            name = f"{name}  tau_d={float(result['taud']):.3g} s"
        figure.add_trace(
            go.Scatter(
                x=lags,
                y=result["acf"].values,
                mode="lines",
                line=dict(color=colour),
                legendgroup=group,
                name=name,
            )
        )
        if "acf_fit" in result:
            figure.add_trace(
                go.Scatter(
                    x=lags,
                    y=result["acf_fit"].values,
                    mode="lines",
                    line=dict(color=colour, dash="dash"),
                    legendgroup=group,
                    showlegend=False,
                )
            )
    figure.update_layout(
        xaxis_title="lag [s]",
        yaxis_title="autocorrelation",
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
        title=f"ACF of {len(items)} traces",
    )
    return figure


SPEC = statistics.register(
    statistics.StatSpec(
        key="acf",
        label="Autocorrelation (ACF)",
        params=AcfParams,
        compute=compute,
        render=render,
        description=(
            "Normalised autocorrelation of each trace with an optional "
            "duration-time fit."
        ),
    )
)
