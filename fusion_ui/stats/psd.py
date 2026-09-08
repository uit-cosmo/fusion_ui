"""Welch power spectrum of one trace, with an optional duration-time fit.

Mirrors :mod:`fusion_ui.plots.spectra` so a number here and a ``taud_psd``
number are the same number: the same ``DurationTimeEstimator`` over
``SecondOrderStatistic.PSD`` / ``Analytics.TwoSided``, the plotted curve from
the same private ``_get_second_order_statistic`` call (a second Welch call
could drift out of step with the fit's normalisation), and the fit from
``estimate_duration_time`` plus ``imaging_methods.utils.power_spectral_density``.

The cached ``taud_psd`` spec stays the per-pixel, scalar-producing spectrum;
this is its exploratory, cross-diagnostic cousin.
"""

from dataclasses import dataclass

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from imaging_methods.utils import power_spectral_density

from fusion_ui.core import statistics


@dataclass
class PsdParams:
    """
    nperseg: Welch segment length, in samples. Sets the lowest frequency that can be estimated.
    cutoff: Frequency cutoff for the PSD fit, in rad/s. None means no cutoff.
    fit: Fit the two-sided-exponential FPP model and draw it dashed.
    """

    nperseg: int = 2000
    cutoff: float = None
    fit: bool = True


def compute(trace, params):
    """Welch PSD of one trace, plus the two-sided-exponential FPP fit to it."""
    values = np.asarray(trace.value, dtype=float)
    dt = float(trace.dt)
    nperseg = int(params.nperseg)
    cutoff = None if params.cutoff is None else float(params.cutoff)

    estimator = im.DurationTimeEstimator(
        im.SecondOrderStatistic.PSD, im.Analytics.TwoSided
    )
    omega, psd = estimator._get_second_order_statistic(
        values, dt, cutoff, nperseg=nperseg
    )

    data = {
        "psd": ("omega", np.asarray(psd, dtype=float)),
    }
    if params.fit:
        taud, lam = estimator.estimate_duration_time(
            values, dt, cutoff=cutoff, nperseg=nperseg
        )
        data["psd_fit"] = (
            "omega",
            np.asarray(power_spectral_density(omega, taud, lam), dtype=float),
        )
        data["taud"] = float(taud)
        data["lam"] = float(lam)
    return xr.Dataset(data, coords={"omega": np.asarray(omega, dtype=float)})


def render(items, params):
    """Log-log spectra; each fit dashed in its curve's colour, tau_d in the legend."""
    from plotly.colors import qualitative

    cycle = qualitative.Plotly
    figure = go.Figure()
    for index, (trace, result) in enumerate(items):
        colour = cycle[index % len(cycle)]
        group = f"trace-{index}"
        omega = result["omega"].values
        name = trace.label or trace.ref.key
        if "taud" in result:
            name = f"{name}  tau_d={float(result['taud']):.3g} s"
        figure.add_trace(
            go.Scatter(
                x=omega,
                y=result["psd"].values,
                mode="lines",
                line=dict(color=colour),
                legendgroup=group,
                name=name,
            )
        )
        if "psd_fit" in result:
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
        title=f"PSD of {len(items)} traces",
    )
    return figure


SPEC = statistics.register(
    statistics.StatSpec(
        key="psd",
        label="Power spectrum (PSD)",
        params=PsdParams,
        compute=compute,
        render=render,
        description=(
            "Welch spectrum of each trace with an optional duration-time fit. "
            "The exploratory cousin of the cached Duration time (PSD fit) plot."
        ),
    )
)
