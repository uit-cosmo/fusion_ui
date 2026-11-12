"""Welch power spectrum of one trace, with an optional duration-time fit.

The estimator is the one :mod:`fusion_ui.plots.spectra` uses, so a duration
time here and a ``taud_psd`` are the same number. The curve comes from the
estimator's own ``_get_second_order_statistic``: a second Welch call could
differ from the fit's normalisation.
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
    figure = go.Figure()
    for index, (trace, result) in enumerate(items):
        statistics.add_fitted(figure, index, trace, result, "psd")
    return statistics.finish(
        figure,
        f"PSD of {len(items)} traces",
        "angular frequency [rad/s]",
        "power spectral density",
        xaxis_type="log",
        yaxis_type="log",
    )


SPEC = statistics.register(
    statistics.StatSpec(
        key="psd",
        label="Power spectrum (PSD)",
        params=PsdParams,
        compute=compute,
        render=render,
        description=(
            "Welch spectrum of each trace with an optional duration-time fit. "
            "The same estimator as the cached Duration time (PSD fit) on the "
            "single-shot page, but across diagnostics and not stored: trust "
            "the cached one for numbers on a multi-shot axis."
        ),
    )
)
