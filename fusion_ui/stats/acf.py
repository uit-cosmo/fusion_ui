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
    """The trace's correlation with itself, and the fit when asked."""
    lags, curve = statistics.correlation(trace, trace, params)
    data = {"acf": ("lag", curve)}
    if params.fit:
        estimator = im.DurationTimeEstimator(
            im.SecondOrderStatistic.ACF, im.Analytics.TwoSided
        )
        taud, lam = estimator.estimate_duration_time(
            np.asarray(trace.value, dtype=float), float(trace.dt)
        )
        data["acf_fit"] = ("lag", np.asarray(autocorrelation(lags, taud, lam)))
        data["taud"] = float(taud)
        data["lam"] = float(lam)
    return xr.Dataset(data, coords={"lag": lags})


def render(items, params):
    """One autocorrelation per trace; each fit dashed in its curve's colour."""
    figure = go.Figure()
    for index, (trace, result) in enumerate(items):
        statistics.add_fitted(figure, index, trace, result, "acf")
    return statistics.finish(
        figure, f"ACF of {len(items)} traces", "lag [s]", "autocorrelation"
    )


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
