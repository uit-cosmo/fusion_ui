"""Probability density of one trace's samples: histogram or KDE.

``fppanalysis`` normalises neither, so ``standardise`` is applied here. It is
on by default: a pixel in counts and a probe in m^-3 share no axis otherwise.
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import xarray as xr

from fusion_ui.core import statistics


@dataclass
class PdfParams:
    """
    bins: Number of bins (histogram) or evaluation points (KDE).
    estimator: Histogram or Gaussian-kernel density estimate.
    standardise: Work with (x - mean) / std so traces in different units share an axis.
    log_y: Log-scaled probability axis.
    """

    bins: int = 64
    estimator: str = "histogram"
    standardise: bool = True
    log_y: bool = True


def _values(trace, params):
    values = np.asarray(trace.value, dtype=float)
    values = values[np.isfinite(values)]
    if params.standardise and values.size:
        std = float(values.std())
        if np.isfinite(std) and std > 0:
            values = (values - float(values.mean())) / std
    return values


def compute(trace, params):
    """The density of one trace's samples."""
    from fppanalysis.distributions import distribution, get_hist

    values = _values(trace, params)
    bins = int(params.bins)
    if params.estimator == "histogram":
        centres, pdf = get_hist(values, bins)
    elif params.estimator == "kde":
        pdf, _, centres = distribution(values, bins, kernel=True)
    else:
        raise ValueError(f"unknown estimator {params.estimator!r}")
    return xr.Dataset(
        {"pdf": ("value", np.asarray(pdf, dtype=float))},
        coords={"value": np.asarray(centres, dtype=float)},
    )


def render(items, params):
    """One curve per trace, log-y when asked. The spline is in the drawing
    only: the bins are as computed."""
    figure = go.Figure()
    for index, (trace, result) in enumerate(items):
        statistics.add_line(
            figure,
            index,
            result["value"].values,
            result["pdf"].values,
            statistics.name(trace),
            line=dict(shape="spline", smoothing=0.8),
        )
    return statistics.finish(
        figure,
        f"PDF of {len(items)} traces",
        "standardised samples" if params.standardise else "samples",
        "probability density",
        yaxis_type="log" if params.log_y else "linear",
    )


SPEC = statistics.register(
    statistics.StatSpec(
        key="pdf",
        label="Probability density (PDF)",
        params=PdfParams,
        compute=compute,
        render=render,
        description=(
            "Histogram or kernel density estimate of each trace's samples, "
            "standardised by default so traces in different units share an axis."
        ),
    )
)
