"""Probability density over one trace: histogram or KDE.

``fppanalysis.distributions.get_hist(v, bins)`` for the histogram,
``distribution(v, bins, kernel=True)`` for the KDE. Neither normalises its
input, so ``standardise`` is applied here -- and it is on by default because a
pixel in counts and a probe in m^-3 are otherwise not on the same axis at all.
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
    """One smooth curve per trace, colour-cycled, log-y when asked.

    Splined in the drawing, not in the data: the stored bins are untouched,
    Plotly just stops drawing the histogram steps.
    """
    from plotly.colors import qualitative

    cycle = qualitative.Plotly
    figure = go.Figure()
    for index, (trace, result) in enumerate(items):
        colour = cycle[index % len(cycle)]
        figure.add_trace(
            go.Scatter(
                x=result["value"].values,
                y=result["pdf"].values,
                mode="lines",
                line=dict(color=colour, shape="spline", smoothing=0.8),
                name=trace.label or trace.ref.key,
            )
        )
    figure.update_layout(
        xaxis_title="standardised samples" if params.standardise else "samples",
        yaxis_title="probability density",
        yaxis_type="log" if params.log_y else "linear",
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
        title=f"PDF of {len(items)} traces",
    )
    return figure


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
