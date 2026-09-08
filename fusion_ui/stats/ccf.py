"""Cross-correlation of every trace against one reference trace.

``compute(trace, reference, params)`` -- the only pairwise statistic. The two
arrays must be the same length on the same base, which is what
:func:`fusion_ui.core.traces.common_grid` guarantees before ``compute`` is
called. The lag at the maximum and the maximum itself go into the result, and
into the legend -- that number is usually the reason someone drew the plot.
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import xarray as xr

from fusion_ui.core import statistics


@dataclass
class CcfParams:
    """
    max_lag: Plotted lag range, in seconds (lags beyond it are dropped).
    biased: Use the biased (1/N) correlation estimator rather than the unbiased (1/(N-|k|)) one.
    """

    max_lag: float = 1e-4
    biased: bool = False


def compute(trace, reference, params):
    """``fppanalysis.corr_fun`` of one trace against the reference, truncated."""
    from fppanalysis import corr_fun

    values = np.asarray(trace.value, dtype=float)
    ref_values = np.asarray(reference.value, dtype=float)
    lags, curve = corr_fun(
        values, ref_values, float(trace.dt), norm=True, biased=bool(params.biased)
    )
    lags = np.asarray(lags, dtype=float)
    curve = np.asarray(curve, dtype=float)
    mask = np.abs(lags) <= float(params.max_lag)
    lags, curve = lags[mask], curve[mask]
    peak = int(np.argmax(curve))
    return xr.Dataset(
        {
            "ccf": ("lag", curve),
            "peak_lag": float(lags[peak]),
            "peak_value": float(curve[peak]),
        },
        coords={"lag": lags},
    )


def render(items, params):
    """One cross-correlation per trace, peak lag and value in the legend."""
    from plotly.colors import qualitative

    cycle = qualitative.Plotly
    figure = go.Figure()
    for index, (trace, result) in enumerate(items):
        colour = cycle[index % len(cycle)]
        name = trace.label or trace.ref.key
        name = (
            f"{name}  peak {float(result['peak_lag']):.3g} s "
            f"({float(result['peak_value']):.3g})"
        )
        figure.add_trace(
            go.Scatter(
                x=result["lag"].values,
                y=result["ccf"].values,
                mode="lines",
                line=dict(color=colour),
                name=name,
            )
        )
    figure.update_layout(
        xaxis_title="lag [s]",
        yaxis_title="cross-correlation",
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
        title=f"CCF of {len(items)} traces",
    )
    return figure


SPEC = statistics.register(
    statistics.StatSpec(
        key="ccf",
        label="Cross-correlation (CCF)",
        params=CcfParams,
        compute=compute,
        render=render,
        pairwise=True,
        description=(
            "Normalised cross-correlation of each trace against a reference "
            "trace from the basket."
        ),
    )
)
