"""Cross-correlation of every trace against one reference trace.

The one pairwise statistic: ``compute(trace, reference, params)``. Both must be
on the same time base, which :func:`fusion_ui.core.traces.common_grid` sees to
before ``compute`` is called. The lag of the maximum and the maximum go in the
result and in the legend.
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
    """The trace's correlation with the reference, and where it peaks."""
    lags, curve = statistics.correlation(trace, reference, params)
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
    figure = go.Figure()
    for index, (trace, result) in enumerate(items):
        statistics.add_line(
            figure,
            index,
            result["lag"].values,
            result["ccf"].values,
            f"{statistics.name(trace)}  peak {float(result['peak_lag']):.3g} s "
            f"({float(result['peak_value']):.3g})",
        )
    return statistics.finish(
        figure, f"CCF of {len(items)} traces", "lag [s]", "cross-correlation"
    )


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
