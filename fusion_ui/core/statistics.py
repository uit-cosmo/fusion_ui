"""``StatSpec``: a statistic over a basket of traces, for the Statistics page.

A statistic draws many traces from many targets on one axis, which a
``PlotSpec`` (one target, one result) cannot express. It registers the same way
and keeps the same purity rule: ``compute`` never touches Streamlit, the
database or the filesystem.

Statistics are live. A histogram or a Welch PSD takes tens of milliseconds and
gives no scalar for the multi-shot axis (the cached ``taud_psd`` plot does), so
nothing is stored.

The helpers at the end are what the modules of ``fusion_ui/stats/`` share.
"""

from dataclasses import dataclass
from typing import Callable

import plotly.graph_objects as go
from plotly.colors import qualitative


@dataclass(frozen=True)
class StatSpec:
    """One statistic: its parameters, how to compute it, and how to draw it."""

    #: Stable identifier. Keys the session state and the params prefix.
    key: str
    #: What the statistic selectbox shows.
    label: str
    #: A dataclass with ``int``/``float``/``str``/``bool`` leaves, walked by
    #: :mod:`fusion_ui.core.params_ui`.
    params: type
    #: ``(trace, params) -> xr.Dataset``, or
    #: ``(trace, reference, params) -> xr.Dataset`` when ``pairwise``.
    compute: Callable
    #: ``(items, params) -> go.Figure | None``. ``items`` is
    #: ``[(Trace, xr.Dataset), ...]`` in basket order. ``None`` means it drew
    #: into Streamlit itself, as the time trace does for its zoom.
    render: Callable
    #: Needs a reference trace, and a common time base
    #: (:func:`fusion_ui.core.traces.common_grid`).
    pairwise: bool = False
    #: One line under the statistic picker.
    description: str = ""


REGISTRY = {}


def register(spec):
    """Add ``spec`` to the registry and return it. A duplicate key raises."""
    if spec.key in REGISTRY:
        raise ValueError(f"a StatSpec is already registered under {spec.key!r}")
    REGISTRY[spec.key] = spec
    return spec


def get(key):
    return REGISTRY[key]


def all_specs():
    """Every registered spec, in registration order."""
    return list(REGISTRY.values())


# ---------------------------------------------------------------------------
# Shared by the statistics
# ---------------------------------------------------------------------------


def name(trace):
    """What the legend calls a trace."""
    return trace.label or trace.ref.key


def add_line(figure, index, x, y, label, line=None, **scatter):
    """One line, in the colour of its place ``index`` in the basket."""
    colour = qualitative.Plotly[index % len(qualitative.Plotly)]
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=colour, **(line or {})),
            name=label,
            **scatter,
        )
    )


def add_fitted(figure, index, trace, result, variable):
    """A trace's ``variable`` and, dashed in its colour, the fit stored beside
    it as ``<variable>_fit``. A fitted duration time goes in the legend."""
    x = result[result[variable].dims[0]].values
    group = f"trace-{index}"
    label = name(trace)
    if "taud" in result:
        label = f"{label}  tau_d={float(result['taud']):.3g} s"
    add_line(figure, index, x, result[variable].values, label, legendgroup=group)
    if f"{variable}_fit" in result:
        add_line(
            figure,
            index,
            x,
            result[f"{variable}_fit"].values,
            None,
            line=dict(dash="dash"),
            legendgroup=group,
            showlegend=False,
        )


def finish(figure, title, x_title, y_title, **layout):
    """The layout every statistic's figure shares."""
    figure.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
        title=title,
        **layout,
    )
    return figure


def correlation(trace, reference, params):
    """``(lags, curve)``: the normalised correlation of ``trace`` with
    ``reference``, cut to ``params.max_lag``."""
    import numpy as np
    from fppanalysis import corr_fun

    lags, curve = corr_fun(
        np.asarray(trace.value, dtype=float),
        np.asarray(reference.value, dtype=float),
        float(trace.dt),
        norm=True,
        biased=bool(params.biased),
    )
    lags = np.asarray(lags, dtype=float)
    curve = np.asarray(curve, dtype=float)
    keep = np.abs(lags) <= float(params.max_lag)
    return lags[keep], curve[keep]
