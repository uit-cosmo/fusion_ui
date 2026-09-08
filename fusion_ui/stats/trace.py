"""The windowed time series themselves, one line per trace.

Raw or preprocessed is a property of the file each trace came from, not of
this view: a trace off a raw file draws raw counts, one off a preprocessed
file draws the normalised series. Long traces are min/max-envelope decimated
to a per-trace point budget on the way in -- Plotly dies above roughly 50k
points, and a full APD window is 583k samples.
"""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import xarray as xr

from fusion_ui.core import decimate, statistics


@dataclass
class TraceParams:
    """
    max_points: Points plotted per trace, after min/max-envelope decimation.
    """

    max_points: int = 4000


def compute(trace, params):
    """The trace's windowed series, decimated for Plotly."""
    time = np.asarray(trace.time, dtype=float)
    values = np.asarray(trace.value, dtype=float)
    time, values = decimate.envelope(time, values, int(params.max_points))
    return xr.Dataset(
        {"signal": ("time", np.asarray(values, dtype=float))},
        coords={"time": np.asarray(time, dtype=float)},
    )


def render(items, params):
    """One line per trace, colour-cycled."""
    from plotly.colors import qualitative

    cycle = qualitative.Plotly
    figure = go.Figure()
    for index, (trace, result) in enumerate(items):
        colour = cycle[index % len(cycle)]
        figure.add_trace(
            go.Scatter(
                x=result["time"].values,
                y=result["signal"].values,
                mode="lines",
                line=dict(color=colour),
                name=trace.label or trace.ref.key,
            )
        )
    figure.update_layout(
        xaxis_title="time [s]",
        yaxis_title="signal",
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
        title=f"Time traces ({len(items)} traces)",
    )
    return figure


SPEC = statistics.register(
    statistics.StatSpec(
        key="trace",
        label="Time trace",
        params=TraceParams,
        compute=compute,
        render=render,
        description=(
            "The windowed series themselves, as stored -- raw or preprocessed "
            "depending on the file each trace came from."
        ),
    )
)
