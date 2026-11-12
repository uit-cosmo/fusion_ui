"""The windowed time series themselves, one line per trace.

A trace off a raw file draws raw counts, one off a preprocessed file the
normalised series. Long traces are min/max-envelope decimated to a point budget
per trace: Plotly dies above roughly 50k points, and a full APD window is 583k
samples.
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
    time, values = decimate.envelope(
        np.asarray(trace.time, dtype=float),
        np.asarray(trace.value, dtype=float),
        int(params.max_points),
    )
    return xr.Dataset({"signal": ("time", values)}, coords={"time": time})


def _figure(series):
    """``[(name, time, signal), …]`` on one axis."""
    figure = go.Figure()
    for index, (label, time, signal) in enumerate(series):
        statistics.add_line(figure, index, time, signal, label)
    # The traces are decimated, so zooming the axes alone shows nothing new: a
    # drag must draw a selection box.
    return statistics.finish(
        figure,
        f"Time traces ({len(series)} traces)",
        "time [s]",
        "signal",
        dragmode="select",
    )


def figure(items, params):
    """The overview: every trace's decimated series on one axis. Pure."""
    return _figure(
        [
            (statistics.name(trace), result["time"].values, result["signal"].values)
            for trace, result in items
        ]
    )


def render(items, params):
    """The zoomable trace view, drawn into Streamlit; returns ``None``.

    A box selection cuts each trace's full-resolution data to that window and
    decimates it again, so a zoom shows what the overview's envelope left out.
    The zoom window is view state, never a parameter.
    """
    import streamlit as st

    zoom_key = "stats.trace_zoom"
    gen_key = "stats.trace_zoomgen"
    gen = st.session_state.get(gen_key, 0)
    window = st.session_state.get(zoom_key)

    dropped = 0
    if window is not None:
        cut = [
            decimate.slice_to_window(trace.time, trace.value, window[0], window[1])
            for trace, _ in items
        ]
        series = [
            (statistics.name(trace), *decimate.envelope(x, y, int(params.max_points)))
            for (trace, _), (x, y) in zip(items, cut)
            if len(x) >= 2
        ]
        dropped = len(items) - len(series)
        if not series:
            # A window left from another basket or time window: drop it.
            st.session_state.pop(zoom_key, None)
            window = None

    event = st.plotly_chart(
        figure(items, params) if window is None else _figure(series),
        on_select="rerun",
        selection_mode=["box", "lasso"],
        key=f"stats.trace.{'overview' if window is None else 'detail'}.{gen}",
        use_container_width=True,
    )
    ranged = decimate.selected_x_range(event)
    if ranged is not None and (window is None or tuple(ranged) != tuple(window)):
        decimate._apply_window(st, zoom_key, gen_key, gen, ranged)
        return None

    if window is None:
        total = sum(len(trace.time) for trace, _ in items)
        st.caption(
            f"{total} samples in window, one envelope-decimated trace per series. "
            "Drag a box to zoom in."
        )
        return None
    st.caption(
        f"Zoomed to {window[0]:.6f}–{window[1]:.6f} s"
        + (f" ({dropped} traces have no samples here)" if dropped else "")
        + ". Drag a box to zoom deeper."
    )
    if st.button("Reset zoom", key="stats.trace.reset"):
        st.session_state.pop(zoom_key, None)
        st.session_state[gen_key] = gen + 1
        st.rerun()
    return None


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
