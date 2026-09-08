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


from plotly.colors import qualitative

_CYCLE = qualitative.Plotly


def figure(items, params):
    """The overview: every trace's (decimated) series on one axis.

    Pure, like every other ``render`` -- this is what the zoomable
    :func:`render` draws first, and what unit tests assert on.
    """
    overview = go.Figure()
    for index, (trace, result) in enumerate(items):
        colour = _CYCLE[index % len(_CYCLE)]
        overview.add_trace(
            go.Scatter(
                x=result["time"].values,
                y=result["signal"].values,
                mode="lines",
                line=dict(color=colour),
                name=trace.label or trace.ref.key,
            )
        )
    overview.update_layout(
        xaxis_title="time [s]",
        yaxis_title="signal",
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
        title=f"Time traces ({len(items)} traces)",
        dragmode="select",
    )
    return overview


def render(items, params):
    """The zoomable trace view, drawn into Streamlit; returns ``None``.

    The overview draws the envelope-decimated series, and zooming the Plotly
    axes alone could never show more than the envelope kept -- so a box/lasso
    selection slices each trace's full-resolution data to that window first
    and the detail redraws off the slices, spending the whole point budget
    where the user is looking. The same loop
    :func:`fusion_ui.core.decimate.zoomable_trace` closes for one trace, done
    here for the whole basket at once. The zoom window is view state, never a
    parameter.
    """
    import streamlit as st

    from fusion_ui.core import decimate

    zoom_key = "stats.trace_zoom"
    gen_key = "stats.trace_zoomgen"
    gen = st.session_state.get(gen_key, 0)
    window = st.session_state.get(zoom_key)

    if window is not None:
        sliced = [
            (trace, decimate.slice_to_window(trace.time, trace.value,
                                             window[0], window[1]))
            for trace, _ in items
        ]
        kept = [(t, xy) for t, xy in sliced if len(xy[0]) >= 2]
        if not kept:
            # Stale window (the basket or window changed underneath it):
            # drop it rather than drawing empty traces.
            st.session_state.pop(zoom_key, None)
        else:
            detail = go.Figure()
            for index, (trace, (x, y)) in enumerate(kept):
                colour = _CYCLE[index % len(_CYCLE)]
                x, y = decimate.envelope(x, y, int(params.max_points))
                detail.add_trace(
                    go.Scatter(
                        x=x, y=y, mode="lines", line=dict(color=colour),
                        name=trace.label or trace.ref.key,
                    )
                )
            detail.update_layout(
                xaxis_title="time [s]", yaxis_title="signal", height=440,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", y=-0.2),
                title=f"Time traces ({len(kept)} traces)",
                # A plain drag must draw a selection box, not a client-side
                # zoom: the traces are decimated, so zooming the axes alone
                # shows nothing new.
                dragmode="select",
            )
            event = st.plotly_chart(
                detail, on_select="rerun", selection_mode=["box", "lasso"],
                key=f"stats.trace.detail.{gen}", use_container_width=True,
            )
            narrowed = decimate.selected_x_range(event)
            if narrowed is not None and tuple(narrowed) != tuple(window):
                # Remount under a new key, as the pixel selector does: the
                # chart's selection lives in widget state and would otherwise
                # overwrite the window that was just chosen on the next run.
                st.session_state[zoom_key] = (float(narrowed[0]),
                                              float(narrowed[1]))
                st.session_state[gen_key] = gen + 1
                st.rerun()
                return None
            dropped = len(sliced) - len(kept)
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

    event = st.plotly_chart(
        figure(items, params), on_select="rerun",
        selection_mode=["box", "lasso"], key=f"stats.trace.overview.{gen}",
        use_container_width=True,
    )
    ranged = decimate.selected_x_range(event)
    if ranged is not None and (window is None or tuple(ranged) != tuple(window)):
        st.session_state[zoom_key] = (float(ranged[0]), float(ranged[1]))
        st.session_state[gen_key] = gen + 1
        st.rerun()
        return None
    total = sum(len(trace.time) for trace, _ in items)
    st.caption(
        f"{total} samples in window, one envelope-decimated trace per series. "
        "Drag a box to zoom in."
    )
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
