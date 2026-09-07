"""Min/max envelope downsampling for 1D traces headed to Plotly.

Plotly dies above roughly 50k points, so every 1D trace goes through here on
its way to a chart -- the single most important performance detail in the
project. Naive striding (``y[::n]``) throws away spikes; bucketing the trace
and keeping both the min and the max of each bucket does not.

Because the overview is decimated, zooming the Plotly axes alone never shows
more detail than the envelope kept. :func:`zoomable_trace` closes that loop:
a box/lasso selection on the overview re-decimates the full-resolution data
to just that window, so the detail view spends the whole point budget where
the user is looking.
"""

import numpy as np


def envelope(x, y, max_points=4000):
    """``(x, y)`` reduced to at most ``max_points`` samples, spikes intact.

    The trace is split into ``max_points // 2`` buckets; each contributes the
    sample at its minimum and the sample at its maximum, in time order. A
    bucket that is all-NaN contributes nothing rather than a spurious point.
    Below ``max_points`` samples, the input is returned unchanged.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = y.shape[0]
    if n <= max_points or max_points < 2:
        return x, y

    n_buckets = max(1, max_points // 2)
    edges = np.linspace(0, n, n_buckets + 1).astype(int)

    out_idx = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        chunk = y[start:stop]
        if np.all(np.isnan(chunk)):
            continue
        lo = start + int(np.nanargmin(chunk))
        hi = start + int(np.nanargmax(chunk))
        out_idx.append(lo)
        out_idx.append(hi)

    if not out_idx:
        return x[:0], y[:0]

    order = np.unique(out_idx)  # sorted, and de-duplicates lo == hi
    return x[order], y[order]


def selection_points(event):
    """The selected points of a ``st.plotly_chart`` event, robustly.

    ``PlotlyState`` supports both dict and attribute access, and older code
    paths hand back either -- try each spelling rather than assuming one.
    Anything unrecognised is no selection, never an exception in the page.
    """
    if event is None:
        return []
    try:
        if hasattr(event, "__getitem__"):
            try:
                selection = event["selection"]
            except Exception:
                selection = getattr(event, "selection", None)
        else:
            selection = getattr(event, "selection", None)
    except Exception:
        return []
    if selection is None:
        return []
    try:
        if hasattr(selection, "__getitem__"):
            try:
                points = selection["points"]
            except Exception:
                points = getattr(selection, "points", [])
        else:
            points = getattr(selection, "points", [])
    except Exception:
        return []
    if not points:
        return []
    try:
        return list(points)
    except TypeError:
        return []


def selected_x_range(event):
    """``(x0, x1)`` spanned by the selected points' x values, or ``None``.

    A box/lasso selection on a decimated trace returns the displayed points
    inside the area; their x extent is the zoom window to re-decimate the
    full-resolution data to. Uses the documented ``points`` list only, never
    the box metadata. Needs at least two distinct x values -- a single
    point-click is not a zoom.
    """
    xs = []
    for point in selection_points(event):
        try:
            value = point.get("x")
        except AttributeError:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            xs.append(number)
    if len(xs) < 2:
        return None
    lo, hi = min(xs), max(xs)
    if lo == hi:
        return None
    return lo, hi


def slice_to_window(x, y, x0, x1):
    """Full-resolution ``(x, y)`` restricted to ``[x0, x1]``.

    An empty selection window stays empty rather than raising; callers decide
    whether that means "nothing to draw" or "fall back to the full trace".
    """
    x = np.asarray(x)
    y = np.asarray(y)
    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
    mask = (x >= lo) & (x <= hi)
    return x[mask], y[mask]


def zoomable_trace(
    x,
    y,
    key,
    x_label="time [s]",
    y_label="signal",
    height=280,
    max_points=4000,
):
    """An overview trace with box-select-to-zoom, drawn into Streamlit.

    The overview shows the envelope-decimated full trace; dragging a box (or
    lasso) over it stores that x window in ``st.session_state`` and reruns,
    and the detail chart below re-decimates the full-resolution data to just
    that window -- so zooming in spends the whole point budget where the user
    is looking. The detail chart is itself selectable, so zooming iterates;
    **Reset zoom** clears the stored window and remounts both charts.

    ``key`` namespaces the charts, the stored window and the reset button, so
    callers pass something unique per trace (target plus pixel, or quantity
    plus position). The zoom window is view state, like the selected pixel:
    it never reaches a params hash.
    """
    import plotly.graph_objects as go
    import streamlit as st

    x = np.asarray(x)
    y = np.asarray(y)
    zoom_key = f"zoom.{key}"
    gen_key = f"zoomgen.{key}"
    gen = st.session_state.get(gen_key, 0)
    window = st.session_state.get(zoom_key)

    ov_x, ov_y = envelope(x, y, max_points)
    overview = go.Figure(go.Scatter(x=ov_x, y=ov_y, mode="lines"))
    overview.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        # A plain drag must draw a selection box, not a client-side zoom:
        # zooming the axes alone can never show more than the envelope kept,
        # so without this the drag gesture the caption asks for silently does
        # nothing on the server.
        dragmode="select",
    )
    ov_event = st.plotly_chart(
        overview,
        on_select="rerun",
        selection_mode=["box", "lasso"],
        key=f"{key}.overview.{gen}",
        use_container_width=True,
    )
    ranged = selected_x_range(ov_event)
    if ranged is not None and (
        window is None or tuple(ranged) != tuple(window)
    ):
        st.session_state[zoom_key] = (float(ranged[0]), float(ranged[1]))
        st.rerun()
        return

    if window is None:
        st.caption(
            f"{x.size} samples in window, {ov_x.size} plotted after "
            "min/max-envelope decimation. Drag a box to zoom in."
        )
        return

    zx, zy = slice_to_window(x, y, window[0], window[1])
    if zx.size < 2:
        # Stale window (the trace changed underneath it): drop it rather
        # than drawing an empty detail.
        st.session_state.pop(zoom_key, None)
        st.caption(
            f"{x.size} samples in window, {ov_x.size} plotted after "
            "min/max-envelope decimation. Drag a box to zoom in."
        )
        return

    d_x, d_y = envelope(zx, zy, max_points)
    detail = go.Figure(go.Scatter(x=d_x, y=d_y, mode="lines"))
    detail.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        dragmode="select",  # as above: drag selects, it must not just zoom
    )
    d_event = st.plotly_chart(
        detail,
        on_select="rerun",
        selection_mode=["box", "lasso"],
        key=f"{key}.detail.{gen}",
        use_container_width=True,
    )
    narrowed = selected_x_range(d_event)
    if narrowed is not None and tuple(narrowed) != tuple(window):
        st.session_state[zoom_key] = (float(narrowed[0]), float(narrowed[1]))
        st.rerun()
        return

    st.caption(
        f"Zoomed to {window[0]:.6f}–{window[1]:.6f} s: {zx.size} samples in "
        f"view, {d_x.size} plotted. Drag a box to zoom deeper."
    )
    if st.button("Reset zoom", key=f"{key}.reset"):
        st.session_state.pop(zoom_key, None)
        st.session_state[gen_key] = gen + 1
        st.rerun()
