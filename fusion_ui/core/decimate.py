"""Min/max envelope downsampling for 1D traces headed to Plotly.

Plotly dies above roughly 50k points, so every 1D trace goes through here on
its way to a chart. Striding (``y[::n]``) throws away spikes; bucketing the
trace and keeping the min and the max of each bucket does not.

Zooming the Plotly axes of a decimated trace never shows more detail than the
envelope kept. :func:`zoomable_trace` closes that loop: a box or lasso
selection re-decimates the full-resolution data to that window.
"""

import numpy as np


def envelope(x, y, max_points=4000):
    """``(x, y)`` reduced to at most ``max_points`` samples, spikes intact.

    The trace is split into ``max_points // 2`` buckets; each contributes the
    sample at its minimum and the sample at its maximum, in time order. An
    all-NaN bucket contributes nothing. At or below ``max_points`` samples the
    input is returned unchanged.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = y.shape[0]
    if n <= max_points or max_points < 2:
        return x, y

    edges = np.linspace(0, n, max_points // 2 + 1).astype(int)
    kept = []
    for start, stop in zip(edges[:-1], edges[1:]):
        chunk = y[start:stop]
        if not np.all(np.isnan(chunk)):
            kept += [start + int(np.nanargmin(chunk)), start + int(np.nanargmax(chunk))]
    order = np.unique(np.asarray(kept, dtype=int))  # sorted, and one point where min == max
    return x[order], y[order]


def _selection_field(event, name):
    """One list of a ``st.plotly_chart`` event's selection (``points``, ``box``
    or ``lasso``); ``[]`` for no event or no such list."""
    try:
        return list(event["selection"][name])
    except (KeyError, TypeError):
        return []


def selection_points(event):
    """The selected points of a ``st.plotly_chart`` event."""
    return _selection_field(event, "points")


def _span(values):
    """``(lo, hi)`` of the finite numbers in ``values``, or ``None`` when they
    do not span a range."""
    numbers = []
    for value in values:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            pass
    numbers = [number for number in numbers if np.isfinite(number)]
    if not numbers or min(numbers) == max(numbers):
        return None
    return min(numbers), max(numbers)


def selected_x_range(event):
    """The zoom window ``(x0, x1)`` a box or lasso selection asks for, or ``None``.

    The drawn shape wins over the points it caught: it is the window the user
    indicated, not clipped to the nearest kept sample, and it is all Plotly
    reports for a trace with neither markers nor text. A single point is not a
    zoom.
    """
    shapes = _selection_field(event, "box") + _selection_field(event, "lasso")
    drawn = _span(x for shape in shapes for x in shape.get("x") or ())
    return drawn or _span(point.get("x") for point in selection_points(event))


def slice_to_window(x, y, x0, x1):
    """Full-resolution ``(x, y)`` restricted to ``[x0, x1]``, in either order;
    empty when nothing lies inside."""
    x = np.asarray(x)
    y = np.asarray(y)
    lo, hi = min(x0, x1), max(x0, x1)
    mask = (x >= lo) & (x <= hi)
    return x[mask], y[mask]


#: A line trace Plotly also reports *points* for: it reports none for a trace
#: with neither markers nor text. Size-1, opacity-0 markers are invisible and
#: make the points list a fallback for the drawn box.
_SELECTABLE_LINE = dict(
    mode="lines+markers",
    marker=dict(size=1, opacity=0),
)


def _apply_window(st, zoom_key, gen_key, gen, ranged):
    """Store a new zoom window and remount both charts.

    Each chart keeps its selection in widget state under its own key, so
    without a remount the chart that did *not* set the new window still reports
    its previous rectangle on the next run and overwrites the window just
    chosen.
    """
    st.session_state[zoom_key] = (float(ranged[0]), float(ranged[1]))
    st.session_state[gen_key] = gen + 1
    st.rerun()


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

    The overview shows the envelope of the full trace. Dragging a box or lasso
    over it stores that x window in ``st.session_state`` and reruns, and a
    detail chart below shows the envelope of the full-resolution data in that
    window. The detail chart is selectable too, so zooming iterates; **Reset
    zoom** clears the window.

    ``key`` namespaces the charts, the stored window and the reset button, so
    it must be unique per trace. The zoom window is view state: it never
    reaches a params hash.
    """
    import plotly.graph_objects as go
    import streamlit as st

    x = np.asarray(x)
    y = np.asarray(y)
    zoom_key = f"zoom.{key}"
    gen_key = f"zoomgen.{key}"
    gen = st.session_state.get(gen_key, 0)
    window = st.session_state.get(zoom_key)

    def chart(name, chart_x, chart_y):
        """Draw one envelope; the number of points drawn and the window selected on it."""
        env_x, env_y = envelope(chart_x, chart_y, max_points)
        figure = go.Figure(go.Scatter(x=env_x, y=env_y, **_SELECTABLE_LINE))
        figure.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=height,
            margin=dict(l=10, r=10, t=20, b=10),
            # A plain drag must draw a selection box, which reaches the server,
            # not zoom the axes in the browser.
            dragmode="select",
        )
        event = st.plotly_chart(
            figure,
            on_select="rerun",
            selection_mode=["box", "lasso"],
            key=f"{key}.{name}.{gen}",
            use_container_width=True,
        )
        return env_x.size, selected_x_range(event)

    plotted, ranged = chart("overview", x, y)
    if ranged is not None and ranged != window:
        _apply_window(st, zoom_key, gen_key, gen, ranged)

    if window is not None:
        zoom_x, zoom_y = slice_to_window(x, y, window[0], window[1])
        if zoom_x.size < 2:  # the trace changed under a stored window: drop it
            st.session_state.pop(zoom_key, None)
            window = None
    if window is None:
        st.caption(
            f"{x.size} samples in window, {plotted} plotted after "
            "min/max-envelope decimation. Drag a box to zoom in."
        )
        return

    plotted, narrowed = chart("detail", zoom_x, zoom_y)
    if narrowed is not None and narrowed != window:
        _apply_window(st, zoom_key, gen_key, gen, narrowed)

    st.caption(
        f"Zoomed to {window[0]:.6f}–{window[1]:.6f} s: {zoom_x.size} samples in "
        f"view, {plotted} plotted. Drag a box to zoom deeper."
    )
    if st.button("Reset zoom", key=f"{key}.reset"):
        st.session_state.pop(zoom_key, None)
        st.session_state[gen_key] = gen + 1
        st.rerun()
