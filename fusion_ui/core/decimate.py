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


def _selection(event):
    """The selection state of a ``st.plotly_chart`` event, or ``None``.

    ``PlotlyState`` supports both dict and attribute access, and older code
    paths hand back either -- try each spelling rather than assuming one.
    Anything unrecognised is no selection, never an exception in the page.
    """
    if event is None:
        return None
    try:
        if hasattr(event, "__getitem__"):
            try:
                return event["selection"]
            except Exception:
                return getattr(event, "selection", None)
        return getattr(event, "selection", None)
    except Exception:
        return None


def _selection_field(event, name):
    """One list off the selection state (``points``, ``box``, ``lasso``)."""
    selection = _selection(event)
    if selection is None:
        return []
    try:
        if hasattr(selection, "__getitem__"):
            try:
                value = selection[name]
            except Exception:
                value = getattr(selection, name, [])
        else:
            value = getattr(selection, name, [])
    except Exception:
        return []
    if not value:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def selection_points(event):
    """The selected points of a ``st.plotly_chart`` event, robustly."""
    return _selection_field(event, "points")


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def selected_shape_range(event):
    """``(x0, x1)`` of the drawn box/lasso, or ``None``.

    This -- not the points list -- is what a drag over a decimated **line**
    trace gives us. Plotly's ``selectPoints`` returns nothing for a trace
    with no markers and no text (``src/traces/scatter/select.js``), so a
    box drawn over ``mode="lines"`` reports zero selected points while still
    reporting the rectangle itself in ``selection["box"]``. Reading only the
    points was why zooming never resampled.
    """
    xs = []
    for shape in _selection_field(event, "box") + _selection_field(event, "lasso"):
        try:
            coords = shape["x"]
        except (TypeError, KeyError, IndexError):
            coords = getattr(shape, "x", None)
        if coords is None:
            continue
        try:
            coords = list(coords)
        except TypeError:
            continue
        for value in coords:
            number = _finite(value)
            if number is not None:
                xs.append(number)
    if len(xs) < 2:
        return None
    lo, hi = min(xs), max(xs)
    return None if lo == hi else (lo, hi)


def selected_points_range(event):
    """``(x0, x1)`` spanned by the selected points' x values, or ``None``.

    Needs at least two distinct x values -- a single point-click is not a
    zoom. Only traces that carry markers or text ever populate this.
    """
    xs = []
    for point in selection_points(event):
        try:
            value = point.get("x")
        except AttributeError:
            continue
        number = _finite(value)
        if number is not None:
            xs.append(number)
    if len(xs) < 2:
        return None
    lo, hi = min(xs), max(xs)
    return None if lo == hi else (lo, hi)


def selected_x_range(event):
    """The zoom window a box/lasso selection asks for, or ``None``.

    The drawn rectangle wins over the points it caught: it is the window the
    user actually indicated, it is not clipped to the nearest kept sample,
    and on a lines-only trace it is the only thing reported at all.
    """
    shaped = selected_shape_range(event)
    if shaped is not None:
        return shaped
    return selected_points_range(event)


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


#: A line trace Plotly will also report *points* for. ``selectPoints`` in
#: ``src/traces/scatter/select.js`` bails out on a trace with neither markers
#: nor text, so a plain ``mode="lines"`` trace reports an empty points list
#: for any box drawn over it. Size-1, opacity-0 markers are invisible and cost
#: nothing to draw, and they make the points list a real fallback for the
#: box metadata.
_SELECTABLE_LINE = dict(
    mode="lines+markers",
    marker=dict(size=1, opacity=0),
)


def _apply_window(st, zoom_key, gen_key, gen, ranged):
    """Store a new zoom window and remount both charts.

    The generation bump is not cosmetic. Both charts keep their selection in
    widget state under their own key, so without a remount the chart that did
    *not* set the new window still reports its previous rectangle on the next
    run -- and immediately overwrites the window that was just chosen. Zooming
    out from the detail view back to a wider box on the overview is exactly
    that case.
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
    overview = go.Figure(go.Scatter(x=ov_x, y=ov_y, **_SELECTABLE_LINE))
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
    if ranged is not None and (window is None or tuple(ranged) != tuple(window)):
        _apply_window(st, zoom_key, gen_key, gen, ranged)
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
    detail = go.Figure(go.Scatter(x=d_x, y=d_y, **_SELECTABLE_LINE))
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
        _apply_window(st, zoom_key, gen_key, gen, narrowed)
        return

    st.caption(
        f"Zoomed to {window[0]:.6f}–{window[1]:.6f} s: {zx.size} samples in "
        f"view, {d_x.size} plotted. Drag a box to zoom deeper."
    )
    if st.button("Reset zoom", key=f"{key}.reset"):
        st.session_state.pop(zoom_key, None)
        st.session_state[gen_key] = gen + 1
        st.rerun()
