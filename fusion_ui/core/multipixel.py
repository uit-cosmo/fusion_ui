"""Many-pixel mode: one analysis on a selection of pixels at once.

A per-pixel spec carries ``refx``/``refy`` in its params, so each pixel is its
own ``param_sets`` row, ``runs`` row and blob. This mode makes one params
instance per selected pixel and calls ``store.result`` once for each: the cache
is shared with single-pixel mode both ways. A live spec with an ``overlay`` (the
frame viewer) needs no run: its traces come off the open dataset.

The pixel set is view state, in ``st.session_state``, never a parameter. Each
pixel is a separate source on the multi-shot page.

The functions that draw import ``streamlit`` inside the function, so the rest
can be tested without it.
"""

import copy
import dataclasses
import statistics
from dataclasses import dataclass
from typing import Optional

from fusion_ui.core import decimate, loader, params_ui, store


def supported(spec) -> bool:
    """Whether ``spec`` can run in many-pixel mode.

    A live spec needs an ``overlay``. A cached spec needs ``refx`` and ``refy``
    in its params, and must be neither chained nor batch only: each pixel is one
    ``store.result`` call from a page, priced by that spec's own runs.
    """
    if not spec.cached:
        return spec.overlay is not None
    if spec.requires is not None or spec.batch_only:
        return False
    return {"refx", "refy"} <= {f.name for f in dataclasses.fields(spec.params)}


def with_pixel(params, x, y):
    """A copy of ``params`` with ``refx``/``refy`` set to ``(x, y)``.

    A copy, because the caller's instance backs the parameter form. Params
    without those fields come back unchanged.
    """
    stamped = copy.deepcopy(params)
    for name, value in (("refx", x), ("refy", y)):
        if hasattr(stamped, name):
            setattr(stamped, name, int(value))
    return stamped


def pixels_from_points(points) -> list:
    """Sorted, distinct ``(x, y)`` pairs from a Plotly selection's points.

    Each point of the selector carries ``[x, y]`` as its ``customdata``; a
    point without it is skipped.
    """
    pairs = set()
    for point in points or []:
        try:
            x, y = point["customdata"][:2]
            pairs.add((int(x), int(y)))
        except (TypeError, KeyError, ValueError):
            continue
    return sorted(pairs)


@dataclass(frozen=True)
class Estimate:
    total: int
    cached: int
    to_compute: int
    eta_seconds: Optional[float]


def median_seconds(conn, plot) -> Optional[float]:
    """Median ``runs.seconds`` over this plot's successful runs, or ``None``."""
    rows = conn.execute(
        "SELECT seconds FROM runs WHERE plot = ? AND status = 'ok'"
        " AND seconds IS NOT NULL",
        (plot,),
    ).fetchall()
    if not rows:
        return None
    return float(statistics.median(r["seconds"] for r in rows))


def estimate(conn, spec, target, params, pixels) -> Estimate:
    """What running ``spec`` on ``pixels`` would cost, from the ledger alone."""
    cached = 0
    for x, y in pixels:
        digest, _ = params_ui.hash_params(spec.key, with_pixel(params, x, y))
        run = store.find_run(conn, target, spec.key, digest)
        cached += run is not None and run["status"] == "ok"
    to_compute = len(pixels) - cached
    median = median_seconds(conn, spec.key)
    return Estimate(
        total=len(pixels),
        cached=cached,
        to_compute=to_compute,
        eta_seconds=None if median is None else median * to_compute,
    )


def describe(est: Estimate) -> str:
    """The one-line cost caption shown under the selector."""
    text = f"{est.total} pixels · {est.cached} cached"
    if est.to_compute == 0:
        return text + " · nothing to compute"
    if est.eta_seconds is None:
        return text + " · time unknown (never run)"
    return text + f" · ~{est.eta_seconds:.0f} s"


def scalar_data(items, spec):
    """``{name: [(x, y, value)]}`` over the per-pixel scalars of ``items``.

    Shot-level scalars (``str`` keys) are left out: they say nothing about
    where in the selection a value came from.
    """
    out = {}
    if spec.scalars is None:
        return out
    for (x, y), result in items:
        for key, value in spec.scalars(result).items():
            if isinstance(key, str):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            out.setdefault(str(key[2]), []).append((int(x), int(y), number))
    return out


def scalar_figure(items, spec, name):
    """One per-pixel scalar against the pixel order: what a spec without an
    ``overlay`` gets."""
    import plotly.graph_objects as go

    values = scalar_data(items, spec)[name]
    figure = go.Figure(
        go.Scatter(
            x=list(range(len(values))),
            y=[v for _, _, v in values],
            mode="markers+lines",
            customdata=[[x, y] for x, y, _ in values],
            hovertemplate="pixel (x=%{customdata[0]}, y=%{customdata[1]}): %{y}<extra></extra>",
            name=name,
        )
    )
    figure.update_layout(
        xaxis_title="pixel (selection order)",
        yaxis_title=name,
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{name} at {len(values)} pixels",
    )
    return figure


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def selector(ds, target) -> list:
    """The pixel map with box/lasso selection; returns the selected ``[(x, y), …]``.

    One marker per pixel at its ``(R, Z)`` (its indices when the file has no
    R/Z), carrying ``[x, y]`` as ``customdata``. Markers are required: a box
    over a lines-only trace selects no points.
    """
    import plotly.graph_objects as go
    import streamlit as st

    pix_key = f"pixels.{target.key}"
    gen_key = f"pixelgen.{target.key}"
    gen = st.session_state.get(gen_key, 0)

    r_grid, z_grid = loader.pixel_grid(ds)
    ny, nx = int(ds.sizes["y"]), int(ds.sizes["x"])
    pairs = [(x, y) for y in range(ny) for x in range(nx)]
    if r_grid is None:
        place = {pair: pair for pair in pairs}
        x_label, y_label = "x", "y"
    else:
        place = {(x, y): (float(r_grid[y, x]), float(z_grid[y, x])) for x, y in pairs}
        x_label, y_label = "R [cm]", "Z [cm]"

    current = [tuple(p) for p in st.session_state.get(pix_key, [])]
    chosen = [place[p] for p in current if p in place]

    figure = go.Figure(
        go.Scatter(
            x=[place[p][0] for p in pairs],
            y=[place[p][1] for p in pairs],
            mode="markers",
            marker=dict(size=9, color="steelblue"),
            customdata=[[x, y] for x, y in pairs],
            hovertemplate="pixel (x=%{customdata[0]}, y=%{customdata[1]})<extra></extra>",
            name="pixels",
        )
    )
    if current:
        figure.add_trace(
            go.Scatter(
                x=[r for r, _ in chosen],
                y=[z for _, z in chosen],
                mode="markers",
                marker=dict(size=13, color="orange", line=dict(width=1, color="black")),
                hoverinfo="skip",
                name="selected",
            )
        )
    figure.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        dragmode="select",
        legend=dict(orientation="h", y=-0.15),
        title=f"Pixel map — {len(current)} selected",
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    event = st.plotly_chart(
        figure,
        on_select="rerun",
        selection_mode=["box", "lasso"],
        key=f"pixelmap.{target.key}.{gen}",
        use_container_width=True,
    )

    def select(pixels):
        # Remount the chart under a new key: its selection lives in widget
        # state and would otherwise overwrite this one on the next run.
        if pixels:
            st.session_state[pix_key] = pixels
        else:
            st.session_state.pop(pix_key, None)
        st.session_state[gen_key] = gen + 1
        st.rerun()

    picked = pixels_from_points(decimate.selection_points(event))
    if picked and picked != sorted(current):
        select(picked)

    left, right = st.columns(2)
    if left.button("Select all", key=f"pixelmap.all.{target.key}"):
        select(sorted(pairs))
    if right.button("Clear", key=f"pixelmap.clear.{target.key}"):
        select([])
    return sorted(current)


def run_all(conn, spec, target, params, ds, pixels) -> list:
    """``store.result`` once per pixel, behind a progress bar.

    Returns ``[((x, y), result, run), …]`` in selection order. A pixel whose
    run fails is reported at the end and has ``result`` ``None``. Each pixel
    commits as it finishes.
    """
    import streamlit as st

    progress = st.progress(0, text="Starting…")
    items = []
    failures = []
    for index, (x, y) in enumerate(pixels):
        progress.progress(
            index / len(pixels), text=f"pixel {index + 1}/{len(pixels)} (x={x}, y={y})"
        )
        result, run = store.result(conn, spec, target, with_pixel(params, x, y), ds)
        items.append(((x, y), result, run))
        if result is None:
            reason = run["error"] if run is not None else "it produced nothing"
            failures.append(f"(x={x}, y={y}): {reason}")
    progress.progress(1.0, text=f"{len(pixels)}/{len(pixels)} pixels")
    if failures:
        st.warning(
            f"{len(failures)} of {len(pixels)} pixels failed and are left out"
            " of the figure:\n\n" + "\n\n".join(failures),
            icon="⚠️",
        )
    return items


def _live_overlay(ds, spec, target, params, pixels) -> None:
    """A live spec's overlay, where a box selection resamples.

    The overlay draws envelope-decimated traces, and zooming the axes alone
    shows no more than the envelope kept. So a box selection cuts the dataset
    to that window (lazily) and the overlay is redrawn from the cut. The zoom
    window is view state, keyed by ``Target.key``.
    """
    import streamlit as st

    zoom_key = f"multizoom.{target.key}"
    gen_key = f"multizoomgen.{target.key}"
    gen = st.session_state.get(gen_key, 0)
    window = st.session_state.get(zoom_key)

    view_ds = ds
    if window is not None:
        view_ds = loader.sliced(ds, window[0], window[1])
        if int(view_ds.sizes[loader.TIME_DIM]) < 2:
            # A window left from another target: drop it.
            st.session_state.pop(zoom_key, None)
            view_ds = ds
            window = None

    figure = spec.overlay([((x, y), view_ds) for x, y in pixels], params, target)
    # A drag must draw a selection box, not zoom the axes in the browser.
    figure.update_layout(dragmode="select")
    event = st.plotly_chart(
        figure,
        on_select="rerun",
        selection_mode=["box", "lasso"],
        key=f"multitraces.{target.key}.{gen}",
        use_container_width=True,
    )
    ranged = decimate.selected_x_range(event)
    if ranged is not None and (window is None or tuple(ranged) != tuple(window)):
        decimate._apply_window(st, zoom_key, gen_key, gen, ranged)

    samples = int(view_ds.sizes[loader.TIME_DIM])
    if window is None:
        st.caption(
            f"{samples} samples in window, one envelope-decimated trace per "
            "pixel. Drag a box to zoom in."
        )
        return
    st.caption(
        f"Zoomed to {window[0]:.6f}–{window[1]:.6f} s: {samples} samples in "
        "view, one envelope-decimated trace per pixel. "
        "Drag a box to zoom deeper."
    )
    if st.button("Reset zoom", key=f"multizoom.reset.{target.key}"):
        st.session_state.pop(zoom_key, None)
        st.session_state[gen_key] = gen + 1
        st.rerun()


def _items_key(spec, target, params, pixels):
    digest, _ = params_ui.hash_params(spec.key, params)
    return (
        f"multiitems.{target.key}.{spec.key}.{digest[:12]}."
        f"{hash(tuple(sorted(pixels))) & 0xFFFFFFFF:08x}"
    )


def view(conn, spec, target, params, ds) -> None:
    """Many-pixel mode: selector, estimate, run button, figure."""
    import streamlit as st

    pixels = selector(ds, target)
    if not pixels:
        st.info(
            "Drag a rectangle over the pixel map to select pixels, "
            "then run the analysis on all of them at once."
        )
        return

    if not spec.cached:
        _live_overlay(ds, spec, target, params, pixels)
        return

    st.caption(describe(estimate(conn, spec, target, params, pixels)))

    key = _items_key(spec, target, params, pixels)
    if st.button(
        f"Run on {len(pixels)} pixels", key=f"multirun.{spec.key}.{target.key}"
    ):
        items = run_all(conn, spec, target, params, ds, pixels)
        st.session_state[key] = [
            (pair, result) for pair, result, _ in items if result is not None
        ]
        st.rerun()

    cached = st.session_state.get(key)
    if not cached:
        st.info(
            f"Press **Run on {len(pixels)} pixels** to run {spec.label.lower()} "
            f"on the selection.",
            icon="▶️",
        )
        return

    if spec.overlay is not None:
        st.plotly_chart(spec.overlay(cached, params, target), use_container_width=True)
        return
    names = sorted(scalar_data(cached, spec))
    if not names:
        st.info(
            "The runs succeeded but carry no per-pixel scalars -- "
            "open one pixel in single-pixel mode instead."
        )
        return
    name = st.selectbox("Scalar", names, key=f"multiscalar.{spec.key}.{target.key}")
    st.plotly_chart(scalar_figure(cached, spec, name), use_container_width=True)
