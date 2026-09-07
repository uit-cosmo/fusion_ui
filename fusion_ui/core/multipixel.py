"""Many-pixel mode: one cached analysis, run on a rectangle of pixels at once.

Every per-pixel analysis in this app is one pixel at a time: ``refx``/``refy``
sit in the params, so each pixel is already its own ``param_sets`` row, ``runs``
row and blob. Comparing twelve pixels means twelve visits to the sidebar and
twelve figures that cannot be overlaid.

This module closes that gap the cheap way: **N separate runs, not one
region-run.** The selected rectangle mints one params instance per pixel
(``refx``/``refy`` stamped) and calls ``store.result`` once per pixel. That
means no schema change, no ``params_ui`` change and no new spec -- and the
cache is shared both ways, so a pixel computed in single-pixel mode is free
here and a pixel computed here is a hit when opened singly. The ``requires``
chain works untouched, because ``upstream_params`` carries ``refx``/``refy``
and each pixel gets its own upstream run.

The pixel set itself is view state, in ``st.session_state``, never in a hash --
which is the registry's rule, and correct here because the thing that actually
varies the answer, the pixel, is in each individual run's own hash.

Known limitation: N pixel-runs are N distinct ``params_hash`` values, so
``multishot.distinct_sources`` lists them as N separate sources on the
multi-shot page. This feature does not aggregate a rectangle into one point
there; that is a follow-up.

Mirrors :mod:`fusion_ui.core.multishot`: the pure logic is unit-testable, and
the few functions that draw import ``streamlit`` inside the function, never at
module top.
"""

import copy
import dataclasses
import statistics
from dataclasses import dataclass
from typing import Optional

from fusion_ui.core import decimate, loader, params_ui, registry, store

#: The params fields that name the reference pixel. A spec is eligible for
#: many-pixel mode when one dataclass in its params tree carries both.
PIXEL_FIELDS = ("refx", "refy")


# ---------------------------------------------------------------------------
# Pure
# ---------------------------------------------------------------------------


def _tree_has_pixel_pair(params_cls, _seen=None):
    """Whether some dataclass in ``params_cls``' tree carries refx *and* refy."""
    _seen = _seen if _seen is not None else set()
    if not (isinstance(params_cls, type) and dataclasses.is_dataclass(params_cls)):
        return False
    if params_cls in _seen:
        return False
    _seen.add(params_cls)
    try:
        fields = dataclasses.fields(params_cls)
    except TypeError:
        return False
    names = {f.name for f in fields}
    if all(n in names for n in PIXEL_FIELDS):
        return True
    try:
        hints = params_ui._hints(params_cls)
    except Exception:  # noqa: BLE001 - an unresolvable tree is not eligible
        return False
    return any(
        _tree_has_pixel_pair(hints.get(f.name, f.type), _seen) for f in fields
    )


def supported(spec) -> bool:
    """Whether ``spec`` can run in many-pixel mode.

    True for a cached spec whose params tree carries a ``refx``/``refy`` pair,
    and for a live spec that brings its own ``overlay`` (today just the frame
    viewer: nothing to stamp and nothing to cache, each trace comes off the
    already-open dataset). Correctly excludes ``probe_trace`` (live, no
    overlay) and ``velocity_field`` (deliberately no ``refx``/``refy`` -- it
    sweeps every pixel internally, so there is nothing to stamp).
    """
    if not spec.cached:
        return spec.overlay is not None
    return _tree_has_pixel_pair(spec.params)


def with_pixel(params, x, y):
    """A deep copy of ``params`` with ``refx``/``refy`` stamped to ``(x, y)``.

    The copy, never the input: the caller's instance backs the still-visible
    parameter form and must not move under it. This is ``precompute._set_pixel``
    promoted; ``precompute.default_params`` calls it.
    """
    stamped = copy.deepcopy(params)
    _stamp(stamped, int(x), int(y))
    return stamped


def _stamp(params, x, y):
    for f in dataclasses.fields(params):
        value = getattr(params, f.name)
        if dataclasses.is_dataclass(value):
            _stamp(value, x, y)
        elif f.name == "refx":
            setattr(params, f.name, x)
        elif f.name == "refy":
            setattr(params, f.name, y)


def pixels_from_points(points) -> list:
    """``(x, y)`` pairs off a Plotly selection event's customdata.

    ``points`` is what :func:`fusion_ui.core.decimate.selection_points`
    returns. Each point carries the ``[x, y]`` index pair the selector stored
    as its ``customdata``; anything malformed is skipped rather than failing
    the page. De-duplicated, sorted.
    """
    pairs = set()
    for point in points or []:
        custom = None
        try:
            custom = point["customdata"]
        except (TypeError, KeyError, IndexError):
            custom = getattr(point, "customdata", None)
        if custom is None:
            try:
                custom = point.get("customdata", point.get("customData"))
            except AttributeError:
                continue
        try:
            x, y = int(custom[0]), int(custom[1])
        except (TypeError, ValueError, IndexError):
            continue
        pairs.add((x, y))
    return sorted(pairs)


@dataclass(frozen=True)
class Estimate:
    total: int
    cached: int
    to_compute: int
    seconds_per_pixel: Optional[float]
    eta_seconds: Optional[float]


def median_seconds(conn, plot) -> Optional[float]:
    """Median ``runs.seconds`` over this plot's successful runs, or ``None``."""
    rows = conn.execute(
        "SELECT seconds FROM runs WHERE plot = ? AND status = 'ok'"
        " AND seconds IS NOT NULL",
        (plot,),
    ).fetchall()
    values = [r["seconds"] for r in rows if r["seconds"] is not None]
    if not values:
        return None
    return float(statistics.median(values))


def _chain(spec):
    """``[spec, its upstream, …]`` down the ``requires`` links."""
    chain = [spec]
    while chain[-1].requires is not None:
        chain.append(registry.get(chain[-1].requires))
    return chain


def _hashes_along_chain(chain, params):
    """``[(spec, params, params_hash)]`` down the chain from one pixel's params.

    Each link's parameters come out of the one above it through
    ``upstream_params`` -- the same derivation the store uses at compute time,
    so the estimate looks up the same ledger rows the run would hit.
    """
    out = []
    current = params
    for index, spec in enumerate(chain):
        if index > 0:
            current = chain[index - 1].upstream_params(current)
        digest, _ = params_ui.hash_params(spec.key, current)
        out.append((spec, current, digest))
    return out


def estimate(conn, spec, target, params, pixels) -> Estimate:
    """What running ``spec`` on ``pixels`` would cost, from the ledger alone.

    No file is opened: each pixel's hashes are looked up with
    ``store.find_run``, the same trick ``precompute.run`` uses to skip hits.
    For a chained spec each link's own median is added for pixels that would
    miss it -- without that the 2DCA chain reads "~2 min" when it means
    "~30 min", because ``runs.seconds`` on the downstream excludes its
    upstream by design.
    """
    chain = _chain(spec)
    medians = {link.key: median_seconds(conn, link.key) for link in chain}

    cached = 0
    eta = 0.0
    known = False
    for x, y in pixels:
        links = _hashes_along_chain(chain, with_pixel(params, x, y))
        run = store.find_run(conn, target, links[0][0].key, links[0][2])
        if run is not None and run["status"] == "ok":
            cached += 1
            continue
        for link, _, digest in links:
            # The downstream always runs on a miss; an upstream link only runs
            # when its own row is missing too.
            if link is not chain[0]:
                upstream_run = store.find_run(conn, target, link.key, digest)
                if upstream_run is not None and upstream_run["status"] == "ok":
                    continue
            median = medians[link.key]
            if median is not None:
                eta += median
                known = True

    total = len(pixels)
    to_compute = total - cached
    if to_compute > 0:
        seconds_per_pixel = eta / to_compute if known else None
        eta_seconds = eta if known else None
    else:
        whole = sum(m for m in medians.values() if m is not None)
        seconds_per_pixel = whole if any(
            m is not None for m in medians.values()
        ) else None
        eta_seconds = 0.0
    return Estimate(
        total=total,
        cached=cached,
        to_compute=to_compute,
        seconds_per_pixel=seconds_per_pixel,
        eta_seconds=eta_seconds,
    )


def describe(est: Estimate) -> str:
    """The one-line cost caption the mode shows under the selector."""
    text = f"{est.total} pixels · {est.cached} cached"
    if est.to_compute == 0:
        return text + " · nothing to compute"
    if est.eta_seconds is None:
        return text + " · time unknown (never run)"
    return text + f" · ~{est.eta_seconds:.0f} s"


def scalar_data(items, spec):
    """``{name: [(x, y, value)]}`` over one overlay run's successful results.

    Only the ``(x, y, name)`` entries -- the per-pixel ones. Shot-level scalars
    (the ``x = y = -1`` sentinel) say nothing about where in the rectangle a
    value came from, so the fallback leaves them out.
    """
    out = {}
    for (x, y), result in items:
        if spec.scalars is None:
            continue
        for key, value in spec.scalars(result).items():
            if isinstance(key, str):
                continue
            _, _, name = key
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            out.setdefault(str(name), []).append((int(x), int(y), number))
    return out


def scalar_figure(items, spec, name=None):
    """The universal fallback: one scalar's value against the pixel order.

    Every eligible spec with ``scalars`` works in many-pixel mode from day one
    through this -- ``velocity_contour`` gives ``vx_c`` across the rectangle,
    ``fwhm_sizes`` gives ``lr``/``lz`` -- without any of them needing an
    ``overlay`` function. Pure, like ``render``.
    """
    import plotly.graph_objects as go

    data = scalar_data(items, spec)
    names = sorted(data)
    figure = go.Figure()
    if not names:
        figure.add_annotation(text="No per-pixel scalars for this plot.")
        figure.update_layout(
            height=380, margin=dict(l=10, r=10, t=40, b=10),
            title="Many pixels",
        )
        return figure
    chosen = name if name in data else names[0]
    values = data[chosen]
    figure.add_trace(
        go.Scatter(
            x=list(range(len(values))),
            y=[v for _, _, v in values],
            mode="markers+lines",
            customdata=[[x, y] for x, y, _ in values],
            hovertemplate="pixel (x=%{customdata[0]}, y=%{customdata[1]}): %{y}<extra></extra>",
            name=chosen,
        )
    )
    figure.update_layout(
        xaxis_title="pixel (selection order)",
        yaxis_title=chosen,
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{chosen} at {len(values)} pixels",
    )
    return figure


# ---------------------------------------------------------------------------
# Drawing (Streamlit inside the function, never at module top)
# ---------------------------------------------------------------------------


def selector(ds, target, height=380) -> list:
    """The pixel map with box/lasso selection, drawn into Streamlit.

    One marker per pixel at its ``(R, Z)`` (index axes when there is no R/Z),
    carrying ``[x, y]`` as its ``customdata``. A markers trace is required: a
    lines-only trace reports no selected points for a box. Returns the selected
    ``[(x, y), …]`` from session state -- the pixel set is view state, never a
    parameter.
    """
    import plotly.graph_objects as go
    import streamlit as st

    pix_key = f"pixels.{target.key}"
    gen_key = f"pixelgen.{target.key}"
    gen = st.session_state.get(gen_key, 0)

    r_grid, z_grid = loader.pixel_grid(ds)
    ny, nx = int(ds.sizes["y"]), int(ds.sizes["x"])
    if r_grid is None:
        xs = [x for _ in range(ny) for x in range(nx)]
        ys = [y for y in range(ny) for _ in range(nx)]
        x_label, y_label = "x", "y"
    else:
        xs = [float(r_grid[y, x]) for y in range(ny) for x in range(nx)]
        ys = [float(z_grid[y, x]) for y in range(ny) for x in range(nx)]
        x_label, y_label = "R [cm]", "Z [cm]"
    pairs = [(x, y) for y in range(ny) for x in range(nx)]

    current = [tuple(p) for p in st.session_state.get(pix_key, [])]

    figure = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=9, color="steelblue"),
            customdata=[[x, y] for x, y in pairs],
            hovertemplate="pixel (x=%{customdata[0]}, y=%{customdata[1]})<extra></extra>",
            name="pixels",
        )
    )
    if current:
        lookup = {pair: index for index, pair in enumerate(pairs)}
        figure.add_trace(
            go.Scatter(
                x=[xs[lookup[p]] for p in current if p in lookup],
                y=[ys[lookup[p]] for p in current if p in lookup],
                mode="markers",
                marker=dict(
                    size=13, color="orange", line=dict(width=1, color="black")
                ),
                hoverinfo="skip",
                name="selected",
            )
        )
    figure.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
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
    picked = pixels_from_points(decimate.selection_points(event))
    if picked and picked != sorted(current):
        # Remount the chart under a new key: its selection lives in widget
        # state, so without this the previous rectangle overwrites the new one
        # on the next run -- the same generation-bump as decimate._apply_window.
        st.session_state[pix_key] = picked
        st.session_state[gen_key] = gen + 1
        st.rerun()
        return picked

    left, right = st.columns(2)
    if left.button("Select all", key=f"pixelmap.all.{target.key}"):
        st.session_state[pix_key] = sorted(pairs)
        st.session_state[gen_key] = gen + 1
        st.rerun()
    if right.button("Clear", key=f"pixelmap.clear.{target.key}"):
        st.session_state.pop(pix_key, None)
        st.session_state[gen_key] = gen + 1
        st.rerun()
    return sorted(current)


def run_all(conn, spec, target, params, ds, pixels) -> list:
    """``store.result`` once per pixel, driving a progress bar.

    A pixel whose run comes back ``failed`` is collected and reported at the
    end, never fatal -- the ``velocity_field`` philosophy ("one NaN pixel
    among many") applied at the run level. Each pixel commits as it finishes,
    so a browser refresh mid-run loses only the pixel in flight. Returns
    ``[((x, y), result, run), …]`` in selection order.
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
    """A live overlay with box-select-to-zoom that resamples, not just zooms.

    The overlay draws envelope-decimated traces, and zooming the Plotly axes
    alone can never show more than the envelope kept (``open_issues.md`` #3).
    So a box/lasso selection slices the dataset to that window first and the
    overlay redraws off the slice -- the whole point budget spent where the
    user is looking. That is the loop :func:`decimate.zoomable_trace` closes
    for one trace, done here without the spec's help: the slice is lazy, so
    zooming never loads the full axis, and any present or future live overlay
    gets it for free. The zoom window is view state keyed off ``Target.key``,
    like the pixel set.
    """
    import streamlit as st

    zoom_key = f"multizoom.{target.key}"
    gen_key = f"multizoomgen.{target.key}"
    gen = st.session_state.get(gen_key, 0)
    window = st.session_state.get(zoom_key)
    has_time = loader.TIME_DIM in ds.dims

    full = int(ds.sizes[loader.TIME_DIM]) if has_time else 0
    view_ds = ds
    if window is not None and has_time:
        view_ds = loader.sliced(ds, window[0], window[1])
        if int(view_ds.sizes[loader.TIME_DIM]) < 2:
            # Stale window (the target changed underneath it): drop it rather
            # than drawing empty traces.
            st.session_state.pop(zoom_key, None)
            view_ds = ds
            window = None

    figure = spec.overlay([((x, y), view_ds) for x, y in pixels], params, target)
    # A plain drag must draw a selection box, not a client-side zoom: the
    # traces are decimated, so zooming the axes alone shows nothing new.
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
        # Remount under a new key, as the pixel selector does: the chart's
        # selection lives in widget state and would otherwise overwrite the
        # window that was just chosen on the next run.
        st.session_state[zoom_key] = (float(ranged[0]), float(ranged[1]))
        st.session_state[gen_key] = gen + 1
        st.rerun()
        return

    shown = int(view_ds.sizes[loader.TIME_DIM]) if has_time else 0
    if window is None:
        st.caption(
            f"{full} samples in window, one envelope-decimated trace per "
            "pixel. Drag a box to zoom in."
        )
        return
    st.caption(
        f"Zoomed to {window[0]:.6f}–{window[1]:.6f} s: {shown} samples in "
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
    """Assemble many-pixel mode: selector, estimate, run button, figure."""
    import streamlit as st

    pixels = selector(ds, target)
    if not pixels:
        st.info(
            "Drag a rectangle over the pixel map to select pixels, "
            "then run the analysis on all of them at once."
        )
        return

    if not spec.cached:
        # Live: nothing to stamp, nothing to compute, nothing to cache. Each
        # pixel's data comes off the already-open dataset, so draw the
        # overlay straight away -- no estimate, no run button.
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
        return

    cached = st.session_state.get(key)
    if not cached:
        st.info(
            f"Press **Run on {len(pixels)} pixels** to run {spec.label.lower()} "
            f"on the selection.",
            icon="▶️",
        )
        return

    if spec.overlay is not None:
        st.plotly_chart(
            spec.overlay(cached, params, target), use_container_width=True
        )
    elif spec.scalars is not None:
        names = sorted(scalar_data(cached, spec))
        if not names:
            st.info(
                "The runs succeeded but carry no per-pixel scalars -- "
                "open one pixel in single-pixel mode instead."
            )
            return
        name = st.selectbox(
            "Scalar", names, key=f"multiscalar.{spec.key}.{target.key}"
        )
        st.plotly_chart(
            scalar_figure(cached, spec, name), use_container_width=True
        )
    else:
        st.success(f"{len(cached)} of {len(pixels)} pixels ran.", icon="✅")
        st.caption(
            "This plot writes no per-pixel scalars and has no overlay -- "
            "open one pixel in single-pixel mode to see its figure."
        )
