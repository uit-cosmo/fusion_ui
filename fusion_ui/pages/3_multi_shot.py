"""Multi shot: one scalar against an axis, across a selection of shots.

Reads the scalar store (:func:`fusion_ui.core.store.scalar_frame`) and the shot
catalog's metadata, collapses per-pixel scalars to one number per shot, and
draws the scatter coloured by confinement mode. Clicking a point jumps to that
shot's single-shot view with the parameters that produced it, so an outlier is
one click from being explained.

The page owns no analysis of its own: the scalar names come from whatever the
store already holds -- the ``density_scan`` seed, or results computed through
the single-shot view or ``fusion-ui precompute``.
"""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

import fusion_ui.plots  # noqa: F401 - importing the package registers every spec
from fusion_ui import config, ui
from fusion_ui.core import multishot, params_ui, registry, store

st.set_page_config(page_title="Multi shot · Shot Explorer", layout="wide")


def source_label(source):
    plot, params_hash, diagnostic, preprocessed = source
    kind = "preprocessed" if preprocessed else "raw"
    return f"{plot} · {diagnostic} ({kind}) · {params_hash[:8]}"


def jump_to_single_shot(conn, machine, shot, source):
    """Set the selection (and, for a real spec, the parameters) and navigate.

    The single-shot page reads ``selection`` and, when ``spec.{diagnostic}`` is
    set, that plot; seeding the parameter widgets from the stored ``params_json``
    and marking the run ready means the click lands directly on the plot that
    produced the point, from cache. Seeded imports (plot ``density_scan_import``)
    have no spec to restore, so the jump is just the shot.
    """
    plot, params_hash, diagnostic, preprocessed = source
    st.session_state["selection"] = {
        "machine": machine,
        "shot": shot,
        "diagnostic": diagnostic,
        "preprocessed": bool(preprocessed),
    }

    if plot in registry.REGISTRY:
        spec = registry.get(plot)
        st.session_state[f"spec.{diagnostic}"] = spec
        text = store.params_json(conn, params_hash)
        if text:
            body = json.loads(text)
            params = params_ui.from_dict(spec.params, body["params"]["values"])
            params_ui.seed_session_state(st.session_state, f"params.{spec.key}", params)
        target_key = f"{machine}_{shot}_{diagnostic}_{'p' if preprocessed else 'r'}"
        st.session_state[f"ready.{spec.key}.{target_key}"] = True

    st.switch_page("pages/2_single_shot.py")


def main():
    st.session_state.setdefault("selection", None)
    st.session_state.setdefault("shot_selection", [])

    st.title("Multi shot")

    conn = ui.get_connection()
    # One machine at a time. `rescan` only ever deletes its own machine's rows,
    # so an index can hold two machines at once (the single-shot page picks
    # between them); shot numbers are not unique across machines, and neither
    # the selection carried over from the browser nor the metadata join is
    # keyed by machine, so an unscoped frame would silently overlay two
    # different discharges on one point.
    machine = config.MACHINE
    frame = store.scalar_frame(conn, machine=machine)
    if frame.empty:
        st.warning(
            f"No results are stored for machine `{machine}` yet. Compute "
            "something on the single-shot page, run `fusion-ui import-results` "
            "to load the density_scan seed, or `fusion-ui precompute` to warm "
            "the cache overnight.",
            icon="⚠️",
        )
        return

    names = multishot.distinct_names(frame)

    # ---- sidebar: what to plot -------------------------------------------
    name = st.sidebar.selectbox("Scalar", names, key="ms.scalar")

    sources = multishot.distinct_sources(frame, name)
    if "ms.source" in st.session_state and st.session_state["ms.source"] not in sources:
        st.session_state["ms.source"] = sources[0]
    source = st.sidebar.selectbox(
        "Source", sources, format_func=source_label, key="ms.source"
    )

    shot_level = multishot.is_shot_level(frame, source, name)
    how = "mean"
    if not shot_level:
        how = st.sidebar.selectbox(
            "Aggregate",
            list(multishot.AGGREGATES),
            format_func=lambda key: multishot.AGGREGATES[key],
            key="ms.aggregate",
        )

    pixel = None
    if how == "pixel":
        pairs = multishot.pixel_choices(frame, source, name)
        if not pairs:
            st.sidebar.warning("No pixel-indexed values for this source.")
            return
        xs = sorted({x for x, _ in pairs})
        if (
            "ms.pixel.x" in st.session_state
            and st.session_state["ms.pixel.x"] not in xs
        ):
            st.session_state["ms.pixel.x"] = xs[0]
        x = st.sidebar.selectbox("x", xs, key="ms.pixel.x")
        ys = sorted({y for px, y in pairs if px == x})
        if (
            "ms.pixel.y" in st.session_state
            and st.session_state["ms.pixel.y"] not in ys
        ):
            st.session_state["ms.pixel.y"] = ys[0]
        y = st.sidebar.selectbox("y", ys, key="ms.pixel.y")
        pixel = (x, y)

    x_axis = st.sidebar.selectbox(
        "x-axis",
        list(multishot.X_AXES),
        format_func=lambda key: multishot.X_AXES[key][1],
        key="ms.xaxis",
    )

    # ---- shot selection, carried over from the browser -------------------
    shot_selection = st.session_state.get("shot_selection") or []
    if shot_selection:
        st.sidebar.caption(
            f"{len(shot_selection)} shots carried over from the browser."
        )
        if st.sidebar.button("Clear selection", use_container_width=True):
            st.session_state["shot_selection"] = []
            st.rerun()
    else:
        st.sidebar.caption("No selection carried over — using every stored shot.")

    # ---- the number itself ----------------------------------------------
    aggregated = multishot.aggregate(frame, source, name, how, pixel)
    if shot_selection:
        selected = set(shot_selection)
        aggregated = aggregated[aggregated["shot"].isin(selected)]

    if aggregated.empty:
        st.info("No stored values fall in this selection.", icon="ℹ️")
        return

    merged = multishot.with_metadata(aggregated, ui.cached_shot_table())
    column, x_label = multishot.X_AXES[x_axis]
    x = (
        merged["shot"].astype(int)
        if x_axis == "shot"
        else pd.to_numeric(merged[column], errors="coerce")
    )
    plot_frame = merged.assign(
        _x=x,
        _mode=merged["mode"].fillna("").replace("", "no metadata"),
    )
    plot_frame = plot_frame[pd.notna(plot_frame["_x"]) & pd.notna(plot_frame["value"])]

    if plot_frame.empty:
        st.info(
            f"No shot in this selection has a value for {name!r} on this axis.",
            icon="ℹ️",
        )
        return

    st.caption(
        f"{name} · {multishot.AGGREGATES.get(how, how)} · "
        f"{len(plot_frame)} shots on `{machine}` · click a point to open that shot"
    )

    figure = px.scatter(
        plot_frame,
        x="_x",
        y="value",
        color="_mode",
        custom_data=["machine", "shot"],
        hover_name="shot",
        hover_data={
            "machine": False,
            "shot": False,
            "f_GW": True,
            "n_e_bar": True,
            "I_p": True,
        },
        labels={
            "_x": x_label,
            "value": f"{name} ({multishot.AGGREGATES.get(how, how)})",
            "_mode": "confinement mode",
        },
    )
    figure.update_layout(legend=dict(orientation="h", y=-0.3), height=520)

    event = st.plotly_chart(
        figure,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="ms.scatter",
    )
    selection = getattr(event, "selection", None)
    if selection is not None and selection.points:
        machine, shot = selection.points[0]["customdata"]
        jump_to_single_shot(conn, machine, int(shot), source)


main()
